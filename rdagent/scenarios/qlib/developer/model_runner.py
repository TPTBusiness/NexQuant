import pandas as pd
from typing import Optional

from rdagent.app.qlib_rd_loop.conf import ModelBasePropSetting
from rdagent.components.runner import CachedRunner
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.exception import ModelEmptyError
from rdagent.core.utils import cache_with_pickle
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.qlib.developer.utils import process_factor_data
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment


class QlibModelRunner(CachedRunner[QlibModelExperiment]):
    """
    Docker run
    Everything in a folder
    - config.yaml
    - Pytorch `model.py`
    - results in `mlflow`

    https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_nn.py
    - pt_model_uri:  hard-code `model.py:Net` in the config
    - let LLM modify model.py
    """

    @cache_with_pickle(CachedRunner.get_cache_key, CachedRunner.assign_cached_result)
    def develop(self, exp: QlibModelExperiment) -> QlibModelExperiment:
        if exp.based_experiments and exp.based_experiments[-1].result is None:
            exp.based_experiments[-1] = self.develop(exp.based_experiments[-1])

        exist_sota_factor_exp = False
        if exp.based_experiments:
            SOTA_factor = None
            # Filter and retain only QlibFactorExperiment instances
            sota_factor_experiments_list = [
                base_exp for base_exp in exp.based_experiments if isinstance(base_exp, QlibFactorExperiment)
            ]
            if len(sota_factor_experiments_list) > 1:
                logger.info(f"SOTA factor processing ...")
                SOTA_factor = process_factor_data(sota_factor_experiments_list)

            if SOTA_factor is not None and not SOTA_factor.empty:
                exist_sota_factor_exp = True
                combined_factors = SOTA_factor
                combined_factors = combined_factors.sort_index()
                combined_factors = combined_factors.loc[:, ~combined_factors.columns.duplicated(keep="last")]
                new_columns = pd.MultiIndex.from_product([["feature"], combined_factors.columns])
                combined_factors.columns = new_columns
                num_features = str(len(exp.base_features) + len(combined_factors.columns))

                target_path = exp.experiment_workspace.workspace_path / "combined_factors_df.parquet"

                # Save the combined factors to the workspace
                combined_factors.to_parquet(target_path, engine="pyarrow")

        if exp.sub_workspace_list[0].file_dict.get("model.py") is None:
            raise ModelEmptyError("model.py is empty")
        # to replace & inject code
        exp.experiment_workspace.inject_files(**{"model.py": exp.sub_workspace_list[0].file_dict["model.py"]})

        mbps = ModelBasePropSetting()
        env_to_use = {
            "PYTHONPATH": "./",
            "train_start": mbps.train_start,
            "train_end": mbps.train_end,
            "valid_start": mbps.valid_start,
            "valid_end": mbps.valid_end,
            "test_start": mbps.test_start,
            "feature_names": str(list(exp.base_features.keys())),
            "feature_expressions": str(list(exp.base_features.values())),
        }
        if mbps.test_end is not None:
            env_to_use.update({"test_end": mbps.test_end})

        training_hyperparameters = exp.sub_tasks[0].training_hyperparameters
        if training_hyperparameters:
            env_to_use.update(
                {
                    "n_epochs": str(training_hyperparameters.get("n_epochs", "100")),
                    "lr": str(training_hyperparameters.get("lr", "2e-4")),
                    "early_stop": str(training_hyperparameters.get("early_stop", 10)),
                    "batch_size": str(training_hyperparameters.get("batch_size", 256)),
                    "weight_decay": str(training_hyperparameters.get("weight_decay", 0.0001)),
                }
            )

        logger.info(f"start to run {exp.sub_tasks[0].name} model")
        if exp.sub_tasks[0].model_type == "TimeSeries":
            if exist_sota_factor_exp:
                env_to_use.update(
                    {"dataset_cls": "TSDatasetH", "num_features": num_features, "step_len": 20, "num_timesteps": 20}
                )
                result, stdout = exp.experiment_workspace.execute(
                    qlib_config_name="conf_sota_factors_model.yaml", run_env=env_to_use
                )
            else:
                env_to_use.update({"dataset_cls": "TSDatasetH", "step_len": 20, "num_timesteps": 20})
                result, stdout = exp.experiment_workspace.execute(
                    qlib_config_name="conf_baseline_factors_model.yaml", run_env=env_to_use
                )
        elif exp.sub_tasks[0].model_type == "Tabular":
            if exist_sota_factor_exp:
                env_to_use.update({"dataset_cls": "DatasetH", "num_features": num_features})
                result, stdout = exp.experiment_workspace.execute(
                    qlib_config_name="conf_sota_factors_model.yaml", run_env=env_to_use
                )
            else:
                env_to_use.update({"dataset_cls": "DatasetH"})
                result, stdout = exp.experiment_workspace.execute(
                    qlib_config_name="conf_baseline_factors_model.yaml", run_env=env_to_use
                )

        exp.result = result
        exp.stdout = stdout

        if result is None:
            logger.error(
                f"Failed to run {exp.sub_tasks[0].name} model (result is None), because {stdout}"
            )
            # Save failed run info for debugging
            self._save_failed_run(exp, stdout, error_type="result_none")
            raise ModelEmptyError(f"Failed to run {exp.sub_tasks[0].name} model, because {stdout}")

        # Validate result before proceeding
        validation_result = self._validate_result(exp, result)
        if validation_result.get("has_issues"):
            logger.warning(
                f"Model result validation warnings for '{exp.sub_tasks[0].name}': "
                f"{validation_result['warnings']}"
            )
            self._save_failed_run(exp, stdout, error_type="validation_warnings", validation=validation_result)

        # Save results to database immediately after Docker execution
        try:
            self._save_result_to_database(exp, result)
        except Exception as e:
            logger.warning(f"Failed to save model results to database: {e}")

        return exp

    def _save_result_to_database(self, exp, result) -> None:
        """
        Save model backtest results to the ResultsDatabase.

        Parameters
        ----------
        exp : QlibModelExperiment
            The experiment with backtest results
        result : dict or pd.Series
            Backtest metrics from Qlib (qlib_res.csv)
        """
        try:
            import pandas as pd
            from rdagent.components.backtesting import ResultsDatabase

            # Get model/factor name from hypothesis
            factor_name = "unknown"
            if hasattr(exp, 'hypothesis') and exp.hypothesis is not None:
                factor_name = getattr(exp.hypothesis, 'hypothesis', 'unknown')

            # Extract metrics from result (pd.Series from qlib_res.csv)
            metrics = {}
            if isinstance(result, pd.Series):
                metrics['ic'] = self._safe_float(result.get('IC', None))
                metrics['sharpe_ratio'] = self._safe_float(
                    result.get('1day.excess_return_with_cost.shar',
                    result.get('1day.excess_return_with_cost.sharpe', None))
                )
                metrics['annualized_return'] = self._safe_float(
                    result.get('1day.excess_return_with_cost.annualized_return', None)
                )
                metrics['max_drawdown'] = self._safe_float(
                    result.get('1day.excess_return_with_cost.max_drawdown', None)
                )
                metrics['win_rate'] = self._safe_float(result.get('win_rate', None))
                metrics['information_ratio'] = self._safe_float(
                    result.get('1day.excess_return_with_cost.information_ratio', None)
                )
                metrics['volatility'] = self._safe_float(
                    result.get('1day.excess_return_with_cost.std',
                    result.get('1day.excess_return_with_cost.volatility', None))
                )
            elif isinstance(result, dict):
                metrics['ic'] = self._safe_float(result.get('IC', result.get('ic', None)))
                metrics['sharpe_ratio'] = self._safe_float(
                    result.get('sharpe', result.get('sharpe_ratio', None))
                )
                metrics['annualized_return'] = self._safe_float(result.get('annualized_return', None))
                metrics['max_drawdown'] = self._safe_float(result.get('max_drawdown', None))
                metrics['win_rate'] = self._safe_float(result.get('win_rate', None))
                metrics['information_ratio'] = None
                metrics['volatility'] = None

            # Only save if we have at least IC or Sharpe
            if metrics.get('ic') is None and metrics.get('sharpe_ratio') is None:
                logger.debug(f"No valid IC/Sharpe for model {factor_name}, skipping DB save")
                return

            # Log warnings about result quality
            self._log_result_warnings(factor_name, result, metrics)

            # Save to database
            db = ResultsDatabase()
            try:
                run_id = db.add_backtest(factor_name=factor_name[:100], metrics=metrics)
                logger.info(
                    f"Model result saved to DB: {factor_name[:50]} "
                    f"(IC={metrics.get('ic')}, Sharpe={metrics.get('sharpe_ratio')}, run_id={run_id})"
                )
            finally:
                db.close()

        except Exception as e:
            logger.warning(f"Database save failed for model {getattr(exp.hypothesis, 'hypothesis', 'unknown')}: {e}")

    def _log_result_warnings(self, factor_name: str, result, metrics: dict) -> None:
        """Log warnings about model result quality before saving."""
        warnings_list = []

        ic = metrics.get('ic')
        if ic is None:
            warnings_list.append("IC is None — model has no predictive power")
        elif abs(ic) < 0.001:
            warnings_list.append(f"IC near zero ({ic:.6f})")

        if isinstance(result, pd.Series):
            pos_value = result.get('1day.pos', None)
            if pos_value is not None:
                try:
                    pos_float = float(pos_value)
                    if pos_float == 0:
                        warnings_list.append(
                            "1day.pos == 0 — ZERO positions! Check config topk=1."
                        )
                except (ValueError, TypeError):
                    pass

        if warnings_list:
            for warn_msg in warnings_list:
                logger.warning(f"[MODEL {factor_name[:50]}] {warn_msg}")

    def _safe_float(self, value):
        """Safely convert value to float, returning None for invalid values."""
        import pandas as pd
        if value is None:
            return None
        try:
            f = float(value)
            if pd.isna(f) or f == float('inf') or f == float('-inf'):
                return None
            return f
        except (ValueError, TypeError):
            return None

    def _validate_result(self, exp, result) -> dict:
        """
        Validate model backtest result for common issues before saving.

        Parameters
        ----------
        exp : QlibModelExperiment
            The experiment with backtest results
        result : pd.Series or dict
            Backtest metrics from Qlib

        Returns
        -------
        dict
            Validation result with 'has_issues' (bool), 'warnings' (list), and 'details' (dict)
        """
        warnings = []
        details = {}
        model_name = exp.sub_tasks[0].name if exp.sub_tasks else "unknown"

        if isinstance(result, pd.Series):
            # Check IC
            ic_value = result.get('IC', None)
            details['ic_raw'] = ic_value
            if ic_value is None or (isinstance(ic_value, float) and (ic_value != ic_value)):
                warnings.append("IC is None/NaN — model has no predictive power")
            else:
                try:
                    ic_float = float(ic_value)
                    details['ic'] = ic_float
                    if abs(ic_float) < 0.001:
                        warnings.append(f"IC is near zero ({ic_float:.6f})")
                except (ValueError, TypeError):
                    warnings.append(f"IC value is not numeric: {ic_value}")

            # Check positions
            pos_value = result.get('1day.pos', None)
            details['positions_raw'] = pos_value
            if pos_value is not None:
                try:
                    pos_float = float(pos_value)
                    details['positions'] = pos_float
                    if pos_float == 0:
                        warnings.append(
                            "1day.pos == 0 — model opened ZERO positions. "
                            "Check Qlib config: topk=1 for single-asset."
                        )
                except (ValueError, TypeError):
                    pass

            non_null_count = result.notna().sum()
            details['non_null_metrics'] = int(non_null_count)
            if non_null_count < 3:
                warnings.append(f"Only {non_null_count} non-null metrics — likely empty results")

        elif isinstance(result, dict):
            ic_value = result.get('IC', result.get('ic', None))
            details['ic_raw'] = ic_value
            if ic_value is None:
                warnings.append("IC is None — model has no predictive power")

        return {
            "has_issues": len(warnings) > 0,
            "warnings": "; ".join(warnings),
            "details": details,
        }

    def _save_failed_run(self, exp, stdout: str, error_type: str = "unknown",
                         validation: Optional[dict] = None) -> None:
        """
        Save failed model run information to results/failed_runs.json.

        Parameters
        ----------
        exp : QlibModelExperiment
            The experiment that failed
        stdout : str
            Standard output from Docker execution
        error_type : str
            Type of error
        validation : dict, optional
            Validation result dict
        """
        import json
        from datetime import datetime
        from pathlib import Path

        try:
            # 5 levels up to project root
            project_root = Path(__file__).parent.parent.parent.parent.parent
            failed_dir = project_root / "results" / "failed_runs"
            failed_dir.mkdir(parents=True, exist_ok=True)

            model_name = exp.sub_tasks[0].name if exp.sub_tasks else "unknown"
            factor_name = "unknown"
            if hasattr(exp, 'hypothesis') and exp.hypothesis is not None:
                factor_name = getattr(exp.hypothesis, 'hypothesis', model_name)

            failed_record = {
                "timestamp": datetime.now().isoformat(),
                "factor_name": f"[MODEL] {factor_name}",
                "model_name": model_name,
                "error_type": error_type,
                "stdout": stdout if stdout else "(empty)",
                "validation": validation,
                "experiment_details": {
                    "model_type": exp.sub_tasks[0].model_type if exp.sub_tasks else "unknown",
                    "hypothesis": factor_name,
                },
            }

            failed_file = failed_dir / "failed_runs.json"
            existing_records = []
            if failed_file.exists():
                try:
                    existing_records = json.loads(failed_file.read_text(encoding="utf-8"))
                    if not isinstance(existing_records, list):
                        existing_records = [existing_records]
                except (json.JSONDecodeError, Exception):
                    existing_records = []

            existing_records.append(failed_record)
            if len(existing_records) > 500:
                existing_records = existing_records[-500:]

            failed_file.write_text(
                json.dumps(existing_records, indent=2, default=str, ensure_ascii=False),
                encoding="utf-8"
            )
            logger.info(f"Failed model run saved: {model_name} (type={error_type}) → {failed_file}")

        except Exception as e:
            logger.warning(f"Could not save failed model run info: {e}")
