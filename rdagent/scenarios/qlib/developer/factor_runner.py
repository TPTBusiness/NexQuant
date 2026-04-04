from pathlib import Path
"""
Qlib Factor Runner - Executes factor backtests in Docker.

NOTE: The @cache_with_pickle decorator was REMOVED from develop() because:
- Backtests should ALWAYS run fresh — caching causes stale results
- Each hypothesis may have different code even with same task info
- Docker-level caching (QlibDockerConf.enable_cache=False) is sufficient
- The pickle cache caused 240+ factor generations but ZERO Docker backtests
"""
from pathlib import Path
from typing import Optional

import pandas as pd
from pandarallel import pandarallel


pandarallel.initialize(verbose=1)

from rdagent.app.qlib_rd_loop.conf import FactorBasePropSetting
from rdagent.components.runner import CachedRunner
from rdagent.core.exception import FactorEmptyError
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.qlib.developer.utils import process_factor_data
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment

DIRNAME = Path(__file__).absolute().resolve().parent
DIRNAME_local = Path.cwd()

# TODO: supporting multiprocessing and keep previous results


class QlibFactorRunner(CachedRunner[QlibFactorExperiment]):
    """
    Docker run
    Everything in a folder
    - config.yaml
    - price-volume data dumper
    - `data.py` + Adaptor to Factor implementation
    - results in `mlflow`
    """

    def calculate_information_coefficient(
        self, concat_feature: pd.DataFrame, SOTA_feature_column_size: int, new_feature_columns_size: int
    ) -> pd.DataFrame:
        res = pd.Series(index=range(SOTA_feature_column_size * new_feature_columns_size))
        for col1 in range(SOTA_feature_column_size):
            for col2 in range(SOTA_feature_column_size, SOTA_feature_column_size + new_feature_columns_size):
                res.loc[col1 * new_feature_columns_size + col2 - SOTA_feature_column_size] = concat_feature.iloc[
                    :, col1
                ].corr(concat_feature.iloc[:, col2])
        return res

    def deduplicate_new_factors(self, SOTA_feature: pd.DataFrame, new_feature: pd.DataFrame) -> pd.DataFrame:
        # calculate the IC between each column of SOTA_feature and new_feature
        # if the IC is larger than a threshold, remove the new_feature column
        # return the new_feature

        concat_feature = pd.concat([SOTA_feature, new_feature], axis=1)
        IC_max = (
            concat_feature.groupby("datetime")
            .parallel_apply(
                lambda x: self.calculate_information_coefficient(x, SOTA_feature.shape[1], new_feature.shape[1])
            )
            .mean()
        )
        IC_max.index = pd.MultiIndex.from_product([range(SOTA_feature.shape[1]), range(new_feature.shape[1])])
        IC_max = IC_max.unstack().max(axis=0)
        return new_feature.iloc[:, IC_max[IC_max < 0.99].index]

    def develop(self, exp: QlibFactorExperiment) -> QlibFactorExperiment:
        """
        Generate the experiment by processing and combining factor data,
        then passing the combined data to Docker for backtest results.

        NOTE: @cache_with_pickle decorator was REMOVED. Every experiment
        triggers a fresh Docker backtest — no cached results are used.
        """
        # Ensure all results directories exist
        self._ensure_results_dirs()

        if exp.based_experiments and exp.based_experiments[-1].result is None:
            logger.info(f"Baseline experiment execution ...")
            exp.based_experiments[-1] = self.develop(exp.based_experiments[-1])

        fbps = FactorBasePropSetting()
        env_to_use = {
            "PYTHONPATH": "./",
            "train_start": fbps.train_start,
            "train_end": fbps.train_end,
            "valid_start": fbps.valid_start,
            "valid_end": fbps.valid_end,
            "test_start": fbps.test_start,
            "feature_names": str(list(exp.base_features.keys())),
            "feature_expressions": str(list(exp.base_features.values())),
        }
        if fbps.test_end is not None:
            env_to_use.update({"test_end": fbps.test_end})

        if exp.based_experiments:
            SOTA_factor = None
            # Filter and retain only QlibFactorExperiment instances
            sota_factor_experiments_list = [
                base_exp for base_exp in exp.based_experiments if isinstance(base_exp, QlibFactorExperiment)
            ]
            if len(sota_factor_experiments_list) > 1:
                logger.info(f"SOTA factor processing ...")
                SOTA_factor = process_factor_data(sota_factor_experiments_list)

            # Process the new factors data
            logger.info(f"New factor processing ...")
            new_factors = process_factor_data(exp)

            if new_factors.empty:
                raise FactorEmptyError("Factors failed to run on the full sample, this round of experiment failed.")

            # Combine the SOTA factor and new factors if SOTA factor exists
            if SOTA_factor is not None and not SOTA_factor.empty:
                new_factors = self.deduplicate_new_factors(SOTA_factor, new_factors)
                if new_factors.empty:
                    raise FactorEmptyError(
                        "The factors generated in this round are highly similar to the previous factors. Please change the direction for creating new factors."
                    )
                combined_factors = pd.concat([SOTA_factor, new_factors], axis=1).dropna()
            else:
                combined_factors = new_factors

            # Sort and nest the combined factors under 'feature'
            combined_factors = combined_factors.sort_index()
            combined_factors = combined_factors.loc[:, ~combined_factors.columns.duplicated(keep="last")]
            new_columns = pd.MultiIndex.from_product([["feature"], combined_factors.columns])
            combined_factors.columns = new_columns
            logger.info(f"Factor data processing completed.")

            num_features = len(exp.base_features) + len(combined_factors.columns)

            # Due to the rdagent and qlib docker image in the numpy version of the difference,
            # the `combined_factors_df.pkl` file could not be loaded correctly in qlib dokcer,
            # so we changed the file type of `combined_factors_df` from pkl to parquet.
            target_path = exp.experiment_workspace.workspace_path / "combined_factors_df.parquet"

            # Save the combined factors to the workspace
            combined_factors.to_parquet(target_path, engine="pyarrow")

            # If model exp exists in the previous experiment
            exist_sota_model_exp = False
            for base_exp in reversed(exp.based_experiments):
                if isinstance(base_exp, QlibModelExperiment):
                    sota_model_exp = base_exp
                    exist_sota_model_exp = True
                    break
            logger.info(f"Experiment execution ...")
            if exist_sota_model_exp:
                exp.experiment_workspace.inject_files(
                    **{"model.py": sota_model_exp.sub_workspace_list[0].file_dict["model.py"]}
                )
                sota_training_hyperparameters = sota_model_exp.sub_tasks[0].training_hyperparameters
                if sota_training_hyperparameters:
                    env_to_use.update(
                        {
                            "n_epochs": str(sota_training_hyperparameters.get("n_epochs", "100")),
                            "lr": str(sota_training_hyperparameters.get("lr", "2e-4")),
                            "early_stop": str(sota_training_hyperparameters.get("early_stop", 10)),
                            "batch_size": str(sota_training_hyperparameters.get("batch_size", 256)),
                            "weight_decay": str(sota_training_hyperparameters.get("weight_decay", 0.0001)),
                        }
                    )
                sota_model_type = sota_model_exp.sub_tasks[0].model_type
                if sota_model_type == "TimeSeries":
                    env_to_use.update(
                        {"dataset_cls": "TSDatasetH", "num_features": num_features, "step_len": 20, "num_timesteps": 20}
                    )
                elif sota_model_type == "Tabular":
                    env_to_use.update({"dataset_cls": "DatasetH", "num_features": num_features})

                # model + combined factors
                result, stdout = exp.experiment_workspace.execute(
                    qlib_config_name="conf_combined_factors_sota_model.yaml", run_env=env_to_use
                )
            else:
                # LGBM + combined factors
                result, stdout = exp.experiment_workspace.execute(
                    qlib_config_name="conf_combined_factors.yaml",
                    run_env=env_to_use,
                )
        else:
            logger.info(f"Experiment execution ...")
            if exp.base_feature_codes:
                factors = process_factor_data(exp)
                factors = factors.sort_index()
                factors = factors.loc[:, ~factors.columns.duplicated(keep="last")]
                new_columns = pd.MultiIndex.from_product([["feature"], factors.columns])
                factors.columns = new_columns
                target_path = exp.experiment_workspace.workspace_path / "combined_factors_df.parquet"
                # Save the combined factors to the workspace
                factors.to_parquet(target_path, engine="pyarrow")
                logger.info(f"Factor data processing completed.")
                result, stdout = exp.experiment_workspace.execute(
                    qlib_config_name="conf_combined_factors.yaml",
                    run_env=env_to_use,
                )
            else:
                result, stdout = exp.experiment_workspace.execute(
                    qlib_config_name="conf_baseline.yaml",
                    run_env=env_to_use,
                )

        if result is None:
            logger.error(
                f"Failed to run this experiment (result is None). "
                f"Factor: {getattr(exp.hypothesis, 'hypothesis', 'unknown')}"
            )
            # Save failed run info for debugging
            self._save_failed_run(exp, stdout, error_type="result_none")
            raise FactorEmptyError(f"Failed to run this experiment, because {stdout}")

        # Validate result before proceeding
        validation_result = self._validate_result(exp, result)
        if validation_result.get("has_issues"):
            logger.warning(
                f"Result validation warnings for factor '{getattr(exp.hypothesis, 'hypothesis', 'unknown')}': "
                f"{validation_result['warnings']}"
            )
            # Save warning info for debugging
            self._save_failed_run(exp, stdout, error_type="validation_warnings", validation=validation_result)

        exp.result = result
        exp.stdout = stdout

        # Protection Manager: Check if factor passes risk criteria
        try:
            self._run_protection_check(exp, result)
        except Exception as e:
            logger.warning(f"Protection check failed for factor {exp.hypothesis.hypothesis}: {e}")
            # Don't block the workflow, just log the warning

        # Save results to database immediately after Docker execution
        try:
            self._save_result_to_database(exp, result)
        except Exception as e:
            logger.warning(f"Failed to save results to database: {e}")

        # Always write a log entry for every run (success or failure)
        self._write_run_log(exp, result)

        return exp

    def _validate_result(self, exp, result) -> dict:
        """
        Validate backtest result for common issues before saving.

        Checks for:
        - Empty/None IC (no predictive power)
        - Zero positions (1day.pos == 0, model stayed neutral)
        - All metrics being None/NaN

        Parameters
        ----------
        exp : QlibFactorExperiment
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

        factor_name = "unknown"
        if hasattr(exp, 'hypothesis') and exp.hypothesis is not None:
            factor_name = getattr(exp.hypothesis, 'hypothesis', 'unknown')

        if isinstance(result, pd.Series):
            # Check IC
            ic_value = result.get('IC', None)
            details['ic_raw'] = ic_value
            if ic_value is None or (isinstance(ic_value, float) and (ic_value != ic_value)):  # NaN check
                warnings.append("IC is None/NaN — factor has no predictive power")
            else:
                try:
                    ic_float = float(ic_value)
                    details['ic'] = ic_float
                    if abs(ic_float) < 0.001:
                        warnings.append(
                            f"IC is near zero ({ic_float:.6f}) — factor may not predict returns"
                        )
                except (ValueError, TypeError):
                    warnings.append(f"IC value is not numeric: {ic_value}")

            # Check positions (1day.pos)
            pos_value = result.get('1day.pos', None)
            details['positions_raw'] = pos_value
            if pos_value is not None:
                try:
                    pos_float = float(pos_value)
                    details['positions'] = pos_float
                    if pos_float == 0:
                        warnings.append(
                            "1day.pos == 0 — model opened ZERO positions (stayed neutral). "
                            "Possible causes: (1) topk too high for single-asset, "
                            "(2) signal threshold too restrictive, (3) no valid predictions"
                        )
                    elif pos_float < 10:
                        warnings.append(
                            f"1day.pos = {pos_float:.0f} — very few positions opened. "
                            f"Check signal threshold and topk settings"
                        )
                except (ValueError, TypeError):
                    pass  # pos might be a string

            # Check if result is essentially empty (all values None or NaN)
            non_null_count = result.notna().sum()
            total_count = len(result)
            details['non_null_metrics'] = int(non_null_count)
            details['total_metrics'] = int(total_count)
            if non_null_count < 3:
                warnings.append(
                    f"Result has only {non_null_count}/{total_count} non-null metrics — "
                    f"backtest likely produced empty results"
                )

            # Check for key metrics
            required_metrics = ['IC', '1day.excess_return_with_cost.shar', '1day.pos']
            for metric_name in required_metrics:
                val = result.get(metric_name, None)
                details[f'has_{metric_name}'] = val is not None

        elif isinstance(result, dict):
            # Dict-based result validation
            ic_value = result.get('IC', result.get('ic', None))
            details['ic_raw'] = ic_value
            if ic_value is None:
                warnings.append("IC is None — factor has no predictive power")

        return {
            "has_issues": len(warnings) > 0,
            "warnings": "; ".join(warnings),
            "details": details,
        }

    def _save_failed_run(self, exp, stdout: str, error_type: str = "unknown",
                         validation: Optional[dict] = None) -> None:
        """
        Save failed run information to results/failed_runs.json for debugging.

        Parameters
        ----------
        exp : QlibFactorExperiment
            The experiment that failed
        stdout : str
            Standard output from the Docker execution
        error_type : str
            Type of error: 'result_none', 'validation_warnings', 'docker_error', etc.
        validation : dict, optional
            Validation result dict if error_type is 'validation_warnings'
        """
        import json
        from datetime import datetime
        from pathlib import Path

        try:
            # Ensure failed_runs directory exists (5 levels up to project root)
            project_root = Path(__file__).parent.parent.parent.parent.parent
            failed_dir = project_root / "results" / "failed_runs"
            failed_dir.mkdir(parents=True, exist_ok=True)

            # Get factor name
            factor_name = "unknown"
            if hasattr(exp, 'hypothesis') and exp.hypothesis is not None:
                factor_name = getattr(exp.hypothesis, 'hypothesis', 'unknown')

            # Build failed run record
            failed_record = {
                "timestamp": datetime.now().isoformat(),
                "factor_name": factor_name,
                "error_type": error_type,
                "stdout": stdout if stdout else "(empty)",
                "validation": validation,
                "experiment_details": {
                    "base_features": list(getattr(exp, 'base_features', {}).keys()) if hasattr(exp, 'base_features') else [],
                    "hypothesis": getattr(exp.hypothesis, 'hypothesis', str(getattr(exp, 'hypothesis', 'N/A')))
                                  if hasattr(exp, 'hypothesis') else "N/A",
                },
            }

            # Append to failed_runs.json
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

            # Keep only last 500 records to prevent file bloat
            if len(existing_records) > 500:
                existing_records = existing_records[-500:]

            failed_file.write_text(
                json.dumps(existing_records, indent=2, default=str, ensure_ascii=False),
                encoding="utf-8"
            )
            logger.info(
                f"Failed run saved: {factor_name} (type={error_type}) "
                f"→ {failed_file}"
            )

        except Exception as e:
            # Don't let failed run logging break the main workflow
            logger.warning(f"Could not save failed run info: {e}")

    def _save_result_to_database(self, exp, result) -> None:
        """
        Save backtest results to the ResultsDatabase and write factor JSON summary.

        This method is called immediately after Docker execution to ensure
        results are persisted before any potential failures.

        Parameters
        ----------
        exp : QlibFactorExperiment
            The experiment with backtest results
        result : pd.Series
            Backtest metrics from Qlib (qlib_res.csv) - a pd.Series with index
            containing metric names like 'IC', '1day.excess_return_with_cost.shar', etc.
        """
        try:
            import json
            import pandas as pd
            from pathlib import Path
            from rdagent.components.backtesting import ResultsDatabase

            # Get factor name from hypothesis
            factor_name = "unknown"
            if hasattr(exp, 'hypothesis') and exp.hypothesis is not None:
                factor_name = getattr(exp.hypothesis, 'hypothesis', 'unknown')

            # Check if already rejected by protection
            if getattr(exp, 'rejected_by_protection', False):
                logger.info(
                    f"Factor rejected by protection, skipping DB save: "
                    f"{getattr(exp, 'protection_reason', 'unknown')}"
                )
                return

            # Log result structure for debugging
            logger.info(f"Saving result for factor: {factor_name}")
            logger.info(f"Result type: {type(result)}")
            if isinstance(result, pd.Series):
                logger.info(f"Result index (metric names): {list(result.index)}")
                logger.info(f"Result values:\n{result.to_string()}")
            elif isinstance(result, dict):
                logger.info(f"Result keys: {list(result.keys())}")

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
                # Store raw metrics for JSON export
                metrics['raw_metrics'] = result.to_dict()
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
                metrics['raw_metrics'] = result

            # Result validation before saving (warnings, not blocking)
            self._log_result_warnings(factor_name, result, metrics)

            # Only save if we have at least IC or Sharpe
            if metrics.get('ic') is None and metrics.get('sharpe_ratio') is None:
                logger.warning(
                    f"No valid IC/Sharpe for factor '{factor_name}', skipping DB save. "
                    f"IC={metrics.get('ic')}, Sharpe={metrics.get('sharpe_ratio')}"
                )
                return

            # Ensure DB directory exists before creating database (5 levels up to project root)
            project_root = Path(__file__).parent.parent.parent.parent.parent
            db_path = project_root / "results" / "db"
            db_path.mkdir(parents=True, exist_ok=True)
            db_file = db_path / "backtest_results.db"

            # Parallel run isolation: use run-specific subdirectory if PARALLEL_RUN_ID is set
            run_id = os.getenv("PARALLEL_RUN_ID", "0")
            if run_id != "0":
                # For parallel runs, save to isolated results directory
                isolated_db_path = project_root / "results" / "runs" / f"run{run_id}" / "db"
                isolated_db_path.mkdir(parents=True, exist_ok=True)
                db_file = isolated_db_path / "backtest_results.db"

            # Save to database
            db = ResultsDatabase(db_path=str(db_file))
            run_id = db.add_backtest(factor_name=factor_name[:100], metrics=metrics)
            logger.info(
                f"Factor result saved to DB: {factor_name[:60]} "
                f"(IC={metrics.get('ic')}, Sharpe={metrics.get('sharpe_ratio')}, run_id={run_id})"
            )

            # Also write a JSON summary to results/factors/ for file-based access
            self._save_factor_json(factor_name, metrics, run_id)

            db.close()

        except Exception as e:
            import traceback
            logger.error(
                f"Database save failed for factor '{getattr(exp.hypothesis, 'hypothesis', 'unknown')}': {e}\n"
                f"Traceback: {traceback.format_exc()}"
            )

    def _save_factor_json(self, factor_name: str, metrics: dict, run_id: int) -> None:
        """
        Save factor metrics as a JSON file for easy file-based access.

        Parameters
        ----------
        factor_name : str
            Name of the factor
        metrics : dict
            Extracted metrics dictionary
        run_id : int
            Database run ID
        """
        import json
        import os as _os
        from datetime import datetime
        from pathlib import Path

        try:
            # Ensure factors directory exists (5 levels up to project root)
            project_root = Path(__file__).parent.parent.parent.parent.parent

            # Parallel run isolation: use run-specific directory if PARALLEL_RUN_ID is set
            parallel_run_id = _os.getenv("PARALLEL_RUN_ID", "0")
            if parallel_run_id != "0":
                factors_dir = project_root / "results" / "runs" / f"run{parallel_run_id}" / "factors"
            else:
                factors_dir = project_root / "results" / "factors"
            factors_dir.mkdir(parents=True, exist_ok=True)

            # Sanitize factor name for filename
            safe_name = factor_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
            # Truncate if too long (filesystem limits)
            if len(safe_name) > 150:
                safe_name = safe_name[:150]

            json_path = factors_dir / f"{safe_name}.json"

            # Build summary document
            factor_summary = {
                "factor_name": factor_name,
                "run_id": run_id,
                "saved_at": datetime.now().isoformat(),
                "metrics": {
                    "ic": metrics.get("ic"),
                    "sharpe_ratio": metrics.get("sharpe_ratio"),
                    "annualized_return": metrics.get("annualized_return"),
                    "max_drawdown": metrics.get("max_drawdown"),
                    "win_rate": metrics.get("win_rate"),
                    "information_ratio": metrics.get("information_ratio"),
                    "volatility": metrics.get("volatility"),
                },
                "raw_metrics": {
                    k: v for k, v in metrics.get("raw_metrics", {}).items()
                    if isinstance(v, (int, float, str, bool, type(None)))
                },
            }

            json_path.write_text(json.dumps(factor_summary, indent=2, default=str), encoding="utf-8")
            logger.info(f"Factor JSON saved: {json_path}")

        except Exception as e:
            logger.warning(f"Failed to save factor JSON for '{factor_name}': {e}")

    def _log_result_warnings(self, factor_name: str, result, metrics: dict) -> None:
        """
        Log warnings about result quality before saving to database.

        These are informational warnings, not errors — they don't block the workflow
        but help identify factors with potential issues.

        Parameters
        ----------
        factor_name : str
            Name of the factor
        result : pd.Series or dict
            Raw backtest result
        metrics : dict
            Extracted metrics dictionary
        """
        warnings_list = []

        # Check IC
        ic = metrics.get('ic')
        if ic is None:
            warnings_list.append("IC is None — factor has no predictive power")
        elif abs(ic) < 0.001:
            warnings_list.append(f"IC is near zero ({ic:.6f}) — weak predictive signal")

        # Check positions (1day.pos) — CRITICAL for EURUSD
        if isinstance(result, pd.Series):
            pos_value = result.get('1day.pos', None)
            if pos_value is not None:
                try:
                    pos_float = float(pos_value)
                    if pos_float == 0:
                        warnings_list.append(
                            "WARNING: 1day.pos == 0 — ZERO positions opened! "
                            "Model stayed completely neutral. Check Qlib config: "
                            "ensure topk=1 and market=eurusd for single-asset trading."
                        )
                    elif pos_float < 10:
                        warnings_list.append(
                            f"Low position count: 1day.pos = {pos_float:.0f} — "
                            f"model traded very rarely"
                        )
                except (ValueError, TypeError):
                    pass

        # Check Sharpe
        sharpe = metrics.get('sharpe_ratio')
        if sharpe is not None and abs(sharpe) < 0.1:
            warnings_list.append(f"Sharpe near zero ({sharpe:.4f}) — no risk-adjusted edge")

        # Check max drawdown
        mdd = metrics.get('max_drawdown')
        if mdd is not None and mdd < -0.5:
            warnings_list.append(f"Extreme drawdown: {mdd:.2%} — high risk factor")

        if warnings_list:
            for warn_msg in warnings_list:
                logger.warning(f"[{factor_name[:60]}] {warn_msg}")

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

    def _run_protection_check(self, exp, result: dict) -> None:
        """
        Run protection checks on backtest results.

        Parameters
        ----------
        exp : QlibFactorExperiment
            The experiment with backtest results
        result : dict
            Backtest metrics dictionary
        """
        from rdagent.components.backtesting.protections import ProtectionManager

        protection_manager = ProtectionManager()
        protection_manager.create_default_protections()

        # Extract returns and equity curve from backtest results
        returns = result.get("returns", [])
        timestamps = result.get("timestamps", [])
        current_equity = result.get("final_equity", 100000)
        peak_equity = result.get("peak_equity", current_equity)

        # Get factor name from hypothesis
        factor_name = "unknown"
        if hasattr(exp, "hypothesis") and exp.hypothesis is not None:
            factor_name = getattr(exp.hypothesis, "hypothesis", "unknown")

        protection_result = protection_manager.check_all(
            returns=returns,
            timestamps=timestamps,
            current_equity=current_equity,
            peak_equity=peak_equity,
            factor_name=factor_name,
        )

        if protection_result.should_block:
            logger.warning(
                f"Factor {factor_name} rejected by protection manager: {protection_result.reason}"
            )
            # Mark factor as rejected by protection
            exp.rejected_by_protection = True
            exp.protection_reason = protection_result.reason

    def _write_run_log(self, exp, result) -> None:
        """
        Write a log entry for EVERY run (success or failure) to results/logs/.

        This ensures we have a record of every factor attempt, even if it failed
        validation or had no valid metrics.

        Parameters
        ----------
        exp : QlibFactorExperiment
            The experiment object
        result : pd.Series or dict or None
            Backtest result (can be None if execution failed)
        """
        import json
        from datetime import datetime
        from pathlib import Path

        factor_name = "unknown"
        if hasattr(exp, 'hypothesis') and exp.hypothesis is not None:
            factor_name = getattr(exp.hypothesis, 'hypothesis', 'unknown')

        # Build log entry
        log_entry = {
            "factor_name": factor_name,
            "timestamp": datetime.now().isoformat(),
            "status": "unknown",
            "ic": None,
            "sharpe": None,
            "annualized_return": None,
            "max_drawdown": None,
            "win_rate": None,
            "rejected_by_protection": getattr(exp, 'rejected_by_protection', False),
            "protection_reason": getattr(exp, 'protection_reason', None),
        }

        # Extract metrics if available
        if result is not None:
            if hasattr(result, 'get'):  # pd.Series or dict
                ic_val = result.get('IC', result.get('ic', None))
                log_entry['ic'] = self._safe_float(ic_val) if ic_val is not None else None

                sharpe_val = result.get('1day.excess_return_with_cost.shar',
                                        result.get('1day.excess_return_with_cost.sharpe',
                                        result.get('sharpe', None)))
                log_entry['sharpe'] = self._safe_float(sharpe_val) if sharpe_val is not None else None

                ann_ret = result.get('1day.excess_return_with_cost.annualized_return',
                                     result.get('annualized_return', None))
                log_entry['annualized_return'] = self._safe_float(ann_ret) if ann_ret is not None else None

                mdd = result.get('1day.excess_return_with_cost.max_drawdown',
                                 result.get('max_drawdown', None))
                log_entry['max_drawdown'] = self._safe_float(mdd) if mdd is not None else None

                wr = result.get('win_rate', None)
                log_entry['win_rate'] = self._safe_float(wr) if wr is not None else None

            # Determine status
            if log_entry['ic'] is not None or log_entry['sharpe'] is not None:
                log_entry['status'] = "success"
            elif getattr(exp, 'rejected_by_protection', False):
                log_entry['status'] = "rejected_protection"
            else:
                log_entry['status'] = "no_valid_metrics"
        else:
            log_entry['status'] = "execution_failed"
            log_entry['reason'] = "Result was None"

        # Write to results/logs/
        try:
            project_root = Path(__file__).parent.parent.parent.parent.parent

            # Parallel run isolation: use run-specific log directory if PARALLEL_RUN_ID is set
            parallel_run_id = os.getenv("PARALLEL_RUN_ID", "0")
            if parallel_run_id != "0":
                log_dir = project_root / "results" / "runs" / f"run{parallel_run_id}" / "logs"
            else:
                log_dir = project_root / "results" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)

            # One file per day
            today = datetime.now().strftime("%Y-%m-%d")
            log_file = log_dir / f"factor_runs_{today}.jsonl"

            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            logger.info(
                f"Run log written for '{factor_name[:50]}': "
                f"status={log_entry['status']}, IC={log_entry['ic']}, Sharpe={log_entry['sharpe']}"
            )
        except Exception as e:
            logger.error(f"Failed to write run log: {e}")

    def _ensure_results_dirs(self) -> None:
        """Ensure all results directories exist."""
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent.parent.parent

        # Parallel run isolation: create run-specific directories if PARALLEL_RUN_ID is set
        parallel_run_id = os.getenv("PARALLEL_RUN_ID", "0")
        if parallel_run_id != "0":
            # Isolated run directories
            run_base = project_root / "results" / "runs" / f"run{parallel_run_id}"
            for subdir in ["factors", "logs", "db"]:
                (run_base / subdir).mkdir(parents=True, exist_ok=True)
        else:
            # Standard shared directories
            for subdir in ["results/runs", "results/factors", "results/logs", "results/backtests", "results/db"]:
                (project_root / subdir).mkdir(parents=True, exist_ok=True)
