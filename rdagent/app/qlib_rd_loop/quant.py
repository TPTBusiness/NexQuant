"""
Quant (Factor & Model) workflow with session control
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import fire
from rdagent.app.qlib_rd_loop.conf import QUANT_PROP_SETTING
from rdagent.components.workflow.conf import BasePropSetting
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.developer import Developer
from rdagent.core.exception import FactorEmptyError, LLMUnavailableError, ModelEmptyError
from rdagent.core.proposal import (
    Experiment2Feedback,
    ExperimentPlan,
    Hypothesis2Experiment,
    HypothesisFeedback,
    HypothesisGen,
)
from rdagent.core.scenario import Scenario
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.qlib.proposal.quant_proposal import QuantTrace
from rdagent.utils.qlib import ALPHA20


class QuantRDLoop(RDLoop):
    skip_loop_error = (
        FactorEmptyError,
        ModelEmptyError,
        LLMUnavailableError,  # LLM timeout after all retries → skip loop, don't crash
    )

    def __init__(self, PROP_SETTING: BasePropSetting):
        scen: Scenario = import_class(PROP_SETTING.scen)()
        logger.log_object(scen, tag="scenario")

        self.hypothesis_gen: HypothesisGen = import_class(PROP_SETTING.quant_hypothesis_gen)(scen)
        logger.log_object(self.hypothesis_gen, tag="quant hypothesis generator")

        self.factor_hypothesis2experiment: Hypothesis2Experiment = import_class(
            PROP_SETTING.factor_hypothesis2experiment,
        )()
        logger.log_object(self.factor_hypothesis2experiment, tag="factor hypothesis2experiment")
        self.model_hypothesis2experiment: Hypothesis2Experiment = import_class(
            PROP_SETTING.model_hypothesis2experiment,
        )()
        logger.log_object(self.model_hypothesis2experiment, tag="model hypothesis2experiment")

        self.factor_coder: Developer = import_class(PROP_SETTING.factor_coder)(scen)
        logger.log_object(self.factor_coder, tag="factor coder")
        self.model_coder: Developer = import_class(PROP_SETTING.model_coder)(scen)
        logger.log_object(self.model_coder, tag="model coder")

        self.factor_runner: Developer = import_class(PROP_SETTING.factor_runner)(scen)
        logger.log_object(self.factor_runner, tag="factor runner")
        self.model_runner: Developer = import_class(PROP_SETTING.model_runner)(scen)
        logger.log_object(self.model_runner, tag="model runner")

        self.factor_summarizer: Experiment2Feedback = import_class(PROP_SETTING.factor_summarizer)(scen)
        logger.log_object(self.factor_summarizer, tag="factor summarizer")
        self.model_summarizer: Experiment2Feedback = import_class(PROP_SETTING.model_summarizer)(scen)
        logger.log_object(self.model_summarizer, tag="model summarizer")

        self.plan: ExperimentPlan = {
            "features": ALPHA20,
            "feature_codes": {},
        }  # for user interaction
        self.trace = QuantTrace(scen=scen)
        super(RDLoop, self).__init__()

    async def direct_exp_gen(self, prev_out: dict[str, Any]):
        while True:
            if self.get_unfinished_loop_cnt(self.loop_idx) < RD_AGENT_SETTINGS.get_max_parallel():
                hypo = self._propose()
                if hypo.action not in ["factor", "model"]:
                    raise ValueError(f"hypo.action must be 'factor' or 'model', got {hypo.action!r}")
                if hypo.action == "factor":
                    exp = self.factor_hypothesis2experiment.convert(hypo, self.trace)
                else:
                    exp = self.model_hypothesis2experiment.convert(hypo, self.trace)
                logger.log_object(exp.sub_tasks, tag="experiment generation")
                exp.base_features = self.plan["features"]
                exp.base_feature_codes = self.plan["feature_codes"]
                if exp.based_experiments:
                    exp.based_experiments[-1].base_features = self.plan["features"]
                    exp.based_experiments[-1].base_feature_codes = self.plan["feature_codes"]
                return {"propose": hypo, "exp_gen": exp}
            await asyncio.sleep(1)

    def coding(self, prev_out: dict[str, Any]):
        exp = None
        try:
            direct = prev_out.get("direct_exp_gen")
            if not direct:
                # Loop was reset (LoopResumeError) while this step was already queued.
                # Treat as empty so skip_loop_error skips this iteration cleanly.
                raise FactorEmptyError("direct_exp_gen result missing after loop reset")
            if direct["propose"].action == "factor":
                exp = self.factor_coder.develop(direct["exp_gen"])
            elif direct["propose"].action == "model":
                exp = self.model_coder.develop(direct["exp_gen"])
            logger.log_object(exp, tag="coder result")
        except (FactorEmptyError, ModelEmptyError) as e:
            logger.warning(f"Coding failed with {type(e).__name__}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected coding error: {e}")
            raise
        finally:
            # Always save results, even on partial failure
            if exp is not None:
                self._save_coder_results(exp)

        return exp

    def _save_coder_results(self, exp) -> None:
        """
        Save CoSTEER-generated code and evaluation to results/ directory.

        This ensures we have a record of generated factors even if
        the full Qlib backtest pipeline fails or is skipped.

        Parameters
        ----------
        exp : Experiment
            The experiment with generated code
        """
        import json
        from datetime import datetime

        try:
            project_root = Path(__file__).parent.parent.parent.parent
            results_dir = project_root / "results" / "runs"
            results_dir.mkdir(parents=True, exist_ok=True)

            # Build result summary
            summary = {
                "timestamp": datetime.now().isoformat(),
                "hypothesis": None,
                "factors": [],
                "status": "generated",
            }

            if hasattr(exp, "hypothesis") and exp.hypothesis is not None:
                summary["hypothesis"] = getattr(exp.hypothesis, "hypothesis", None)

            # Extract generated code from sub_workspace_list
            if hasattr(exp, "sub_workspace_list") and exp.sub_workspace_list:
                for i, ws in enumerate(exp.sub_workspace_list):
                    factor_info = {
                        "index": i,
                        "code": None,
                        "file_count": 0,
                    }
                    if hasattr(ws, "file_dict") and ws.file_dict:
                        factor_info["file_count"] = len(ws.file_dict)
                        factor_info["code"] = ws.file_dict.get("factor.py", None)
                    summary["factors"].append(factor_info)

            # Check if experiment was accepted or rejected
            if hasattr(exp, "accepted_tasks"):
                accepted = getattr(exp, "accepted_tasks", [])
                summary["accepted_count"] = len(accepted)
                summary["status"] = "accepted" if accepted else "rejected"

            # Write JSON summary
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = (summary["hypothesis"] or "unknown_factor")[:80].replace("/", "_").replace(" ", "_")
            json_path = results_dir / f"{timestamp}_{safe_name}.json"

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

            logger.info(f"CoSTEER result saved to {json_path}")

            # Also write a consolidated log entry
            log_dir = project_root / "results" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            today = datetime.now().strftime("%Y-%m-%d")
            log_file = log_dir / f"coder_runs_{today}.jsonl"

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary, ensure_ascii=False, default=str) + "\n")

        except Exception as e:
            logger.warning(f"Failed to save CoSTEER results: {e}")

    def running(self, prev_out: dict[str, Any]):
        if prev_out["direct_exp_gen"]["propose"].action == "factor":
            exp = self.factor_runner.develop(prev_out["coding"])
            if exp is None:
                logger.error("Factor extraction failed.")
                raise FactorEmptyError("Factor extraction failed.")

            # Increment factor count for tracking
            if hasattr(self, "trace") and hasattr(self.trace, "increment_factor_count"):
                self.trace.increment_factor_count()

            # Handle failed experiments gracefully (don't break the loop)
            if getattr(exp, "failed", False):
                reason = getattr(exp, "failure_reason", "unknown")
                factor_name = "unknown"
                if hasattr(exp, "hypothesis") and exp.hypothesis is not None:
                    factor_name = getattr(exp.hypothesis, "hypothesis", "unknown")
                logger.warning(
                    f"Factor '{factor_name}' failed evaluation: {reason}. "
                    f"Continuing with next factor.",
                )
                # Return exp anyway - loop will continue
        elif prev_out["direct_exp_gen"]["propose"].action == "model":
            exp = self.model_runner.develop(prev_out["coding"])
        logger.log_object(exp, tag="runner result")
        return exp

    def feedback(self, prev_out: dict[str, Any]):
        e = prev_out.get(self.EXCEPTION_KEY)
        if e is not None:
            feedback = HypothesisFeedback(
                observations=str(e),
                hypothesis_evaluation="",
                new_hypothesis="",
                reason="",
                decision=False,
            )
        else:
            # Handle cases where the experiment failed during execution (e.g., Docker error)
            exp = prev_out.get("running")
            if exp is not None and getattr(exp, "failed", False):
                reason = getattr(exp, "failure_reason", "Unknown failure reason")
                factor_name = "unknown"
                if hasattr(exp, "hypothesis") and exp.hypothesis is not None:
                    factor_name = getattr(exp.hypothesis, "hypothesis", "unknown")

                logger.warning(f"Skipping feedback for failed factor '{factor_name}'. Reason: {reason}")
                feedback = HypothesisFeedback(
                    observations=f"Factor '{factor_name}' failed execution.",
                    hypothesis_evaluation="Failed",
                    new_hypothesis="Try a different approach.",
                    reason=reason,
                    decision=False,
                )
            elif prev_out["direct_exp_gen"]["propose"].action == "factor":
                feedback = self.factor_summarizer.generate_feedback(prev_out["running"], self.trace)
            elif prev_out["direct_exp_gen"]["propose"].action == "model":
                feedback = self.model_summarizer.generate_feedback(prev_out["running"], self.trace)

            # NOTE: DB save is handled by factor_runner.py _save_result_to_database()
            # which runs immediately after Docker execution. No duplicate save needed here.

        # Periodically build strategies using AI when enough factors are available
        factor_count = self.trace.get_factor_count()

        # Check for auto-strategies trigger
        auto_strategies = getattr(self, "_auto_strategies", False)
        auto_threshold = getattr(self, "_auto_strategies_threshold", 500)

        if auto_strategies and factor_count > 0 and factor_count % auto_threshold == 0:
            logger.info(
                f"Auto-strategy trigger: {factor_count} factors evaluated. "
                f"Suggesting strategy generation now...",
            )
            self._build_strategies_with_ai()
        elif factor_count > 0 and factor_count % 50 == 0 and not auto_strategies:
            # Standard periodic suggestion (every 50 factors)
            logger.info(
                f"Periodic check: {factor_count} factors evaluated. "
                f"Consider running 'rdagent generate_strategies' for AI strategy generation.",
            )

        feedback = self._interact_feedback(feedback)
        logger.log_object(feedback, tag="feedback")
        return feedback

    def _build_strategies_with_ai(self) -> None:
        """
        Build trading strategies using StrategyOrchestrator with Optuna optimization.

        This method is called periodically during the factor generation loop
        to convert accumulated factors into trading strategies.

        Features:
        - Uses improved LLM prompt (strategy_generation_v2.yaml)
        - Forward-fills daily factors to 1-min OHLCV
        - Realistic backtesting with real OHLCV data
        - Optuna hyperparameter optimization
        """
        try:
            from pathlib import Path

            import yaml
            from rdagent.components.coder.strategy_orchestrator import StrategyOrchestrator

            # Load improved prompt
            project_root = Path(__file__).parent.parent.parent.parent
            prompt_path = project_root / "prompts" / "strategy_generation_v2.yaml"
            if prompt_path.exists():
                with open(prompt_path) as f:
                    improved_prompt = yaml.safe_load(f)
            else:
                improved_prompt = None

            # Load factors from results
            results_dir = project_root / "results"
            factors_dir = results_dir / "factors"

            if not factors_dir.exists():
                logger.debug("StrategyOrchestrator: No factors directory found. Skipping.")
                return

            # Load evaluated factors
            factors = []
            for f in factors_dir.glob("*.json"):
                try:
                    with open(f) as fh:
                        data = json.load(fh)
                    if data.get("status") == "success" and data.get("ic") is not None:
                        factors.append(data)
                except Exception:
                    logger.warning("Failed to load factor file %s", f, exc_info=True)
                    continue

            if len(factors) < 10:
                logger.debug(f"StrategyOrchestrator: Only {len(factors)} factors available. Need at least 10. Skipping.")
                return

            # Sort by IC and take top 50
            factors.sort(key=lambda x: abs(x.get("ic", 0) or 0), reverse=True)
            top_factors = factors[:50]

            logger.info(f"StrategyOrchestrator: Building strategies from {len(top_factors)} top factors...")
            logger.info(f"  - Using improved prompt: {improved_prompt is not None}")
            logger.info("  - Optuna optimization: enabled (20 trials)")
            logger.info("  - Real OHLCV backtest: enabled")

            # Initialize orchestrator with Optuna
            orchestrator = StrategyOrchestrator(
                top_factors=20,
                trading_style="swing",
                min_sharpe=0.5,
                max_drawdown=-0.20,
                min_win_rate=0.40,
                use_optuna=True,
                optuna_trials=20,
            )

            # Override with improved prompt if available
            if improved_prompt:
                orchestrator.strategy_prompt = improved_prompt.get("strategy_generation", {})

            # Generate 3 strategies per cycle
            n_strategies = 3
            logger.info(f"Generating {n_strategies} strategies...")

            # Load top factors for generation
            orch_factors = orchestrator.load_top_factors()
            if len(orch_factors) < 2:
                logger.warning(f"Not enough factors for strategy generation (need >= 2, got {len(orch_factors)}). Skipping.")
                return

            for i in range(n_strategies):
                strategy_name = f"auto_gen_v{i+1}"
                try:
                    # Select random factor combination
                    import random
                    n_factors = random.randint(2, min(5, len(orch_factors)))
                    factor_subset = random.sample(orch_factors, n_factors)

                    code = orchestrator.generate_strategy_code(factor_subset, strategy_name)

                    if code:
                        result = orchestrator.evaluate_strategy(code, strategy_name, factor_subset)

                        if result.get("status") == "accepted":
                            logger.info(f"✅ Strategy {strategy_name} accepted!")
                            logger.info(f"   Sharpe: {result.get('sharpe_ratio', 0):.2f}")
                            logger.info(f"   Max DD: {result.get('max_drawdown', 0):.4f}")
                            logger.info(f"   Win Rate: {result.get('win_rate', 0):.4f}")
                        else:
                            logger.info(f"❌ Strategy {strategy_name} rejected: {result.get('reason', 'unknown')[:100]}")
                except Exception as e:
                    logger.warning(f"Strategy generation failed for {strategy_name}: {e}")

            logger.info("StrategyOrchestrator: Cycle complete.")

        except ImportError as e:
            logger.warning(f"StrategyOrchestrator: Import failed ({e}). Skipping strategy building.")
        except Exception as e:
            # Don't break the main loop for strategy building failures
            logger.warning(f"StrategyOrchestrator: Unexpected error: {e}. Skipping strategy building.")


def main(
    path=None,
    step_n: int | None = None,
    loop_n: int | None = None,
    all_duration: str | None = None,
    checkout: bool = True,
    base_features_path: str | None = None,
    auto_strategies: bool = False,
    auto_strategies_threshold: int = 500,
    **kwargs,
):
    """
    Auto R&D Evolving loop for fintech factors.
    You can continue running session by
    .. code-block:: python
        dotenv run -- python rdagent/app/qlib_rd_loop/quant.py $LOG_PATH/__session__/1/0_propose  --step_n 1   # `step_n` is a optional paramter

    Parameters
    ----------
    auto_strategies : bool
        Automatically generate strategies after factor threshold
    auto_strategies_threshold : int
        Number of factors before triggering strategy generation
    """
    if path is None:
        quant_loop = QuantRDLoop(QUANT_PROP_SETTING)
    else:
        quant_loop = QuantRDLoop.load(path, checkout=checkout)
    quant_loop._init_base_features(base_features_path)
    if "user_interaction_queues" in kwargs and kwargs["user_interaction_queues"] is not None:
        quant_loop._set_interactor(*kwargs["user_interaction_queues"])
        quant_loop._interact_init_params()

    # Store auto_strategies settings for use in feedback loop
    if auto_strategies:
        quant_loop._auto_strategies = True
        quant_loop._auto_strategies_threshold = auto_strategies_threshold
        logger.info(
            f"Auto-strategies enabled. Will trigger after {auto_strategies_threshold} factors.",
        )
    else:
        quant_loop._auto_strategies = False
        quant_loop._auto_strategies_threshold = auto_strategies_threshold

    asyncio.run(quant_loop.run(step_n=step_n, loop_n=loop_n, all_duration=all_duration))


if __name__ == "__main__":
    fire.Fire(main)
