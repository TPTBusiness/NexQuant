import logging
import json
import os
import random
from typing import Tuple

from rdagent.app.qlib_rd_loop.conf import QUANT_PROP_SETTING
from rdagent.components.proposal import FactorAndModelHypothesisGen
from rdagent.core.proposal import Hypothesis, Scenario, Trace
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.qlib.proposal.bandit import (
    EnvController,
    extract_metrics_from_experiment,
)
from rdagent.utils.agent.tpl import T


class QuantTrace(Trace):
    def __init__(self, scen: Scenario) -> None:
        super().__init__(scen)
        self._factor_count = 0
        self.controller = EnvController()  # Initialize immediately, not lazily

    def get_factor_count(self) -> int:
        """Return the number of factors generated so far."""
        return self._factor_count

    def increment_factor_count(self) -> None:
        """Increment the factor count."""
        self._factor_count += 1


class QlibQuantHypothesis(Hypothesis):
    def __init__(
        self,
        hypothesis: str,
        reason: str,
        concise_reason: str,
        concise_observation: str,
        concise_justification: str,
        concise_knowledge: str,
        action: str,
    ) -> None:
        super().__init__(
            hypothesis, reason, concise_reason, concise_observation, concise_justification, concise_knowledge
        )
        self.action = action

    def __str__(self) -> str:
        return f"""Chosen Action: {self.action}
Hypothesis: {self.hypothesis}
Reason: {self.reason}
"""


class QlibQuantHypothesisGen(FactorAndModelHypothesisGen):
    def __init__(self, scen: Scenario) -> Tuple[dict, bool]:
        super().__init__(scen)

    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:

        # ========= Bandit ==========
        if QUANT_PROP_SETTING.action_selection == "bandit":
            # Find the most recent hist entry that has a valid experiment+hypothesis.
            # Entries can be None/corrupt when a loop was reset mid-way (LoopResumeError).
            last_valid = next(
                (entry for entry in reversed(trace.hist)
                 if entry[0] is not None and getattr(entry[0], "hypothesis", None) is not None),
                None,
            )
            if last_valid is not None:
                metric = extract_metrics_from_experiment(last_valid[0])
                prev_action = last_valid[0].hypothesis.action
                trace.controller.record(metric, prev_action)
                action = trace.controller.decide(metric)
            else:
                action = "factor"
        # ========= LLM ==========
        elif QUANT_PROP_SETTING.action_selection == "llm":
            hypothesis_and_feedback = (
                T("scenarios.qlib.prompts:hypothesis_and_feedback").r(trace=trace)
                if len(trace.hist) > 0
                else "No previous hypothesis and feedback available since it's the first round."
            )

            last_hypothesis_and_feedback = (
                T("scenarios.qlib.prompts:last_hypothesis_and_feedback").r(
                    experiment=trace.hist[-1][0], feedback=trace.hist[-1][1]
                )
                if len(trace.hist) > 0
                else "No previous hypothesis and feedback available since it's the first round."
            )

            system_prompt = T("scenarios.qlib.prompts:action_gen.system").r()
            user_prompt = T("scenarios.qlib.prompts:action_gen.user").r(
                hypothesis_and_feedback=hypothesis_and_feedback,
                last_hypothesis_and_feedback=last_hypothesis_and_feedback,
            )
            resp = APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt, json_mode=True)

            action = json.loads(resp).get("action", "factor")
        # ========= random ==========
        elif QUANT_PROP_SETTING.action_selection == "random":
            action = random.choice(["factor", "model"])
        self.targets = action

        qaunt_rag = None
        if action == "factor":
            if len(trace.hist) < 6:
                qaunt_rag = "Try the easiest and fastest factors to experiment with from various perspectives first."
            else:
                qaunt_rag = "Now, you need to try factors that can achieve high IC (e.g., machine learning-based factors)! Do not include factors that are similar to those in the SOTA factor library!"
        elif action == "model":
            qaunt_rag = "1. In Quantitative Finance, market data could be time-series, and GRU model/LSTM model are suitable for them. Do not generate GNN model as for now.\n2. The training data consists of approximately 478,000 samples for the training set and about 128,000 samples for the validation set. Please design the hyperparameters accordingly and control the model size. This has a significant impact on the training results. If you believe that the previous model itself is good but the training hyperparameters or model hyperparameters are not optimal, you can return the same model and adjust these parameters instead.\n"

        if len(trace.hist) == 0:
            hypothesis_and_feedback = "No previous hypothesis and feedback available since it's the first round."
        else:
            specific_trace = Trace(trace.scen)
            # Limit history to avoid exceeding the LLM context window.
            # With 2000+ experiments the prompt easily hits 76k+ tokens on an 80k ctx model.
            MAX_FACTOR_HISTORY = int(os.environ.get("QLIB_QUANT_MAX_FACTOR_HISTORY", "20"))
            MAX_MODEL_HISTORY = int(os.environ.get("QLIB_QUANT_MAX_MODEL_HISTORY", "10"))
            if action == "factor":
                # Most-recent N factor experiments + best SOTA model experiment
                model_inserted = False
                factor_count = 0
                for i in range(len(trace.hist) - 1, -1, -1):  # Reverse iteration
                    if trace.hist[i][0].hypothesis.action == "factor" and factor_count < MAX_FACTOR_HISTORY:
                        specific_trace.hist.insert(0, trace.hist[i])
                        factor_count += 1
                    elif (
                        trace.hist[i][0].hypothesis.action == "model"
                        and trace.hist[i][1].decision is True
                        and model_inserted == False
                    ):
                        specific_trace.hist.insert(0, trace.hist[i])
                        model_inserted = True
            elif action == "model":
                # Most-recent N model experiments + best SOTA factor experiment
                factor_inserted = False
                model_count = 0
                for i in range(len(trace.hist) - 1, -1, -1):  # Reverse iteration
                    if trace.hist[i][0].hypothesis.action == "model" and model_count < MAX_MODEL_HISTORY:
                        specific_trace.hist.insert(0, trace.hist[i])
                        model_count += 1
                    elif (
                        trace.hist[i][0].hypothesis.action == "factor"
                        and trace.hist[i][1].decision is True
                        and factor_inserted == False
                    ):
                        specific_trace.hist.insert(0, trace.hist[i])
                        factor_inserted = True
            if len(specific_trace.hist) > 0:
                specific_trace.hist.reverse()
                # Keep only the 2 most recent experiments in full detail; compress older ones
                # to brief bullet points to stay within the LLM context window.
                FULL_DETAIL_COUNT = 2
                old_hist = specific_trace.hist[:-FULL_DETAIL_COUNT] if len(specific_trace.hist) > FULL_DETAIL_COUNT else []
                recent_hist = specific_trace.hist[-FULL_DETAIL_COUNT:] if len(specific_trace.hist) > FULL_DETAIL_COUNT else specific_trace.hist

                parts = []
                if old_hist:
                    summary_lines = ["## Earlier experiments (summarized):"]
                    for exp, fb in old_hist:
                        factor_names = []
                        for task in exp.sub_tasks:
                            if task is not None and hasattr(task, "factor_name"):
                                factor_names.append(task.factor_name)
                            elif task is not None and hasattr(task, "model_type"):
                                factor_names.append(getattr(task, "model_type", "model"))
                        names_str = ", ".join(factor_names) if factor_names else "unknown"
                        ic_str = ""
                        try:
                            if exp.result is not None:
                                ic_val = exp.result.loc["IC"] if "IC" in exp.result.index else ""
                                ic_str = f" IC={ic_val:.4f}" if ic_val != "" else ""
                        except Exception:
                            logging.debug("Error getting IC", exc_info=True)
                        decision_str = "PASS" if fb.decision else "FAIL"
                        obs_short = (fb.observations or "")[:120].replace("\n", " ")
                        summary_lines.append(f"- [{decision_str}]{ic_str} {names_str}: {obs_short}")
                    parts.append("\n".join(summary_lines))

                if recent_hist:
                    recent_trace = Trace(specific_trace.scen)
                    recent_trace.hist = recent_hist
                    parts.append(T("scenarios.qlib.prompts:hypothesis_and_feedback").r(trace=recent_trace))

                hypothesis_and_feedback = "\n\n".join(parts)
            else:
                hypothesis_and_feedback = "No previous hypothesis and feedback available."

        last_hypothesis_and_feedback = None
        for i in range(len(trace.hist) - 1, -1, -1):
            if trace.hist[i][0].hypothesis.action == action:
                last_hypothesis_and_feedback = T("scenarios.qlib.prompts:last_hypothesis_and_feedback").r(
                    experiment=trace.hist[i][0], feedback=trace.hist[i][1]
                )
                break

        sota_hypothesis_and_feedback = None
        if action == "model":
            for i in range(len(trace.hist) - 1, -1, -1):
                if trace.hist[i][0].hypothesis.action == "model" and trace.hist[i][1].decision is True:
                    sota_hypothesis_and_feedback = T("scenarios.qlib.prompts:sota_hypothesis_and_feedback").r(
                        experiment=trace.hist[i][0], feedback=trace.hist[i][1]
                    )
                    break

        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "last_hypothesis_and_feedback": last_hypothesis_and_feedback,
            "SOTA_hypothesis_and_feedback": sota_hypothesis_and_feedback,
            "RAG": qaunt_rag,
            "hypothesis_output_format": T("scenarios.qlib.prompts:hypothesis_output_format_with_action").r(),
            "hypothesis_specification": (
                T("scenarios.qlib.prompts:factor_hypothesis_specification").r()
                if action == "factor"
                else T("scenarios.qlib.prompts:model_hypothesis_specification").r()
            ),
        }
        return context_dict, True

    def convert_response(self, response: str) -> Hypothesis:
        response_dict = json.loads(response)
        hypothesis = QlibQuantHypothesis(
            hypothesis=response_dict.get("hypothesis"),
            reason=response_dict.get("reason"),
            concise_reason=response_dict.get("concise_reason"),
            concise_observation=response_dict.get("concise_observation"),
            concise_justification=response_dict.get("concise_justification"),
            concise_knowledge=response_dict.get("concise_knowledge"),
            action=response_dict.get("action"),
        )
        return hypothesis
