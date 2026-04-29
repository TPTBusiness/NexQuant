import logging
import json
import os
from typing import List, Tuple

from rdagent.components.coder.factor_coder.factor import FactorExperiment, FactorTask
from rdagent.components.proposal import FactorHypothesis2Experiment, FactorHypothesisGen
from rdagent.core.proposal import Hypothesis, Scenario, Trace
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment
from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario
from rdagent.utils.agent.tpl import T


def _build_compressed_history(trace: Trace, max_history: int) -> str:
    """Return hypothesis_and_feedback string with only `max_history` entries.

    Older entries beyond the last 2 are compressed to one bullet line each.
    """
    if len(trace.hist) == 0:
        return "No previous hypothesis and feedback available since it's the first round."

    FULL_DETAIL = 2
    old_hist = trace.hist[:-FULL_DETAIL] if len(trace.hist) > FULL_DETAIL else []
    recent_hist = trace.hist[-FULL_DETAIL:] if len(trace.hist) > FULL_DETAIL else trace.hist

    parts = []
    if old_hist:
        lines = ["## Earlier experiments (summarized):"]
        for exp, fb in old_hist:
            names = []
            for task in exp.sub_tasks:
                if task is not None and hasattr(task, "factor_name"):
                    names.append(task.factor_name)
                elif task is not None and hasattr(task, "model_type"):
                    names.append(getattr(task, "model_type", "model"))
            ic_str = ""
            try:
                if exp.result is not None and "IC" in exp.result.index:
                    ic_str = f" IC={exp.result.loc['IC']:.4f}"
            except Exception:
                logging.debug("Could not extract IC from experiment result", exc_info=True)
            decision = "PASS" if fb.decision else "FAIL"
            obs = (fb.observations or "")[:120].replace("\n", " ")
            lines.append(f"- [{decision}]{ic_str} {', '.join(names) or 'unknown'}: {obs}")
        parts.append("\n".join(lines))

    if recent_hist:
        rt = Trace(trace.scen)
        rt.hist = recent_hist
        parts.append(T("scenarios.qlib.prompts:hypothesis_and_feedback").r(trace=rt))

    return "\n\n".join(parts)

QlibFactorHypothesis = Hypothesis


class QlibFactorHypothesisGen(FactorHypothesisGen):
    def __init__(self, scen: Scenario) -> Tuple[dict, bool]:
        super().__init__(scen)

    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        max_h = int(os.environ.get("QLIB_QUANT_MAX_FACTOR_HISTORY", "20"))
        limited = Trace(trace.scen)
        limited.hist = trace.hist[-max_h:] if len(trace.hist) > max_h else trace.hist
        hypothesis_and_feedback = _build_compressed_history(limited, max_h)
        last_hypothesis_and_feedback = (
            T("scenarios.qlib.prompts:last_hypothesis_and_feedback").r(
                experiment=trace.hist[-1][0], feedback=trace.hist[-1][1]
            )
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )

        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "last_hypothesis_and_feedback": last_hypothesis_and_feedback,
            "RAG": (
                "Try the easiest and fastest factors to experiment with from various perspectives first."
                if len(trace.hist) < 15
                else "Now, you need to try factors that can achieve high IC (e.g., machine learning-based factors)."
            ),
            "hypothesis_output_format": T("scenarios.qlib.prompts:factor_hypothesis_output_format").r(),
            "hypothesis_specification": T("scenarios.qlib.prompts:factor_hypothesis_specification").r(),
        }
        return context_dict, True

    def convert_response(self, response: str) -> Hypothesis:
        response_dict = json.loads(response)
        hypothesis = QlibFactorHypothesis(
            hypothesis=response_dict.get("hypothesis"),
            reason=response_dict.get("reason"),
            concise_reason=response_dict.get("concise_reason"),
            concise_observation=response_dict.get("concise_observation"),
            concise_justification=response_dict.get("concise_justification"),
            concise_knowledge=response_dict.get("concise_knowledge"),
        )
        return hypothesis


class QlibFactorHypothesis2Experiment(FactorHypothesis2Experiment):
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict | bool]:
        if isinstance(trace.scen, QlibQuantScenario):
            scenario = trace.scen.get_scenario_all_desc(action="factor")
        else:
            scenario = trace.scen.get_scenario_all_desc()

        experiment_output_format = T("scenarios.qlib.prompts:factor_experiment_output_format").r()

        if len(trace.hist) == 0:
            hypothesis_and_feedback = "No previous hypothesis and feedback available since it's the first round."
        else:
            max_h = int(os.environ.get("QLIB_QUANT_MAX_FACTOR_HISTORY", "20"))
            factor_hist = [
                e for e in trace.hist
                if not hasattr(e[0].hypothesis, "action") or e[0].hypothesis.action == "factor"
            ][-max_h:]
            specific_trace = Trace(trace.scen)
            specific_trace.hist = factor_hist
            if specific_trace.hist:
                hypothesis_and_feedback = _build_compressed_history(specific_trace, max_h)
            else:
                hypothesis_and_feedback = "No previous hypothesis and feedback available."

        return {
            "target_hypothesis": str(hypothesis),
            "scenario": scenario,
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "experiment_output_format": experiment_output_format,
            "target_list": [],
            "RAG": None,
        }, True

    def convert_response(self, response: str, hypothesis: Hypothesis, trace: Trace) -> FactorExperiment:
        response_dict = json.loads(response)
        tasks = []

        for factor_name in response_dict:
            description = response_dict[factor_name]["description"]
            formulation = response_dict[factor_name]["formulation"]
            variables = response_dict[factor_name]["variables"]
            tasks.append(
                FactorTask(
                    factor_name=factor_name,
                    factor_description=description,
                    factor_formulation=formulation,
                    variables=variables,
                )
            )

        exp = QlibFactorExperiment(tasks, hypothesis=hypothesis)
        exp.based_experiments = [QlibFactorExperiment(sub_tasks=[])] + [
            t[0] for t in trace.hist if t[1] and isinstance(t[0], FactorExperiment)
        ]

        unique_tasks = []
        for task in tasks:
            duplicate = False
            for based_exp in exp.based_experiments:
                if isinstance(based_exp, QlibModelExperiment):
                    continue
                for sub_task in based_exp.sub_tasks:
                    if task.factor_name == sub_task.factor_name:
                        duplicate = True
                        break
                if duplicate:
                    break
            if not duplicate:
                unique_tasks.append(task)

        exp.tasks = unique_tasks
        return exp
