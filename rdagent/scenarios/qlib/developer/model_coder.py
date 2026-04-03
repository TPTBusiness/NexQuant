"""
Qlib Model Coder - Generates and improves ML models using LLM.

Integrates with model_loader to load local models as baselines
for the LLM to reference and improve upon.
"""

import inspect
from typing import Any

from rdagent.components.coder.model_coder import ModelCoSTEER
from rdagent.core.scenario import Scenario


class QlibModelCoSTEER(ModelCoSTEER):
    """
    Qlib-specific Model Coder that integrates local model baselines.

    Loads available local models and includes their source code
    in the LLM prompt as reference implementations to improve upon.
    """

    def __init__(self, scen: Scenario, *args, **kwargs) -> None:
        super().__init__(scen, *args, **kwargs)
        self._baseline_code = self._load_baseline_models()

    def _load_baseline_models(self) -> str:
        """
        Load available local models as baseline references.

        Returns
        -------
        str
            Source code of available baseline models, or empty string if none found.
        """
        try:
            from rdagent.components.model_loader import load_model, list_available_models

            available = list_available_models()
            local_models = available.get("local", [])

            if not local_models:
                return ""

            # Load the first available local model as baseline
            baseline_code_parts = []
            for model_name in local_models[:2]:  # Load up to 2 baselines
                try:
                    model_factory = load_model(model_name)
                    source = inspect.getsource(model_factory)
                    baseline_code_parts.append(
                        f"### Baseline Model: {model_name}\n```python\n{source}\n```\n"
                    )
                except Exception as e:
                    # Skip models that fail to load
                    pass

            if baseline_code_parts:
                return (
                    "\n## Reference Baseline Models\n"
                    "Here are existing local models you can improve upon:\n\n"
                    + "\n".join(baseline_code_parts)
                )
        except Exception as e:
            # If model_loader fails entirely, return empty baseline
            pass

        return ""

    def develop(self, exp: Any) -> Any:
        """
        Develop a model experiment with baseline reference.

        If baseline models are available, they are referenced in the
        development process to guide the LLM toward better implementations.
        """
        # Store baseline code in scenario for prompt injection
        if self._baseline_code and hasattr(self, "scen"):
            self.scen.baseline_model_code = self._baseline_code

        return super().develop(exp)


# Backward compatibility alias
QlibModelCoSTEER = QlibModelCoSTEER
