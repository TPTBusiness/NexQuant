"""Smith benchmarks — dynamic discovery via config.yaml.

Scans SMITH_BENCH_DIR/*/config.yaml and builds BenchmarkConfig entries
automatically. The actual benchmark code/data lives outside the repo;
default location is ``<repo-root>/../rl-smith/benchmarks/``.
"""

import logging
import os
from pathlib import Path

import yaml

from rdagent.scenarios.rl.autorl_bench.benchmarks import BenchmarkConfig

logger = logging.getLogger(__name__)

# Default: rl-smith/benchmarks as a sibling of the repo root
import rdagent

_REPO_ROOT = Path(rdagent.__path__[0]).resolve().parent  # rdagent pkg dir → RD-Agent/
_SMITH_BENCH_DIR = Path(os.environ.get("SMITH_BENCH_DIR", str(_REPO_ROOT.parent / "rl-smith" / "benchmarks")))
_PKG = "rdagent.scenarios.rl.autorl_bench"


def discover_smith_benchmarks() -> dict[str, BenchmarkConfig]:
    """Scan SMITH_BENCH_DIR/*/config.yaml and build BenchmarkConfig dict."""
    if not _SMITH_BENCH_DIR.is_dir():
        logger.warning(
            "SMITH_BENCH_DIR=%s does not exist; returning empty smith registry",
            _SMITH_BENCH_DIR,
        )
        return {}

    result = {}
    for cfg_path in sorted(_SMITH_BENCH_DIR.glob("*/config.yaml")):
        bench_dir = cfg_path.parent
        if bench_dir.name.startswith("_"):
            continue
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not raw.get("name"):
            continue

        name = raw["name"]
        eval_mode = raw.get("eval_mode", "per_sample")  # nosec
        bench_id = f"smith-{name}"

        if eval_mode == "opencompass":  # nosec
            evaluator_class = f"{_PKG}.core.opencompass.OpenCompassEvaluator"  # nosec
            eval_config = {"dataset": raw.get("opencompass_dataset", "")}  # nosec
        elif eval_mode == "per_sample":  # nosec
            evaluator_class = f"{_PKG}.benchmarks.smith.per_sample_eval.PerSampleEvaluator"  # nosec
            eval_config = {"eval_script": str(bench_dir / "eval.py")}  # nosec
        else:
            # Skip benchmarks with unsupported eval modes (e.g. custom_model)  # nosec
            # that are already registered as standalone benchmarks.
            logger.info("Skipping smith-%s: unsupported eval_mode=%s", name, eval_mode)  # nosec
            continue

        result[bench_id] = BenchmarkConfig(
            id=bench_id,
            evaluator_class=evaluator_class,  # nosec
            data_module="",
            description=raw.get("description", ""),
            eval_config=eval_config,  # nosec
            expose_files=raw.get("expose_files", []),
            bench_dir=str(bench_dir),
        )
    return result
