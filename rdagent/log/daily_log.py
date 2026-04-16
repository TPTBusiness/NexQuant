"""
Daily-rotating log for all Predix commands.

Automatically organizes logs by date:

  logs/
    YYYY-MM-DD/
      fin_quant.log       ← R&D loop (structured)
      strategies.log      ← strategy generation
      strategies_bt.log   ← parallel strategy generator script
      evaluate.log        ← factor evaluation
      parallel.log        ← parallel runs
      all.log             ← every command combined

Usage:
    from rdagent.log.daily_log import setup, session

    # One-shot setup (returns a bound logger):
    log = setup("fin_quant", model="local", loops=10)
    log.info("Loop started")

    # Context manager (logs start + elapsed + stop automatically):
    with session("strategies", style="swing", count=10) as log:
        log.info("Generating…")
        run_generation()
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger as _root

# ── paths ─────────────────────────────────────────────────────────────────────
LOGS_ROOT: Path = Path(__file__).parent.parent.parent / "logs"

# ── format ────────────────────────────────────────────────────────────────────
_FILE_FMT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[cmd]: <18} | {message}"
)

# ── internal state ─────────────────────────────────────────────────────────────
_registered: set[str] = set()   # command keys that already have a file sink
_all_added: bool = False         # whether the combined all.log sink is active


# ── helpers ───────────────────────────────────────────────────────────────────

def _today_dir() -> Path:
    d = LOGS_ROOT / datetime.now().strftime("%Y-%m-%d")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fmt_td(td) -> str:
    s = int(td.total_seconds())
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    if m:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


def _banner(log, title: str, meta: dict[str, Any]) -> None:
    sep = "─" * 76
    log.info(sep)
    log.info(f"  {title}")
    if meta:
        pairs = "   ".join(f"{k}={v}" for k, v in meta.items())
        log.info(f"  {pairs}")
    log.info(sep)


# ── public API ────────────────────────────────────────────────────────────────

def setup(command: str, **context: Any):
    """
    Initialise daily log sinks for *command* and return a bound logger.

    Idempotent — safe to call multiple times within the same process.

    Args:
        command: Short slug, e.g. "fin_quant", "strategies", "evaluate".
        **context: Key/value pairs printed in the startup banner.

    Returns:
        loguru.Logger bound with extra["cmd"] = command.upper().
    """
    global _all_added

    log_dir = _today_dir()
    key = command.lower()

    if key not in _registered:
        # Per-command rotating file
        _root.add(
            str(log_dir / f"{key}.log"),
            format=_FILE_FMT,
            filter=lambda r, k=key: r["extra"].get("cmd", "").lower() == k,
            rotation="00:00",      # new file at midnight
            retention="30 days",
            encoding="utf-8",
            enqueue=True,
            backtrace=False,
            diagnose=False,
        )
        _registered.add(key)

    if not _all_added:
        # Combined log — all commands
        _root.add(
            str(log_dir / "all.log"),
            format=_FILE_FMT,
            filter=lambda r: "cmd" in r["extra"],
            rotation="00:00",
            retention="60 days",
            encoding="utf-8",
            enqueue=True,
            backtrace=False,
            diagnose=False,
        )
        _all_added = True

    bound = _root.bind(cmd=command.upper())
    _banner(bound, f"▶  START   {command.upper()}", context)
    return bound


@contextmanager
def session(command: str, **context: Any):
    """
    Context manager: logs start, stop, and elapsed duration automatically.

    Usage::

        with daily_log.session("fin_quant", model="local", loops=10) as log:
            log.info("Step 1 complete")
            run_loop()

    On success logs: ``◼  DONE   FIN_QUANT  (12m 34s)``
    On interrupt:    ``⚠  INTERRUPTED   FIN_QUANT  (2m 01s)``
    On error:        ``✖  FAILED  FIN_QUANT  (0s)  — <exception>``
    """
    log = setup(command, **context)
    t0 = datetime.now()
    try:
        yield log
        elapsed = datetime.now() - t0
        _banner(log, f"◼  DONE    {command.upper()}   ({_fmt_td(elapsed)})", {})
    except KeyboardInterrupt:
        elapsed = datetime.now() - t0
        _banner(log, f"⚠  INTERRUPTED   {command.upper()}   ({_fmt_td(elapsed)})", {})
        raise
    except Exception as exc:
        elapsed = datetime.now() - t0
        log.error(f"✖  FAILED  {command.upper()}  ({_fmt_td(elapsed)})  — {exc}")
        raise
