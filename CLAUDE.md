# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Setup
- **Use conda**: `conda activate predix` (required, plain Python won't work)
- **Docker required**: For sandboxed factor/model code execution
- **LLM setup**: Start llama.cpp server before running — `rdagent fin_quant --model local` waits for the health endpoint automatically
  ```bash
  ~/llama.cpp/build/bin/llama-server \
    --model ~/models/qwen3.6/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf \
    --n-gpu-layers 26 \
    --no-mmap \
    --port 8081 \
    --ctx-size 240000 \
    --parallel 3 \
    --batch-size 512 --ubatch-size 512 \
    --host 0.0.0.0 \
    -ctk q4_0 -ctv q4_0 \
    --reasoning off
  ```
  - **`--reasoning off`** — KRITISCH. Deaktiviert Chain-of-Thought vollständig (`thinking=0`). **`--reasoning-budget 0` reicht nicht** — es startet Reasoning und bricht sofort ab, was bei JSON-Anfragen leere Antworten (`char 0` Fehler) erzeugt. Nur `--reasoning off` verhindert das komplett.
  - **`--ctx-size 240000 --parallel 3`** — 3 Slots à **80.128 Tokens** pro Slot (240k ÷ 3). fin_quant-Prompts sind ~17.5k Tokens → 4.5× Puffer. KV-Cache = 1.3 GB (Q4_0), VRAM-Verbrauch ~13.6 GB
  - **`--n-gpu-layers 26`** — 4 Layer weniger als Maximal, gibt ~500 MB VRAM frei für den größeren KV-Cache
  - **Formel für Slot-Größe**: `ctx_size / parallel = n_ctx_slot`. Muss gelten: `n_ctx_slot > MAX_FACTOR_HISTORY × 2500 + 5000`

## 🚨 LLM-Abbrüche sind NICHT vernachlässigbar — immer Ursache beheben

Wiederkehrende `LLMUnavailableError` / "Failed to create chat completion after N retries" sind **immer ein Symptom eines echten Problems** und dürfen nicht einfach als "transient" ignoriert werden. Bisherige Root Causes:

### Root Cause: Prompt-Größe überschreitet llama-server Slot-Kapazität
- **Problem**: `llama-server --ctx-size 100000 --parallel 3` ergibt nur **33.333 Tokens/Slot**
- `fin_quant` sendet Prompts mit bis zu 30k+ Tokens (Trace-History × `MAX_FACTOR_HISTORY`)
- Sobald Experimente etwas größer werden, sprengen die Prompts den Slot → leere/fehlerhafte Antwort → JSON-Parsefehler → Retry-Schleife → Absturz
- **strategies_bt ist nicht betroffen** weil deren Prompts nur ~1.5k-3k Tokens haben
- **Korrekte Lösung**: `--parallel 1 --ctx-size 120000` → voller Kontext pro Anfrage, kein Splitting
- **Diagnosebefehle**:
  ```bash
  grep "task.n_tokens" ~/llama-server.log | grep -oP "n_tokens = \d+" | sort -n | tail -20
  grep "n_ctx_slot" ~/llama-server.log | tail -3
  ```
- **Formel**: `MAX_FACTOR_HISTORY × 2500 + 5000 < n_ctx_slot` muss gelten.  
  Bei `--parallel 1 --ctx-size 120000` → Slot = 120k → bis zu **46** Experimente möglich.  
  Bei `--parallel 3 --ctx-size 100000` → Slot = 33k → max. **11** (knapp, besser **5**).

### Allgemeine Diagnose-Checkliste bei LLM-Abbrüchen
1. Prompt-Größe prüfen: `grep "task.n_tokens" ~/llama-server.log | tail -30`
2. Slot-Kapazität prüfen: `grep "n_ctx_slot" ~/llama-server.log | tail -3`
3. llama-server Health: `curl http://localhost:8081/health`
4. Modell-Pfad/VRAM prüfen: `pgrep -fa llama-server`
5. Fehlertyp im Log: JSON-Parsefehler → Prompt zu groß / Timeout → Server überlastet

## Core Commands
- **Main trading loop**: `rdagent fin_quant` (or `predix quant`)
  - `--model local` / `--model openrouter` — selects LLM backend
  - `--loop-n N` — number of R&D loop iterations
  - `--step-n N` — steps per loop
- **Parallel execution**: `python predix_parallel.py --runs 5 --api-keys 1 -m openrouter`
- **Strategy generation**: `python predix_gen_strategies_real_bt.py [count]`
- **Factor evaluation**: `python predix.py evaluate --all`
- **Top factors**: `python predix.py top -n 20`
- **Best strategies (safe — no source code exposed)**: `python predix.py best`
  - `-n 20` show top N (default 10); `-m sharpe|ic|composite|monthly_return|annual_return` (default `composite` = `sharpe × (1+dd) × trade_penalty`)
  - `--min-trades 30` filter; `--no-realistic` includes numerically suspicious runs (DD<−50% or total_return>100×)
  - `--show NAME` full metadata for one strategy; `--export path.json` writes top-N metadata (code stripped)
  - **Safe for sharing**: CLI never prints or exports the `code` field — use it when discussing strategies with external assistants
- **UI Dashboard**: `rdagent server_ui --port 19899 --log-dir git_ignore_folder/RD-Agent_workspace/`

## Environment
- **Required vars in .env**: `OPENAI_API_KEY`, `OPENAI_API_BASE`, `CHAT_MODEL`, `LITELLM_PROXY_API_KEY`, `LITELLM_PROXY_API_BASE`, `EMBEDDING_MODEL`, `QLIB_DATA_DIR`
- **Data path**: `~/.qlib/qlib_data/eurusd_1min_data` (1-min EUR/USD 2020-2026, 96 bars/day)
- **Config**: Edit `data_config.yaml` for walk-forward splits; runtime config via env vars prefixed `QLIB_QUANT_`

## Testing
- **Run all tests**: `pytest`
- **Run single test**: `pytest tests/path/test_file.py::test_name`
- **Markers**: `offline`, `slow`, `integration`
- **Avoid**: `workspace` directory (excluded from test collection)
- **Mandatory**: Run `pytest` before every commit — no exceptions
- **Extend tests**: Every medium or larger change requires adding or updating tests covering the new/changed behavior

## Code Style
- **Formatter**: Ruff — `ruff check .` / `ruff check --fix .`
- **Type checking**: `mypy`
- **Line length**: 120

## Available CLI Tools

### GitHub CLI (`gh`) — authenticated as TPTBusiness
```bash
unset GITHUB_TOKEN  # required — env var from .env interferes
gh pr list          # list open PRs
gh pr merge <n>     # merge a PR
gh issue list       # list issues
gh issue create     # create issue
gh api repos/TPTBusiness/Predix/code-scanning/alerts  # CodeQL/Bandit alerts
```
**Note**: Always `unset GITHUB_TOKEN` first in the same command — the `.env` value overrides stored credentials.

### PR Merge Policy
- **Dependabot / dependency PRs**: merge autonomously after `pytest test/backtesting/ -v` passes — no user confirmation needed
- **release-please PRs**: merge autonomously after every push if CI is green — `gh pr merge <n> --squash`. This cuts a release automatically with each batch of commits.
- **Feature/fix PRs**: run relevant tests, then merge if green
- Always use `gh pr merge <n> --squash` for clean history

### Auto-release schema — run this after every `git push`

```
1. unset GITHUB_TOKEN && gh pr list --state open -R TPTBusiness/Predix
2. For each open PR:
   - release-please PR  → merge immediately (--squash), then gh release list to verify tag
   - Dependabot PR      → merge immediately if CI green (--squash --auto)
   - Other PRs          → merge if tests pass and change is clearly ready
```

**Always merge the release-please PR right after pushing** — no user confirmation needed.
This is the standing authorization: every push → release.

```bash
# Full post-push sequence:
unset GITHUB_TOKEN
gh pr list --state open -R TPTBusiness/Predix --json number,title,author
# merge release-please PR:
gh pr merge <n> --squash -R TPTBusiness/Predix
# verify:
gh release list -R TPTBusiness/Predix --limit 3
```

### Release Cadence
- Version bumps: `fix:` / `feat:` → patch (1.2.x), `feat!:` / BREAKING CHANGE → minor (1.x.0)
- Only mention open-source changes in release notes — never closed-source strategies/models/prompts
- If the release-please PR doesn't exist yet (CI still running), wait ~60 s then check again

### Git Commit Signing — SSH ("Verified" badge)
- Configured globally: `gpg.format=ssh`, `commit.gpgsign=true`
- Signing key: `~/.ssh/id_ed25519`
- All commits are automatically signed — no extra steps needed

### Active GitHub Actions
| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | push/PR | Bandit security scan + pytest |
| `codacy.yml` | push/PR/weekly | Codacy bandit-only SARIF scan |
| `release.yml` | push master | release-please auto-changelog + PR |
| `conventional-commits.yml` | PR | Enforce conventional commit titles |
| `scheduled-tests.yml` | Monday 07:00 UTC | Weekly pytest py3.10+3.11 + safety |
| `dependabot.yml` | Monday 06:00 UTC | Auto-update pip + GitHub Actions deps |

## 🔍 Verification & Log Discipline

**Always verify before reporting success:**
- After any code change: run a targeted test or smoke-check, don't assume it works
- After starting/restarting a process: confirm via health endpoint or log tail
- After a bug fix: reproduce the failure first, then verify the fix resolves it
- After a commit/push: confirm the git command actually succeeded (check exit code / remote confirmation)

**Always read the logs precisely before drawing conclusions:**
- Tail the relevant log file — don't guess the state of a background process
- Check both the high-level summary log (e.g. `fin_quant.log`) AND the detailed stdout/stderr log
- For LLM issues: always check `~/llama-server.log` for `send_error` / `n_ctx_slot` / `n_tokens` — the root cause is almost always there
- For optimization issues: check the actual result `.json` files in `results/optimization/` to see what metrics and stage values are being produced
- Look at the llama-server log paths:
  ```bash
  grep "send_error" ~/llama-server.log | tail -10          # context overflow errors
  grep "n_ctx_slot" ~/llama-server.log | tail -3           # slot capacity
  grep "task.n_tokens" ~/llama-server.log | sort -t= -k2 -n | tail -10  # largest prompts
  ```
- **Never assume a failure is "transient"** — check the log, find the root cause, fix it

---

## Architecture

### The R&D Loop (`rdagent/app/qlib_rd_loop/quant.py`)

The system runs an async `QuantRDLoop` (extends `LoopBase`) with these steps per iteration:

```
direct_exp_gen → coding → running → feedback → record
```

- **direct_exp_gen**: LLM proposes a hypothesis with `action == "factor"` or `action == "model"` (bandit-based selection balancing past success rates)
- **coding**: CoSTEER generates Python code for the proposed factor or model
- **running**: Executes code in Docker, produces IC/Sharpe/backtest metrics
- **feedback**: Evaluates results, generates natural-language improvement notes
- **record**: Persists trace state; triggers auto-strategy generation every N factors

Session state is pickled after every step to `__session__/{loop_idx}/{step_idx}_{step_name}` — runs are fully resumable.

### CoSTEER Code Generation (`rdagent/components/coder/CoSTEER/`)

CoSTEER is the LLM-based code evolution engine used for both factors and models:

1. Wraps the `Experiment` in an `EvolvingItem` (one sub-task per factor/model)
2. `RAGEvoAgent` retrieves relevant past examples from a knowledge base
3. `MultiProcessEvolvingStrategy` generates/patches code per task via LLM
4. `RAGEvaluator` runs partial evaluation, yields feedback per code segment
5. Best-of-N selection: falls back to highest-scoring checkpoint if later iterations regress

Factor-specific post-processing: `auto_fixer.py` patches common issues (rolling `min_periods`, inf/NaN from division, `groupby().apply()` → `.transform()`, MultiIndex corrections).

### Factor vs Model Tracks

Both tracks use CoSTEER but with separate instances and evaluators:

| | Factor | Model |
|---|---|---|
| Coder | `factor_coder` | `model_coder` |
| Runner | `factor_runner` (Docker) | `model_runner` |
| Feedback | `factor_summarizer` | `model_summarizer` |
| Output | `result.h5` (MultiIndex DataFrame) | predictions + metrics |

Factor output format: MultiIndex `(datetime, instrument)` with a single float64 column named after the factor. Data must span the full 2020–2026 range.

### Strategy Orchestrator + Optuna (`rdagent/components/coder/`)

After enough factors accumulate, `StrategyOrchestrator` runs automatically:
1. Loads top-ranked factors from `results/factors/`
2. LLM generates strategy code combining those factors
3. Real OHLCV backtest on 1-min data (forward-fill daily factors to minute bars)
4. Acceptance: Sharpe ≥ 0.3, max drawdown ≥ −0.30, win rate ≥ 0.40
5. `OptunaOptimizer` tunes rejected strategies in 3 stages (10 → 15 → 5 trials)

### Configuration System

Settings are Pydantic classes in `rdagent/app/qlib_rd_loop/conf.py` (`QuantBasePropSetting`) and `rdagent/core/conf.py` (`RDAgentSettings`). Override any field via env var with prefix `QLIB_QUANT_`.

Key runtime settings:
- `workspace_path` — where generated code lives (`git_ignore_folder/RD-Agent_workspace/`)
- `step_semaphore` — controls parallelism per step
- `evolving_n` — CoSTEER iterations per coding call

### Scenario System (`rdagent/scenarios/qlib/`)

Each `Scenario` subclass injects domain context into LLM prompts: market background, data schema, output format spec, function interface, date ranges. `QlibQuantScenario` is the combined factor+model scenario used by `fin_quant`.

---

## 🚨 CRITICAL: BEFORE EVERY PUSH — CHECK FOR CLOSED SOURCE!

### Never commit:
- `git_ignore_folder/` — trading scripts, OHLCV data, credentials
- `rdagent/scenarios/qlib/local/` — advanced closed-source components
- `models/local/` — improved models (Transformer, TCN, etc.)
- `prompts/local/` — improved prompts
- `.env` — API keys
- `results/` — backtest results, strategies, logs
- `*.db`, `*.log`

### Before every push:
```bash
git status
git diff --staged --name-only
```
Stop if any of the above paths appear in the output.

### Commit message rules:
- **Never mention closed-source changes** in commit messages — not even vaguely ("improved internal strategy logic", "tuned private model", etc.)
- Only describe open-source changes; omit closed-source work entirely

### When adding features or changes:
- **New feature** → extend `README.md` with usage/description
- **New `rdagent` command** → also update:
  1. The `app = typer.Typer(help=...)` block in `rdagent/app/cli.py` (Available Commands section)
  2. The `cmd_table` in `rdagent/app/cli_welcome.py` (shown by `rdagent predix`)
- **New dependency** → add it to `requirements.txt` (and `requirements-dev.txt` if dev-only)
- **Medium+ change** → add or update tests before committing

### Release guidelines:
- Only mention open-source features in release notes
- Never expose internal trading strategies, models, or prompts
- Focus on: framework capabilities, CLI commands, integrations, backtest engine improvements
- Focus on: framework capabilities, CLI commands, integrations

---

## Behavioral Guidelines

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
