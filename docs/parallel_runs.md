# NexQuant Parallel Run System

## Overview

The Parallel Run System enables concurrent execution of 5+ factor generation experiments with automatic API key distribution and complete isolation between runs.

## Architecture

### Components

| File | Purpose |
|------|---------|
| `nexquant.py` | Extended with `--run-id` parameter for isolated single runs |
| `nexquant_parallel.py` | Parallel runner manager with Rich live dashboard |
| `factor_runner.py` | Modified to use `PARALLEL_RUN_ID` for path isolation |
| `CoSTEER/__init__.py` | Modified to use `PARALLEL_RUN_ID` for intermediate results |

### Directory Structure (Per Run)

```
results/
├── db/                          # Shared database
├── runs/
│   ├── run1/                    # Run #1 isolated results
│   │   ├── factors/             # Factor JSON files
│   │   ├── logs/                # Run-specific logs
│   │   ├── db/                  # Run-specific database
│   │   └── costeer/             # CoSTEER intermediate results
│   ├── run2/                    # Run #2 isolated results
│   │   └── ...
│   └── runN/                    # Run #N isolated results
│       └── ...
└── logs/                        # Default (non-parallel) logs
```

### Log Files

```
fin_quant.log                    # Single run (run_id=0)
fin_quant_run1.log               # Parallel run #1
fin_quant_run2.log               # Parallel run #2
...
```

### Workspaces

```
RD-Agent_workspace/              # Single run (run_id=0)
RD-Agent_workspace_run1/         # Parallel run #1
RD-Agent_workspace_run2/         # Parallel run #2
...
```

## Usage

### CLI - Single Parallel Run

```bash
# Run with isolated results
nexquant quant --run-id 1 -m openrouter
```

### CLI - Parallel Runner (Direct)

```bash
# Run 5 experiments with 2 API keys
python nexquant_parallel.py --runs 5 --api-keys 2

# Run 3 experiments with local model
python nexquant_parallel.py --runs 3 --model local

# Custom configuration
python nexquant_parallel.py -n 10 -k 2 -m openrouter
```

### Programmatic Usage

```python
from nexquant_parallel import main

result = main(runs=5, api_keys=2, model="openrouter")
print(f"Success: {result['success']}/{result['total']}")
```

## API Key Distribution

The system distributes API keys using round-robin assignment:

| Run ID | API Key | Model |
|--------|---------|-------|
| 1 | Key 1 | openrouter |
| 2 | Key 2 | openrouter |
| 3 | Key 1 | openrouter |
| 4 | Key 2 | openrouter |
| 5 | Key 1 | openrouter |

**With 2 API keys:**
- Runs 1, 3, 5 → Key 1
- Runs 2, 4 → Key 2

**LiteLLM Load Balancing:**
When 2 API keys are available, the system configures LiteLLM for parallel request handling:
```
OPENAI_API_KEY=key1,key2
LITELLM_PARALLEL_CALLS=2
```

## Isolation Guarantees

Each parallel run is completely isolated:

### Environment Variables
- `PARALLEL_RUN_ID=N` - Identifies the run
- `RD_AGENT_WORKSPACE` - Points to run-specific workspace
- `OPENAI_API_KEY` - Assigned API key for this run

### No Shared State
- ✅ Separate log files
- ✅ Separate result directories
- ✅ Separate workspace directories
- ✅ Separate database files (optional)
- ✅ No race conditions (no shared mutable state)

### Graceful Degradation
- If a run fails, others continue unaffected
- Each run is independently restartable
- Results are persisted immediately after completion

## Live Dashboard

The parallel runner shows a Rich-based live dashboard:

```
┌─────────────────────────────────────────────────────────┐
│  🔀 NexQuant Parallel Run Dashboard                       │
├──────┬──────────┬──────────┬─────────┬──────────┬───────┤
│ Run  │ Status   │ Elapsed  │ API Key │ Model    │ Exit  │
├──────┼──────────┼──────────┼─────────┼──────────┼───────┤
│ #1   │ ✅ success│ 02:15:30│ 1       │openrouter│ 0     │
│ #2   │ 🔄 running│ 01:45:12│ 2       │openrouter│ --    │
│ #3   │ 🔄 running│ 01:42:08│ 1       │openrouter│ --    │
│ #4   │ ⏳ pending│ --:--:--│ 2       │openrouter│ --    │
│ #5   │ ❌ failed │ 00:05:23│ 1       │openrouter│ 1     │
├──────┴──────────┴──────────┴─────────┴──────────┴───────┤
│  Summary: 5 total | 1 done | 2 running | 1 pending | 1 failed │
└─────────────────────────────────────────────────────────┘
```

## Signal Handling

- **First Ctrl+C:** Gracefully stops all running subprocesses
- **Second Ctrl+C:** Force kills all remaining processes
- Dashboard updates in real-time during shutdown

## Configuration

### Environment Variables (`.env`)

```bash
# Required for openrouter mode
OPENROUTER_API_KEY=sk-or-your-first-key
OPENROUTER_API_KEY_2=sk-or-your-second-key  # Optional

# Required for local mode
OPENAI_API_KEY=local
OPENAI_API_BASE=http://localhost:8081/v1
CHAT_MODEL=qwen3.5-35b

# Optional: Custom model
OPENROUTER_MODEL=openrouter/qwen/qwen3.6-plus:free
```

## Performance

**Expected Speedup:**
- 5 runs with 2 API keys ≈ 2.5× faster than sequential
- 5 runs with local model ≈ 5× faster than sequential (no API rate limits)

**Overhead:**
- ~1 second per run for subprocess startup
- Dashboard refresh: 2 Hz (negligible CPU)

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Run fails | Logged, others continue |
| API key exhausted | Retry with next key |
| Ctrl+C pressed | Graceful shutdown of all runs |
| Disk full | Error logged, run marked failed |
| LLM timeout | Run fails, others unaffected |

## Integration with Existing Code

### factor_runner.py Changes

```python
# Before (shared paths)
log_dir = project_root / "results" / "logs"
factors_dir = project_root / "results" / "factors"

# After (parallel-aware)
parallel_run_id = os.getenv("PARALLEL_RUN_ID", "0")
if parallel_run_id != "0":
    log_dir = project_root / "results" / "runs" / f"run{parallel_run_id}" / "logs"
    factors_dir = project_root / "results" / "runs" / f"run{parallel_run_id}" / "factors"
```

### CoSTEER/__init__.py Changes

```python
# Intermediate results isolation
parallel_run_id = os.getenv("PARALLEL_RUN_ID", "0")
if parallel_run_id != "0":
    results_dir = project_root / "results" / "runs" / f"run{parallel_run_id}" / "costeer"
```

## Testing

```bash
# Run all integration tests
pytest test/integration/test_all_features.py -v

# Test parallel runner imports
python -c "from nexquant_parallel import ParallelRunner, main; print('✅ OK')"

# Test CLI options
nexquant quant --help  # Should show --run-id option
```

## Future Enhancements

- [ ] Auto-detect optimal number of parallel runs based on API rate limits
- [ ] Result aggregation and comparison across runs
- [ ] Dynamic API key rebalancing (assign more runs to faster key)
- [ ] Support for >2 API keys
- [ ] Run prioritization (run high-priority experiments first)
- [ ] Slack/email notifications on completion
