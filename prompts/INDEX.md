# Predix Prompts Index

Centralized location for all LLM prompts used in the Predix trading system.

## Structure

```
prompts/
├── standard_prompts.yaml          # Main EURUSD trading prompts (Factor Discovery, Evolution, Model Coder)
├── local/                         # Your improved prompts (NOT in Git!)
├── patches/                       # Override patches for Qlib scenarios
│   ├── qlib_experiment_prompts.yaml
│   ├── qlib_rd_loop_prompts.yaml
│   └── qlib_scenarios_prompts.yaml
├── app/                           # Application-level prompts
│   ├── ci/prompts.yaml            # CI/CD prompts
│   ├── qlib_rd_loop/prompts.yaml  # Qlib RD Loop hypothesis generation
│   ├── utils/prompts.yaml         # APE prompts
│   └── finetune/prompts.yaml      # Finetune prompts
├── components/                    # Component prompts
│   ├── agent/prompts.yaml         # Context7 MCP documentation search
│   ├── proposal/prompts.yaml      # Hypothesis proposal generation
│   ├── coder/
│   │   ├── factor_coder/prompts.yaml   # Factor code evaluator
│   │   ├── model_coder/prompts.yaml    # Model code evaluator
│   │   ├── rl/prompts.yaml             # RL trading coder (Chinese)
│   │   ├── CoSTEER/prompts.yaml        # Component analysis
│   │   ├── finetune/prompts.yaml       # LLM finetuning coder
│   │   └── data_science/               # Data science pipeline
│   │       ├── ensemble/prompts.yaml
│   │       ├── feature/prompts.yaml
│   │       ├── model/prompts.yaml
│   │       ├── pipeline/prompts.yaml
│   │       ├── raw_data_loader/prompts.yaml
│   │       ├── share/prompts.yaml
│   │       └── workflow/prompts.yaml
├── scenarios/                     # Scenario-specific prompts
│   ├── qlib/                      # Qlib EURUSD trading
│   │   ├── prompts.yaml           # Main Qlib scenario
│   │   ├── experiment/prompts.yaml
│   │   └── factor_experiment_loader/prompts.yaml
│   ├── data_science/              # Data science scenarios
│   │   ├── dev/prompts.yaml
│   │   ├── runner/dev/prompts.yaml
│   │   ├── proposal/exp_gen/prompts.yaml
│   │   ├── proposal/exp_gen/prompts_v2.yaml    # Largest file (82KB)
│   │   ├── proposal/exp_gen/select/prompts.yaml
│   │   └── scen/prompts.yaml
│   ├── finetune/                  # LLM finetuning
│   │   ├── dev/prompts.yaml
│   │   ├── proposal/prompts.yaml
│   │   └── scen/prompts.yaml
│   ├── kaggle/                    # Kaggle competition
│   │   ├── prompts.yaml
│   │   ├── experiment/prompts.yaml
│   │   └── knowledge_management/prompts.yaml
│   ├── rl/                        # Reinforcement learning (Chinese)
│   │   ├── dev/prompts.yaml
│   │   └── proposal/prompts.yaml
│   └── general_model/prompts.yaml
└── utils/                         # Utility prompts
    └── prompts.yaml               # Filter redundant text
```

## Active Prompts for EURUSD Trading

The following prompts are actively used in the `rdagent fin_quant` trading loop:

| Priority | File | Purpose |
|----------|------|---------|
| 1 | `standard_prompts.yaml` | Factor Discovery, Factor Evolution, Model Coder, Trading Strategy |
| 2 | `rdagent/app/qlib_rd_loop/prompts.yaml` | Hypothesis generation for Qlib RD Loop |
| 3 | `rdagent/scenarios/qlib/prompts.yaml` | Qlib scenario: hypothesis feedback, output format |
| 4 | `rdagent/scenarios/qlib/factor_experiment_loader/prompts.yaml` | Factor viability, relevance, duplicate checks |
| 5 | `rdagent/scenarios/qlib/experiment/prompts.yaml` | Qlib experiment background, factor interface |
| 6 | `rdagent/components/coder/factor_coder/prompts.yaml` | Code evaluation, final decision |
| 7 | `patches/qlib_scenarios_prompts.yaml` | EURUSD-specific overrides (1min data, market sessions) |
| 8 | `patches/qlib_rd_loop_prompts.yaml` | EURUSD hypothesis generation overrides |

## Key Changes (April 2026)

- **Fixed:** All "daily frequency" references changed to "intraday 1-minute bars"
- **Fixed:** `daily_pv.h5` renamed to `intraday_pv.h5` in data descriptions
- **Fixed:** `FactorDatetimeDailyEvaluator` now accepts 1min-30min bars as correct for EURUSD

## Total Files: 44 YAML files
## Total Size: ~486 KB
