# Predix

<h4 align="center">
  <strong>AI-powered Quantitative Trading Agent for EUR/USD Forex</strong>
</h4>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#features">Features</a>
</p>

<p align="center">
  <a href="https://github.com/TPTBusiness/Predix/blob/main/LICENSE"><img src="https://img.shields.io/github/license/TPTBusiness/Predix" alt="License"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://github.com/TPTBusiness/Predix/stargazers"><img src="https://img.shields.io/github/stars/TPTBusiness/Predix" alt="Stars"></a>
</p>

---

## Overview

**Predix** is an autonomous AI agent for quantitative trading strategies in the EUR/USD forex market. Built on a multi-agent framework, Predix automates the full research and development cycle:

- 📊 **Data Analysis** – Automatically analyzes market patterns and microstructure
- 💡 **Strategy Discovery** – Proposes novel trading factors and signals
- 🧠 **Model Evolution** – Iteratively improves predictive models
- 📈 **Backtesting** – Validates strategies on historical 1-minute data

Predix is optimized for **1-minute EUR/USD FX data** (2020–2026) and uses Qlib as the underlying backtesting engine.

## Acknowledgments

This project draws inspiration from various open-source projects in the AI trading and multi-agent systems space. We thank all the authors for their innovative work that helped shape our understanding of these patterns.

Special thanks to:

- **[Microsoft RD-Agent](https://github.com/microsoft/RD-Agent)** (MIT License) - Foundation for our autonomous R&D agent framework. We extend our gratitude to the RD-Agent team for their excellent foundational work.

- **[TradingAgents](https://github.com/TauricResearch/TradingAgents)** (Apache 2.0 License) - Inspiration for our multi-agent debate system, reflection mechanism, and memory management modules.

- **[ai-hedge-fund](https://github.com/virattt/ai-hedge-fund)** - Inspiration for macro analysis (Stanley Druckenmiller agent), risk management concepts, and market regime detection.

All code in Predix is originally written and implemented independently. Predix extends these frameworks with EUR/USD forex-specific features, 1-minute backtesting capabilities, comprehensive risk management, and trading dashboards.

---

## Installation

### Prerequisites

- **Python 3.10 or 3.11**
- **Docker** (required for sandboxed code execution)
- **Linux** (officially supported; macOS/Windows may work with adjustments)

### Quick Install

```bash
# Clone repository
git clone https://github.com/TPTBusiness/Predix
cd predix

# Create conda environment
conda create -n predix python=3.10
conda activate predix

# Install in editable mode
pip install -e .[test,lint]
```

### Configuration

1. **Create `.env` file:**
```bash
# Local LLM (llama.cpp)
OPENAI_API_KEY=local
OPENAI_API_BASE=http://localhost:8081/v1
CHAT_MODEL=qwen3.5-35b

# Embedding (Ollama)
LITELLM_PROXY_API_KEY=local
LITELLM_PROXY_API_BASE=http://localhost:11434/v1
EMBEDDING_MODEL=nomic-embed-text

# Paths
QLIB_DATA_DIR=~/.qlib/qlib_data/eurusd_1min_data
```

2. **Start LLM server (llama.cpp):**
```bash
~/llama.cpp/build/bin/llama-server \
  --model ~/models/qwen3.5/Qwen3.5-35B-A3B-Q3_K_M.gguf \
  --n-gpu-layers 36 \
  --ctx-size 80000 \
  --port 8081
```

---

## Quick Start

### 1. Run Trading Loop

```bash
# Activate conda environment
conda activate predix

# Start EURUSD trading loop
rdagent fin_quant

# With options
rdagent fin_quant --loop-n 5 --step-n 2
```

### 2. Monitor Results

```bash
# Start the UI dashboard
rdagent server_ui --port 19899 --log-dir git_ignore_folder/RD-Agent_workspace/

# Or open in browser
# http://127.0.0.1:19899
```

### 3. Loop Continuously

To run the trading loop continuously with auto-restart:

```bash
# Simple loop
while true; do
    rdagent fin_quant
    sleep 5
done
```

---

## Configuration

```bash
# Start the UI dashboard
rdagent ui --port 19899 --log-dir log/ --data-science
```

Then open `http://127.0.0.1:19899` in your browser.

---

## Configuration

### Data Configuration

Edit [`data_config.yaml`](data_config.yaml) to customize:

```yaml
instrument: EURUSD
frequency: 1min
data_path: ~/.qlib/qlib_data/eurusd_1min_data

# Walk-forward split
train_start: "2022-03-14"
train_end:   "2024-06-30"
valid_start: "2024-07-01"
valid_end:   "2024-12-31"
test_start:  "2025-01-01"
test_end:    "2026-03-20"

# Market context for LLM prompts
market_context:
  spread_bps: 1.5
  target_arr: 9.62          # Target annual return (%)
  max_drawdown: 20          # Max drawdown (%)
```

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `CHAT_MODEL` | LLM for reasoning | `gpt-4o`, `deepseek-chat` |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `OPENAI_API_KEY` | API key for OpenAI | `sk-...` |
| `DEEPSEEK_API_KEY` | API key for DeepSeek | `sk-...` |
| `DS_LOCAL_DATA_PATH` | Local data directory | `./data` |

---

## Features

### 🔄 Iterative Factor Evolution

Predix continuously proposes, implements, and validates new alpha factors:

- Learns from backtest feedback
- Avoids overfitting through walk-forward validation
- Discovers non-obvious patterns in order flow, volatility, and session dynamics

### 🧠 Model Architecture Search

Automatically explores and refines predictive models:

- Linear baselines (LightGBM, XGBoost)
- Deep learning (LSTM, Transformer, Temporal CNN)
- Ensemble methods

### 📚 Knowledge Base

Built-in knowledge accumulation across loops:

- Successful factors are archived
- Failed attempts inform future proposals
- Cross-loop learning improves robustness

### 🖥️ Interactive UI

Real-time dashboard for monitoring:

- Factor performance metrics
- Model architecture evolution
- Cumulative returns and drawdowns
- Code diffs and implementation history

---

## Project Structure

```
predix/
├── rdagent/                 # Core agent framework
│   ├── app/                 # CLI and scenario apps
│   ├── components/          # Reusable agent components
│   ├── core/                # Core abstractions
│   ├── scenarios/           # Domain-specific scenarios
│   └── utils/               # Utilities
├── constraints/             # Constraint definitions
├── docs/                    # Documentation
├── web/                     # Web UI frontend
├── data_config.yaml         # Data configuration
├── pyproject.toml           # Project metadata
└── requirements.txt         # Dependencies
```

---

## Data Setup

Predix uses 1-minute EUR/USD data. To prepare your dataset:

```bash
# Run the data setup script (if provided)
./setup_predix_eurusd.sh

# Or manually place data in:
# ~/.qlib/qlib_data/eurusd_1min_data/
```

Expected data columns: `$open`, `$close`, `$high`, `$low`, `$volume`

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `rdagent fin_quant` | Full factor & model co-evolution |
| `rdagent fin_factor` | Factor-only evolution |
| `rdagent fin_model` | Model-only evolution |
| `rdagent fin_factor_report --report-folder=<path>` | Extract factors from financial reports |
| `rdagent general_model <paper-url>` | Extract model from research paper |
| `rdagent data_science --competition <name>` | Kaggle/data science competition mode |
| `rdagent ui --port 19899 --log-dir <path>` | Start monitoring dashboard |
| `rdagent health_check` | Validate environment setup |

---

## Requirements

Core dependencies (see [`requirements.txt`](requirements.txt) for full list):

- **LLM**: `openai`, `litellm`
- **Data**: `pandas`, `numpy`, `pyarrow`
- **ML**: `scikit-learn`, `lightgbm`, `xgboost`
- **Backtesting**: `qlib` (via Docker)
- **UI**: `streamlit`, `plotly`, `flask`

---

## License

This project is licensed under the **MIT License** – see the [`LICENSE`](LICENSE) file for details.

### Attribution Requirements

If you use this code or concepts in your project, you **must**:
1. Include the MIT License text
2. Keep the copyright notice: "Copyright (c) 2025 Predix Team"
3. Provide attribution to the original project

See [`ATTRIBUTION.md`](ATTRIBUTION.md) for detailed guidelines and examples.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

For major changes, please open an issue first to discuss your approach.

---

## Citation

If you use Predix in your research, please cite the underlying framework:

```bibtex
@misc{yang2025rdagentllmagentframeworkautonomous,
  title={R&D-Agent: An LLM-Agent Framework Towards Autonomous Data Science},
  author={Yang, Xu and Yang, Xiao and Fang, Shikai and Zhang, Yifei and Wang, Jian and Xian, Bowen and Li, Qizheng and Li, Jingyuan and Xu, Minrui and Li, Yuante and others},
  year={2025},
  eprint={2505.14738},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```

---

## Support

- **Issues**: [GitHub Issues](https://github.com/TPTBusiness/Predix/issues)
- **Documentation**: [Read the Docs](https://rdagent.readthedocs.io/)

---

## Disclaimer

Predix is provided "as is" for **research and educational purposes only**. It is **not** intended for:

- Live trading or financial advice
- Production use without thorough testing
- Replacement of qualified financial professionals

Users assume all liability and should comply with applicable laws and regulations in their jurisdiction. Past performance does not guarantee future results.
