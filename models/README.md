# NexQuant Models

This directory contains all ML model definitions for NexQuant trading factors.

---

## 📁 Directory Structure

```
models/
├── standard/               # Default models (committed to Git)
│   ├── xgboost_factor.py   # XGBoost for tabular data
│   ├── lightgbm_factor.py  # LightGBM (faster than XGBoost)
│   └── randomforest_factor.py  # Baseline model
│
├── local/                  # YOUR IMPROVED MODELS (not in Git!)
│   ├── transformer_factor.py   # Your Transformer
│   ├── tcn_factor.py           # Your TCN
│   ├── patchtst_factor.py      # Your PatchTST
│   ├── cnn_lstm_hybrid.py      # Your Hybrid model
│   └── optimized_xgboost.py    # Your optimized XGBoost
│
└── README.md               # This file
```

---

## 🎯 How It Works

**Model Loading Priority:**

1. **`models/local/*.py`** ← Your improved models (loaded first!)
2. **`models/standard/*.py`** ← Default models (fallback)

**Example:**
```python
from rdagent.components.model_loader import load_model

# Load XGBoost model
# If models/local/xgboost_factor*.py exists → loads that
# Otherwise → loads from models/standard/
model_factory = load_model("xgboost_factor")

# Create model instance
model = model_factory(max_depth=8, learning_rate=0.1)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
```

---

## 📝 Available Standard Models

| Model | File | Use Case |
|-------|------|----------|
| **XGBoost** | `xgboost_factor.py` | Tabular factors, fast training |
| **LightGBM** | `lightgbm_factor.py` | Large datasets, faster than XGBoost |
| **RandomForest** | `randomforest_factor.py` | Baseline, robust |

---

## 🚀 Creating Your Improved Models

### Step 1: Create Local Model File

```bash
# Create local directory (if not exists)
mkdir -p models/local

# Copy standard model as template
cp models/standard/xgboost_factor.py models/local/optimized_xgboost.py
```

### Step 2: Improve Your Model

```python
# models/local/optimized_xgboost.py

class XGBoostFactorModel:
    """Your optimized version with better hyperparameters."""
    
    def __init__(self, **params):
        self.params = {
            'objective': 'reg:squarederror',
            'max_depth': 8,  # Deeper trees
            'learning_rate': 0.03,  # Slower learning
            'n_estimators': 1000,  # More estimators
            'subsample': 0.9,  # Less dropout
            'colsample_bytree': 0.9,
            'random_state': 42,
            # Your custom params
            'gamma': 0.1,  # Regularization
            'min_child_weight': 3,
            **params
        }
        # ... rest of implementation
```

### Step 3: Use in Trading

Your improved models are automatically used when running:

```python
from rdagent.components.model_loader import load_model

# Auto-loads your optimized version!
model_factory = load_model("xgboost_factor")
```

---

## 🔐 Security

**What to keep in `models/local/`:**

✅ Your proprietary model architectures
✅ Optimized hyperparameters
✅ Custom feature engineering
✅ Ensemble methods
✅ Trade secrets & alpha-generating logic

**What NOT to commit to Git:**

❌ Anything in `models/local/` (already in .gitignore)
❌ Files with `.local.py` suffix
❌ Files with `_private.py` suffix

---

## 📊 Best Practices

### 1. Version Your Models

```python
# Good naming:
models/local/
├── xgboost_v2.py          # Version 2
├── xgboost_v3_optimized.py  # Version 3 optimized
└── lightgbm_lstm_hybrid_v1.py  # Hybrid v1
```

### 2. Document Changes

```python
# models/local/optimized_xgboost_v2.py
"""
XGBoost Factor Model v2.0

Changes from v1:
- Increased max_depth from 6 to 8
- Added gamma regularization
- Increased n_estimators from 500 to 1000
- Target: +2% ARR, +0.2 Sharpe

Author: Your Name
Date: 2026-04-02
"""
```

### 3. Test Performance

```python
# Compare model versions
from rdagent.components.model_loader import load_model

# Load standard
std_model = load_model("xgboost_factor", local_only=False)

# Load local (if exists)
local_model = load_model("xgboost_factor", local_only=True)

# Backtest both and compare
# ...
```

---

## 🔧 Advanced Usage

### Load All Models

```python
from rdagent.components.model_loader import list_available_models

all_models = list_available_models()
print(f"Standard: {all_models['standard']}")
print(f"Local: {all_models['local']}")
```

### Force Local Model

```python
# Raise error if local model not found
model = load_model("transformer_factor", local_only=True)
```

### Custom Model Path

```python
from rdagent.components.model_loader import load_module_from_path
from pathlib import Path

# Load from custom location
module = load_module_from_path(
    Path("/path/to/my/custom_model.py"),
    "custom_model"
)
```

---

## 📈 Model Selection Guide

| Scenario | Recommended Model | Why |
|----------|------------------|-----|
| **Tabular Factors** | XGBoost / LightGBM | Fast, interpretable |
| **Large Dataset** | LightGBM | Lower memory, faster |
| **Baseline** | RandomForest | Robust, no tuning needed |
| **Time-Series Patterns** | LSTM / GRU (local) | Sequential dependencies |
| **Multi-Scale** | TCN (local) | Different time horizons |
| **Long-Range** | Transformer (local) | Attention mechanism |
| **Best Performance** | Ensemble (local) | Combine multiple models |

---

## 🎯 Next Steps

1. **Review standard models:** `cat models/standard/*.py`
2. **Create your improved version:** `mkdir -p models/local`
3. **Test:** `python rdagent/components/model_loader.py`
4. **Run trading:** `rdagent fin_quant`

---

**Your improved models in `models/local/` are your competitive edge! 🚀**
