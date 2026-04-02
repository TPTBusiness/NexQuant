# Predix Prompts

This directory contains all LLM prompts for the Predix trading agent.

---

## 📁 Directory Structure

```
prompts/
├── standard_prompts.yaml    # Default prompts (committed to Git)
├── local/                   # YOUR IMPROVED PROMPTS (not in Git!)
│   ├── factor_discovery_v2.yaml
│   ├── optimized_prompts.yaml
│   └── best_performing.yaml
└── README.md                # This file
```

---

## 🎯 How It Works

**Prompt Loading Priority:**

1. **`prompts/local/*.yaml`** ← Your improved prompts (loaded first!)
2. **`prompts/standard_prompts.yaml`** ← Default prompts (fallback)

**Example:**
```python
from rdagent.components.loader import load_prompt

# Load factor discovery prompt
# If prompts/local/factor_discovery.yaml exists → loads that
# Otherwise → loads from standard_prompts.yaml
prompt = load_prompt("factor_discovery")

# Load specific section
system_prompt = load_prompt("factor_discovery", section="system")
user_prompt = load_prompt("factor_discovery", section="user")

# Force local only (raise error if not found)
prompt = load_prompt("factor_discovery", local_only=True)
```

---

## 📝 Available Standard Prompts

| Prompt Name | Description | Used By |
|-------------|-------------|---------|
| `factor_discovery` | Generate new trading factor hypotheses | Hypothesis Agent |
| `factor_evolution` | Improve existing factors | Evolution Agent |
| `model_coder` | Generate ML model code | Model Coder Agent |
| `trading_strategy` | Design complete trading strategies | Strategy Agent |

---

## 🚀 Creating Your Improved Prompts

### Step 1: Create Local Prompt File

```bash
# Create local directory (if not exists)
mkdir -p prompts/local

# Copy standard prompt as template
cp prompts/standard_prompts.yaml prompts/local/factor_discovery_v2.yaml
```

### Step 2: Edit Your Prompt

```yaml
# prompts/local/factor_discovery_v2.yaml

factor_discovery:
  system: |-
    YOUR IMPROVED SYSTEM PROMPT HERE
    
    Add your proprietary insights:
    - Specific EURUSD patterns you've discovered
    - Your unique factor formulas
    - Custom session filters
    - Proprietary risk management rules
    
  user: |-
    YOUR IMPROVED USER PROMPT HERE
```

### Step 3: Test Your Prompt

```bash
# Test prompt loading
python rdagent/components/loader.py

# Should show:
# ✓ Loading prompt 'factor_discovery' from local: prompts/local/factor_discovery_v2.yaml
```

### Step 4: Use in Trading

Your improved prompts are automatically used when running:

```bash
rdagent fin_quant
```

The loader checks `prompts/local/` first, so your improved prompts take precedence!

---

## 🔐 Security

**What to keep in `prompts/local/`:**

✅ Your proprietary factor discovery logic
✅ Optimized prompt templates
✅ Best-performing configurations
✅ Custom evolution strategies
✅ Trade secrets & alpha-generating logic

**What NOT to commit to Git:**

❌ Anything in `prompts/local/` (already in .gitignore)
❌ Files with `.local.yaml` suffix
❌ Files with `_private.yaml` suffix

---

## 📊 Best Practices

### 1. Version Your Prompts

```yaml
# Good naming:
prompts/local/factor_discovery_v2.yaml
prompts/local/factor_discovery_v3_optimized.yaml
prompts/local/model_coder_xgboost_v1.yaml
```

### 2. Document Changes

```yaml
# Add metadata to your prompts
# prompts/local/factor_discovery_v2.yaml

# Version: 2.0
# Author: Your Name
# Date: 2026-04-02
# Changes:
#   - Added session-specific filters
#   - Improved spread cost modeling
#   - Target ARR: 12% (up from 9.62%)

factor_discovery:
  system: |-
    ...
```

### 3. Test Performance

```python
# Compare prompt versions
from rdagent.components.loader import load_prompt

# Load different versions
prompt_v1 = load_yaml_file("prompts/standard_prompts.yaml")
prompt_v2 = load_yaml_file("prompts/local/factor_discovery_v2.yaml")

# Run backtests and compare
# ...
```

### 4. Backup Your Prompts

```bash
# Backup to private repo
cd ~/Predix
git archive --format=tar prompts/local/ | gzip > ~/backups/prompts_local_$(date +%Y%m%d).tar.gz

# Or sync to private GitHub repo
git clone git@github.com:TPTBusiness/predix-prompts-private.git
cp -r prompts/local/* predix-prompts-private/
cd predix-prompts-private && git push
```

---

## 🔧 Advanced Usage

### Load All Prompts

```python
from rdagent.components.loader import load_all_prompts

all_prompts = load_all_prompts()
print(all_prompts['standard'])  # Standard prompts
print(all_prompts['local'])     # Your improved prompts
```

### List Available Prompts

```python
from rdagent.components.loader import list_available_prompts

available = list_available_prompts()
print(f"Standard: {available['standard']}")
print(f"Local: {available['local']}")
```

### Custom Prompt Path

```python
from rdagent.components.loader import load_yaml_file

# Load from custom location
custom_prompt = load_yaml_file("/path/to/my/prompts.yaml")
```

---

## 📈 Performance Tips

### 1. Be Specific

**Bad:**
```yaml
system: "Generate a good trading factor."
```

**Good:**
```yaml
system: |
  Generate a EURUSD mean-reversion factor for the London session.
  Target: 8-12% ARR, <15% max drawdown.
  Use 5-minute lookback with RSI filter.
```

### 2. Include Domain Knowledge

```yaml
system: |
  EURUSD domain knowledge:
  - London session (08:00-16:00 UTC): highest volume
  - Spread cost: 1.5 bps
  - Mean-reverting on <1h windows
  - Trending on >4h windows
```

### 3. Specify Output Format

```yaml
system: |
  Your response must be in JSON format:
  {
    "hypothesis": "...",
    "reason": "...",
    "target_session": "london/ny/asian/all",
    "expected_arr_range": "8-12%"
  }
```

### 4. Provide Examples

```yaml
user: |
  Example of a good factor:
  
  Name: Momentum_8Bar_London
  Logic: Long if 8-bar return > 0 and is_london=True
  Filter: ADX > 1.2 (trending regime)
  Expected ARR: 9.5%
  
  Now generate a NEW factor with different logic.
```

---

## 🎯 Next Steps

1. **Review standard prompts:** `cat prompts/standard_prompts.yaml`
2. **Create your improved version:** `mkdir -p prompts/local`
3. **Test:** `python rdagent/components/loader.py`
4. **Run trading:** `rdagent fin_quant`

---

**Your improved prompts in `prompts/local/` are your competitive edge! 🚀**
