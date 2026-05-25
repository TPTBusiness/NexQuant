#!/usr/bin/env python3
"""Price-Action R&D Loop — Generates, evaluates, and optimizes technical strategies.

Unlike the factor-based R&D loop (CoSTEER → Docker → Qlib), this loop uses
deterministic technical indicators evaluated via backtest_signal.

Loop steps:
    1. Hypothesize: Randomly sample indicator + parameter combination
    2. Evaluate: Run backtest_signal on 1-min data
    3. Feedback: Compare against best-so-far, adjust search space
    4. Record: Save top-N strategies to results/

Usage:
    python scripts/nexquant_priceaction_loop.py --iterations 100
    python scripts/nexquant_priceaction_loop.py --live  # Continuously optimize
"""

import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
OHLCV_PATH = Path(os.getenv("PREDIX_OHLCV_PATH",
    str(PROJECT / "git_ignore_folder" / "intraday_pv_all.h5")))
RESULTS_DIR = PROJECT / "results" / "strategies_new"

# ── Indicator Library ────────────────────────────────────────────────────────

def _macd_signal(c, fast, slow, sig):
    ema_f = c.ewm(span=fast, adjust=False).mean()
    ema_s = c.ewm(span=slow, adjust=False).mean()
    ml = ema_f - ema_s
    sl = ml.ewm(span=sig, adjust=False).mean()
    s = pd.Series(0, index=c.index)
    s[ml > sl] = 1
    s[ml < sl] = -1
    return s.fillna(0).astype(int).clip(-1, 1)

def _donchian_signal(c, period, hold):
    s = pd.Series(0, index=c.index)
    s[c > c.rolling(period).max().shift(1)] = 1
    s[c < c.rolling(period).min().shift(1)] = -1
    return s.replace(0, np.nan).ffill(limit=hold).fillna(0).astype(int).clip(-1, 1)

def _rsi_signal(c, period, oversold, overbought):
    d = c.diff()
    g = d.clip(lower=0).rolling(period).mean()
    l = (-d.clip(upper=0)).rolling(period).mean()
    rsi = 100 - 100 / (1 + g / l.replace(0, 1e-8))
    s = pd.Series(0, index=c.index)
    s[rsi < oversold] = 1
    s[rsi > overbought] = -1
    return s.fillna(0).astype(int).clip(-1, 1)

def _sma_signal(c, fast, slow):
    s = pd.Series(0, index=c.index)
    s[c.rolling(fast).mean() > c.rolling(slow).mean()] = 1
    s[c.rolling(fast).mean() < c.rolling(slow).mean()] = -1
    return s.fillna(0).astype(int).clip(-1, 1)

def _bb_signal(c, period, std):
    ma = c.rolling(period).mean()
    st = c.rolling(period).std()
    s = pd.Series(0, index=c.index)
    s[c < ma - std * st] = 1
    s[c > ma + std * st] = -1
    return s.fillna(0).astype(int).clip(-1, 1)

def _atr_signal(c, period, mult):
    atr = (c.diff().abs()).rolling(period).mean()
    ma = c.rolling(period).mean()
    s = pd.Series(0, index=c.index)
    s[c > ma + mult * atr] = 1
    s[c < ma - mult * atr] = -1
    return s.replace(0, np.nan).ffill(limit=2).fillna(0).astype(int).clip(-1, 1)

def _ma_env_signal(c, period, pct):
    ma = c.rolling(period).mean()
    s = pd.Series(0, index=c.index)
    s[c < ma * (1 - pct)] = 1
    s[c > ma * (1 + pct)] = -1
    return s.replace(0, np.nan).ffill(limit=3).fillna(0).astype(int).clip(-1, 1)


def _stoch_signal(c, period, smooth):
    """Stochastic Oscillator — oversold/overbought crossover."""
    lo = c.rolling(period).min()
    hi = c.rolling(period).max()
    k = 100 * (c - lo) / (hi - lo + 1e-8)
    d = k.rolling(smooth).mean()
    s = pd.Series(0, index=c.index)
    s[(k > d) & (k < 30)] = 1
    s[(k < d) & (k > 70)] = -1
    return s.fillna(0).astype(int).clip(-1, 1)

def _cci_signal(c, h, l, period):
    """Commodity Channel Index."""
    tp = (h + l + c) / 3
    ma = tp.rolling(period).mean()
    md = (tp - ma).abs().rolling(period).mean()
    cci = (tp - ma) / (0.015 * md + 1e-8)
    s = pd.Series(0, index=c.index)
    s[cci < -100] = 1
    s[cci > 100] = -1
    return s.fillna(0).astype(int).clip(-1, 1)

def _williams_r(c, h, l, period):
    """Williams %R — overbought/oversold."""
    hi = h.rolling(period).max()
    lo = l.rolling(period).min()
    wr = -100 * (hi - c) / (hi - lo + 1e-8)
    s = pd.Series(0, index=c.index)
    s[wr < -80] = 1
    s[wr > -20] = -1
    return s.fillna(0).astype(int).clip(-1, 1)

def _roc_signal(c, period, threshold):
    """Rate of Change — momentum threshold."""
    roc = c.pct_change(period) * 100
    s = pd.Series(0, index=c.index)
    s[roc > threshold] = 1
    s[roc < -threshold] = -1
    return s.fillna(0).astype(int).clip(-1, 1)

def _ema_cross(c, fast, slow):
    """EMA Crossover (separate from SMA)."""
    ef = c.ewm(span=fast, adjust=False).mean()
    es = c.ewm(span=slow, adjust=False).mean()
    s = pd.Series(0, index=c.index)
    s[ef > es] = 1
    s[ef < es] = -1
    return s.fillna(0).astype(int).clip(-1, 1)

def _keltner(c, h, l, period, mult):
    """Keltner Channel breakout."""
    ma = c.rolling(period).mean()
    atr = ((h - l).abs()).rolling(period).mean()
    s = pd.Series(0, index=c.index)
    s[c > ma + mult * atr] = 1
    s[c < ma - mult * atr] = -1
    return s.replace(0, np.nan).ffill(limit=2).fillna(0).astype(int).clip(-1, 1)

def _adx_filter(c, h, l, period, threshold):
    """ADX trend-strength filter — only trade when ADX > threshold."""
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    up = h - h.shift()
    dn = l.shift() - l
    pdi = 100 * (up.clip(lower=0).rolling(period).mean() / (atr + 1e-8))
    ndi = 100 * (dn.clip(lower=0).rolling(period).mean() / (atr + 1e-8))
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-8)
    adx = dx.rolling(period).mean()
    s = pd.Series(0, index=c.index)
    s[(pdi > ndi) & (adx > threshold)] = 1
    s[(ndi > pdi) & (adx > threshold)] = -1
    return s.fillna(0).astype(int).clip(-1, 1)


INDICATORS = {
    "MACD":       {"params": {"fast": [3,5,8,12], "slow": [10,15,20,26,35], "sig": [3,5,9]},
                   "build": _macd_signal, "desc": "MACD({fast},{slow},{sig})"},
    "Donchian":   {"params": {"period": [5,10,20,30,50,100], "hold": [1,2,3,5]},
                   "build": _donchian_signal, "desc": "Donchian({period},{hold})"},
    "RSI":        {"params": {"period": [7,14,21], "oversold": [20,25,30,35], "overbought": [65,70,75,80]},
                   "build": _rsi_signal, "desc": "RSI({period})[{oversold}/{overbought}]"},
    "SMA_Cross":  {"params": {"fast": [5,10,20,50], "slow": [20,50,100,200]},
                   "build": _sma_signal, "desc": "SMA({fast},{slow})"},
    "EMA_Cross":  {"params": {"fast": [3,5,8,12], "slow": [15,26,50,100]},
                   "build": _ema_cross, "desc": "EMA({fast},{slow})"},
    "Bollinger":  {"params": {"period": [10,20,40], "std": [1.5,2.0,2.5]},
                   "build": _bb_signal, "desc": "BB({period},{std}s)"},
    "Keltner":    {"params": {"period": [10,20,40], "mult": [1.0,1.5,2.0,2.5]},
                   "build": lambda c, period, mult: _keltner(c, c, c, period, mult), "desc": "Keltner({period},{mult})"},
    "ATR_Channel":{"params": {"period": [10,20,40], "mult": [1.0,1.5,2.0,2.5]},
                   "build": _atr_signal, "desc": "ATR({period},{mult})"},
    "MA_Envelope":{"params": {"period": [20,50,100], "pct": [0.01,0.02,0.03,0.05]},
                   "build": _ma_env_signal, "desc": "MA_Env({period},{pct})"},
    "Stochastic": {"params": {"period": [5,9,14], "smooth": [3,5]},
                   "build": lambda c, period, smooth: _stoch_signal(c, period, smooth), "desc": "Stoch({period},{smooth})"},
    "CCI":        {"params": {"period": [14,20,50]},
                   "build": lambda c, period: _cci_signal(c, c, c, period), "desc": "CCI({period})"},
    "WilliamsR":  {"params": {"period": [7,14,21]},
                   "build": lambda c, period: _williams_r(c, c, c, period), "desc": "WR({period})"},
    "ROC_Momentum":{"params": {"period": [5,10,20], "threshold": [0.1,0.2,0.5,1.0]},
                   "build": _roc_signal, "desc": "ROC({period},{threshold}%)"},
    "ADX":        {"params": {"period": [7,14,21], "threshold": [15,20,25]},
                   "build": lambda c, period, threshold: _adx_filter(c, c, c, period, threshold), "desc": "ADX({period}>{threshold})"},
}

TIMEFRAMES = ["15min", "30min", "1h", "4h", "1d"]
VOTE_THRESHOLD = 0.25
MIN_SHARPE = 1.0
MIN_TRADES = 20
TOP_N = 20


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy Generation & Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def random_hypothesis() -> dict:
    """Generate a random strategy hypothesis."""
    strategy_type = random.choice(["single", "multi_tf", "portfolio"])

    if strategy_type == "single":
        # One indicator on one timeframe
        tf = random.choice(TIMEFRAMES)
        ind_name = random.choice(list(INDICATORS.keys()))
        ind = INDICATORS[ind_name]
        params = {k: random.choice(v) for k, v in ind["params"].items()}
        # Filter invalid combos
        if ind_name == "SMA_Cross" and params["fast"] >= params["slow"]:
            params["fast"] = min(params["fast"], params["slow"] // 2)
        if ind_name == "RSI" and params["oversold"] >= params["overbought"]:
            params["oversold"], params["overbought"] = 30, 70
        return {
            "type": "single",
            "indicator": ind_name,
            "timeframe": tf,
            "params": params,
            "description": ind['desc'].format(**params) + f" on {tf}",
        }

    elif strategy_type == "multi_tf":
        # Same indicator on multiple timeframes, majority vote
        ind_name = random.choice(list(INDICATORS.keys()))
        ind = INDICATORS[ind_name]
        tfs = random.sample(TIMEFRAMES, k=random.randint(2, 4))
        params = {k: random.choice(v) for k, v in ind["params"].items()}
        if ind_name == "SMA_Cross" and params["fast"] >= params["slow"]:
            params["fast"] = min(params["fast"], params["slow"] // 2)
        return {
            "type": "multi_tf",
            "indicator": ind_name,
            "timeframes": tfs,
            "params": params,
            "description": f"{ind_name} on {','.join(tfs)} majority-vote",
        }

    else:  # portfolio
        # Two different indicators, daily timeframe, majority vote
        i1, i2 = random.sample(list(INDICATORS.keys()), 2)
        ind1 = INDICATORS[i1]
        ind2 = INDICATORS[i2]
        p1 = {k: random.choice(v) for k, v in ind1["params"].items()}
        p2 = {k: random.choice(v) for k, v in ind2["params"].items()}
        if i1 == "SMA_Cross" and p1["fast"] >= p1["slow"]:
            p1["fast"] = min(p1["fast"], p1["slow"] // 2)
        if i2 == "SMA_Cross" and p2["fast"] >= p2["slow"]:
            p2["fast"] = min(p2["fast"], p2["slow"] // 2)
        return {
            "type": "portfolio",
            "indicators": [{"name": i1, "params": p1}, {"name": i2, "params": p2}],
            "timeframe": "1d",
            "description": f"{i1} + {i2} portfolio on daily",
        }


def build_signal(close_1min: pd.Series, hypothesis: dict) -> pd.Series:
    """Build a 1-min signal from a hypothesis."""
    hp = hypothesis

    if hp["type"] == "single":
        tf = hp["timeframe"]
        bars = close_1min.resample(tf).last().dropna()
        ind = INDICATORS[hp["indicator"]]
        s = ind["build"](bars, **hp["params"])
        return s.reindex(close_1min.index).ffill().fillna(0).astype(int).clip(-1, 1)

    elif hp["type"] == "multi_tf":
        signals = {}
        ind = INDICATORS[hp["indicator"]]
        for tf in hp["timeframes"]:
            bars = close_1min.resample(tf).last().dropna()
            signals[tf] = ind["build"](bars, **hp["params"]).reindex(
                close_1min.index).ffill().fillna(0).astype(int).clip(-1, 1)
        port = pd.DataFrame(signals).dropna()
        vote = port.mean(axis=1)
        result = pd.Series(0, index=vote.index)
        result[vote > VOTE_THRESHOLD] = 1
        result[vote < -VOTE_THRESHOLD] = -1
        return result

    else:  # portfolio
        signals = []
        daily = close_1min.resample("1d").last().dropna()
        for cfg in hp["indicators"]:
            ind = INDICATORS[cfg["name"]]
            s = ind["build"](daily, **cfg["params"])
            signals.append(s.reindex(close_1min.index).ffill().fillna(0).astype(int).clip(-1, 1))
        port = pd.DataFrame({f"s{i}": s for i, s in enumerate(signals)}).dropna()
        vote = port.mean(axis=1)
        result = pd.Series(0, index=vote.index)
        result[vote > VOTE_THRESHOLD] = 1
        result[vote < -VOTE_THRESHOLD] = -1
        return result


def evaluate_strategy(close: pd.Series, hypothesis: dict) -> dict:
    """Evaluate a strategy via backtest_signal."""
    signal = build_signal(close, hypothesis)
    from rdagent.components.backtesting.vbt_backtest import backtest_signal

    bt = backtest_signal(close=close, signal=signal)
    return {
        "hypothesis": hypothesis,
        "sharpe": bt.get("sharpe", 0) or 0,
        "monthly_pct": bt.get("monthly_return_pct", 0) or 0,
        "max_dd": bt.get("max_drawdown", 0) or 0,
        "n_trades": bt.get("n_trades", 0) or 0,
        "win_rate": bt.get("win_rate", 0) or 0,
        "total_return": bt.get("total_return", 0) or 0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main Loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_loop(iterations: int = 100, continuous: bool = False):
    print("=" * 60)
    print("  Price-Action R&D Loop")
    print("  Indicators:", ", ".join(INDICATORS.keys()))
    print("  Iterations:", iterations if not continuous else "continuous")
    print("=" * 60)

    df = pd.read_hdf(OHLCV_PATH, key="data")
    close = df.xs("EURUSD", level="instrument")["$close"].sort_index()

    top_strategies = []
    best_sharpe = 0
    total_evaluated = 0
    iteration = 0

    while True:
        iteration += 1
        if not continuous and iteration > iterations:
            break

        # Generate hypothesis
        hp = random_hypothesis()
        total_evaluated += 1

        # Evaluate
        result = evaluate_strategy(close, hp)
        result["iteration"] = iteration
        result["timestamp"] = datetime.now().isoformat()

        # Track top strategies
        if (result["sharpe"] >= MIN_SHARPE and result["n_trades"] >= MIN_TRADES
                and result["monthly_pct"] > 0):
            top_strategies.append(result)
            top_strategies.sort(key=lambda r: r["sharpe"], reverse=True)
            top_strategies = top_strategies[:TOP_N]

        # Progress
        if iteration % 10 == 0 or result["sharpe"] > best_sharpe:
            if result["sharpe"] > best_sharpe:
                best_sharpe = result["sharpe"]
                print(f"\n  ★ NEW BEST (#{iteration}): {hp['description']}")
                print(f"    Sharpe={result['sharpe']:.2f} Mon={result['monthly_pct']:.2f}% "
                      f"DD={result['max_dd']:.4f} Tr={result['n_trades']} WR={result['win_rate']:.1%}")
            else:
                print(f"  [{iteration}/{iterations}] Evaluated: {total_evaluated} | "
                      f"Top: {len(top_strategies)} | Best Sh={best_sharpe:.2f}")

        # Save checkpoint every 50 iterations
        if iteration % 50 == 0 and top_strategies:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            cp_path = RESULTS_DIR / f"pal_loop_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            cp_path.write_text(json.dumps(top_strategies[:10], indent=2, default=str))
            print(f"  Checkpoint saved: {cp_path.name}")

    # Final results
    print(f"\n{'=' * 60}")
    print(f"  Loop Complete: {total_evaluated} strategies evaluated")
    print(f"  Top strategies: {len(top_strategies)} meeting criteria")
    print(f"{'=' * 60}")

    if top_strategies:
        print(f"\n{'#':>3s} {'Description':<50s} {'Sharpe':>7s} {'Mon%':>7s} {'DD':>7s} {'Tr':>5s} {'WR':>6s}")
        print("-" * 90)
        for i, r in enumerate(top_strategies[:15], 1):
            hp = r["hypothesis"]
            print(f"{i:>3d} {hp['description'][:50]:<50s} {r['sharpe']:>+7.2f} "
                  f"{r['monthly_pct']:>+6.2f}% {r['max_dd']:>+6.4f} "
                  f"{r['n_trades']:>5d} {r['win_rate']:>6.1%}")

        final_path = RESULTS_DIR / f"pal_loop_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        final_path.write_text(json.dumps(top_strategies, indent=2, default=str))
        print(f"\n  Final results saved: {final_path}")


if __name__ == "__main__":
    import sys
    iterations = 100
    continuous = False
    if "--iterations" in sys.argv:
        idx = sys.argv.index("--iterations")
        iterations = int(sys.argv[idx + 1])
    if "--live" in sys.argv:
        continuous = True
    run_loop(iterations=iterations, continuous=continuous)
