#!/usr/bin/env python3
"""Price-Action R&D Loop — TA-Lib powered. 17 indicators, deterministic.

Uses TA-Lib (161 indicators) for standardized technical analysis.
Generates random strategy hypotheses and evaluates via backtest_signal.
"""

import json, os, random, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
import talib

PROJECT = Path(__file__).resolve().parent.parent
OHLCV_PATH = Path(os.getenv("PREDIX_OHLCV_PATH",
    str(PROJECT / "git_ignore_folder" / "intraday_pv_all.h5")))
RESULTS_DIR = PROJECT / "results" / "strategies_new"

TIMEFRAMES = ["15min", "30min", "1h", "4h", "1d"]
VOTE_THRESHOLD = 0.25
MIN_SHARPE, MIN_TRADES, TOP_N = 1.0, 20, 20

# ═══════════════════════════════════════════════════════════════════════════════
# Indicator functions — all use (close, high, low, volume, **params) signature
# ═══════════════════════════════════════════════════════════════════════════════

def _macd(c, h, l, v, fast, slow, sig):
    mc, sc, _ = talib.MACD(c.values.astype(np.float64), fastperiod=fast, slowperiod=slow, signalperiod=sig)
    s = pd.Series(0, index=c.index); s[mc > sc] = 1; s[mc < sc] = -1
    return s.fillna(0).astype(int).clip(-1, 1)

def _rsi(c, h, l, v, period, oversold, overbought):
    vv = talib.RSI(c.values.astype(np.float64), timeperiod=period)
    s = pd.Series(0, index=c.index); s[vv < oversold] = 1; s[vv > overbought] = -1
    return s.fillna(0).astype(int).clip(-1, 1)

def _bbands(c, h, l, v, period, std):
    up, mi, lo = talib.BBANDS(c.values.astype(np.float64), timeperiod=period, nbdevup=std, nbdevdn=std)
    s = pd.Series(0, index=c.index); s[c.values < lo] = 1; s[c.values > up] = -1
    return s.fillna(0).astype(int).clip(-1, 1)

def _stoch(c, h, l, v, fastk, slowk, slowd):
    k, d = talib.STOCH(h.values.astype(np.float64), l.values.astype(np.float64), c.values.astype(np.float64),
                       fastk_period=fastk, slowk_period=slowk, slowd_period=slowd)
    s = pd.Series(0, index=c.index); s[(k > d) & (k < 30)] = 1; s[(k < d) & (k > 70)] = -1
    return s.fillna(0).astype(int).clip(-1, 1)

def _cci(c, h, l, v, period):
    vv = talib.CCI(h.values.astype(np.float64), l.values.astype(np.float64), c.values.astype(np.float64), timeperiod=period)
    s = pd.Series(0, index=c.index); s[vv < -100] = 1; s[vv > 100] = -1
    return s.fillna(0).astype(int).clip(-1, 1)

def _willr(c, h, l, v, period):
    vv = talib.WILLR(h.values.astype(np.float64), l.values.astype(np.float64), c.values.astype(np.float64), timeperiod=period)
    s = pd.Series(0, index=c.index); s[vv < -80] = 1; s[vv > -20] = -1
    return s.fillna(0).astype(int).clip(-1, 1)

def _adx(c, h, l, v, period, threshold):
    pdi = talib.PLUS_DI(h.values.astype(np.float64), l.values.astype(np.float64), c.values.astype(np.float64), timeperiod=period)
    ndi = talib.MINUS_DI(h.values.astype(np.float64), l.values.astype(np.float64), c.values.astype(np.float64), timeperiod=period)
    adx = talib.ADX(h.values.astype(np.float64), l.values.astype(np.float64), c.values.astype(np.float64), timeperiod=period)
    s = pd.Series(0, index=c.index); s[(pdi > ndi) & (adx > threshold)] = 1; s[(ndi > pdi) & (adx > threshold)] = -1
    return s.fillna(0).astype(int).clip(-1, 1)

def _sar(c, h, l, v, accel, max_accel):
    vv = talib.SAR(h.values.astype(np.float64), l.values.astype(np.float64), acceleration=accel, maximum=max_accel)
    s = pd.Series(0, index=c.index); s[c.values > vv] = 1; s[c.values < vv] = -1
    return s.fillna(0).astype(int).clip(-1, 1)

def _roc(c, h, l, v, period, threshold):
    vv = talib.ROC(c.values.astype(np.float64), timeperiod=period)
    s = pd.Series(0, index=c.index); s[vv > threshold] = 1; s[vv < -threshold] = -1
    return s.fillna(0).astype(int).clip(-1, 1)

def _mom(c, h, l, v, period):
    vv = talib.MOM(c.values.astype(np.float64), timeperiod=period)
    s = pd.Series(0, index=c.index); s[vv > 0] = 1; s[vv < 0] = -1
    return s.fillna(0).astype(int).clip(-1, 1)

def _aroon(c, h, l, v, period):
    up, dn = talib.AROON(h.values.astype(np.float64), l.values.astype(np.float64), timeperiod=period)
    s = pd.Series(0, index=c.index); s[up > dn] = 1; s[up < dn] = -1
    return s.fillna(0).astype(int).clip(-1, 1)

def _mfi(c, h, l, v, period):
    vv = talib.MFI(h.values.astype(np.float64), l.values.astype(np.float64), c.values.astype(np.float64), v.values.astype(np.float64), timeperiod=period)
    s = pd.Series(0, index=c.index); s[vv < 20] = 1; s[vv > 80] = -1
    return s.fillna(0).astype(int).clip(-1, 1)

def _ultosc(c, h, l, v, p1, p2, p3):
    vv = talib.ULTOSC(h.values.astype(np.float64), l.values.astype(np.float64), c.values.astype(np.float64), timeperiod1=p1, timeperiod2=p2, timeperiod3=p3)
    s = pd.Series(0, index=c.index); s[vv < 30] = 1; s[vv > 70] = -1
    return s.fillna(0).astype(int).clip(-1, 1)

def _natr(c, h, l, v, period):
    vv = talib.NATR(h.values.astype(np.float64), l.values.astype(np.float64), c.values.astype(np.float64), timeperiod=period)
    m, s = vv[-200:].mean(), vv[-200:].std()
    s = pd.Series(0, index=c.index); s[c.values > m+s] = 1; s[c.values < m-s] = -1
    return s.fillna(0).astype(int).clip(-1, 1)

def _donchian(c, h, l, v, period, hold):
    hi, lo = c.rolling(period).max(), c.rolling(period).min()
    s = pd.Series(0, index=c.index); s[c > hi.shift(1)] = 1; s[c < lo.shift(1)] = -1
    return s.replace(0, np.nan).ffill(limit=hold).fillna(0).astype(int).clip(-1, 1)

def _sma(c, h, l, v, fast, slow):
    s = pd.Series(0, index=c.index)
    s[c.rolling(fast).mean() > c.rolling(slow).mean()] = 1
    s[c.rolling(fast).mean() < c.rolling(slow).mean()] = -1
    return s.fillna(0).astype(int).clip(-1, 1)

def _ema(c, h, l, v, fast, slow):
    ef, es = c.ewm(span=fast, adjust=False).mean(), c.ewm(span=slow, adjust=False).mean()
    s = pd.Series(0, index=c.index); s[ef > es] = 1; s[ef < es] = -1
    return s.fillna(0).astype(int).clip(-1, 1)

# ═══════════════════════════════════════════════════════════════════════════════

INDICATORS = {
    "MACD":    ({"fast":[3,5,8,12], "slow":[10,15,20,26,35], "sig":[3,5,9]}, _macd, "MACD({fast},{slow},{sig})"),
    "RSI":     ({"period":[7,14,21], "oversold":[20,25,30,35], "overbought":[65,70,75,80]}, _rsi, "RSI({period})[{oversold}/{overbought}]"),
    "BBands":  ({"period":[10,20,40], "std":[1.5,2.0,2.5]}, _bbands, "BB({period},{std}s)"),
    "Stoch":   ({"fastk":[5,9,14], "slowk":[3], "slowd":[3,5]}, _stoch, "Stoch({fastk},{slowk},{slowd})"),
    "CCI":     ({"period":[14,20,50]}, _cci, "CCI({period})"),
    "WillR":   ({"period":[7,14,21]}, _willr, "WR({period})"),
    "ADX":     ({"period":[7,14,21], "threshold":[15,20,25]}, _adx, "ADX({period}>{threshold})"),
    "SAR":     ({"accel":[0.02,0.05,0.08], "max_accel":[0.2,0.3,0.5]}, _sar, "SAR({accel},{max_accel})"),
    "ROC":     ({"period":[5,10,20], "threshold":[0.1,0.2,0.5]}, _roc, "ROC({period},{threshold}%)"),
    "MOM":     ({"period":[5,10,20,50]}, _mom, "MOM({period})"),
    "AROON":   ({"period":[7,14,21]}, _aroon, "AROON({period})"),
    "MFI":     ({"period":[7,14,21]}, _mfi, "MFI({period})"),
    "UltOsc":  ({"p1":[7], "p2":[14], "p3":[28]}, _ultosc, "UltOsc(7,14,28)"),
    "NATR":    ({"period":[7,14,21]}, _natr, "NATR({period})"),
    "Donchian":({"period":[5,10,20,30,50,100], "hold":[1,2,3,5]}, _donchian, "Donchian({period},{hold})"),
    "SMA":     ({"fast":[5,10,20,50], "slow":[20,50,100,200]}, _sma, "SMA({fast},{slow})"),
    "EMA":     ({"fast":[3,5,8,12], "slow":[15,26,50,100]}, _ema, "EMA({fast},{slow})"),
}

# ═══════════════════════════════════════════════════════════════════════════════
# Strategy generation
# ═══════════════════════════════════════════════════════════════════════════════

def _resample_ohlc(close_1min, tf):
    """Resample to timeframe, producing OHLCV bars."""
    bars = close_1min.resample(tf).ohlc()
    # Flatten MultiIndex columns
    o = bars['close']['close'] if isinstance(bars.columns, pd.MultiIndex) else bars['close']
    h = bars['high']['high'] if isinstance(bars.columns, pd.MultiIndex) else bars['high']
    l = bars['low']['low'] if isinstance(bars.columns, pd.MultiIndex) else bars['low']
    c = bars['close']['close'] if isinstance(bars.columns, pd.MultiIndex) else bars['close']
    v = pd.Series(1000, index=c.index)  # dummy volume
    return c, h, l, v

def random_hypothesis():
    stype = random.choice(["single", "multi_tf", "portfolio"])
    if stype == "single":
        ind_name = random.choice(list(INDICATORS.keys()))
        params_def, _, desc_tpl = INDICATORS[ind_name]
        params = {k: random.choice(v) for k, v in params_def.items()}
        if ind_name == "SMA" and params["fast"] >= params["slow"]:
            params["fast"] = min(params["fast"], params["slow"] // 2)
        return {"type": "single", "indicator": ind_name, "timeframe": random.choice(TIMEFRAMES),
                "params": params, "description": desc_tpl.format(**params)}
    elif stype == "multi_tf":
        ind_name = random.choice(list(INDICATORS.keys()))
        params_def, _, desc_tpl = INDICATORS[ind_name]
        params = {k: random.choice(v) for k, v in params_def.items()}
        tfs = random.sample(TIMEFRAMES, k=random.randint(2, 4))
        return {"type": "multi_tf", "indicator": ind_name, "timeframes": tfs,
                "params": params, "description": f"{ind_name} on {','.join(tfs)} maj-vote"}
    else:
        i1, i2 = random.sample(list(INDICATORS.keys()), 2)
        p1_def, _, _ = INDICATORS[i1]; p2_def, _, _ = INDICATORS[i2]
        p1 = {k: random.choice(v) for k, v in p1_def.items()}
        p2 = {k: random.choice(v) for k, v in p2_def.items()}
        return {"type": "portfolio", "indicators": [{"name": i1, "params": p1}, {"name": i2, "params": p2}],
                "timeframe": "1d", "description": f"{i1} + {i2} portfolio daily"}


def build_signal(close_1min, hypothesis):
    hp = hypothesis
    if hp["type"] == "single":
        _, fn, _ = INDICATORS[hp["indicator"]]
        c, h, l, v = _resample_ohlc(close_1min, hp["timeframe"])
        s = fn(c, h, l, v, **hp["params"])
        return s.reindex(close_1min.index).ffill().fillna(0).astype(int).clip(-1, 1)
    elif hp["type"] == "multi_tf":
        _, fn, _ = INDICATORS[hp["indicator"]]
        sigs = {}
        for tf in hp["timeframes"]:
            c, h, l, v = _resample_ohlc(close_1min, tf)
            sigs[tf] = fn(c, h, l, v, **hp["params"]).reindex(close_1min.index).ffill().fillna(0).astype(int).clip(-1, 1)
        port_df = pd.DataFrame(sigs).dropna()
        vote = port_df.mean(axis=1)
        result = pd.Series(0, index=vote.index)
        result[vote > VOTE_THRESHOLD] = 1; result[vote < -VOTE_THRESHOLD] = -1
        return result
    else:
        sigs = []
        daily, dh, dl, dv = _resample_ohlc(close_1min, "1d")
        for cfg in hp["indicators"]:
            _, fn, _ = INDICATORS[cfg["name"]]
            s = fn(daily, dh, dl, dv, **cfg["params"]).reindex(close_1min.index).ffill().fillna(0).astype(int).clip(-1, 1)
            sigs.append(s)
        port_df = pd.DataFrame({f"s{i}": s for i, s in enumerate(sigs)}).dropna()
        vote = port_df.mean(axis=1)
        result = pd.Series(0, index=vote.index)
        result[vote > VOTE_THRESHOLD] = 1; result[vote < -VOTE_THRESHOLD] = -1
        return result


def evaluate(hp, close):
    signal = build_signal(close, hp)
    from rdagent.components.backtesting.vbt_backtest import backtest_signal
    bt = backtest_signal(close=close, signal=signal)
    return {"hypothesis": hp, "sharpe": bt.get("sharpe", 0) or 0,
            "monthly_pct": bt.get("monthly_return_pct", 0) or 0,
            "max_dd": bt.get("max_drawdown", 0) or 0, "n_trades": bt.get("n_trades", 0) or 0,
            "win_rate": bt.get("win_rate", 0) or 0}

# ═══════════════════════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    iterations = 100; continuous = False
    if "--iterations" in sys.argv:
        iterations = int(sys.argv[sys.argv.index("--iterations") + 1])
    if "--live" in sys.argv: continuous = True

    print("=" * 60)
    print(f"  Price-Action R&D Loop — TA-Lib ({len(INDICATORS)} indicators)")
    print(f"  Iterations: {'continuous' if continuous else iterations}")
    print("=" * 60)

    df = pd.read_hdf(OHLCV_PATH, key="data")
    close = df.xs("EURUSD", level="instrument")["$close"].sort_index()

    top, best_sh, total, iteration = [], 0, 0, 0

    while True:
        iteration += 1
        if not continuous and iteration > iterations: break

        hp = random_hypothesis()
        result = evaluate(hp, close)
        result["iteration"] = iteration
        result["timestamp"] = datetime.now().isoformat()
        total += 1

        if result["sharpe"] >= MIN_SHARPE and result["n_trades"] >= MIN_TRADES and result["monthly_pct"] > 0:
            top.append(result)
            top.sort(key=lambda r: r["sharpe"], reverse=True)
            top = top[:TOP_N]

        if iteration % 10 == 0 or result["sharpe"] > best_sh:
            if result["sharpe"] > best_sh:
                best_sh = result["sharpe"]
                print(f"\n  * NEW BEST (#{iteration}): {hp['description']}")
                print(f"    Sharpe={result['sharpe']:.2f} Mon={result['monthly_pct']:.2f}% "
                      f"DD={result['max_dd']:.4f} Tr={result['n_trades']} WR={result['win_rate']:.1%}")
            else:
                print(f"  [{iteration}/{iterations}] Evals: {total} | Top: {len(top)} | Best Sh={best_sh:.2f}")

        if iteration % 50 == 0 and top:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            cp = RESULTS_DIR / f"pal_talib_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            cp.write_text(json.dumps(top[:10], indent=2, default=str))
            print(f"  Checkpoint: {cp.name}")

    # Final
    print(f"\n{'=' * 60}")
    print(f"  Done: {total} evaluated, {len(top)} strategies")
    if top:
        print(f"\n{'#':>3s} {'Strategy':<50s} {'Sharpe':>7s} {'Mon%':>7s} {'DD':>7s} {'Tr':>5s}")
        print("-" * 80)
        for i, r in enumerate(top[:15], 1):
            print(f"{i:>3d} {r['hypothesis']['description'][:50]:<50s} {r['sharpe']:>+7.2f} {r['monthly_pct']:>+6.2f}% {r['max_dd']:>+6.4f} {r['n_trades']:>5d}")
        final = RESULTS_DIR / f"pal_talib_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        final.write_text(json.dumps(top, indent=2, default=str))
        print(f"\n  Saved: {final}")


if __name__ == "__main__":
    main()
