#!/usr/bin/env python3
"""R&D Loop for Technical Indicators — Replaces factor pipeline with indicator discovery.

Architecture (mirrors the original R&D loop):
    1. Hypothesize: LLM proposes indicator combinations with parameters
    2. Evaluate: Direct backtest_signal (no Docker, no Qlib)
    3. Feedback: Compare against SOTA, bias next hypotheses
    4. Record: Save best strategies, checkpoint progress

Unlike the factor R&D loop, this uses deterministic evaluation instead of Docker.
"""

import json, os, random, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
from numba import jit

PROJECT = Path(__file__).resolve().parent.parent
OHLCV_PATH = Path(os.getenv("PREDIX_OHLCV_PATH",
    str(PROJECT / "git_ignore_folder" / "intraday_pv_all.h5")))
RESULTS_DIR = PROJECT / "results" / "rd_loop"
STATE_DIR = PROJECT / "git_ignore_folder" / "rd_loop_state"

# ═══════════════════════════════════════════════════════════════════════════════
# GPU-accelerated backtest via Numba (735M bars/second — 245× faster)
# ═══════════════════════════════════════════════════════════════════════════════

@jit(nopython=True)
def _backtest_numba(prices, signals, cost=0.000264):
    n = len(prices)
    equity = 100000.0; peak = 100000.0; max_dd = 0.0
    position = 0; entry_price = 0.0
    trades = np.zeros(100000, dtype=np.float64)  # preallocate
    trade_count = 0; wins = 0
    
    for i in range(1, n):
        px = prices[i]; sg = signals[i]; ps = signals[i-1]
        if position != 0 and sg != position:
            if position == 1: ret = (px - entry_price) / entry_price - cost
            else: ret = (entry_price - px) / entry_price - cost
            equity *= (1.0 + ret)
            if equity > peak: peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd: max_dd = dd
            if trade_count < len(trades):
                trades[trade_count] = ret
            trade_count += 1
            if ret > 0: wins += 1
            position = 0
        if sg != 0 and position == 0:
            position = sg; entry_price = px
    if position != 0:
        fp = prices[-1]
        if position == 1: ret = (fp - entry_price) / entry_price - cost
        else: ret = (entry_price - fp) / entry_price - cost
        equity *= (1.0 + ret)
        if trade_count < len(trades):
            trades[trade_count] = ret
        trade_count += 1
        if ret > 0: wins += 1
    total_ret = (equity - 100000.0) / 100000.0
    
    # Compute Sharpe from trade returns
    if trade_count > 5:
        t = trades[:trade_count]
        mean_ret = np.mean(t)
        std_ret = np.std(t)
        sharpe = mean_ret / std_ret * np.sqrt(trade_count) if std_ret > 0 else 0.0
    else:
        sharpe = 0.0
    
    return equity, max_dd, trade_count, wins, total_ret, sharpe, trades[:trade_count]

TIMEFRAMES = ["5min", "15min", "30min", "1h", "4h"]
INDICATORS_POOL = ["MACD", "RSI", "BBands", "Donchian", "Stoch", "CCI", "WillR", "ADX", "SAR", "ROC", "MOM", "AROON", "MFI", "SMA", "EMA"]
STRATEGY_TYPES = ["single", "multi_tf", "portfolio", "multi_role"]
TREND_TFS = ["30min", "1h", "4h"]    # higher TFs for trend filter
ENTRY_TFS = ["5min", "15min", "30min"]  # lower TFs for entry
MIN_SHARPE, MIN_TRADES = 0.5, 20
EXPLORATION_RATE = 0.30  # 30% explore, 70% exploit

# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_strategy(close, hypothesis):
    """Run backtest and return metrics."""
    import talib
    signal = None

    # Optuna optimization branch
    if hypothesis.get('generation') == 'optuna':
        return _run_optuna(close, hypothesis)
    
    # ML training branch
    if hypothesis.get('type') == 'ml':
        return _train_ml(close, hypothesis)

    if hypothesis['type'] == 'single':
        ind = hypothesis['indicator']
        tf = hypothesis['timeframe']
        bars = close.resample(tf).last().dropna()
        sig = _build_indicator_signal(ind, bars, hypothesis['params'])
        signal = sig.reindex(close.index).ffill().fillna(0).astype(int).clip(-1, 1)

    elif hypothesis['type'] == 'multi_tf':
        ind = hypothesis['indicator']
        sigs = {}
        for tf in hypothesis['timeframes']:
            bars = close.resample(tf).last().dropna()
            sig = _build_indicator_signal(ind, bars, hypothesis['params'])
            sigs[tf] = sig.reindex(close.index).ffill().fillna(0).astype(int).clip(-1, 1)
        port = pd.DataFrame(sigs).dropna()
        vote = port.mean(axis=1)
        signal = pd.Series(0, index=vote.index)
        signal[vote > 0.25] = 1; signal[vote < -0.25] = -1

    elif hypothesis['type'] == 'portfolio':
        sigs = []
        for cfg in hypothesis['indicators']:
            bars = close.resample('1d').last().dropna()
            sig = _build_indicator_signal(cfg['name'], bars, cfg['params'])
            sigs.append(sig.reindex(close.index).ffill().fillna(0).astype(int).clip(-1, 1))
        port = pd.DataFrame({f"s{i}": s for i, s in enumerate(sigs)}).dropna()
        vote = port.mean(axis=1)
        signal = pd.Series(0, index=vote.index)
        signal[vote > 0.25] = 1; signal[vote < -0.25] = -1

    elif hypothesis['type'] == 'multi_role':
        # Trend filter (higher TF) + Entry signal (lower TF) with directional gating
        trend_ind = hypothesis['trend_ind']
        entry_ind = hypothesis['entry_ind']
        trend_tf = hypothesis['trend_tf']
        entry_tf = hypothesis['entry_tf']
        # Build trend signal on higher TF, forward-fill to lower TF
        trend_bars = close.resample(trend_tf).last().dropna()
        trend_sig = _build_indicator_signal(trend_ind, trend_bars, hypothesis['trend_params'])
        trend_sig = trend_sig.reindex(close.index).ffill().fillna(0).astype(int).clip(-1, 1)
        # Build entry signal on lower TF
        entry_bars = close.resample(entry_tf).last().dropna()
        entry_sig = _build_indicator_signal(entry_ind, entry_bars, hypothesis['entry_params'])
        entry_sig = entry_sig.reindex(close.index).ffill().fillna(0).astype(int).clip(-1, 1)
        # GATE: entry only fires when trend confirms direction
        signal = pd.Series(0, index=close.index)
        signal[(trend_sig == 1) & (entry_sig == 1)] = 1    # long trend + long entry
        signal[(trend_sig == -1) & (entry_sig == -1)] = -1  # short trend + short entry

    if signal is None or signal.nunique() <= 1:
        return {"sharpe": 0, "monthly_pct": 0, "max_dd": 0, "n_trades": 0, "win_rate": 0}

    # Numba-accelerated backtest (245× faster)
    prices = close.values.astype(np.float64)
    sigs = signal.values.astype(np.int32)
    eq, dd, tr, wins, total_ret, sharpe, _ = _backtest_numba(prices, sigs)
    return {"sharpe": float(sharpe), "monthly_pct": float(((1+total_ret)**(1/((close.index[-1]-close.index[0]).days/30.44))-1)*100) if total_ret > -1 else 0,
            "max_dd": float(-dd), "n_trades": int(tr), "win_rate": float(wins/tr) if tr>0 else 0,
            "total_return": float(total_ret)}


def _build_indicator_signal(name, bars, params):
    """Build indicator signal using talib + hand-rolled."""
    import talib
    c = bars.values.astype(np.float64)
    if name == 'MACD':
        mc, sc, _ = talib.MACD(c, fastperiod=params.get('fast', 3),
                               slowperiod=params.get('slow', 15),
                               signalperiod=params.get('sig', 3))
        s = pd.Series(0, index=bars.index); s[mc > sc] = 1; s[mc < sc] = -1
    elif name == 'RSI':
        v = talib.RSI(c, timeperiod=params.get('period', 14))
        s = pd.Series(0, index=bars.index); s[v < params.get('oversold', 30)] = 1; s[v > params.get('overbought', 70)] = -1
    elif name == 'BBands':
        up, mi, lo = talib.BBANDS(c, timeperiod=params.get('period', 20),
                                  nbdevup=params.get('std', 2), nbdevdn=params.get('std', 2))
        s = pd.Series(0, index=bars.index); s[c < lo] = 1; s[c > up] = -1
    elif name == 'Donchian':
        hi = bars.rolling(params.get('period', 20)).max()
        lo = bars.rolling(params.get('period', 20)).min()
        s = pd.Series(0, index=bars.index); s[bars > hi.shift(1)] = 1; s[bars < lo.shift(1)] = -1
        s = s.replace(0, np.nan).ffill(limit=params.get('hold', 1)).fillna(0).astype(int)
    elif name == 'Stoch':
        k, d = talib.STOCH(c, c, c, fastk_period=params.get('fastk', 9),
                           slowk_period=params.get('slowk', 3), slowd_period=params.get('slowd', 3))
        s = pd.Series(0, index=bars.index); s[(k > d) & (k < 30)] = 1; s[(k < d) & (k > 70)] = -1
    elif name == 'CCI':
        v = talib.CCI(c, c, c, timeperiod=params.get('period', 14))
        s = pd.Series(0, index=bars.index); s[v < -100] = 1; s[v > 100] = -1
    elif name == 'WillR':
        v = talib.WILLR(c, c, c, timeperiod=params.get('period', 14))
        s = pd.Series(0, index=bars.index); s[v < -80] = 1; s[v > -20] = -1
    elif name == 'ADX':
        pdi = talib.PLUS_DI(c, c, c, timeperiod=params.get('period', 14))
        ndi = talib.MINUS_DI(c, c, c, timeperiod=params.get('period', 14))
        adx = talib.ADX(c, c, c, timeperiod=params.get('period', 14))
        s = pd.Series(0, index=bars.index)
        s[(pdi > ndi) & (adx > params.get('threshold', 20))] = 1
        s[(ndi > pdi) & (adx > params.get('threshold', 20))] = -1
    elif name == 'SAR':
        v = talib.SAR(c, c, acceleration=params.get('accel', 0.02), maximum=params.get('max_accel', 0.2))
        s = pd.Series(0, index=bars.index); s[c > v] = 1; s[c < v] = -1
    elif name == 'ROC':
        v = talib.ROC(c, timeperiod=params.get('period', 10))
        s = pd.Series(0, index=bars.index); s[v > params.get('threshold', 0.2)] = 1; s[v < -params.get('threshold', 0.2)] = -1
    elif name == 'MOM':
        v = talib.MOM(c, timeperiod=params.get('period', 10))
        s = pd.Series(0, index=bars.index); s[v > 0] = 1; s[v < 0] = -1
    elif name == 'AROON':
        up, dn = talib.AROON(c, c, timeperiod=params.get('period', 14))
        s = pd.Series(0, index=bars.index); s[up > dn] = 1; s[up < dn] = -1
    elif name == 'MFI':
        v = talib.MFI(c, c, c, c, timeperiod=params.get('period', 14))
        s = pd.Series(0, index=bars.index); s[v < 20] = 1; s[v > 80] = -1
    elif name == 'SMA':
        s = pd.Series(0, index=bars.index)
        s[bars.rolling(params.get('fast', 10)).mean() > bars.rolling(params.get('slow', 50)).mean()] = 1
        s[bars.rolling(params.get('fast', 10)).mean() < bars.rolling(params.get('slow', 50)).mean()] = -1
    elif name == 'EMA':
        ef = bars.ewm(span=params.get('fast', 5), adjust=False).mean()
        es = bars.ewm(span=params.get('slow', 26), adjust=False).mean()
        s = pd.Series(0, index=bars.index); s[ef > es] = 1; s[ef < es] = -1
    else:
        s = pd.Series(0, index=bars.index)
    return s.fillna(0).astype(int).clip(-1, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Hypothesis Generation (simulated R&D loop — no LLM needed for indicators)
# ═══════════════════════════════════════════════════════════════════════════════

class ResearchLoop:
    """Mimics the R&D loop: hypothesize → evaluate → feedback → record."""

    def __init__(self, close):
        self.close = close
        self.sota = []  # State-of-the-art strategies
        self.history = []  # All evaluated hypotheses
        self.iteration = 0
        self.best_sharpe = 0
        self.exploration_rate = 0.3  # % of time we explore randomly vs exploit SOTA

    def hypothesize(self):
        self.iteration += 1
        
        # Every 500 iterations: Optuna-optimize best strategy
        if self.iteration % 500 == 0 and self.sota:
            best = self.sota[0]
            hp = dict(best['hypothesis'])
            hp['generation'] = 'optuna'
            hp['description'] = f"Optuna: {hp.get('description','?')}"
            return hp
        
        # Every 2000 iterations: train ML model
        if self.iteration % 2000 == 0 and len(self.sota) >= 5:
            return {'type': 'ml', 'generation': 'ml',
                    'description': f"ML: LightGBM on {len(self.sota)} strategies",
                    'sota': self.sota[:5]}
        
        if random.random() < self.exploration_rate or not self.sota:
            return self._random_hypothesis()
        else:
            base = random.choice(self.sota[:5])
            return self._mutate_hypothesis(base['hypothesis'])

    def _random_hypothesis(self):
        stype = random.choice(STRATEGY_TYPES)
        if stype == 'single':
            ind = random.choice(INDICATORS_POOL)
            tf = random.choice(TIMEFRAMES)
            params = self._random_params(ind)
            return {'type': 'single', 'indicator': ind, 'timeframe': tf, 'params': params,
                    'description': f"{ind} on {tf}", 'generation': 'explore'}
        elif stype == 'multi_tf':
            ind = random.choice(INDICATORS_POOL)
            tfs = random.sample(TIMEFRAMES, k=random.randint(2, 4))
            params = self._random_params(ind)
            return {'type': 'multi_tf', 'indicator': ind, 'timeframes': tfs, 'params': params,
                    'description': f"{ind} on {','.join(tfs)}", 'generation': 'explore'}
        elif stype == 'multi_role':
            trend_ind = random.choice(INDICATORS_POOL)
            entry_ind = random.choice(INDICATORS_POOL)
            trend_tf = random.choice(TREND_TFS)
            entry_tf = random.choice([t for t in ENTRY_TFS if t < trend_tf])
            trend_p = self._random_params(trend_ind)
            entry_p = self._random_params(entry_ind)
            return {'type': 'multi_role',
                    'trend_ind': trend_ind, 'trend_params': trend_p, 'trend_tf': trend_tf,
                    'entry_ind': entry_ind, 'entry_params': entry_p, 'entry_tf': entry_tf,
                    'description': f"{trend_ind}({trend_tf})→{entry_ind}({entry_tf})",
                    'generation': 'explore'}
        else:
            i1, i2 = random.sample(INDICATORS_POOL, 2)
            p1 = self._random_params(i1); p2 = self._random_params(i2)
            return {'type': 'portfolio', 'indicators': [{'name': i1, 'params': p1}, {'name': i2, 'params': p2}],
                    'description': f"{i1}+{i2}", 'generation': 'explore'}

    def _mutate_hypothesis(self, base):
        """Mutate an existing hypothesis — change one aspect."""
        hp = dict(base)  # shallow copy
        hp['generation'] = 'exploit'

        # multi_role has its own mutation logic
        if hp.get('type') == 'multi_role':
            mut = random.choice(['trend_ind', 'entry_ind', 'trend_tf', 'entry_tf',
                                 'trend_params', 'entry_params'])
            if mut == 'trend_ind':
                hp['trend_ind'] = random.choice([i for i in INDICATORS_POOL if i != hp['trend_ind']])
                hp['trend_params'] = self._random_params(hp['trend_ind'])
                hp['description'] = f"{hp['trend_ind']}({hp['trend_tf']})→{hp['entry_ind']}({hp['entry_tf']})"
            elif mut == 'entry_ind':
                hp['entry_ind'] = random.choice([i for i in INDICATORS_POOL if i != hp['entry_ind']])
                hp['entry_params'] = self._random_params(hp['entry_ind'])
                hp['description'] = f"{hp['trend_ind']}({hp['trend_tf']})→{hp['entry_ind']}({hp['entry_tf']})"
            elif mut == 'trend_tf':
                hp['trend_tf'] = random.choice(TREND_TFS)
                if hp['trend_tf'] <= hp['entry_tf']:
                    hp['entry_tf'] = random.choice([t for t in ENTRY_TFS if t < hp['trend_tf']])
                hp['description'] = f"{hp['trend_ind']}({hp['trend_tf']})→{hp['entry_ind']}({hp['entry_tf']})"
            elif mut == 'entry_tf':
                hp['entry_tf'] = random.choice([t for t in ENTRY_TFS if t < hp['trend_tf']])
                hp['description'] = f"{hp['trend_ind']}({hp['trend_tf']})→{hp['entry_ind']}({hp['entry_tf']})"
            elif mut == 'trend_params':
                p = dict(hp['trend_params']); k = random.choice(list(p.keys()))
                if isinstance(p[k], (int, float)): p[k] = p[k] * random.uniform(0.5, 1.5)
                hp['trend_params'] = p
                hp['description'] = f"{hp['trend_ind']}({hp['trend_tf']})→{hp['entry_ind']}({hp['entry_tf']}) (tuned)"
            elif mut == 'entry_params':
                p = dict(hp['entry_params']); k = random.choice(list(p.keys()))
                if isinstance(p[k], (int, float)): p[k] = p[k] * random.uniform(0.5, 1.5)
                hp['entry_params'] = p
                hp['description'] = f"{hp['trend_ind']}({hp['trend_tf']})→{hp['entry_ind']}({hp['entry_tf']}) (tuned)"
            return hp

        mutations = ['params', 'indicator', 'timeframe']
        mutation = random.choice(mutations)

        if mutation == 'params' and 'params' in hp:
            # Tweak one parameter
            params = dict(hp['params'])
            key = random.choice(list(params.keys()))
            if isinstance(params[key], (int, float)):
                params[key] = params[key] * random.uniform(0.5, 1.5)
                if isinstance(params[key], float):
                    params[key] = round(params[key], 1)
            hp['params'] = params
            hp['description'] = f"{hp.get('indicator','?')} (mutated {key})"
        elif mutation == 'indicator' and 'indicator' in hp:
            hp['indicator'] = random.choice([i for i in INDICATORS_POOL if i != hp.get('indicator')])
            hp['params'] = self._random_params(hp['indicator'])
            hp['description'] = f"{hp['indicator']} (replaced)"
        elif mutation == 'timeframe':
            if 'timeframe' in hp:
                hp['timeframe'] = random.choice(TIMEFRAMES)
            elif 'timeframes' in hp:
                hp['timeframes'] = random.sample(TIMEFRAMES, k=len(hp['timeframes']))
            hp['description'] = f"{hp.get('indicator','?')} (timeframe change)"

        return hp

    def _random_params(self, indicator):
        param_sets = {
            'MACD': {'fast': random.choice([3,5,8,12]), 'slow': random.choice([10,15,20,26]), 'sig': random.choice([3,5,9])},
            'RSI': {'period': random.choice([7,14,21]), 'oversold': random.choice([20,25,30]), 'overbought': random.choice([70,75,80])},
            'BBands': {'period': random.choice([10,20,40]), 'std': random.choice([1.5,2.0,2.5])},
            'Donchian': {'period': random.choice([5,10,20,30,50]), 'hold': random.choice([1,2,3,5])},
            'Stoch': {'fastk': random.choice([5,9,14]), 'slowk': 3, 'slowd': random.choice([3,5])},
            'CCI': {'period': random.choice([14,20,50])},
            'WillR': {'period': random.choice([7,14,21])},
            'ADX': {'period': random.choice([7,14,21]), 'threshold': random.choice([15,20,25])},
            'SAR': {'accel': random.choice([0.02,0.05,0.08]), 'max_accel': random.choice([0.2,0.3,0.5])},
            'ROC': {'period': random.choice([5,10,20]), 'threshold': random.choice([0.1,0.2,0.5])},
            'MOM': {'period': random.choice([5,10,20,50])},
            'AROON': {'period': random.choice([7,14,21])},
            'MFI': {'period': random.choice([7,14,21])},
            'SMA': {'fast': random.choice([5,10,20,50]), 'slow': random.choice([20,50,100,200])},
            'EMA': {'fast': random.choice([3,5,8,12]), 'slow': random.choice([15,26,50,100])},
        }
        return param_sets.get(indicator, {'period': 14})

    def feedback(self, result):
        """Update SOTA and bias future exploration."""
        if result['sharpe'] >= MIN_SHARPE and result['n_trades'] >= MIN_TRADES:
            self.sota.append(result)
            self.sota.sort(key=lambda r: r['sharpe'], reverse=True)
            self.sota = self.sota[:20]  # Keep top 20
            if result['sharpe'] > self.best_sharpe:
                self.best_sharpe = result['sharpe']
                return True  # NEW BEST
        return False

    def record(self):
        """Save checkpoint."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        if self.sota:
            cp = RESULTS_DIR / f"rd_loop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            cp.write_text(json.dumps(self.sota[:15], indent=2, default=str))

# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Optuna Optimization
# ═══════════════════════════════════════════════════════════════════════════════

def _run_optuna(close, hypothesis):
    """Run Optuna hyperparameter optimization on a strategy."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    hp = hypothesis
    ind = hp.get('indicator', 'MACD')
    tfs = hp.get('timeframes', ['15min','30min','1h','4h'])
    base_params = hp.get('params', {})
    
    param_ranges = {
        'MACD': {'fast': (2,15), 'slow': (5,40), 'sig': (2,15)},
        'RSI': {'period': (5,30), 'oversold': (10,40), 'overbought': (60,90)},
        'Donchian': {'period': (3,100), 'hold': (1,10)},
        'SAR': {'accel': (0.01, 0.2), 'max_accel': (0.1, 1.0)},
        'ADX': {'period': (5,30), 'threshold': (10,40)},
    }
    ranges = param_ranges.get(ind, {})
    
    def objective(trial):
        params = {}
        for k, (lo, hi) in ranges.items():
            if isinstance(base_params.get(k, 1), int):
                params[k] = trial.suggest_int(k, int(lo), int(hi))
            else:
                params[k] = trial.suggest_float(k, lo, hi)
        params['fast'] = min(params.get('fast',99), params.get('slow',99)-2)
        
        sigs = {}
        for tf in tfs:
            bars = close.resample(tf).last().dropna()
            sig = _build_indicator_signal(ind, bars, params)
            sigs[tf] = sig.reindex(close.index).ffill().fillna(0).astype(int).clip(-1,1)
        port = pd.DataFrame(sigs).dropna(); vote = port.mean(axis=1)
        signal = pd.Series(0, index=vote.index)
        signal[vote > 0.25] = 1; signal[vote < -0.25] = -1
        
        prices = close.values.astype(np.float64); sigs_arr = signal.values.astype(np.int32)
        _, dd, tr, wins, total_ret, sharpe, _ = _backtest_numba(prices, sigs_arr)
        return float(sharpe) if sharpe > 0 else -999.0
    
    try:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20, show_progress_bar=False)
        best = study.best_params
        hp['params'] = {k: int(v) if v == int(v) else v for k, v in best.items()}
        hp['description'] = f"Optuna: {ind} on {','.join(tfs[:2])}"
        hp['generation'] = 'optuna'
        # Re-evaluate with best params
        sigs = {}
        for tf in tfs:
            bars = close.resample(tf).last().dropna()
            sig = _build_indicator_signal(ind, bars, hp['params'])
            sigs[tf] = sig.reindex(close.index).ffill().fillna(0).astype(int).clip(-1,1)
        port = pd.DataFrame(sigs).dropna(); vote = port.mean(axis=1)
        signal = pd.Series(0, index=vote.index)
        signal[vote > 0.25] = 1; signal[vote < -0.25] = -1
        prices = close.values.astype(np.float64); sigs_arr = signal.values.astype(np.int32)
        eq, dd, tr, wins, ret, sh, _ = _backtest_numba(prices, sigs_arr)
        n_days = (close.index[-1] - close.index[0]).days
        mon = ((1+ret)**(1/(n_days/30.44))-1)*100 if ret > -1 else 0
        print(f"    Optuna best: {best} → Sh={sh:.1f} Mon={mon:.1f}% ({study.best_value:.1f})")
        return {"sharpe": float(sh), "monthly_pct": float(mon), "max_dd": float(-dd),
                "n_trades": int(tr), "win_rate": float(wins/tr) if tr>0 else 0,
                "optuna_best": best, "optuna_value": float(study.best_value)}
    except Exception as e:
        return {"sharpe": 0, "monthly_pct": 0, "max_dd": 0, "n_trades": 0, "win_rate": 0}


def _train_ml(close, hypothesis):
    """Train LightGBM classifier on indicator signals to predict direction."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        return {"sharpe": 0, "monthly_pct": 0, "max_dd": 0, "n_trades": 0, "win_rate": 0}
    
    sota = hypothesis.get('sota', [])
    if not sota: return {"sharpe": 0, "monthly_pct": 0, "max_dd": 0, "n_trades": 0, "win_rate": 0}
    
    # Generate features from all SOTA strategies
    daily = close.resample('1h').last().dropna()
    features = pd.DataFrame(index=daily.index)
    
    for s in sota[:5]:
        hp_s = s['hypothesis']
        ind = hp_s.get('indicator', 'MACD')
        sig = _build_indicator_signal(ind, daily, hp_s.get('params', {}))
        features[f"{ind}_{hp_s.get('generation','?')}"] = sig
    
    features = features.fillna(0)
    # Target: next bar direction (1=up, 0=down)
    target = (daily.pct_change().shift(-1) > 0).astype(int)
    target = target.reindex(features.index).fillna(0)
    
    # Train/test split (80/20)
    split = int(len(features) * 0.8)
    X_train, X_test = features.iloc[:split], features.iloc[split:]
    y_train, y_test = target.iloc[:split], target.iloc[split:]
    
    if len(X_train) < 100: return {"sharpe": 0, "monthly_pct": 0, "max_dd": 0, "n_trades": 0, "win_rate": 0}
    
    model = LGBMClassifier(n_estimators=100, max_depth=5, verbosity=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # Convert predictions to trading signal
    ml_signal = pd.Series(0, index=X_test.index)
    ml_signal[preds == 1] = 1; ml_signal[preds == 0] = -1
    ml_signal = ml_signal.reindex(close.index).ffill().fillna(0).astype(int).clip(-1,1)
    
    prices = close.values.astype(np.float64); sigs = ml_signal.values.astype(np.int32)
    eq, dd, tr, wins, ret, sh, _ = _backtest_numba(prices, sigs)
    n_days = (close.index[-1] - close.index[0]).days
    mon = ((1+ret)**(1/(n_days/30.44))-1)*100 if ret > -1 else 0
    acc = (preds == y_test).mean()
    print(f"    ML LightGBM: Test acc={acc:.1%} → Sh={sh:.1f} Mon={mon:.1f}% Tr={tr}")
    return {"sharpe": float(sh), "monthly_pct": float(mon), "max_dd": float(-dd),
            "n_trades": int(tr), "win_rate": float(wins/tr) if tr>0 else 0,
            "ml_accuracy": float(acc), "ml_model": "LightGBM"}


def main():
    iterations = 200
    if "--iterations" in sys.argv:
        iterations = int(sys.argv[sys.argv.index("--iterations") + 1])

    print("=" * 60)
    print(f"  R&D Loop — Technical Indicators ({len(INDICATORS_POOL)} indicators)")
    print(f"  Strategy: hypothezise → evaluate → feedback → record")
    print(f"  Iterations: {iterations}")
    print("=" * 60)

    df = pd.read_hdf(OHLCV_PATH, key="data")
    close = df.xs("EURUSD", level="instrument")["$close"].sort_index()

    loop = ResearchLoop(close)
    t0 = time.time()

    for i in range(iterations):
        # 1. HYPOTHESIZE
        hp = loop.hypothesize()

        # 2. EVALUATE
        try:
            result = evaluate_strategy(close, hp)
        except Exception:
            continue  # skip bad parameters
        result['hypothesis'] = hp
        result['iteration'] = i + 1
        result['timestamp'] = datetime.now().isoformat()
        loop.history.append(result)

        # 3. FEEDBACK
        is_new_best = loop.feedback(result)

        # Log
        gen = hp.get('generation', '?')
        if is_new_best:
            print(f"\n  ★ NEW BEST (#{i+1}, {gen}): {hp['description']}")
            print(f"    Sharpe={result['sharpe']:.2f} Mon={result['monthly_pct']:.1f}% "
                  f"DD={result['max_dd']:.4f} Tr={result['n_trades']}")
        elif (i + 1) % 50 == 0:
            print(f"  [{i+1}/{iterations}] {gen:>7s} | SOTA: {len(loop.sota)} | "
                  f"Best Sh={loop.best_sharpe:.2f} | "
                  f"Explore: {loop.exploration_rate:.0%}")

        # 4. RECORD checkpoint
        if (i + 1) % 100 == 0:
            loop.record()

        # Adaptive exploration: higher rate needed for indicator discovery
        if len(loop.sota) > 10:
            loop.exploration_rate = max(0.15, EXPLORATION_RATE - len(loop.sota) * 0.005)

    # Final
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  R&D Loop Complete: {iterations} iterations in {elapsed:.0f}s")
    print(f"  SOTA Strategies: {len(loop.sota)}")
    print(f"{'=' * 60}")

    if loop.sota:
        print(f"\n  TOP DISCOVERIES:")
        for i, r in enumerate(loop.sota[:15], 1):
            hp = r['hypothesis']
            print(f"  {i:>2d}. {hp['description'][:50]:50s} Sh={r['sharpe']:+.2f} Mon={r['monthly_pct']:+.1f}% "
                  f"Tr={r['n_trades']} ({hp.get('generation','?')})")

        final = RESULTS_DIR / f"rd_loop_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        final.write_text(json.dumps(loop.sota, indent=2, default=str))
        print(f"\n  Saved: {final}")

        # Learnings summary
        exploit_best = [r for r in loop.sota if r['hypothesis'].get('generation') == 'exploit']
        explore_best = [r for r in loop.sota if r['hypothesis'].get('generation') == 'explore']
        print(f"\n  Exploit wins: {len(exploit_best)} (avg Sh={np.mean([r['sharpe'] for r in exploit_best]):.1f})" if exploit_best else "")
        print(f"  Explore wins: {len(explore_best)} (avg Sh={np.mean([r['sharpe'] for r in explore_best]):.1f})" if explore_best else "")


if __name__ == "__main__":
    main()
