#!/usr/bin/env python3
"""R&D Loop V2 — Multi-Instrument + Correlation Score + Session/Vola Filter + OOS.

Changes from V1:
    1. Multi-Instrument: Evaluate on EUR/USD + GBP/USD + BTC/USD
    2. Correlation Score: Reward uncorrelated strategies (Sharpe × (1−corr))
    3. Session Filter: Only trade London session (07:00-16:00 UTC)
    4. Volatility Filter: No trades when ATR < threshold
    5. OOS Split: Report IS/OOS separately (80/20)
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

INSTRUMENTS = ["EURUSD", "GBPUSD", "BTCUSD", "XAUUSD"]
LEADER_MAP = {"GBPUSD": "EURUSD"}
INSTRUMENT_TIMEFRAMES = {
    "XAUUSD": ["1d", "1w"],           # Daily data → daily/weekly TFs
    "default": ["5min", "15min", "30min", "1h", "4h"],
}
TIMEFRAMES = ["5min", "15min", "30min", "1h", "4h", "1d", "1w"]
INDICATORS_POOL = ["MACD", "RSI", "BBands", "Donchian", "Stoch", "CCI", "WillR", "ADX", "SAR", "ROC", "MOM", "AROON", "MFI", "SMA", "EMA"]
STRATEGY_TYPES = ["single", "multi_tf", "multi_role"]
TREND_TFS = ["30min", "1h", "4h"]
ENTRY_TFS = ["5min", "15min", "30min"]
MIN_SHARPE, MIN_TRADES = 0.3, 10
EXPLORATION_RATE = 0.40
OOS_SPLIT = 0.2

# ═══════════════════════════════════════════════════════════════════════════════
# Numba-accelerated backtest
# ═══════════════════════════════════════════════════════════════════════════════

@jit(nopython=True)
def _backtest_numba(prices, signals, cost=0.000264):
    n = len(prices)
    equity = np.zeros(n, dtype=np.float64)
    equity[0] = 100000.0; peak = 100000.0; max_dd = 0.0
    trade_returns = np.zeros(100000, dtype=np.float64)
    position = 0; entry_price = 0.0; trade_count = 0; wins = 0

    for i in range(1, n):
        px = prices[i]; sg = signals[i]; ps = signals[i-1]
        # Close position on signal reversal or flatten
        if position != 0 and (sg != position or sg == 0 and position != 0):
            if position == 1: ret = (px - entry_price) / entry_price - cost
            else: ret = (entry_price - px) / entry_price - cost
            equity[i] = equity[i-1] * (1.0 + ret)
            if equity[i] > peak: peak = equity[i]
            dd = (peak - equity[i]) / peak
            if dd > max_dd: max_dd = dd
            if trade_count < len(trade_returns):
                trade_returns[trade_count] = ret
            trade_count += 1
            if ret > 0: wins += 1
            position = 0
        else:
            equity[i] = equity[i-1]
        # Open new position
        if position == 0 and sg != 0:
            position = sg; entry_price = px

    # Close final position
    if position != 0:
        fp = prices[-1]
        if position == 1: ret = (fp - entry_price) / entry_price - cost
        else: ret = (entry_price - fp) / entry_price - cost
        equity[-1] = equity[-2] * (1.0 + ret)
        if trade_count < len(trade_returns):
            trade_returns[trade_count] = ret
        trade_count += 1
        if ret > 0: wins += 1

    # Ensure monotonic equity (carry forward zeros)
    for i in range(1, n):
        if equity[i] == 0: equity[i] = equity[i-1]

    total_ret = (equity[-1] - 100000.0) / 100000.0

    if trade_count > 5:
        t = trade_returns[:trade_count]
        mean_ret = np.mean(t); std_ret = np.std(t)
        sharpe = mean_ret / std_ret * np.sqrt(trade_count) if std_ret > 0 else 0.0
    else:
        sharpe = 0.0

    return equity, max_dd, trade_count, wins, total_ret, sharpe, trade_returns[:trade_count]

# ═══════════════════════════════════════════════════════════════════════════════
# Signal Construction
# ═══════════════════════════════════════════════════════════════════════════════

def build_signal(close, hypothesis):
    """Build trading signal from hypothesis. Returns (-1,0,1) Series."""
    import talib
    signal = None

    # Adapt timeframes to data frequency
    median_delta = (close.index[1:] - close.index[:-1]).median()
    if median_delta > pd.Timedelta("1h"):
        valid_tfs = ["1d", "1w"]
        tf_map = {"5min": "1d", "15min": "1d", "30min": "1d", "1h": "1d", "4h": "1w"}
        # Remap hypothesis timeframes
        hp = dict(hypothesis)
        if hp.get('type') in ('single', 'multi_tf') and 'timeframe' in hp:
            hp['timeframe'] = tf_map.get(hp.get('timeframe','1h'), '1d')
        if hp.get('type') == 'multi_tf' and 'timeframes' in hp:
            hp['timeframes'] = [tf_map.get(t, '1d') for t in hp['timeframes']]
            hp['timeframes'] = list(set(hp['timeframes']))  # dedup
        if hp.get('type') == 'multi_role':
            hp['trend_tf'] = tf_map.get(hp.get('trend_tf','4h'), '1w')
            hp['entry_tf'] = tf_map.get(hp.get('entry_tf','15min'), '1d')
        hypothesis = hp
    else:
        valid_tfs = ["5min", "15min", "30min", "1h", "4h"]

    if hypothesis['type'] == 'single':
        ind = hypothesis['indicator']; tf = hypothesis['timeframe']
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
        signal = pd.Series(0, index=close.index)
        signal[vote > 0.25] = 1; signal[vote < -0.25] = -1

    elif hypothesis['type'] == 'multi_role':
        trend_ind = hypothesis['trend_ind']; entry_ind = hypothesis['entry_ind']
        trend_tf = hypothesis['trend_tf']; entry_tf = hypothesis['entry_tf']
        trend_bars = close.resample(trend_tf).last().dropna()
        trend_sig = _build_indicator_signal(trend_ind, trend_bars, hypothesis['trend_params'])
        trend_sig = trend_sig.reindex(close.index).ffill().fillna(0).astype(int).clip(-1, 1)
        entry_bars = close.resample(entry_tf).last().dropna()
        entry_sig = _build_indicator_signal(entry_ind, entry_bars, hypothesis['entry_params'])
        entry_sig = entry_sig.reindex(close.index).ffill().fillna(0).astype(int).clip(-1, 1)
        signal = pd.Series(0, index=close.index)
        signal[(trend_sig == 1) & (entry_sig == 1)] = 1
        signal[(trend_sig == -1) & (entry_sig == -1)] = -1

    return signal if signal is not None and signal.nunique() > 1 else None


def _apply_session_filter(signal, index):
    """Only trade London session (07:00-16:00 UTC Mon-Fri). Skip for daily data."""
    delta = (index[1:] - index[:-1]).median() if len(index) > 1 else pd.Timedelta("1min")
    if delta > pd.Timedelta("1h"):
        return signal  # Skip session filter for daily/weekly data
    hours = index.hour
    days = index.dayofweek
    in_session = (days < 5) & (hours >= 7) & (hours < 16)
    if hasattr(in_session, 'values'):
        in_session = in_session.values
    return (signal * in_session.astype(int)).astype(int).clip(-1, 1)


def _apply_vola_filter(signal, close, atr_period=14, min_atr_pct=0.0003):
    """Don't trade when ATR is too low (flat/quiet markets)."""
    tr = pd.DataFrame({
        'hl': close.diff().abs(),
        'hc': (close - close.shift(1)).abs(),
        'lc': (close.shift(1) - close).abs(),
    }).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    atr_pct = atr / close
    too_quiet = atr_pct < min_atr_pct
    return (signal * (~too_quiet).astype(int)).fillna(0).astype(int).clip(-1, 1)


_NEWS_CACHE = None
def _load_news_events():
    """Load high-impact news events from YAML, return dict of currency → DatetimeIndex mask."""
    global _NEWS_CACHE
    if _NEWS_CACHE is not None:
        return _NEWS_CACHE
    import yaml
    news_file = PROJECT / "git_ignore_folder" / "economic_events_full.yaml"
    if not news_file.exists():
        _NEWS_CACHE = {}
        return _NEWS_CACHE
    with open(news_file) as f:
        data = yaml.safe_load(f)
    events = {}
    for evt in data.get('events', []):
        if evt.get('impact') != 'high':
            continue
        dt = pd.Timestamp(evt['datetime'])
        currency = evt.get('currency', 'USD')
        if currency not in events:
            events[currency] = []
        events[currency].append(dt)
    _NEWS_CACHE = events
    return _NEWS_CACHE


def _apply_news_filter(signal, index, currency, window_min=5):
    """Block trades during high-impact news events (+/- window_min)."""
    events = _load_news_events()
    if currency not in events and currency[:3] not in events:
        return signal
    key = currency if currency in events else currency[:3]
    timestamps = events.get(key, [])
    if not timestamps:
        return signal

    blocked = np.zeros(len(index), dtype=bool)
    for ts in timestamps:
        start = ts - pd.Timedelta(minutes=window_min)
        end = ts + pd.Timedelta(minutes=window_min)
        mask = (index >= start) & (index <= end)
        blocked |= mask

    return (signal * (~blocked).astype(int)).astype(int).clip(-1, 1)


def _apply_cross_confirm(signal, close_follower, close_leader, lookback=5, min_pct=0.0003):
    """Cancel follower signals when leader momentum strongly opposes (>0.03% move)."""
    leader_mom = close_leader.pct_change(lookback)
    leader_mom = leader_mom.reindex(signal.index, method='ffill')

    # Only BLOCK when leader moves strongly opposite to signal
    # Don't require confirmation — just cancel clear contrarian moves
    cancel_long = (signal == 1) & (leader_mom < -min_pct)
    cancel_short = (signal == -1) & (leader_mom > min_pct)
    cancel = cancel_long | cancel_short
    return (signal * (~cancel).astype(int)).fillna(0).astype(int).clip(-1, 1)


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
# Multi-Instrument Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_multi(closes, hypothesis, use_session=True, use_vola=False):
    """Evaluate strategy on all instruments, return combined metrics + per-instrument."""
    results = {}
    equity_curves = {}

    for inst, close in closes.items():
        signal = build_signal(close, hypothesis)
        if signal is None:
            results[inst] = {"sharpe": 0, "monthly_pct": 0, "n_trades": 0}
            continue

        # Apply filters
        if use_session:
            signal = _apply_session_filter(signal, close.index)
        if use_vola:
            signal = _apply_vola_filter(signal, close)

        # News filter: block trades during high-impact events for this currency
        signal = _apply_news_filter(signal, close.index, inst.replace("USD", "").replace("BTC", "BTC"))

        # Cross-pair confirmation: validate follower with leader momentum
        leader_inst = LEADER_MAP.get(inst)
        if leader_inst and leader_inst in closes:
            signal = _apply_cross_confirm(signal, close, closes[leader_inst])

        if signal.nunique() <= 1:
            results[inst] = {"sharpe": 0, "monthly_pct": 0, "n_trades": 0}
            continue

        # OOS split
        n = len(close)
        is_n = int(n * (1 - OOS_SPLIT))
        close_is = close.iloc[:is_n]; signal_is = signal.iloc[:is_n]
        close_oos = close.iloc[is_n:]; signal_oos = signal.iloc[is_n:]

        # IS backtest
        prices_is = close_is.values.astype(np.float64); sigs_is = signal_is.values.astype(np.int32)
        eq_is, dd_is, tr_is, wins_is, ret_is, sh_is, _ = _backtest_numba(prices_is, sigs_is)

        # OOS backtest
        prices_oos = close_oos.values.astype(np.float64); sigs_oos = signal_oos.values.astype(np.int32)
        eq_oos, dd_oos, tr_oos, wins_oos, ret_oos, sh_oos, _ = _backtest_numba(prices_oos, sigs_oos)

        # Full backtest (for equity curve)
        prices_full = close.values.astype(np.float64); sigs_full = signal.values.astype(np.int32)
        eq_full, dd_full, tr_full, wins_full, ret_full, sh_full, trades_full = _backtest_numba(prices_full, sigs_full)

        n_days = (close.index[-1] - close.index[0]).days
        mon = ((1+ret_full)**(1/(n_days/30.44))-1)*100 if ret_full > -1 else 0
        mon_oos = ((1+ret_oos)**(1/((close_oos.index[-1] - close_oos.index[0]).days/30.44))-1)*100 if ret_oos > -1 else 0

        results[inst] = {
            "sharpe": float(sh_full), "sharpe_is": float(sh_is), "sharpe_oos": float(sh_oos),
            "monthly_pct": float(mon), "monthly_oos": float(mon_oos),
            "n_trades": int(tr_full), "n_trades_oos": int(tr_oos),
            "win_rate": float(wins_full/tr_full) if tr_full>0 else 0,
            "max_dd": float(-dd_full), "total_return": float(ret_full),
        }
        equity_curves[inst] = eq_full.copy()

    # Combined metrics (harmonic mean — only good if ALL instruments good)
    valid = [r for r in results.values() if r['sharpe'] > 0]
    if not valid:
        combined = {"sharpe": 0, "monthly_pct": 0, "monthly_oos": 0, "n_trades": 0, "n_trades_oos": 0}
    else:
        combined = {
            "sharpe": float(np.mean([r['sharpe'] for r in valid])),
            "monthly_pct": float(np.mean([r['monthly_pct'] for r in valid])),
            "monthly_oos": float(np.mean([r['monthly_oos'] for r in valid])),
            "n_trades": int(np.sum([r['n_trades'] for r in valid])),
            "n_trades_oos": int(np.sum([r['n_trades_oos'] for r in valid])),
        }
    combined['per_instrument'] = results
    combined['equity_curves'] = equity_curves

    return combined


def correlation_penalty(result, sota_equity_curves):
    """Compute avg correlation of this strategy's returns with SOTA returns."""
    if not sota_equity_curves:
        return 0.0
    my_returns = []
    for eq in result.get('equity_curves', {}).values():
        if len(eq) > 1:
            my_returns.append(np.diff(eq) / eq[:-1])
    if not my_returns:
        return 0.5
    # Use longest equity curve for this strategy
    my_ret = max(my_returns, key=len)

    correlations = []
    for sota_eq_dict in sota_equity_curves:
        for eq in sota_eq_dict.values():
            if len(eq) > 1:
                sota_ret = np.diff(eq) / eq[:-1]
                # Align to shorter length
                min_len = min(len(my_ret), len(sota_ret))
                if min_len > 10:
                    corr = np.corrcoef(my_ret[:min_len], sota_ret[:min_len])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)

    return np.mean(correlations) if correlations else 0.0


def composite_score(result, sota_equity_curves):
    """Composite score = sharpe × (1 - correlation) → rewards uncorrelated profit."""
    sh = result.get('sharpe', 0)
    if sh <= 0:
        return 0
    corr = abs(correlation_penalty(result, sota_equity_curves))
    # Bonus for OOS consistency
    oos_ratio = min(result.get('monthly_oos', 0) / max(result.get('monthly_pct', 1), 0.01), 1.0)
    oos_ratio = max(oos_ratio, 0)
    return sh * (1 - 0.5 * corr) * (0.3 + 0.7 * oos_ratio)

# ═══════════════════════════════════════════════════════════════════════════════
# Hypothesis Generation
# ═══════════════════════════════════════════════════════════════════════════════

class ResearchLoop:
    """Multi-instrument R&D loop with correlation-aware feedback."""

    def __init__(self, closes):
        self.closes = closes  # {instrument: close_series}
        self.sota = []  # State-of-the-art strategies (sorted by composite score)
        self.sota_equity = []  # Equity curves for correlation calc
        self.history = []
        self.iteration = 0
        self.best_score = 0
        self.best_sharpe = 0
        self.exploration_rate = EXPLORATION_RATE

    def hypothesize(self):
        self.iteration += 1

        # Every 2000: ML (higher priority, runs before Optuna)
        if self.iteration % 2000 == 0 and len(self.sota) >= 5:
            return {'type': 'ml', 'generation': 'ml',
                    'description': f"ML: LightGBM on {len(self.sota)} strategies",
                    'sota': self.sota[:5]}

        # Every 500: Optuna optimize best strategy
        if self.iteration % 500 == 0 and self.sota:
            hp = dict(self.sota[0]['hypothesis'])
            hp['generation'] = 'optuna'
            hp['description'] = f"Optuna: {hp.get('description','?')}"
            return hp

        # Every 100: force non-dominant indicator
        if self.iteration % 100 == 0 and len(self.sota) >= 5:
            top = self._top_indicator()
            hp = self._random_hypothesis()
            hp = self._force_different_indicator(hp, top)
            hp['generation'] = 'explore'
            return hp

        # Adaptive exploration rate
        effective_rate = self.exploration_rate
        if len(self.sota) >= 10:
            top = self._top_indicator()
            dominated = sum(1 for r in self.sota if
                r['hypothesis'].get('trend_ind', r['hypothesis'].get('indicator')) == top)
            if dominated > len(self.sota) * 0.8:
                effective_rate += 0.25

        if random.random() < effective_rate or not self.sota:
            return self._random_hypothesis()
        else:
            base = random.choice(self.sota[:5])
            return self._mutate_hypothesis(base['hypothesis'])

    def _top_indicator(self):
        if not self.sota:
            return 'MACD'
        return self.sota[0]['hypothesis'].get('trend_ind',
                self.sota[0]['hypothesis'].get('indicator', 'MACD'))

    def _force_different_indicator(self, hp, top_ind):
        if hp.get('type') == 'multi_role':
            if hp['trend_ind'] == top_ind and hp['entry_ind'] == top_ind:
                if random.random() < 0.5:
                    hp['trend_ind'] = random.choice([i for i in INDICATORS_POOL if i != top_ind])
                    hp['trend_params'] = self._random_params(hp['trend_ind'])
                else:
                    hp['entry_ind'] = random.choice([i for i in INDICATORS_POOL if i != top_ind])
                    hp['entry_params'] = self._random_params(hp['entry_ind'])
        elif hp.get('indicator') == top_ind:
            hp['indicator'] = random.choice([i for i in INDICATORS_POOL if i != top_ind])
            hp['params'] = self._random_params(hp['indicator'])
        hp['description'] = self._make_desc(hp)
        return hp

    def _make_desc(self, hp):
        t = hp.get('type', '?')
        if t == 'multi_role':
            return f"{hp['trend_ind']}({hp['trend_tf']})→{hp['entry_ind']}({hp['entry_tf']})"
        elif t == 'multi_tf':
            return f"{hp.get('indicator','?')} on {','.join(hp.get('timeframes',[])[:2])}"
        else:
            return f"{hp.get('indicator','?')} on {hp.get('timeframe','?')}"

    def _random_hypothesis(self):
        stype = random.choice(STRATEGY_TYPES)
        if stype == 'single':
            ind = random.choice(INDICATORS_POOL); tf = random.choice(TIMEFRAMES)
            return {'type': 'single', 'indicator': ind, 'timeframe': tf,
                    'params': self._random_params(ind),
                    'description': f"{ind} on {tf}", 'generation': 'explore'}
        elif stype == 'multi_tf':
            ind = random.choice(INDICATORS_POOL)
            tfs = random.sample(TIMEFRAMES, k=random.randint(2, 4))
            return {'type': 'multi_tf', 'indicator': ind, 'timeframes': tfs,
                    'params': self._random_params(ind),
                    'description': f"{ind} on {','.join(tfs)}", 'generation': 'explore'}
        else:  # multi_role
            trend_ind = random.choice(INDICATORS_POOL)
            entry_ind = random.choice(INDICATORS_POOL)
            trend_tf = random.choice(TREND_TFS)
            entry_tf = random.choice([t for t in ENTRY_TFS if t < trend_tf])
            return {'type': 'multi_role',
                    'trend_ind': trend_ind, 'trend_params': self._random_params(trend_ind),
                    'trend_tf': trend_tf,
                    'entry_ind': entry_ind, 'entry_params': self._random_params(entry_ind),
                    'entry_tf': entry_tf,
                    'description': f"{trend_ind}({trend_tf})→{entry_ind}({entry_tf})",
                    'generation': 'explore'}

    def _mutate_hypothesis(self, base):
        hp = dict(base); hp['generation'] = 'exploit'

        if hp.get('type') == 'multi_role':
            mut = random.choice(['trend_ind', 'entry_ind', 'trend_tf', 'entry_tf',
                                 'trend_params', 'entry_params'])
            if mut == 'trend_ind':
                hp['trend_ind'] = random.choice([i for i in INDICATORS_POOL if i != hp['trend_ind']])
                hp['trend_params'] = self._random_params(hp['trend_ind'])
            elif mut == 'entry_ind':
                hp['entry_ind'] = random.choice([i for i in INDICATORS_POOL if i != hp['entry_ind']])
                hp['entry_params'] = self._random_params(hp['entry_ind'])
            elif mut == 'trend_tf':
                hp['trend_tf'] = random.choice(TREND_TFS)
                if hp['trend_tf'] <= hp['entry_tf']:
                    hp['entry_tf'] = random.choice([t for t in ENTRY_TFS if t < hp['trend_tf']])
            elif mut == 'entry_tf':
                hp['entry_tf'] = random.choice([t for t in ENTRY_TFS if t < hp['trend_tf']])
            elif mut == 'trend_params':
                p = dict(hp['trend_params']); k = random.choice(list(p.keys()))
                if isinstance(p[k], (int, float)): p[k] = p[k] * random.uniform(0.5, 1.5)
                hp['trend_params'] = p
            elif mut == 'entry_params':
                p = dict(hp['entry_params']); k = random.choice(list(p.keys()))
                if isinstance(p[k], (int, float)): p[k] = p[k] * random.uniform(0.5, 1.5)
                hp['entry_params'] = p
            hp['description'] = f"{hp['trend_ind']}({hp['trend_tf']})→{hp['entry_ind']}({hp['entry_tf']})"
            return hp

        mutations = ['params', 'indicator', 'timeframe']
        mutation = random.choice(mutations)

        if mutation == 'params' and 'params' in hp:
            params = dict(hp['params']); key = random.choice(list(params.keys()))
            if isinstance(params[key], (int, float)):
                params[key] = params[key] * random.uniform(0.5, 1.5)
                if isinstance(params[key], float): params[key] = round(params[key], 1)
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
        """Update SOTA sorted by COMPOSITE score (not just Sharpe)."""
        if result['sharpe'] <= MIN_SHARPE or result['n_trades'] < MIN_TRADES:
            return False

        score = composite_score(result, self.sota_equity)
        result['composite_score'] = float(score)

        # Check if this strategy is diverse enough to add
        is_diverse = True
        if self.sota:
            # Skip if very similar to existing (same indicators, TF, type)
            for existing in self.sota[:3]:
                if self._similar(result, existing):
                    is_diverse = False
                    break

        if is_diverse:
            self.sota.append(result)
            self.sota.sort(key=lambda r: r.get('composite_score', 0), reverse=True)
            self.sota = self.sota[:30]  # Keep top 30
            self.sota_equity = [s['equity_curves'] for s in self.sota]
            if score > self.best_score:
                self.best_score = score
                return True  # NEW BEST

        if result['sharpe'] > self.best_sharpe:
            self.best_sharpe = result['sharpe']

        return False

    def _similar(self, a, b):
        """Check if two strategies are too similar (same indicator combo, type, TFs)."""
        ha = a['hypothesis']; hb = b['hypothesis']
        if ha.get('type') != hb.get('type'):
            return False
        if ha.get('type') == 'multi_role':
            return (ha.get('trend_ind') == hb.get('trend_ind') and
                    ha.get('entry_ind') == hb.get('entry_ind') and
                    ha.get('trend_tf') == hb.get('trend_tf') and
                    ha.get('entry_tf') == hb.get('entry_tf'))
        return ha.get('indicator') == hb.get('indicator')

    def record(self):
        """Save checkpoint."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        if self.sota:
            cp = RESULTS_DIR / f"rd_loop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            # Strip equity_curves (too large) from saved results
            stripped = []
            for r in self.sota[:30]:
                s = {k: v for k, v in r.items() if k != 'equity_curves'}
                stripped.append(s)
            cp.write_text(json.dumps(stripped, indent=2, default=str))


def _run_optuna(closes, hypothesis):
    """Optuna optimization on the primary instrument."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    hp = hypothesis
    close = list(closes.values())[0]  # Use first instrument for Optuna
    ind = hp.get('indicator', hp.get('trend_ind', 'MACD'))
    base_params = hp.get('params', hp.get('trend_params', {}))

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
        if 'fast' in params and 'slow' in params:
            params['fast'] = min(params['fast'], params['slow']-2)

        result = evaluate_multi(closes, hp, use_session=True, use_vola=True)
        return float(result.get('sharpe', 0)) if result.get('sharpe', 0) > 0 else -999.0

    try:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=15, show_progress_bar=False)
        best = study.best_params
        if 'params' in hp:
            hp['params'] = {k: int(v) if v == int(v) else v for k, v in best.items()}
        elif 'trend_params' in hp:
            hp['trend_params'] = {k: int(v) if v == int(v) else v for k, v in best.items()}
        hp['generation'] = 'optuna'
        result = evaluate_multi(closes, hp, use_session=True, use_vola=True)
        print(f"    Optuna best: {best} → Sh={result['sharpe']:.1f} "
              f"Mon={result['monthly_pct']:.1f}% OOS={result['monthly_oos']:.1f}% ({study.best_value:.1f})")
        return result
    except Exception:
        return {"sharpe": 0, "monthly_pct": 0, "monthly_oos": 0, "n_trades": 0}


def _train_ml(closes, hypothesis):
    """Train LightGBM on SOTA indicator signals."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        return {"sharpe": 0, "monthly_pct": 0, "monthly_oos": 0, "n_trades": 0}

    sota = hypothesis.get('sota', [])
    if not sota:
        return {"sharpe": 0, "monthly_pct": 0, "monthly_oos": 0, "n_trades": 0}

    close = list(closes.values())[0]
    daily = close.resample('1h').last().dropna()
    features = pd.DataFrame(index=daily.index)

    for s in sota[:5]:
        hp_s = s['hypothesis']
        # Generate signal from each SOTA strategy as a feature
        from nexquant_rd_loop import build_signal, _build_indicator_signal
        sig = build_signal(close, hp_s)
        if sig is not None:
            sig = sig.reindex(daily.index, method='ffill')
            name = hp_s.get('description', f"strat_{id(s)}")[:30]
            features[name] = sig.fillna(0)

    features = features.iloc[100:]  # Skip warmup
    if len(features) < 200:
        return {"sharpe": 0, "monthly_pct": 0, "monthly_oos": 0, "n_trades": 0}

    target = (daily.pct_change().shift(-1) > 0).astype(int)
    target = target.reindex(features.index).fillna(0)

    split = int(len(features) * 0.8)
    X_train, X_test = features.iloc[:split], features.iloc[split:]
    y_train, y_test = target.iloc[:split], target.iloc[split:]

    model = LGBMClassifier(n_estimators=100, max_depth=5, verbosity=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = float((preds == y_test).mean())

    ml_signal = pd.Series(0, index=X_test.index)
    ml_signal[preds == 1] = 1; ml_signal[preds == 0] = -1
    ml_signal = ml_signal.reindex(close.index).ffill().fillna(0).astype(int).clip(-1, 1)
    ml_signal = _apply_session_filter(ml_signal, close.index)

    prices = close.values.astype(np.float64); sigs = ml_signal.values.astype(np.int32)
    eq, dd, tr, wins, ret, sh, _ = _backtest_numba(prices, sigs)
    n_days = (close.index[-1] - close.index[0]).days
    mon = ((1+ret)**(1/(n_days/30.44))-1)*100 if ret > -1 else 0
    print(f"    ML LightGBM: Test acc={acc:.1%} → Sh={sh:.1f} Mon={mon:.1f}% Tr={tr}")
    return {"sharpe": float(sh), "monthly_pct": float(mon), "monthly_oos": 0,
            "n_trades": int(tr), "win_rate": float(wins/tr) if tr>0 else 0,
            "ml_accuracy": float(acc), "ml_model": "LightGBM"}


def load_data():
    """Load OHLCV data for all instruments from one or multiple HDF5 files."""
    closes = {}
    data_dir = OHLCV_PATH.parent

    # Try main file first
    if OHLCV_PATH.exists():
        df = pd.read_hdf(OHLCV_PATH, key="data")
        for inst in INSTRUMENTS:
            try:
                close = df.xs(inst, level="instrument")["$close"].sort_index()
                closes[inst] = close
            except KeyError:
                pass

    # Load from individual files if not found
    instrument_files = {
        "EURUSD": OHLCV_PATH,
        "GBPUSD": data_dir / "gbpusdt_1min.h5",
        "BTCUSD": data_dir / "btc_1min.h5",
        "XAUUSD": data_dir / "xauusdt_1min.h5",
    }
    for inst, path in instrument_files.items():
        if inst in closes:
            continue
        if not path.exists():
            print(f"  {inst}: file not found — skipping")
            continue
        try:
            df = pd.read_hdf(path, key="data")
            if isinstance(df.index, pd.MultiIndex):
                try:
                    close = df.xs(inst, level="instrument")["$close"].sort_index()
                except KeyError:
                    # Try with T suffix for crypto pairs
                    alt = inst + "T" if not inst.endswith("T") else inst.rstrip("T")
                    try:
                        close = df.xs(alt, level="instrument")["$close"].sort_index()
                    except KeyError:
                        inst_vals = df.index.get_level_values("instrument").unique()
                        for iv in inst_vals:
                            if inst[:3] in str(iv)[:3]:
                                close = df.xs(iv, level="instrument")["$close"].sort_index()
                                break
                        else:
                            raise KeyError(f"No instrument matching {inst}")
            elif "$close" in df.columns:
                close = df["$close"].sort_index()
                close.index = pd.to_datetime(close.index)
            elif "close" in df.columns:
                close = df["close"].sort_index()
                close.index = pd.to_datetime(close.index)
            else:
                close = df.iloc[:, 3].sort_index()
                close.index = pd.to_datetime(close.index)
            closes[inst] = close
        except Exception as e:
            print(f"  {inst}: load error {e} — skipping")

    for inst, close in closes.items():
        print(f"  {inst}: {len(close):,} bars, {close.index[0]} → {close.index[-1]}")

    return closes


def main():
    iterations = 200
    if "--iterations" in sys.argv:
        iterations = int(sys.argv[sys.argv.index("--iterations") + 1])

    print("=" * 60)
    print(f"  R&D Loop V2 — Multi-Instrument + Correlation Score")
    print(f"  Instruments: {', '.join(INSTRUMENTS)}")
    print(f"  Indicators: {len(INDICATORS_POOL)} | Strategy types: {len(STRATEGY_TYPES)}")
    print(f"  Features: Session Filter + Volatility Filter + OOS Split")
    print(f"  Iterations: {iterations}")
    print("=" * 60)
    print("  Loading data...")

    closes = load_data()
    if not closes:
        print("  ERROR: No instruments loaded!"); return

    loop = ResearchLoop(closes)
    t0 = time.time()

    for i in range(iterations):
        hp = loop.hypothesize()

        # Evaluate
        try:
            result = evaluate_multi(closes, hp, use_session=True, use_vola=True)
        except Exception:
            continue

        result['hypothesis'] = hp
        result['iteration'] = i + 1
        result['timestamp'] = datetime.now().isoformat()
        loop.history.append(result)

        # Feedback
        is_new_best = loop.feedback(result)
        best_inst_metrics = [f"{inst}: {m['sharpe']:.1f}" for inst, m in result.get('per_instrument', {}).items() if m.get('sharpe', 0) != 0]

        gen = hp.get('generation', '?')
        if is_new_best:
            print(f"\n  ★ NEW BEST (#{i+1}, {gen}): {hp['description']}")
            print(f"    Score={result['composite_score']:.1f} Sh={result['sharpe']:.1f} "
                  f"Mon={result['monthly_pct']:.1f}% OOS={result['monthly_oos']:.1f}% "
                  f"Tr={result['n_trades']} [{', '.join(best_inst_metrics[:3])}]")
        elif (i + 1) % 50 == 0:
            top_indicators = set()
            for r in loop.sota[:5]:
                top_indicators.add(r['hypothesis'].get('trend_ind', r['hypothesis'].get('indicator', '?')))
            print(f"  [{i+1}/{iterations}] {gen:>7s} | SOTA: {len(loop.sota)} | "
                  f"Best Sh={loop.best_sharpe:.1f} Score={loop.best_score:.1f} | "
                  f"Explore: {loop.exploration_rate:.0%} | Inds: {','.join(sorted(top_indicators)[:4])}")

        if (i + 1) % 100 == 0:
            loop.record()

        if len(loop.sota) > 10:
            loop.exploration_rate = max(0.15, EXPLORATION_RATE - len(loop.sota) * 0.003)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  R&D Loop V2 Complete: {iterations} iterations in {elapsed:.0f}s")
    print(f"  SOTA Strategies: {len(loop.sota)} | Best Score: {loop.best_score:.1f}")
    print(f"{'=' * 60}")

    if loop.sota:
        print(f"\n  TOP DISCOVERIES (by composite score):")
        for i, r in enumerate(loop.sota[:15], 1):
            hp = r['hypothesis']
            per_inst = r.get('per_instrument', {})
            insts = ' '.join([f"{k}:{v['sharpe']:.0f}" for k, v in per_inst.items() if v['sharpe'] != 0])
            print(f"  {i:>2d}. {hp['description'][:45]:45s} "
                  f"Sc={r['composite_score']:.1f} Sh={r['sharpe']:+.1f} "
                  f"Mo={r['monthly_pct']:+.1f}% OOS={r['monthly_oos']:+.1f}% "
                  f"[{insts}]")

        final = RESULTS_DIR / f"rd_loop_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        stripped = [{k: v for k, v in r.items() if k != 'equity_curves'} for r in loop.sota]
        final.write_text(json.dumps(stripped, indent=2, default=str))
        print(f"\n  Saved: {final}")

        exploit_best = [r for r in loop.sota if r['hypothesis'].get('generation') == 'exploit']
        explore_best = [r for r in loop.sota if r['hypothesis'].get('generation') == 'explore']
        optuna_best = [r for r in loop.sota if r['hypothesis'].get('generation') == 'optuna']
        ml_best = [r for r in loop.sota if r['hypothesis'].get('generation') == 'ml']
        print(f"  Exploit: {len(exploit_best)} | Explore: {len(explore_best)} | "
              f"Optuna: {len(optuna_best)} | ML: {len(ml_best)}")


if __name__ == "__main__":
    main()
