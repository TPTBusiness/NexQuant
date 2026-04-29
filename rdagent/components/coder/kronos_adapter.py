"""
Kronos Foundation Model Adapter for Predix.

Wraps the Kronos-mini OHLCV foundation model (4.1M params, AAAI 2026, MIT)
for use as:
  - Factor (Option A): predicted next-day return signal
  - Model alongside LightGBM (Option B): IC/Sharpe evaluation

Kronos repo: https://github.com/shiyu-coder/Kronos
HuggingFace:  NeoQuasar/Kronos-mini  |  NeoQuasar/Kronos-Tokenizer-2k
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


KRONOS_REPO = Path.home() / "Kronos"
_KRONOS_AVAILABLE: Optional[bool] = None


def _ensure_kronos() -> bool:
    global _KRONOS_AVAILABLE
    if _KRONOS_AVAILABLE is not None:
        return _KRONOS_AVAILABLE
    if not KRONOS_REPO.exists():
        logger.warning(f"Kronos repo not found at {KRONOS_REPO}. Clone with: git clone https://github.com/shiyu-coder/Kronos ~/Kronos")
        _KRONOS_AVAILABLE = False
        return False
    repo_str = str(KRONOS_REPO)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    try:
        import model as _  # noqa: F401
        _KRONOS_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"Failed to import Kronos model: {e}")
        _KRONOS_AVAILABLE = False
    return _KRONOS_AVAILABLE


def _ohlcv_from_predix(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Predix HDF5 format ($open/$close/...) to Kronos format (open/close/...)."""
    col_map = {"$open": "open", "$high": "high", "$low": "low", "$close": "close", "$volume": "volume"}
    renamed = df.rename(columns=col_map)
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in renamed.columns]
    return renamed[cols].astype(float)


def _build_window_inputs(
    ohlcv_df: pd.DataFrame,
    pred_bars: int,
    freq: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Prepare (ctx_df, x_timestamp, y_timestamp) for one Kronos window."""
    last_ts = ohlcv_df.index[-1]
    future_idx = pd.date_range(start=last_ts, periods=pred_bars + 1, freq=freq)[1:]
    x_timestamp = pd.Series(ohlcv_df.index.values)
    y_timestamp = pd.Series(future_idx)
    ctx = ohlcv_df.copy().reset_index(drop=True)
    return ctx, x_timestamp, y_timestamp


class KronosAdapter:
    """
    Loads Kronos-mini once and provides rolling-window OHLCV inference.

    Usage:
        adapter = KronosAdapter(device="cuda")
        adapter.load()
        pred_return = adapter.predict_return(ohlcv_df, context_bars=512, pred_bars=96)
    """

    MODEL_ID = "NeoQuasar/Kronos-mini"
    TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-2k"

    def __init__(self, device: Optional[str] = None, max_context: int = 512):
        self.device = device or ("cuda" if _cuda_available() else "cpu")
        self.max_context = max_context
        self._predictor = None

    def load(self) -> "KronosAdapter":
        if self._predictor is not None:
            return self
        if not _ensure_kronos():
            raise RuntimeError("Kronos not available — see warning above.")
        from model import Kronos, KronosTokenizer, KronosPredictor  # type: ignore

        logger.info(f"Loading Kronos-mini from HuggingFace ({self.MODEL_ID})...")
        tokenizer = KronosTokenizer.from_pretrained(self.TOKENIZER_ID)
        model = Kronos.from_pretrained(self.MODEL_ID)
        self._predictor = KronosPredictor(model, tokenizer, device=self.device, max_context=self.max_context)
        logger.info("Kronos-mini loaded.")
        return self

    def predict_next_bars(
        self,
        ohlcv_df: pd.DataFrame,
        context_bars: int,
        pred_bars: int,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> pd.DataFrame:
        """
        Run Kronos on `context_bars` of OHLCV data, returning `pred_bars` predicted bars.

        Args:
            ohlcv_df: DataFrame with columns open/high/low/close[/volume], DatetimeIndex.
            context_bars: Number of history bars to feed as context.
            pred_bars: Number of future bars to predict.

        Returns:
            DataFrame with predicted open/high/low/close/volume, indexed by future timestamps.
        """
        if self._predictor is None:
            raise RuntimeError("Call .load() first.")
        if len(ohlcv_df) < context_bars:
            raise ValueError(f"Need at least {context_bars} bars, got {len(ohlcv_df)}")

        freq = ohlcv_df.index.freq or pd.infer_freq(ohlcv_df.index[:100]) or "1min"
        ctx, x_timestamp, y_timestamp = _build_window_inputs(ohlcv_df.iloc[-context_bars:], pred_bars, freq)
        future_idx = pd.DatetimeIndex(y_timestamp)

        pred_df = self._predictor.predict(
            df=ctx,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_bars,
            T=temperature,
            top_p=top_p,
            sample_count=1,
            verbose=False,
        )
        pred_df.index = future_idx
        return pred_df

    def predict_next_bars_batch(
        self,
        ohlcv_windows: list,
        pred_bars: int,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> list:
        """
        Batch inference: run Kronos on multiple context windows simultaneously.

        All windows must have the same number of bars. Processing them together
        saturates the GPU and is typically 5-20x faster than sequential calls.

        Args:
            ohlcv_windows: List of OHLCV DataFrames, each with a DatetimeIndex.
            pred_bars: Number of future bars to predict per window.

        Returns:
            List of prediction DataFrames (one per input window), same order.
        """
        if self._predictor is None:
            raise RuntimeError("Call .load() first.")
        if not ohlcv_windows:
            return []

        freq = ohlcv_windows[0].index.freq or pd.infer_freq(ohlcv_windows[0].index[:100]) or "1min"

        df_list, x_ts_list, y_ts_list, future_idxs = [], [], [], []
        for win in ohlcv_windows:
            ctx, x_ts, y_ts = _build_window_inputs(win, pred_bars, freq)
            df_list.append(ctx)
            x_ts_list.append(x_ts)
            y_ts_list.append(y_ts)
            future_idxs.append(pd.DatetimeIndex(y_ts))

        pred_dfs = self._predictor.predict_batch(
            df_list=df_list,
            x_timestamp_list=x_ts_list,
            y_timestamp_list=y_ts_list,
            pred_len=pred_bars,
            T=temperature,
            top_p=top_p,
            sample_count=1,
            verbose=False,
        )

        for pred_df, future_idx in zip(pred_dfs, future_idxs):
            pred_df.index = future_idx
        return pred_dfs

    def predict_return(
        self,
        ohlcv_df: pd.DataFrame,
        context_bars: int = 512,
        pred_bars: int = 1,
    ) -> float:
        """
        Predict the average return over the next `pred_bars` using the last `context_bars`.
        Returns the predicted log-return (predicted_close / last_close - 1).
        """
        pred = self.predict_next_bars(ohlcv_df, context_bars=context_bars, pred_bars=pred_bars)
        last_close = float(ohlcv_df["close"].iloc[-1])
        pred_close = float(pred["close"].iloc[-1])
        return pred_close / last_close - 1.0


def build_kronos_factor(
    hdf5_path,
    context_bars: int = 512,
    pred_bars: int = 96,
    stride_bars: int = 96,
    device: Optional[str] = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Generate the Kronos predicted-return factor for all EUR/USD 1-min bars.

    Strategy:
        Every `stride_bars` bars, run Kronos on the previous `context_bars` and
        predict the next `pred_bars`. Windows are processed in GPU batches of
        `batch_size` for full GPU utilization. The predicted log-return is
        forward-filled across the predicted window.

    Returns:
        MultiIndex (datetime, instrument) DataFrame with column "KronosPredReturn".
    """
    device = device or ("cuda" if _cuda_available() else "cpu")
    logger.info(f"Loading data from {hdf5_path}...")
    raw = pd.read_hdf(hdf5_path, key="data")

    instrument = raw.index.get_level_values("instrument").unique()[0]
    df = raw.xs(instrument, level="instrument")
    ohlcv = _ohlcv_from_predix(df)

    adapter = KronosAdapter(device=device, max_context=min(context_bars, 512))
    adapter.load()

    bar_indices = list(range(context_bars, len(ohlcv), stride_bars))
    n_windows = len(bar_indices)
    logger.info(
        f"Running Kronos batch inference: {n_windows} windows "
        f"(batch={batch_size}, stride={stride_bars}, ctx={context_bars}, pred={pred_bars}, device={device})"
    )

    factor_values: dict = {}

    for batch_start in range(0, n_windows, batch_size):
        batch_idx = bar_indices[batch_start : batch_start + batch_size]
        windows = [ohlcv.iloc[i - context_bars : i] for i in batch_idx]
        last_closes = [float(ohlcv["close"].iloc[i - 1]) for i in batch_idx]

        try:
            pred_dfs = adapter.predict_next_bars_batch(windows, pred_bars=pred_bars)
            for pred_df, last_close in zip(pred_dfs, last_closes):
                for ts, row in pred_df.iterrows():
                    factor_values[ts] = float(row["close"]) / last_close - 1.0
        except Exception as e:
            logger.warning(f"Batch {batch_start // batch_size + 1} failed ({e}), retrying individually...")
            for bar_idx, win, last_close in zip(batch_idx, windows, last_closes):
                try:
                    pred = adapter.predict_next_bars(win, context_bars=context_bars, pred_bars=pred_bars)
                    for ts, row in pred.iterrows():
                        factor_values[ts] = float(row["close"]) / last_close - 1.0
                except Exception as e2:
                    logger.warning(f"  Single inference failed at bar {bar_idx}: {e2}")

        done = min(batch_start + batch_size, n_windows)
        if done % max(batch_size, 100) < batch_size or done == n_windows:
            logger.info(f"  {done}/{n_windows} windows done")

    if not factor_values:
        raise RuntimeError("No Kronos predictions were generated.")

    factor_series = pd.Series(factor_values, name="KronosPredReturn")
    factor_series = factor_series.reindex(ohlcv.index, method="ffill")

    result = factor_series.to_frame()
    result.index = pd.MultiIndex.from_arrays(
        [ohlcv.index, [instrument] * len(ohlcv)],
        names=["datetime", "instrument"],
    )
    logger.info(f"Kronos factor built: {len(result)} bars, {result['KronosPredReturn'].notna().sum()} non-NaN")
    return result


def evaluate_kronos_model(
    hdf5_path,
    context_bars: int = 512,
    pred_bars: int = 30,
    stride_bars: int = 30,
    device: Optional[str] = None,
    batch_size: int = 32,
) -> dict:
    """
    Evaluate Kronos as a standalone model (Option B, alongside LightGBM).

    Computes IC (Information Coefficient) between Kronos predicted returns and
    actual realized returns on the test set.

    Returns:
        dict with keys: IC_mean, IC_std, IC_IR (IC / std), hit_rate, n_predictions
    """
    device = device or ("cuda" if _cuda_available() else "cpu")
    raw = pd.read_hdf(hdf5_path, key="data")
    instrument = raw.index.get_level_values("instrument").unique()[0]
    df = raw.xs(instrument, level="instrument")
    ohlcv = _ohlcv_from_predix(df)

    adapter = KronosAdapter(device=device, max_context=min(context_bars, 512))
    adapter.load()

    n = len(ohlcv)
    bar_indices = list(range(context_bars, n - pred_bars, stride_bars))
    logger.info(
        f"Evaluating Kronos: {len(bar_indices)} windows "
        f"(batch={batch_size}, ctx={context_bars}, pred={pred_bars}, device={device})"
    )

    predicted_returns = []
    actual_returns = []

    for batch_start in range(0, len(bar_indices), batch_size):
        batch_idx = bar_indices[batch_start : batch_start + batch_size]
        windows = [ohlcv.iloc[i - context_bars : i] for i in batch_idx]
        last_closes = [float(ohlcv["close"].iloc[i - 1]) for i in batch_idx]
        actuals = [
            float(ohlcv["close"].iloc[i + pred_bars - 1]) / float(ohlcv["close"].iloc[i - 1]) - 1.0
            for i in batch_idx
        ]

        try:
            pred_dfs = adapter.predict_next_bars_batch(windows, pred_bars=pred_bars)
            for pred_df, last_close, actual_ret in zip(pred_dfs, last_closes, actuals):
                pred_ret = float(pred_df["close"].iloc[-1]) / last_close - 1.0
                predicted_returns.append(pred_ret)
                actual_returns.append(actual_ret)
        except Exception as e:
            logger.warning(f"Batch {batch_start // batch_size + 1} failed ({e}), retrying individually...")
            for bar_idx, win, last_close, actual_ret in zip(batch_idx, windows, last_closes, actuals):
                try:
                    pred = adapter.predict_next_bars(win, context_bars=context_bars, pred_bars=pred_bars)
                    pred_ret = float(pred["close"].iloc[-1]) / last_close - 1.0
                    predicted_returns.append(pred_ret)
                    actual_returns.append(actual_ret)
                except Exception:
                    logging.debug("Exception caught", exc_info=True)

    pred_arr = np.array(predicted_returns)
    actual_arr = np.array(actual_returns)

    ic = np.corrcoef(pred_arr, actual_arr)[0, 1] if len(pred_arr) > 1 else float("nan")
    ic_std = float(
        np.std([
            np.corrcoef(pred_arr[i : i + 50], actual_arr[i : i + 50])[0, 1]
            for i in range(0, len(pred_arr) - 50, 10)
        ])
    ) if len(pred_arr) > 60 else float("nan")
    hit_rate = float(np.mean(np.sign(pred_arr) == np.sign(actual_arr)))

    return {
        "IC_mean": float(ic),
        "IC_std": ic_std,
        "IC_IR": float(ic / ic_std) if ic_std and ic_std > 0 else float("nan"),
        "hit_rate": hit_rate,
        "n_predictions": len(pred_arr),
    }

# BATCH_INFERENCE_v2
