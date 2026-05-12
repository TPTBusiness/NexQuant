"""
FTMO Live Trader — momentum_volatility_mixture on EUR/USD
FIX 4.4 protocol via cTrader FIX API (two sessions: QUOTE + TRADE)

Required .env vars:
    FTMO_FIX_HOST           live-uk-eqx-01.p.c-trader.com
    FTMO_FIX_QUOTE_PORT     5211
    FTMO_FIX_TRADE_PORT     5212
    FTMO_FIX_SENDER_COMP_ID live.ftmo.17104129
    FTMO_FIX_TARGET_COMP_ID cServer
    FTMO_FIX_ACCOUNT_ID     17104129
    FTMO_FIX_PASSWORD       <account password>
    FTMO_ACCOUNT_SIZE       10000   (equity in USD — cTrader FIX has no equity query API)

Architecture:
    QuoteSession  — SSL to port 5211, subscribes EUR/USD ticks, aggregates → 1-min bars
    TradeSession  — SSL to port 5212, sends MarketOrders with SL/TP
    FTMOFIXTrader — signal logic (daily factors), risk management, coordinates both sessions

Strategy: momentum_volatility_mixture
    Factors computed once per UTC day at day close:
        daily_ret_close_1d   = (close_today - close_prev) / close_prev
        daily_ret_vol_adj_1d = daily_ret / realized_vol[prev_day]
        daily_ret_1d         = same as daily_ret_close_1d
    Signal: weighted z-score composite vs 70th/30th percentile of rolling history

FTMO rules enforced:
    - Max daily loss: 5%  → flat all, no new trades today
    - Max total loss: 10% → emergency stop
    - Risk per trade: 0.5% equity with 10-pip hard stop
    - Take profit:    20 pip (2:1 R/R)
    - Max leverage:   1:30
    - No trading Fri 21:00 UTC → Sun 22:00 UTC
"""

from __future__ import annotations

import logging
import os
import random
import socket
import ssl
import string
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone, date
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import simplefix

load_dotenv(Path(__file__).parent.parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("ftmo_live_trader.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
HOST = os.environ["FTMO_FIX_HOST"]
QUOTE_PORT = int(os.environ["FTMO_FIX_QUOTE_PORT"])
TRADE_PORT = int(os.environ["FTMO_FIX_TRADE_PORT"])
SENDER_COMP_ID = os.environ["FTMO_FIX_SENDER_COMP_ID"]
TARGET_COMP_ID = os.environ["FTMO_FIX_TARGET_COMP_ID"]
ACCOUNT_ID = os.environ["FTMO_FIX_ACCOUNT_ID"]
PASSWORD = os.environ["FTMO_FIX_PASSWORD"]

SYMBOL_NAME = "EURUSD"
SYMBOL = "EURUSD"
PIP = 0.0001
RISK_PCT = 0.005  # 0.5% equity per trade
STOP_PIPS = 10
TP_PIPS = 20
MAX_LEVERAGE = 30
FTMO_DAILY_LIMIT = 0.05
FTMO_TOTAL_LIMIT = 0.10
HEARTBEAT_SEC = 30
MAX_POSITIONS   = 1      # max concurrent open positions (hard cap)

# momentum_volatility_mixture weights
FACTOR_WEIGHTS = {
    "daily_ret_close_1d": 0.2548,
    "daily_ret_vol_adj_1d": 0.2347,
    "daily_ret_1d": 0.1291,
}
DAILY_WINDOW = 252  # rolling window for z-score and percentile (1 trading year)
MIN_DAILY_BARS = 30  # warmup: minimum daily bars before trading

LOCAL_DATA_CSV = os.path.expanduser("~/.qlib/qlib_data/eurusd_1min_data/eurusd_1min.csv")
WARMUP_DAYS = 300  # daily bars to pre-load from local history


# ── FIX helpers ───────────────────────────────────────────────────────────────


def _clordid() -> str:
    return "predix_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=8))


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3]


class FIXSession:
    """Low-level FIX 4.4 session over SSL TCP."""

    def __init__(self, host: str, port: int, sender_sub_id: str, label: str):
        self.host = host
        self.port = port
        self.sender_sub_id = sender_sub_id
        self.label = label
        self._sock: socket.socket | None = None
        self._parser = simplefix.FixParser()
        self._send_seq = 1
        self._recv_seq = 1
        self._lock = threading.Lock()
        self._connected = False
        self._stop = threading.Event()
        self._recv_thread: threading.Thread | None = None
        self._hb_thread: threading.Thread | None = None   # Bug 4: track to prevent duplicates
        self._auth_failures = 0
        self.on_message = None  # callback(msg: simplefix.FixMessage)
        self.on_logon = None  # callback() — called once logon is confirmed

    def connect(self) -> None:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        raw = socket.create_connection((self.host, self.port), timeout=30)
        self._sock = ctx.wrap_socket(raw, server_hostname=self.host)
        self._sock.settimeout(None)  # blocking after connect — recv_loop runs in dedicated thread
        self._connected = True
        self._stop.clear()
        self._parser = simplefix.FixParser()  # reset parser on new connection
        self._send_seq = 1   # Bug 7: must reset to 1 so logon with ResetSeqNumFlag=Y is seq=1
        self._recv_seq = 1
        logger.info(f"[{self.label}] Connected to {self.host}:{self.port}")
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True, name=f"{self.label}-recv")
        self._recv_thread.start()
        self._send_logon()
        self._start_heartbeat()

    def disconnect(self) -> None:
        self._stop.set()
        self._send_logout()
        if self._sock:
            self._sock.close()
        self._connected = False

    # ── Send ──────────────────────────────────────────────────────────────────

    def send(self, msg_type: str, body_pairs: list[tuple]) -> None:
        """Build a correctly-ordered FIX 4.4 message and send it.

        FIX 4.4 header order: 8, 9(auto), 35, 49, 50, 56, 34, 52 — then body — then 10(auto).
        simplefix inserts 9 after 8 and appends 10 automatically on encode().
        """
        with self._lock:
            msg = simplefix.FixMessage()
            msg.append_pair(8, "FIX.4.4")
            msg.append_pair(35, msg_type)
            msg.append_pair(49, SENDER_COMP_ID)
            msg.append_pair(50, self.sender_sub_id)  # SenderSubID
            msg.append_pair(56, TARGET_COMP_ID)
            msg.append_pair(57, self.sender_sub_id)  # TargetSubID — cTrader requires this
            msg.append_pair(34, self._send_seq)
            msg.append_pair(52, _utcnow())
            for tag, value in body_pairs:
                msg.append_pair(tag, value)
            raw = msg.encode()
            pipe = b"|"
            sep = b"\x01"
            logger.debug(f"[{self.label}] SEND: {raw.replace(sep, pipe)}")
            self._sock.sendall(raw)
            self._send_seq += 1

    def _send_logon(self) -> None:
        self.send(
            "A",
            [
                (98, "0"),  # EncryptMethod = None
                (108, HEARTBEAT_SEC),  # HeartBtInt
                (141, "Y"),  # ResetSeqNumFlag
                (553, ACCOUNT_ID),  # Username
                (554, PASSWORD),  # Password
            ],
        )
        logger.info(f"[{self.label}] Logon sent")

    def _send_logout(self) -> None:
        try:
            self.send("5", [])
        except Exception:
            pass

    def _send_heartbeat(self, test_req_id: str | None = None) -> None:
        body = [(112, test_req_id)] if test_req_id else []
        self.send("0", body)

    def _start_heartbeat(self) -> None:
        # Bug 4: only start a new thread if no existing heartbeat thread is alive
        if self._hb_thread is not None and self._hb_thread.is_alive():
            return

        def _hb_loop():
            while not self._stop.is_set():
                time.sleep(HEARTBEAT_SEC)
                if self._connected:
                    self._send_heartbeat()

        self._hb_thread = threading.Thread(target=_hb_loop, daemon=True, name=f"{self.label}-hb")
        self._hb_thread.start()

    # ── Receive ───────────────────────────────────────────────────────────────

    def _recv_loop(self) -> None:
        while not self._stop.is_set():
            try:
                data = self._sock.recv(4096)
                if not data:
                    logger.warning(f"[{self.label}] Connection closed by server")
                    self._on_disconnect()
                    break
                self._parser.append_buffer(data)
                while True:
                    msg = self._parser.get_message()
                    if msg is None:
                        break
                    self._dispatch(msg)
            except Exception as e:
                if not self._stop.is_set():
                    logger.error(f"[{self.label}] Recv error: {e}")
                    self._on_disconnect()
                break

    def _dispatch(self, msg: simplefix.FixMessage) -> None:
        msg_type = msg.get(35)
        if msg_type == b"0":  # Heartbeat
            return
        if msg_type == b"1":  # TestRequest → send Heartbeat back
            self._send_heartbeat(msg.get(112))
            return
        if msg_type == b"A":  # Logon confirmed
            logger.info(f"[{self.label}] Logon confirmed")
            if self.on_logon:
                self.on_logon()
            return
        if msg_type == b"5":  # Logout
            reason = (msg.get(58) or b"").decode(errors="replace")
            ref_tag = (msg.get(371) or b"").decode(errors="replace")
            ref_id = (msg.get(372) or b"").decode(errors="replace")
            rej_rsn = (msg.get(373) or b"").decode(errors="replace")
            logger.warning(
                f"[{self.label}] Logout received — "
                f"text={reason!r} RefTagID={ref_tag} RefMsgType={ref_id} Reason={rej_rsn}"
            )
            self._auth_failures += 1
            if self._auth_failures >= 3:
                logger.error(f"[{self.label}] 3 consecutive auth failures — stopping reconnect")
                self._stop.set()
                return
            self._on_disconnect()
            return
        if self.on_message:
            self.on_message(msg)

    def _on_disconnect(self) -> None:
        if not self._connected:
            return
        self._connected = False
        if self._stop.is_set():
            return
        logger.warning(f"[{self.label}] Disconnected — reconnecting in 10s")
        threading.Thread(target=self._reconnect_worker, daemon=True).start()

    def _reconnect_worker(self) -> None:
        time.sleep(10)
        if self._stop.is_set():
            return
        try:
            self.connect()
        except Exception as e:
            logger.error(f"[{self.label}] Reconnect failed: {e}")


# ── Main trader ───────────────────────────────────────────────────────────────


class FTMOFIXTrader:
    def __init__(self):
        # Sessions
        self.quote_session = FIXSession(HOST, QUOTE_PORT, "QUOTE", "QUOTE")
        self.trade_session = FIXSession(HOST, TRADE_PORT, "TRADE", "TRADE")
        self.quote_session.on_message = self._on_quote_message
        self.trade_session.on_message = self._on_trade_message

        # Tick → 1-min bar state
        self._current_bar_time: datetime | None = None
        self._bar_open = 0.0
        self._bar_high = 0.0
        self._bar_low = 0.0
        self._bar_close = 0.0

        # Daily factor state — factors computed once per UTC day at close
        self._current_date: date | None = None  # UTC date currently accumulating
        self._day_prev_close: float = 0.0  # close of previous bar (for log-return & daily close)
        self._minute_closes: deque = deque(maxlen=60 * 48)  # ~48h for 1h SMA
        self._day_log_rets: list[float] = []  # intraday log-returns for _current_date

        self._daily_closes: deque = deque(maxlen=DAILY_WINDOW + 5)
        self._daily_vols: deque = deque(maxlen=DAILY_WINDOW + 5)  # realized vol per day
        self._f1_hist: deque = deque(maxlen=DAILY_WINDOW)  # daily_ret_close_1d history
        self._f2_hist: deque = deque(maxlen=DAILY_WINDOW)  # daily_ret_vol_adj_1d history
        self._f3_hist: deque = deque(maxlen=DAILY_WINDOW)  # daily_ret_1d history
        self._composite_hist: deque = deque(maxlen=DAILY_WINDOW)

        self.current_signal = 0  # held constant during the day
        self._signal_lock = threading.Lock()  # prevent re-entrant signal processing
        self.daily_bars_seen = 0  # count of finalized days

        # Equity / risk state
        self.initial_equity = None
        self.current_equity = None
        self.day_start_equity = None
        self.current_day = None
        self.daily_blocked = False
        self.total_blocked = False

        # Position state
        self.position_side       = 0      # 1=long, -1=short, 0=flat
        self.position_qty        = 0
        self.open_positions      = 0      # confirmed open positions counter
        self.open_clordid        = None
        self.last_signal         = 0
        self._pending_sl: float | None = None
        self._pending_tp: float | None = None
        self._pending_open_side: int   = 0    # direction to open after close confirmed
        self._last_mid:          float = 0.0  # latest mid price for deferred opens
        self._close_in_flight:   bool  = False  # Bug 3: prevents double-close

        # MD subscription ID
        self._md_req_id = "MD_EURUSD_1"
        self._symbol_id = None

    def _resolve_symbol_id(self) -> None:
        """Connect briefly to QUOTE session, send SecurityListRequest, find EURUSD numeric ID."""
        import simplefix as _sf

        def fix_msg(fields):
            body = "".join(f"{k}={v}\x01" for k, v in fields)
            hdr = f"8=FIX.4.4\x019={len(body)}\x01"
            raw = hdr + body
            cs = sum(raw.encode()) % 256
            return (raw + f"10={cs:03d}\x01").encode()

        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            s = ctx.wrap_socket(
                socket.create_connection((HOST, QUOTE_PORT), timeout=10),
                server_hostname=HOST,
            )
            now = time.strftime("%Y%m%d-%H:%M:%S.000")
            s.sendall(
                fix_msg(
                    [
                        (35, "A"),
                        (49, SENDER_COMP_ID),
                        (50, "QUOTE"),
                        (56, TARGET_COMP_ID),
                        (57, "QUOTE"),
                        (34, 1),
                        (52, now),
                        (98, 0),
                        (108, 30),
                        (141, "Y"),
                        (553, ACCOUNT_ID),
                        (554, PASSWORD),
                    ]
                )
            )
            time.sleep(0.5)
            s.sendall(
                fix_msg(
                    [
                        (35, "x"),
                        (49, SENDER_COMP_ID),
                        (50, "QUOTE"),
                        (56, TARGET_COMP_ID),
                        (57, "QUOTE"),
                        (34, 2),
                        (52, time.strftime("%Y%m%d-%H:%M:%S.000")),
                        (320, "SLR1"),
                        (559, "0"),
                    ]
                )
            )
            s.settimeout(6)
            buf = b""
            try:
                while True:
                    buf += s.recv(65536)
            except OSError:
                pass
            s.close()

            text = buf.decode("ascii", errors="replace")
            parts = text.split("\x01")
            cur_id = None
            for part in parts:
                if part.startswith("55="):
                    cur_id = part[3:]
                elif part.startswith("1007=") and cur_id is not None:
                    if part[5:] == SYMBOL_NAME:
                        self._symbol_id = cur_id
                        logger.info(f"Resolved {SYMBOL_NAME} → symbolId={cur_id}")
                        return
                    cur_id = None
        except Exception as e:
            logger.warning(f"Symbol resolution failed: {e}")

        if self._symbol_id is None:
            logger.warning(f"Could not resolve {SYMBOL_NAME} — defaulting to id=1")
            self._symbol_id = "1"

    def _load_historical_daily(self) -> None:
        """Pre-load WARMUP_DAYS daily bars from local 1-min CSV to skip live warm-up."""
        if not os.path.exists(LOCAL_DATA_CSV):
            logger.warning(f"Local data not found: {LOCAL_DATA_CSV} — will warm up live")
            return
        try:
            df = pd.read_csv(LOCAL_DATA_CSV, usecols=["datetime", "close"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["date"] = df["datetime"].dt.normalize()

            # Daily close = last 1-min bar of each UTC day
            daily = df.groupby("date")["close"].last().reset_index()
            daily.columns = ["date", "close"]
            daily = daily.tail(WARMUP_DAYS).reset_index(drop=True)

            # Realized vol per day = std of intraday log-returns
            def _day_vol(group):
                closes = group["close"].values.astype(float)
                if len(closes) < 2:
                    return np.nan
                return float(np.std(np.diff(np.log(closes)), ddof=1))

            vol_series = df.groupby("date").apply(_day_vol, include_groups=False)
            vol_series = vol_series.reindex(daily["date"])

            # Populate histories
            for i, row in daily.iterrows():
                dc = float(row["close"])
                dv = float(vol_series.iloc[i]) if not np.isnan(vol_series.iloc[i]) else None
                self._daily_vols.append(dv)
                self._daily_closes.append(dc)

                if i > 0:
                    prev_close = float(daily.iloc[i - 1]["close"])
                    daily_ret = (dc - prev_close) / prev_close if prev_close > 0 else 0.0
                    # vol-adjusted return uses lagged vol (two positions back = day before yesterday)
                    prev_vol = (
                        float(vol_series.iloc[i - 1]) if i >= 1 and not np.isnan(vol_series.iloc[i - 1]) else None
                    )
                    vol_adj = daily_ret / prev_vol if (prev_vol and prev_vol > 0) else 0.0

                    self._f1_hist.append(daily_ret)
                    self._f2_hist.append(vol_adj)
                    self._f3_hist.append(daily_ret)

                    if len(self._f1_hist) >= 2:
                        composite = self._calc_composite()
                        self._composite_hist.append(composite)
                    self.daily_bars_seen += 1

            if len(daily) > 0:
                self._day_prev_close = float(daily.iloc[-1]["close"])
                logger.info(
                    f"Pre-loaded {len(daily)} daily bars "
                    f"(last={daily.iloc[-1]['date'].date() if hasattr(daily.iloc[-1]['date'], 'date') else daily.iloc[-1]['date']})"
                )
        except Exception as e:
            logger.warning(f"Failed to load historical daily data: {e} — will warm up live")

    def _calc_1h_signal(self) -> int:
        """Compute 1h SMA10/30 crossover signal from LIVE minute closes.

        Uses self._minute_closes deque (appended every bar) to build 1h SMA.
        Backtest proven: +0.40%/month OOS, -0.86% worst day, FTMO-safe.
        Only trades during London+NY session (07-17 UTC).
        """
        if len(self._minute_closes) < 30 * 60:  # Need at least 30 hours
            return 0

        import pandas as pd

        # Build 1h bars from accumulated minute closes
        closes = list(self._minute_closes)
        # Take last (close) of each 60-min block
        h1_bars = closes[-61::60]  # Every 60th minute = 1h close
        if len(h1_bars) < 30:
            return 0

        h1 = pd.Series(h1_bars)
        sma10 = h1.rolling(10).mean()
        sma30 = h1.rolling(30).mean()

        now = pd.Timestamp.now(tz="UTC")
        hour = now.hour

        if hour < 7 or hour >= 17:
            return 0

        if pd.isna(sma10.iloc[-1]) or pd.isna(sma30.iloc[-1]):
            return 0

        if sma10.iloc[-1] > sma30.iloc[-1]:
            return 1
        elif sma10.iloc[-1] < sma30.iloc[-1]:
            return -1
        return 0

    def _calc_composite(self) -> float:
        """Compute weighted z-score composite from current factor histories."""

        def _zscore(hist: deque) -> float:
            arr = np.array(hist, dtype=float)
            std = float(arr.std())
            return (arr[-1] - arr.mean()) / std if std > 0 else 0.0

        return (
            _zscore(self._f1_hist) * FACTOR_WEIGHTS["daily_ret_close_1d"]
            + _zscore(self._f2_hist) * FACTOR_WEIGHTS["daily_ret_vol_adj_1d"]
            + _zscore(self._f3_hist) * FACTOR_WEIGHTS["daily_ret_1d"]
        )

    def start(self) -> None:
        logger.info("=" * 60)
        logger.info("FTMO FIX Trader — momentum_volatility_mixture / EUR/USD")
        logger.info(f"Host:             {HOST}")
        logger.info(f"Risk per trade:   {RISK_PCT * 100:.1f}%  SL={STOP_PIPS}pip  TP={TP_PIPS}pip")
        logger.info(f"FTMO limits:      daily -{FTMO_DAILY_LIMIT:.0%}  total -{FTMO_TOTAL_LIMIT:.0%}")
        logger.info("=" * 60)

        self._load_historical_daily()
        self._resolve_symbol_id()
        # Re-subscribe on every (re-)logon so ticks resume after reconnects
        self.quote_session.on_logon = self._subscribe_market_data
        self.quote_session.connect()
        time.sleep(1)
        self._init_equity_from_env()
        # Bug 6: request open positions on every trade logon to reconcile state after restart
        self.trade_session.on_logon = self._request_open_positions
        self.trade_session.connect()
        time.sleep(1)

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self.quote_session.disconnect()
            self.trade_session.disconnect()

    # ── Market data subscription ──────────────────────────────────────────────

    def _subscribe_market_data(self) -> None:
        self.quote_session.send(
            "V",
            [
                (262, self._md_req_id),
                (263, "1"),
                (264, "1"),
                (265, "0"),
                (267, "2"),
                (269, "0"),
                (269, "1"),
                (146, "1"),
                (55, self._symbol_id),
            ],
        )
        logger.info(f"Subscribed to {SYMBOL_NAME} (id={self._symbol_id}) market data")

    def _request_open_positions(self) -> None:
        """Bug 6: query open positions on (re-)logon to reconcile state after restart."""
        self.trade_session.send(
            "AF",
            [
                (584, "POS_QUERY_1"),  # MassStatusReqID
                (585, "7"),            # MassStatusReqType = all positions
                (1, ACCOUNT_ID),       # Account
            ],
        )
        logger.info("Requested open position status for reconciliation")

    # ── Quote message handler ─────────────────────────────────────────────────

    def _on_quote_message(self, msg: simplefix.FixMessage) -> None:
        msg_type = msg.get(35)
        if msg_type in (b"W", b"X"):
            self._on_tick(msg)

    def _on_tick(self, msg: simplefix.FixMessage) -> None:
        # Bug 5: parse MDEntry repeating groups (269=type, 270=price) for true mid
        bid: float | None = None
        ask: float | None = None
        current_type: bytes | None = None
        for tag, val in msg.pairs:
            if tag == b"269":
                current_type = val
            elif tag == b"270":
                try:
                    px = float(val)
                except (ValueError, TypeError):
                    continue
                if current_type == b"0":
                    bid = px
                elif current_type == b"1":
                    ask = px
        if bid is None and ask is None:
            return
        mid = (bid + ask) / 2.0 if bid is not None and ask is not None else (bid or ask)
        if not mid:
            return

        now = datetime.now(timezone.utc)
        bar_time = now.replace(second=0, microsecond=0)

        if self._current_bar_time is None:
            self._current_bar_time = bar_time
            self._bar_open = self._bar_high = self._bar_low = self._bar_close = mid
            return

        self._last_mid = mid

        if bar_time > self._current_bar_time:
            self._on_bar_closed(self._bar_close)
            self._current_bar_time = bar_time
            self._bar_open = mid
            self._bar_high = mid
            self._bar_low = mid
        else:
            self._bar_high = max(self._bar_high, mid)
            self._bar_low = min(self._bar_low, mid)

        self._bar_close = mid

    def _on_bar_closed(self, close: float) -> None:
        """Called with the close price of the just-completed 1-min bar."""
        logger.info(
            f"Bar: {self._current_bar_time.strftime('%H:%M')}  "
            f"O={self._bar_open:.5f} H={self._bar_high:.5f} "
            f"L={self._bar_low:.5f} C={close:.5f}"
        )
        bar_date = self._current_bar_time.date()

        # Bug 8: weekend blackout check BEFORE daily-close so stray ticks can't open positions
        now = datetime.now(timezone.utc)
        weekend = (
            (now.weekday() == 4 and now.hour >= 21)
            or now.weekday() == 5
            or (now.weekday() == 6 and now.hour < 22)
        )

        # Day change: finalize yesterday's daily close (only outside weekend blackout)
        if self._current_date is not None and bar_date != self._current_date:
            if not weekend:
                # _day_prev_close is the close of the last bar of _current_date (finalized day)
                self._on_daily_close(self._day_prev_close)
            self._current_date = bar_date
            self._day_log_rets = []
        elif self._current_date is None:
            self._current_date = bar_date

        # Accumulate intraday log-return (skip first bar of day since no prev reference yet)
        if self._day_prev_close > 0 and bar_date == self._current_date:
            log_ret = float(np.log(close / self._day_prev_close))
            self._day_log_rets.append(log_ret)

        self._day_prev_close = close
        self._minute_closes.append(close)

        if weekend:
            return

        # Risk limits (checked every bar, not only on day change)
        self._check_risk_limits()

        # ── 1h SMA10/30 signal (proven +0.40%/month OOS) ──
        # Check every minute if this is the top of the hour
        if self._current_bar_time.minute == 0 and not self.total_blocked:
            signal_1h = self._calc_1h_signal()
            if signal_1h != self.current_signal and signal_1h != 0:
                logger.info(
                    f"1h SMA signal: {signal_1h:+d}  "
                    f"SMA10/30 crossover  |  replacing daily signal ({self.current_signal:+d})"
                )
                self.current_signal = signal_1h
                self._on_signal(signal_1h, close)

    def _on_daily_close(self, daily_close: float) -> None:
        """Finalize one UTC day: compute factors, update signal."""
        # Daily equity reset
        today = self._current_date
        if self.current_day is None:
            self.current_day = today
            self.day_start_equity = self.current_equity
        elif today != self.current_day:
            self.current_day = today
            self.day_start_equity = self.current_equity
            self.daily_blocked = False
            logger.info(f"New trading day {today} — equity: ${self.current_equity:,.2f}")

        # Compute realized vol for the day that just closed
        if len(self._day_log_rets) > 1:
            realized_vol: float | None = float(np.std(self._day_log_rets, ddof=1))
        else:
            realized_vol = None
        self._daily_vols.append(realized_vol)

        # Daily return
        daily_ret = 0.0
        if len(self._daily_closes) > 0 and self._daily_closes[-1] > 0:
            daily_ret = (daily_close - self._daily_closes[-1]) / self._daily_closes[-1]
        self._daily_closes.append(daily_close)

        # Vol-adjusted return uses previous day's realized vol (lagged by 1)
        prev_vol = self._daily_vols[-2] if len(self._daily_vols) >= 2 else None
        vol_adj = daily_ret / prev_vol if (prev_vol and prev_vol > 0) else 0.0

        self._f1_hist.append(daily_ret)
        self._f2_hist.append(vol_adj)
        self._f3_hist.append(daily_ret)
        self.daily_bars_seen += 1

        if len(self._f1_hist) < 3:
            logger.debug(f"Warming up daily factors… {self.daily_bars_seen}/{MIN_DAILY_BARS}")
            return

        composite = self._calc_composite()
        self._composite_hist.append(composite)

        # Signal via 70th/30th percentile threshold
        if self.daily_bars_seen < MIN_DAILY_BARS or len(self._composite_hist) < 5:
            logger.debug(f"Warming up composite… {self.daily_bars_seen}/{MIN_DAILY_BARS}")
            return

        arr = np.array(self._composite_hist, dtype=float)
        p70 = float(np.percentile(arr, 70))
        p30 = float(np.percentile(arr, 30))
        if composite > p70:
            new_signal = 1
        elif composite < p30:
            new_signal = -1
        else:
            new_signal = 0

        self.current_signal = new_signal
        logger.info(
            f"Daily close={daily_close:.5f}  ret={daily_ret:.4%}  "
            f"vol_adj={vol_adj:.4f}  composite={composite:.3f}  "
            f"p30={p30:.3f}  p70={p70:.3f}  signal={new_signal:+d}"
        )

        if not self.total_blocked:
            self._on_signal(new_signal, daily_close)

    # ── Trade message handler ─────────────────────────────────────────────────

    def _init_equity_from_env(self) -> None:
        """Set initial equity from FTMO_ACCOUNT_SIZE env var (cTrader FIX has no equity query)."""
        try:
            size = float(os.environ.get("FTMO_ACCOUNT_SIZE", "0"))
        except ValueError:
            size = 0.0
        if size > 0:
            self.initial_equity = size
            self.current_equity = size
            self.day_start_equity = size
            logger.info(f"Account equity set from env: ${size:,.2f}")

    def _on_trade_message(self, msg: simplefix.FixMessage) -> None:
        msg_type = msg.get(35)

        if msg_type == b"8":
            self._on_exec_report(msg)
        elif msg_type == b"BA":  # Bug 2: CollateralReport — contains updated equity
            self._on_collateral_report(msg)
        elif msg_type == b"9":
            logger.warning(f"Order cancel rejected: {msg.get(58)}")
        elif msg_type == b"j":
            logger.error(f"Business reject: {msg.get(58)}")
        else:
            raw_type = (msg_type or b"").decode(errors="replace")
            logger.debug(f"[TRADE] unhandled msg type={raw_type!r} tag58={msg.get(58)}")

    def _on_exec_report(self, msg: simplefix.FixMessage) -> None:
        exec_type = msg.get(150)
        text = (msg.get(58) or b"").decode()

        if exec_type in (b"2", b"F"):
            side_raw = msg.get(54)
            qty_raw = msg.get(32)
            price_raw = msg.get(31) or msg.get(669) or msg.get(6)
            side = 1 if side_raw == b"1" else -1
            qty = int(float(qty_raw)) if qty_raw else 0
            price = float(price_raw) if price_raw else 0.0

            # Bug 1: distinguish closing fill (opposite side) from opening fill
            is_closing = self.position_side != 0 and side != self.position_side
            if is_closing:
                self.position_side  = 0
                self.position_qty   = 0
                self.open_positions = max(self.open_positions - 1, 0)
                self._close_in_flight = False  # Bug 3: close completed
                logger.info(f"Close fill: {'BUY' if side == 1 else 'SELL'} {qty} @ {price:.5f}  open_pos={self.open_positions}")
                pending = self._pending_open_side
                if pending != 0 and not self.daily_blocked and not self.total_blocked:
                    self._pending_open_side = 0
                    open_price = self._last_mid or price
                    if self.open_positions < MAX_POSITIONS and open_price > 0:
                        logger.info(f"Deferred open after close fill: signal={pending:+d} @ {open_price:.5f}")
                        self._open_position(pending, open_price)
            else:
                self.position_side  = side
                self.position_qty   = qty
                self.open_positions = min(self.open_positions + 1, MAX_POSITIONS)
                logger.info(f"Open fill: {'BUY' if side == 1 else 'SELL'} {qty} @ {price:.5f}  open_pos={self.open_positions}")
                self._pending_sl = None
                self._pending_tp = None

        elif exec_type in (b"4", b"3"):
            self.position_side  = 0
            self.position_qty   = 0
            self.open_positions = max(self.open_positions - 1, 0)
            self._close_in_flight = False  # Bug 3: close completed
            logger.info(f"Order cancelled/closed ({text})  open_pos={self.open_positions}")
            pending = self._pending_open_side
            if pending != 0 and not self.daily_blocked and not self.total_blocked:
                self._pending_open_side = 0
                price = self._last_mid or 0.0
                if self.open_positions < MAX_POSITIONS and price > 0:
                    logger.info(f"Deferred open: signal={pending:+d} @ {price:.5f}")
                    self._open_position(pending, price)

        elif exec_type == b"C":
            self.position_side  = 0
            self.position_qty   = 0
            self.open_positions = max(self.open_positions - 1, 0)
            self._close_in_flight = False  # Bug 3
            self._pending_open_side = 0   # expired order — discard pending
            logger.warning(f"Order expired ({text})  open_pos={self.open_positions}")

        elif exec_type == b"I":
            # Bug 6: OrderMassStatusRequest response — reconcile position state on (re)start
            ord_status = msg.get(39)
            side_raw = msg.get(54)
            qty_raw = msg.get(151) or msg.get(38)
            if ord_status in (b"1", b"2") and side_raw and qty_raw:
                # OrdStatus 1=PartiallyFilled, 2=Filled but still open (position exists)
                reconciled_side = 1 if side_raw == b"1" else -1
                reconciled_qty  = int(float(qty_raw))
                if self.position_side == 0 and reconciled_qty > 0:
                    self.position_side  = reconciled_side
                    self.position_qty   = reconciled_qty
                    self.open_positions = 1
                    logger.warning(
                        f"Reconciled open position from status: "
                        f"{'BUY' if reconciled_side == 1 else 'SELL'} {reconciled_qty} units"
                    )

        if text:
            logger.debug(f"ExecReport text: {text}")

    def _on_collateral_report(self, msg: simplefix.FixMessage) -> None:
        equity_raw = msg.get(900) or msg.get(899) or msg.get(402) or msg.get(730)
        if equity_raw is None:
            return
        try:
            equity = float(equity_raw)
        except ValueError:
            return
        if equity <= 0:
            return
        if self.initial_equity is None:
            self.initial_equity = equity
            self.day_start_equity = equity
            logger.info(f"Account equity: ${equity:,.2f} (initial)")
        else:
            self.current_equity = equity
            logger.debug(f"Equity update: ${equity:,.2f}")
        self.current_equity = equity

    # ── Risk management ───────────────────────────────────────────────────────

    def _check_risk_limits(self) -> None:
        if self.current_equity is None or self.initial_equity is None or self.day_start_equity is None:
            return
        daily_loss = (self.current_equity - self.day_start_equity) / self.initial_equity
        total_loss = (self.current_equity - self.initial_equity) / self.initial_equity

        if daily_loss < -FTMO_DAILY_LIMIT and not self.daily_blocked:
            self.daily_blocked = True
            logger.warning(
                f"DAILY LOSS LIMIT HIT: {daily_loss:.2%} (limit -{FTMO_DAILY_LIMIT:.0%}) — closing, no new trades today"
            )
            self._close_position("daily loss limit")

        if total_loss < -FTMO_TOTAL_LIMIT and not self.total_blocked:
            self.total_blocked = True
            logger.critical(f"TOTAL LOSS LIMIT HIT: {total_loss:.2%} (limit -{FTMO_TOTAL_LIMIT:.0%}) — EMERGENCY STOP")
            self._close_position("total loss limit EMERGENCY")

    # ── Signal → Order ────────────────────────────────────────────────────────

    def _on_signal(self, signal: int, price: float) -> None:
        if not self._signal_lock.acquire(blocking=False):
            return
        try:
            if self.daily_blocked or self.total_blocked:
                return
            if signal == self.last_signal:
                return
            self.last_signal = signal
            logger.info(f"Signal → {signal:+d}  price={price:.5f}")

            if self.position_side != 0 and signal != self.position_side:
                # Close first — open will fire from _on_exec_report after fill confirmed
                self._pending_open_side = signal
                self._close_position("signal flip")
                return   # do NOT open here — avoid simultaneous long+short

            if signal != 0 and self.open_positions < MAX_POSITIONS:
                self._open_position(signal, price)
            elif signal == 0 and self.position_side != 0:
                self._pending_open_side = 0
                self._close_position("signal exit")
        finally:
            self._signal_lock.release()

    def _position_qty_units(self, price: float) -> int:
        if not self.current_equity:
            return 1000
        stop_dist = STOP_PIPS * PIP
        risk_amt = self.current_equity * RISK_PCT
        units_by_risk = risk_amt / stop_dist
        units_by_lev = self.current_equity * MAX_LEVERAGE / price
        return max(int(min(units_by_risk, units_by_lev)), 1000)

    def _open_position(self, side: int, price: float) -> None:
        units = self._position_qty_units(price)
        stop_price = round(price - side * STOP_PIPS * PIP, 5)
        tp_price = round(price + side * TP_PIPS * PIP, 5)
        clordid = _clordid()
        self.open_clordid = clordid

        self.trade_session.send(
            "D",
            [
                (11, clordid),
                (55, self._symbol_id),
                (54, "1" if side == 1 else "2"),
                (60, _utcnow()),
                (40, "1"),
                (38, units),
                (99, stop_price),   # Stop-Loss
                (44, tp_price),     # Take-Profit
            ],
        )
        self._pending_sl = stop_price
        self._pending_tp = tp_price

        logger.info(
            f"{'BUY' if side == 1 else 'SELL'} {units:,} units {SYMBOL} "
            f"@ {price:.5f}  SL={stop_price}  TP={tp_price}  "
            f"risk=${(self.current_equity or 0) * RISK_PCT:,.0f}"
        )

    def _close_position(self, reason: str) -> None:
        if self.position_side == 0:
            return
        # Bug 3: prevent double-close if a close order is already in flight
        if self._close_in_flight:
            logger.debug(f"_close_position({reason}) skipped — close already in flight")
            return
        self._close_in_flight = True
        clordid = _clordid()
        self.trade_session.send(
            "D",
            [
                (11, clordid),
                (55, self._symbol_id),
                (54, "2" if self.position_side == 1 else "1"),
                (60, _utcnow()),
                (40, "1"),
                (38, self.position_qty),
            ],
        )
        logger.info(f"Closing position ({reason})")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trader = FTMOFIXTrader()
    trader.start()
