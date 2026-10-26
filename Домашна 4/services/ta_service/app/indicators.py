# analytics_api/indicators.py
from __future__ import annotations

import pandas as pd

from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import (
    MACD,
    ADXIndicator,
    WMAIndicator,
    EMAIndicator,
    SMAIndicator,
    CCIIndicator,
)
from ta.volatility import BollingerBands


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _ensure_dtindex(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # normalize column names
    out.columns = [str(c).strip().lower() for c in out.columns]

    if not isinstance(out.index, pd.DatetimeIndex):
        if "date" in out.columns:
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            out = out.dropna(subset=["date"]).set_index("date")
        else:
            raise TypeError("OHLCV frame must have DatetimeIndex or a 'date' column.")

    if not out.index.is_monotonic_increasing:
        out = out.sort_index()

    return out


# ------------------------------------------------------------------------------
# Resampling
# ------------------------------------------------------------------------------
def resample_tf(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    df = _ensure_dtindex(df)

    if tf == "1d":
        return df

    rule = {"1w": "W-SUN", "1m": "ME"}[tf]

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    out = df.resample(rule).agg(agg).dropna()
    out.index.name = "date"
    return out


# ------------------------------------------------------------------------------
# Adaptive window helper
# ------------------------------------------------------------------------------
def _adaptive(window: int, n: int) -> int:
    if n >= window * 3:
        return window
    if n >= window * 2:
        return max(3, window // 2)
    return max(3, window // 3)


# ------------------------------------------------------------------------------
# Indicators
# ------------------------------------------------------------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_dtindex(df)

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    n = len(df)

    # adaptive windows
    w_rsi = _adaptive(14, n)
    w_macd_fast = _adaptive(12, n)
    w_macd_slow = max(w_macd_fast + 1, _adaptive(26, n))
    w_macd_sig = _adaptive(9, n)
    w_stoch = _adaptive(14, n)
    w_adx = _adaptive(14, n)
    w_cci = _adaptive(20, n)
    w_ma_mid = _adaptive(20, n)
    w_ma_slow = max(w_ma_mid + 1, _adaptive(50, n))
    w_bb = _adaptive(20, n)
    w_vma = _adaptive(20, n)

    # --------------------------------------------------------------------------
    # Oscillators
    # --------------------------------------------------------------------------
    rsi = RSIIndicator(df["close"], window=w_rsi).rsi()

    macd_obj = MACD(
        close=df["close"],
        window_slow=w_macd_slow,
        window_fast=w_macd_fast,
        window_sign=w_macd_sig,
    )

    stoch = StochasticOscillator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=w_stoch,
    )

    adx = ADXIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=w_adx,
    )

    cci = CCIIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=w_cci,
    ).cci()

    # --------------------------------------------------------------------------
    # Moving averages & volatility
    # --------------------------------------------------------------------------
    sma20 = SMAIndicator(df["close"], window=w_ma_mid).sma_indicator()
    sma50 = SMAIndicator(df["close"], window=w_ma_slow).sma_indicator()
    ema12 = EMAIndicator(df["close"], window=12).ema_indicator()
    ema26 = EMAIndicator(df["close"], window=26).ema_indicator()
    wma20 = WMAIndicator(df["close"], window=w_ma_mid).wma()

    bb = BollingerBands(df["close"], window=w_bb)
    vma20 = df["volume"].rolling(w_vma).mean()

    # --------------------------------------------------------------------------
    # Output
    # --------------------------------------------------------------------------
    out = df.copy()

    out["rsi"] = rsi

    out["macd"] = macd_obj.macd()
    out["macd_signal"] = macd_obj.macd_signal()
    out["macd_hist"] = macd_obj.macd_diff()

    out["stoch_k"] = stoch.stoch()
    out["stoch_d"] = stoch.stoch_signal()

    out["adx"] = adx.adx()
    out["+di"] = adx.adx_pos()
    out["-di"] = adx.adx_neg()

    out["cci"] = cci

    out["sma20"] = sma20
    out["sma50"] = sma50
    out["ema12"] = ema12
    out["ema26"] = ema26
    out["wma20"] = wma20

    out["bb_high"] = bb.bollinger_hband()
    out["bb_low"] = bb.bollinger_lband()

    out["vma20"] = vma20

    return out.dropna()

#############################################################
