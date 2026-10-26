# analytics_api/signals.py
from __future__ import annotations

import numpy as np
import pandas as pd


def _slope(series: pd.Series, n: int = 3) -> float:
    s = series.dropna()
    if len(s) < n:
        return 0.0
    y = s.iloc[-n:].values
    x = np.arange(n, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = (x ** 2).sum() or 1.0
    return float((x * y).sum() / denom)


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based signal engine.
    Produces a 'score' in [-5, 5] and a 'signal' in {'BUY','SELL','HOLD'}.
    Expects columns computed by indicators.compute_indicators.
    """
    d = df.copy()

    # ------------------------------------------------------------------
    # Scores
    # ------------------------------------------------------------------
    score = np.zeros(len(d), dtype=float)

    # 1) RSI thresholds
    score += np.where(d["rsi"] < 30, 1.0, 0.0)
    score += np.where(d["rsi"] > 70, -1.0, 0.0)

    # 2) MACD cross
    score += np.where(d["macd"] > d["macd_signal"], 0.8, -0.8)
    score += np.where(d["macd_hist"] > 0, 0.2, -0.2)

    # 3) Stochastic cross and extremes
    score += np.where(
        (d["stoch_k"] > d["stoch_d"]) & (d["stoch_k"] < 20),
        0.6,
        0.0,
    )
    score += np.where(
        (d["stoch_k"] < d["stoch_d"]) & (d["stoch_k"] > 80),
        -0.6,
        0.0,
    )

    # 4) ADX trend direction via DI
    score += np.where((d["adx"] > 20) & (d["+di"] > d["-di"]), 0.6, 0.0)
    score += np.where((d["adx"] > 20) & (d["+di"] < d["-di"]), -0.6, 0.0)

    # 5) CCI extremes (SAFE)
    if "cci" in d.columns:
        cci = d["cci"]
        score += np.where(cci < -100, 0.5, 0.0)
        score += np.where(cci > 100, -0.5, 0.0)

    # 6) Price vs SMAs
    score += np.where(
        (d["close"] > d["sma20"]) & (d["sma20"] > d["sma50"]),
        0.8,
        0.0,
    )
    score += np.where(
        (d["close"] < d["sma20"]) & (d["sma20"] < d["sma50"]),
        -0.8,
        0.0,
    )

    # EMA trend
    score += np.where(d["ema12"] > d["ema26"], 0.6, -0.6)

    # 7) WMA slope
    wma_slope = d["wma20"].rolling(4).apply(
        lambda s: _slope(pd.Series(s), n=len(s)),
        raw=False,
    )
    score += np.where(wma_slope > 0, 0.4, -0.4)

    # 8) Bollinger interactions
    score += np.where(d["close"] < d["bb_low"], 0.5, 0.0)
    score += np.where(d["close"] > d["bb_high"], -0.5, 0.0)

    # 9) Volume confirmation
    score += np.where(d["volume"] > d["vma20"], 0.2, -0.2)

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------
    d["score"] = np.round(score, 2)
    d["signal"] = np.where(
        d["score"] >= 1.2,
        "BUY",
        np.where(d["score"] <= -1.2, "SELL", "HOLD"),
    )

    return d

##################################################################
