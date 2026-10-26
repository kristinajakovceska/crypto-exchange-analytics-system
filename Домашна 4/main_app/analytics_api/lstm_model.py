# analytics_api/lstm_model.py
from __future__ import annotations
from typing import Dict, Callable
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def _make_windows(features: np.ndarray, target: np.ndarray, lookback: int):
    """
    Build sliding windows X[t-lookback:t, :] -> y[t]
    features: shape (N, F)
    target:   shape (N,)
    returns X (N-lookback, lookback, F), y (N-lookback,)
    """
    X, y = [], []
    for t in range(lookback, len(features)):
        X.append(features[t - lookback : t, :])
        y.append(target[t])
    return np.array(X), np.array(y)


def train_lstm(
    ohlcv: pd.DataFrame,
    lookback: int = 30,
    epochs: int = 20,
    batch_size: int = 32,
    verbose: int = 0,
) -> Dict[str, object]:
    """
    Multivariate LSTM using OHLCV as inputs, predicting Close.
    - Data split: 70% train, 30% validation
    - Loss: MSE
    - Metrics returned: RMSE, MAPE, R^2
    - Forecast function: forecast_next(n_steps) (uses last window, rolls forward)

    ohlcv must contain columns: open, high, low, close, volume
    Index should be time-ordered (ascending); if not, we sort it.
    """
    req = {"open", "high", "low", "close", "volume"}
    if not req.issubset(set(c.lower() for c in ohlcv.columns)):
        raise ValueError("ohlcv must contain columns: open, high, low, close, volume")

    df = ohlcv.copy()
    # enforce ascending time
    if isinstance(df.index, pd.DatetimeIndex) and not df.index.is_monotonic_increasing:
        df = df.sort_index()
    elif not isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index(drop=True)

    feats = df[["open", "high", "low", "close", "volume"]].astype(float).values
    tgt = df["close"].astype(float).values

    N = len(df)
    if N < lookback + 10:
        raise ValueError(f"Not enough history ({N}) for lookback={lookback}")

    # 70/30 split on the raw series (before windowing)
    split_idx = int(N * 0.7)

    # Fit scalers on TRAIN ONLY to avoid leakage
    feat_scaler = MinMaxScaler()
    targ_scaler = MinMaxScaler()

    feats_train = feats[:split_idx]
    tgt_train = tgt[:split_idx].reshape(-1, 1)

    feats_train_s = feat_scaler.fit_transform(feats_train)
    tgt_train_s = targ_scaler.fit_transform(tgt_train).ravel()

    # Transform full sequences (train+val) using the fitted scalers
    feats_all_s = np.vstack([feats_train_s, feat_scaler.transform(feats[split_idx:])])
    tgt_all_s = np.hstack([tgt_train_s, targ_scaler.transform(tgt[split_idx:].reshape(-1, 1)).ravel()])

    # Build windows across the entire series, then split windows into train/val by time
    X_all, y_all = _make_windows(feats_all_s, tgt_all_s, lookback)
    # The label at position t corresponds to original index t
    # Windows start at index 'lookback', so adjust split for windows:
    split_w = max(0, split_idx - lookback)
    X_train, y_train = X_all[:split_w], y_all[:split_w]
    X_val, y_val = X_all[split_w:], y_all[split_w:]

    # Define model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, feats_all_s.shape[1])),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dense(16, activation="relu"),
        Dense(1, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse")

    # Train
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=verbose,
    )

    # Validation predictions (inverse transform)
    val_pred_s = model.predict(X_val, verbose=0).ravel()
    val_true_s = y_val

    val_pred = targ_scaler.inverse_transform(val_pred_s.reshape(-1, 1)).ravel()
    val_true = targ_scaler.inverse_transform(val_true_s.reshape(-1, 1)).ravel()

    # Metrics
    rmse = float(np.sqrt(mean_squared_error(val_true, val_pred)))
    # Protect against division-by-zero in MAPE
    nonzero = np.where(np.abs(val_true) > 1e-8, val_true, 1e-8)
    mape = float(np.mean(np.abs((val_true - val_pred) / nonzero)) * 100.0)
    r2 = float(r2_score(val_true, val_pred))

    # Prepare last window for forecasting
    last_window_feats_s = feats_all_s[-lookback:, :].copy()

    def forecast_next(n_steps: int = 30) -> np.ndarray:
        """
        Roll forward using predicted close. For other inputs (open/high/low/volume),
        we keep the last observed values as a simple heuristic.
        Returns denormalized (original scale) predictions for Close.
        """
        window = last_window_feats_s.copy()
        preds = []
        # Keep a snapshot of "static" non-close features from the last row
        # (you can replace with more advanced heuristics if desired)
        static_others = window[-1, :].copy()
        # close is column index 3 in [open,high,low,close,volume]
        CLOSE_IDX = 3

        for _ in range(n_steps):
            x = window[np.newaxis, :, :]  # (1, lookback, F)
            y_hat_s = model.predict(x, verbose=0).ravel()[0]

            # Denorm for output
            y_hat = targ_scaler.inverse_transform([[y_hat_s]]).ravel()[0]
            preds.append(y_hat)

            # Build next step input (keep others fixed, update close)
            next_row = static_others.copy()
            next_row[CLOSE_IDX] = y_hat_s  # keep in scaled space for the model input

            # Slide window
            window = np.vstack([window[1:], next_row])

        return np.array(preds, dtype=float)

    return {
        "model": model,
        "metrics": {"rmse": rmse, "mape": mape, "r2": r2},
        "forecast_next": forecast_next,
    }
