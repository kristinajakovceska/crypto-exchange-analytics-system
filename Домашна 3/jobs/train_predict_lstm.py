import argparse
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from analytics_api.db import SessionLocal
from analytics_api.utils import load_symbol_df
from analytics_api.lstm_model import train_lstm

def main(symbol: str, lookback: int, horizon: int):
    with SessionLocal() as db:  # type: Session
        df = load_symbol_df(db, symbol)

    res = train_lstm(df["close"], lookback=lookback)
    metrics = res["metrics"]
    preds = res["forecast_next"](n_steps=horizon)

    pd.Series(preds).to_csv(f"..\\data\\{symbol}_lstm_forecast.csv", index=False)
    print("RMSE.", metrics["rmse"], "MAPE.", metrics["mape"], "R2.", metrics["r2"])
    print(f"Saved ..\\data\\{symbol}_lstm_forecast.csv with {horizon} steps")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--lookback", type=int, default=30)
    p.add_argument("--horizon", type=int, default=30)
    args = p.parse_args()
    main(args.symbol, args.lookback, args.horizon)
