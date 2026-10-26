import argparse
import pandas as pd
from sqlalchemy.orm import Session
from analytics_api.db import SessionLocal
from analytics_api.utils import load_symbol_df
from analytics_api.indicators import compute_indicators, resample_tf
from analytics_api.signals import generate_signals

def main(symbol: str, timeframe: str):
    with SessionLocal() as db:  # type: Session
        base = load_symbol_df(db, symbol)

    tf_df = base if timeframe == "1d" else resample_tf(base, timeframe)
    ind = compute_indicators(tf_df)
    sig = generate_signals(ind)
    out = sig.tail(120)  # limit output
    out.to_csv(f"..\\data\\{symbol}_{timeframe}_indicators.csv")
    print(f"Saved ..\\data\\{symbol}_{timeframe}_indicators.csv")
    print("Last row signal.", out.iloc[-1]["signal"], "score", out.iloc[-1]["score"])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--tf", choices=["1d","1w","1m"], default="1d")
    args = p.parse_args()
    main(args.symbol, args.tf)
