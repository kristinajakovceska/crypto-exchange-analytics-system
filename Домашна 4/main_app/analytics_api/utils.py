import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session
from .db import Candle

def load_symbol_df(db: Session, symbol: str) -> pd.DataFrame:
    q = (
        select(Candle)
        .where(Candle.symbol == symbol)
        .order_by(Candle.date.asc())
    )
    rows = db.execute(q).scalars().all()
    if not rows:
        raise ValueError(f"Symbol '{symbol}' not found in database")

    df = pd.DataFrame([{
        "date": r.date,
        "open": r.open,
        "high": r.high,
        "low": r.low,
        "close": r.close,
        "volume": r.volume
    } for r in rows])
    df.set_index("date", inplace=True)
    return df
