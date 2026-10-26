from __future__ import annotations

from datetime import date
from pathlib import Path

from sqlalchemy import Date, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

# services/ta_service/app/db.py
# app/ is inside services/ta_service/, so parent.parent is services/ta_service
SERVICE_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = (SERVICE_ROOT / "cryptos.db").resolve()

if not DB_PATH.exists():
    raise FileNotFoundError(f"cryptos.db not found at: {DB_PATH}")

DATABASE_URL = f"sqlite:///{DB_PATH.as_posix()}"

engine = create_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


class Candle(Base):
    __tablename__ = "candles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    base_asset: Mapped[str] = mapped_column(String)
    quote_asset: Mapped[str] = mapped_column(String)
    date: Mapped[date] = mapped_column(Date, index=True)

    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)
