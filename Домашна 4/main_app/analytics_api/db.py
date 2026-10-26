# analytics_api/db.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

from sqlalchemy import Date, Float, Index, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


# ------------------------------------------------------------
# Database location. always resolves to analytics_api/data/cryptos.db
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "cryptos.db"
DATABASE_URL = f"sqlite:///{DB_PATH.as_posix()}"

engine = create_engine(DATABASE_URL, future=True, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


# ------------------------------------------------------------
# Declarative base
# ------------------------------------------------------------
class Base(DeclarativeBase):
    pass


# ------------------------------------------------------------
# ORM model that matches your existing SQLite table schema
# (includes the extra optional columns present in your DB)
# ------------------------------------------------------------
class Candle(Base):
    __tablename__ = "candles"
    __table_args__ = (
        Index("idx_candles_symbol_date", "symbol", "date", unique=True),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    exchange: Mapped[str] = mapped_column(String(50))
    base_asset: Mapped[str] = mapped_column(String(50))
    quote_asset: Mapped[str] = mapped_column(String(50))
    symbol: Mapped[str] = mapped_column(String(50), index=True)
    date: Mapped[Date] = mapped_column(Date, index=True)

    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)
    quote_volume: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Extra columns found in your table. keep them optional
    last_price_24h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    volume_24h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    high_24h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    low_24h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    liquidity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
