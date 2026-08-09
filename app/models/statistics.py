import uuid
from datetime import datetime

from sqlalchemy import Column, Integer, Float, DateTime, String
from sqlalchemy.dialects.postgresql import UUID

from app.database import Base


class DailyStatistic(Base):
    """Aggregated per-day stats, refreshed by a scheduled job for fast admin dashboard reads."""
    __tablename__ = "daily_statistics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    date = Column(DateTime, unique=True, nullable=False, index=True)
    total_signals = Column(Integer, default=0)
    buy_signals = Column(Integer, default=0)
    sell_signals = Column(Integer, default=0)
    avg_confidence = Column(Float, default=0.0)
    signals_hit_tp1 = Column(Integer, default=0)
    signals_hit_tp2 = Column(Integer, default=0)
    signals_hit_sl = Column(Integer, default=0)
    new_users = Column(Integer, default=0)
    active_users = Column(Integer, default=0)
    most_active_pair = Column(String(20), nullable=True)
