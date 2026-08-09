import uuid
import enum
from datetime import datetime

from sqlalchemy import Column, String, Float, DateTime, Enum, Integer, Text, Boolean
from sqlalchemy.dialects.postgresql import UUID

from app.database import Base


class SignalDirection(str, enum.Enum):
    buy = "BUY"
    sell = "SELL"


class SignalStatus(str, enum.Enum):
    active = "ACTIVE"
    hit_tp1 = "HIT_TP1"
    hit_tp2 = "HIT_TP2"
    hit_sl = "HIT_SL"
    expired = "EXPIRED"
    cancelled = "CANCELLED"


class Signal(Base):
    __tablename__ = "signals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pair = Column(String(20), nullable=False, index=True)
    direction = Column(Enum(SignalDirection), nullable=False)
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    take_profit_1 = Column(Float, nullable=False)
    take_profit_2 = Column(Float, nullable=False)
    risk_reward_ratio = Column(Float, nullable=False)
    confidence_score = Column(Integer, nullable=False)
    timeframe = Column(String(10), default="1h")
    status = Column(Enum(SignalStatus), default=SignalStatus.active, index=True)
    explanation = Column(Text, nullable=True)
    confirmations = Column(Text, nullable=True)  # JSON-encoded list of confluences that fired
    is_published = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    closed_at = Column(DateTime, nullable=True)
