import uuid
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel


class SignalOut(BaseModel):
    id: uuid.UUID
    pair: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_reward_ratio: float
    confidence_score: int
    timeframe: str
    status: str
    explanation: Optional[str]
    confirmations: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class SignalListResponse(BaseModel):
    total: int
    items: List[SignalOut]


class PriceOut(BaseModel):
    pair: str
    price: float
    change_percent: Optional[float] = None
    timestamp: datetime
