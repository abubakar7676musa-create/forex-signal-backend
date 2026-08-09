import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.database import get_db
from app.config import SUPPORTED_PAIRS
from app.models.signal import Signal
from app.models.user import User
from app.schemas.signal import SignalOut, SignalListResponse
from app.core.deps import get_current_user

router = APIRouter(prefix="/api/v1/signals", tags=["signals"])


@router.get("", response_model=SignalListResponse)
def list_signals(
    pair: Optional[str] = Query(None, description="Filter by currency pair, e.g. EUR/USD"),
    direction: Optional[str] = Query(None, description="BUY or SELL"),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = db.query(Signal).filter(Signal.is_published == True)  # noqa: E712

    if pair:
        if pair not in SUPPORTED_PAIRS:
            raise HTTPException(status_code=400, detail=f"Unsupported pair: {pair}")
        query = query.filter(Signal.pair == pair)
    if direction:
        query = query.filter(Signal.direction == direction.upper())

    total = query.count()
    items = query.order_by(desc(Signal.created_at)).offset(offset).limit(limit).all()
    return SignalListResponse(total=total, items=items)


@router.get("/latest", response_model=SignalListResponse)
def latest_signals(
    limit: int = Query(10, le=50),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """The most recent published signal per pair — powers the dashboard view."""
    items = (
        db.query(Signal)
        .filter(Signal.is_published == True)  # noqa: E712
        .order_by(desc(Signal.created_at))
        .limit(limit)
        .all()
    )
    return SignalListResponse(total=len(items), items=items)


@router.get("/favorites", response_model=SignalListResponse)
def favorite_pair_signals(
    limit: int = Query(50, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not current_user.favorite_pairs:
        return SignalListResponse(total=0, items=[])
    items = (
        db.query(Signal)
        .filter(Signal.is_published == True, Signal.pair.in_(current_user.favorite_pairs))  # noqa: E712
        .order_by(desc(Signal.created_at))
        .limit(limit)
        .all()
    )
    return SignalListResponse(total=len(items), items=items)


@router.get("/{signal_id}", response_model=SignalOut)
def get_signal(signal_id: uuid.UUID, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    signal = db.query(Signal).filter(Signal.id == signal_id).first()
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")
    return signal
