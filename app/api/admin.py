import uuid
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from app.database import get_db
from app.models.user import User
from app.models.signal import Signal
from app.models.notification import Notification
from app.schemas.user import UserOut
from app.schemas.signal import SignalOut
from app.core.deps import get_current_admin
from app.services.fcm import send_to_topic
from app.config import settings

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


# ---------- API Key status ----------
# For security, API keys are managed via server environment variables / secrets manager,
# never through a database or API write path. This endpoint only reports masked status
# so admins can verify configuration without ever exposing the real key value.

def _mask(key: str) -> str:
    if not key or len(key) < 8:
        return "not configured"
    return f"{key[:4]}{'*' * (len(key) - 8)}{key[-4:]}"


@router.get("/api-keys/status")
def api_key_status(admin: User = Depends(get_current_admin)):
    return {
        "twelve_data_api_key": _mask(settings.TWELVE_DATA_API_KEY),
        "firebase_configured": bool(settings.FIREBASE_CREDENTIALS_PATH),
        "note": "API keys are managed via server environment variables / secrets manager only.",
    }


# ---------- Users ----------

@router.get("/users", response_model=list[UserOut])
def list_users(
    limit: int = Query(100, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    return db.query(User).order_by(desc(User.created_at)).offset(offset).limit(limit).all()


@router.patch("/users/{user_id}/toggle-active", response_model=UserOut)
def toggle_user_active(user_id: uuid.UUID, db: Session = Depends(get_db), admin: User = Depends(get_current_admin)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.is_active = not user.is_active
    db.commit()
    db.refresh(user)
    return user


# ---------- Signals ----------

@router.get("/signals", response_model=list[SignalOut])
def list_all_signals(
    limit: int = Query(100, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    return db.query(Signal).order_by(desc(Signal.created_at)).offset(offset).limit(limit).all()


@router.delete("/signals/{signal_id}")
def delete_signal(signal_id: uuid.UUID, db: Session = Depends(get_db), admin: User = Depends(get_current_admin)):
    signal = db.query(Signal).filter(Signal.id == signal_id).first()
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")
    db.delete(signal)
    db.commit()
    return {"status": "deleted"}


# ---------- Notifications / Broadcast ----------

class BroadcastRequest(BaseModel):
    title: str
    body: str


@router.post("/notifications/broadcast")
def broadcast_notification(
    payload: BroadcastRequest,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    notification = Notification(title=payload.title, body=payload.body, is_broadcast=True)
    db.add(notification)
    db.commit()
    send_to_topic(topic="signals", title=payload.title, body=payload.body, data={"type": "broadcast"})
    return {"status": "broadcast sent"}


# ---------- Analytics ----------

@router.get("/analytics/overview")
def analytics_overview(db: Session = Depends(get_db), admin: User = Depends(get_current_admin)):
    total_users = db.query(func.count(User.id)).scalar()
    active_users = db.query(func.count(User.id)).filter(User.is_active == True).scalar()  # noqa: E712
    total_signals = db.query(func.count(Signal.id)).scalar()

    last_7d = datetime.utcnow() - timedelta(days=7)
    signals_7d = db.query(func.count(Signal.id)).filter(Signal.created_at >= last_7d).scalar()
    avg_confidence = db.query(func.avg(Signal.confidence_score)).scalar() or 0

    by_pair = (
        db.query(Signal.pair, func.count(Signal.id))
        .group_by(Signal.pair)
        .order_by(desc(func.count(Signal.id)))
        .limit(10)
        .all()
    )

    return {
        "total_users": total_users,
        "active_users": active_users,
        "total_signals": total_signals,
        "signals_last_7_days": signals_7d,
        "average_confidence": round(float(avg_confidence), 1),
        "signals_by_pair": [{"pair": p, "count": c} for p, c in by_pair],
    }
