from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.database import get_db
from app.models.notification import Notification
from app.models.user import User
from app.core.deps import get_current_user

router = APIRouter(prefix="/api/v1/notifications", tags=["notifications"])


@router.get("")
def list_notifications(
    limit: int = Query(50, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    items = (
        db.query(Notification)
        .filter(
            (Notification.is_broadcast == True) |  # noqa: E712
            (Notification.target_user_id == current_user.id)
        )
        .order_by(desc(Notification.created_at))
        .limit(limit)
        .all()
    )
    return {
        "total": len(items),
        "items": [
            {
                "id": str(n.id),
                "title": n.title,
                "body": n.body,
                "signal_id": str(n.signal_id) if n.signal_id else None,
                "created_at": n.created_at.isoformat(),
            }
            for n in items
        ],
    }
