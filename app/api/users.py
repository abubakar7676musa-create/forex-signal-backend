from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from firebase_admin import auth as firebase_auth
from loguru import logger

from app.database import get_db
from app.config import SUPPORTED_PAIRS
from app.models.user import User
from app.schemas.user import UserOut, UserUpdate, FcmTokenUpdate
from app.core.deps import get_current_user

router = APIRouter(prefix="/api/v1/users", tags=["users"])


@router.get("/me", response_model=UserOut)
def get_profile(current_user: User = Depends(get_current_user)):
    return current_user


@router.patch("/me", response_model=UserOut)
def update_profile(
    payload: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if payload.full_name is not None:
        current_user.full_name = payload.full_name

    if payload.favorite_pairs is not None:
        invalid = [p for p in payload.favorite_pairs if p not in SUPPORTED_PAIRS]
        if invalid:
            raise HTTPException(status_code=400, detail=f"Unsupported pairs: {invalid}")
        current_user.favorite_pairs = payload.favorite_pairs

    db.commit()
    db.refresh(current_user)
    return current_user


@router.post("/me/fcm-token")
def update_fcm_token(
    payload: FcmTokenUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    current_user.fcm_token = payload.fcm_token
    db.commit()
    return {"status": "ok"}


@router.delete("/me")
def delete_account(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    current_user.is_active = False
    db.commit()
    try:
        firebase_auth.update_user(current_user.firebase_uid, disabled=True)
    except Exception as e:
        logger.warning(f"Failed to disable Firebase account for {current_user.email}: {e}")
    return {"status": "account deactivated"}
