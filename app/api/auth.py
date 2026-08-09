from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User, UserRole
from app.schemas.auth import RegisterRequest, LoginRequest, RefreshRequest, AuthResponse
from app.schemas.user import UserOut
from app.core.deps import get_current_user
from app.services.firebase_auth import (
    create_firebase_user, sign_in_with_password, refresh_id_token, delete_firebase_user,
)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    """
    1. Creates the user in Firebase Authentication.
    2. Saves the corresponding profile row in PostgreSQL, linked by firebase_uid.
    3. Immediately signs the new user in so the app gets usable tokens without
       a second round trip.
    """
    existing = db.query(User).filter(User.email == payload.email.lower()).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    firebase_user = create_firebase_user(
        email=payload.email.lower(),
        password=payload.password,
        full_name=payload.full_name,
    )

    try:
        user = User(
            firebase_uid=firebase_user.uid,
            email=payload.email.lower(),
            full_name=payload.full_name,
            role=UserRole.user,
            favorite_pairs=[],
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    except Exception:
        # Roll back the Firebase-side account if we couldn't persist the profile,
        # so we never end up with an orphaned Firebase user with no Postgres row.
        db.rollback()
        delete_firebase_user(firebase_user.uid)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create profile")

    tokens = await sign_in_with_password(payload.email.lower(), payload.password)

    return AuthResponse(
        user=UserOut.model_validate(user),
        id_token=tokens["id_token"],
        refresh_token=tokens["refresh_token"],
        expires_in=tokens["expires_in"],
    )


@router.post("/login", response_model=AuthResponse)
async def login(payload: LoginRequest, db: Session = Depends(get_db)):
    """Validates email/password against Firebase Authentication, then returns
    the matching PostgreSQL profile plus fresh Firebase tokens."""
    tokens = await sign_in_with_password(payload.email.lower(), payload.password)

    user = db.query(User).filter(User.firebase_uid == tokens["uid"]).first()
    if not user:
        # Firebase account exists (e.g. created outside this API) but has no
        # local profile yet — create one on the fly so login still succeeds.
        user = User(
            firebase_uid=tokens["uid"],
            email=tokens["email"].lower(),
            full_name=None,
            role=UserRole.user,
            favorite_pairs=[],
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is inactive")

    return AuthResponse(
        user=UserOut.model_validate(user),
        id_token=tokens["id_token"],
        refresh_token=tokens["refresh_token"],
        expires_in=tokens["expires_in"],
    )


@router.post("/refresh")
async def refresh(payload: RefreshRequest):
    """Exchanges a Firebase refresh token for a new ID token."""
    tokens = await refresh_id_token(payload.refresh_token)
    return {
        "id_token": tokens["id_token"],
        "refresh_token": tokens["refresh_token"],
        "expires_in": tokens["expires_in"],
        "token_type": "bearer",
    }


@router.get("/me", response_model=UserOut)
def me(current_user: User = Depends(get_current_user)):
    return current_user
