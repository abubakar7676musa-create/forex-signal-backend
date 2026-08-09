"""
Firebase Authentication integration.

Two distinct Firebase surfaces are used here, and it's important to keep them
straight:

1. Firebase Admin SDK (server-side, uses the service account from
   FIREBASE_CREDENTIALS_PATH): creates users and verifies ID tokens. This is
   a trusted, secret credential — never expose it to the Flutter app.

2. Firebase Identity Toolkit REST API (uses FIREBASE_WEB_API_KEY, the
   project's public *Web API key*): this is the only way to validate an
   email/password pair server-side, because the Admin SDK deliberately does
   NOT support password verification — that's a client-SDK-only operation.
   The Web API key is not a secret in the same sense as the service account
   (it's the same key embedded in any Firebase web/mobile client config),
   but we still keep it in an environment variable rather than hardcoding it.
"""
import httpx
from fastapi import HTTPException, status
from firebase_admin import auth as firebase_auth
from loguru import logger

from app.config import settings
from app.core.firebase import init_firebase

IDENTITY_TOOLKIT_BASE = "https://identitytoolkit.googleapis.com/v1"
SECURE_TOKEN_BASE = "https://securetoken.googleapis.com/v1"


def _ensure_initialized():
    if init_firebase() is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Firebase Authentication is not configured on the server.",
        )


def create_firebase_user(email: str, password: str, full_name: str) -> firebase_auth.UserRecord:
    """Creates a new user directly in Firebase Authentication."""
    _ensure_initialized()
    try:
        return firebase_auth.create_user(
            email=email,
            password=password,
            display_name=full_name,
            email_verified=False,
        )
    except firebase_auth.EmailAlreadyExistsError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")
    except ValueError as e:
        # Raised by the SDK for malformed input, e.g. password under 6 chars
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Firebase user creation failed: {e}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Failed to create account")


def verify_id_token(id_token: str) -> dict:
    """
    Verifies a Firebase ID token sent by the Flutter app (Authorization: Bearer <token>).
    Returns the decoded claims (includes 'uid', 'email', etc.) or raises 401.
    """
    _ensure_initialized()
    try:
        return firebase_auth.verify_id_token(id_token, check_revoked=True)
    except firebase_auth.RevokedIdTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has been revoked")
    except firebase_auth.ExpiredIdTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")
    except firebase_auth.InvalidIdTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token")
    except Exception as e:
        logger.warning(f"ID token verification failed: {e}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not verify authentication token")


async def sign_in_with_password(email: str, password: str) -> dict:
    """
    Validates email/password against Firebase Authentication via the Identity
    Toolkit REST API and returns {id_token, refresh_token, uid, email, expires_in}.
    """
    if not settings.FIREBASE_WEB_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="FIREBASE_WEB_API_KEY is not configured on the server.",
        )

    url = f"{IDENTITY_TOOLKIT_BASE}/accounts:signInWithPassword"
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            url,
            params={"key": settings.FIREBASE_WEB_API_KEY},
            json={"email": email, "password": password, "returnSecureToken": True},
        )

    if resp.status_code != 200:
        error_message = resp.json().get("error", {}).get("message", "INVALID_LOGIN_CREDENTIALS")
        logger.info(f"Firebase login rejected for {email}: {error_message}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")

    data = resp.json()
    return {
        "id_token": data["idToken"],
        "refresh_token": data["refreshToken"],
        "uid": data["localId"],
        "email": data["email"],
        "expires_in": int(data["expiresIn"]),
    }


async def refresh_id_token(refresh_token: str) -> dict:
    """Exchanges a Firebase refresh token for a new ID token via the securetoken endpoint."""
    if not settings.FIREBASE_WEB_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="FIREBASE_WEB_API_KEY is not configured on the server.",
        )

    url = f"{SECURE_TOKEN_BASE}/token"
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            url,
            params={"key": settings.FIREBASE_WEB_API_KEY},
            data={"grant_type": "refresh_token", "refresh_token": refresh_token},
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token")

    data = resp.json()
    return {
        "id_token": data["id_token"],
        "refresh_token": data["refresh_token"],
        "uid": data["user_id"],
        "expires_in": int(data["expires_in"]),
    }


def delete_firebase_user(uid: str) -> None:
    _ensure_initialized()
    try:
        firebase_auth.delete_user(uid)
    except Exception as e:
        logger.warning(f"Failed to delete Firebase user {uid}: {e}")
