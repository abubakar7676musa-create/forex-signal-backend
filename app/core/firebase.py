"""
Single shared Firebase Admin SDK app instance. Both Firebase Authentication
(user creation, ID token verification) and Firebase Cloud Messaging (push
notifications) initialize from this one place so the service account is
only ever loaded once per process.
"""
import os
import firebase_admin
from firebase_admin import credentials
from loguru import logger

from app.config import settings

_firebase_app: firebase_admin.App | None = None


def init_firebase() -> firebase_admin.App | None:
    """
    Initializes the Firebase Admin SDK from the service account JSON file
    pointed to by FIREBASE_CREDENTIALS_PATH. Safe to call multiple times —
    returns the already-initialized app after the first call.
    Returns None (and logs a warning) if credentials aren't configured, so
    the rest of the app can fail gracefully instead of crashing on startup.
    """
    global _firebase_app
    if _firebase_app is not None:
        return _firebase_app

    if not settings.FIREBASE_CREDENTIALS_PATH or not os.path.exists(settings.FIREBASE_CREDENTIALS_PATH):
        logger.warning(
            "FIREBASE_CREDENTIALS_PATH not set or file not found — "
            "Firebase Authentication and push notifications are disabled."
        )
        return None

    cred = credentials.Certificate(settings.FIREBASE_CREDENTIALS_PATH)
    _firebase_app = firebase_admin.initialize_app(cred)
    logger.info("Firebase Admin SDK initialized.")
    return _firebase_app


def get_firebase_app() -> firebase_admin.App | None:
    return _firebase_app if _firebase_app is not None else init_firebase()
