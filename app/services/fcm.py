"""
Firebase Cloud Messaging wrapper for sending push notifications to Android devices
whenever a new AI signal is published, or when the admin broadcasts a message.
Uses the same shared Firebase Admin SDK app as Firebase Authentication (app.core.firebase).
"""
from loguru import logger

from app.core.firebase import init_firebase


def send_to_token(token: str, title: str, body: str, data: dict | None = None) -> bool:
    if init_firebase() is None:
        return False

    from firebase_admin import messaging

    message = messaging.Message(
        notification=messaging.Notification(title=title, body=body),
        data={k: str(v) for k, v in (data or {}).items()},
        token=token,
        android=messaging.AndroidConfig(priority="high"),
    )
    try:
        messaging.send(message)
        return True
    except Exception as e:
        logger.error(f"FCM send failed for token {token[:12]}...: {e}")
        return False


def send_to_topic(topic: str, title: str, body: str, data: dict | None = None) -> bool:
    if init_firebase() is None:
        return False

    from firebase_admin import messaging

    message = messaging.Message(
        notification=messaging.Notification(title=title, body=body),
        data={k: str(v) for k, v in (data or {}).items()},
        topic=topic,
    )
    try:
        messaging.send(message)
        return True
    except Exception as e:
        logger.error(f"FCM topic send failed for '{topic}': {e}")
        return False


def broadcast_new_signal(pair: str, direction: str, confidence: int, signal_id: str):
    """All devices subscribe to the 'signals' topic on app startup (see Flutter app)."""
    send_to_topic(
        topic="signals",
        title=f"New {direction} Signal: {pair}",
        body=f"AI confidence {confidence}% — tap to view entry, SL and TP.",
        data={"type": "new_signal", "signal_id": signal_id, "pair": pair},
    )
