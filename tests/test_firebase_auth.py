"""
These tests mock firebase_admin/httpx so they verify OUR error-handling and
response-shaping logic without requiring live Firebase credentials or network
access. They do not prove the real Firebase project is reachable — that's
what the manual /docs smoke test in the README's testing section covers.
"""
from unittest.mock import patch, MagicMock

import pytest
from fastapi import HTTPException
from firebase_admin import auth as firebase_auth_sdk

from app.services import firebase_auth as fb_service


class FakeUserRecord:
    def __init__(self, uid):
        self.uid = uid


@patch("app.services.firebase_auth.init_firebase", return_value=MagicMock())
@patch("app.services.firebase_auth.firebase_auth.create_user")
def test_create_firebase_user_success(mock_create_user, _mock_init):
    mock_create_user.return_value = FakeUserRecord(uid="abc123")
    result = fb_service.create_firebase_user("test@example.com", "password123", "Test User")
    assert result.uid == "abc123"
    mock_create_user.assert_called_once()


@patch("app.services.firebase_auth.init_firebase", return_value=MagicMock())
@patch("app.services.firebase_auth.firebase_auth.create_user")
def test_create_firebase_user_duplicate_email_raises_409(mock_create_user, _mock_init):
    mock_create_user.side_effect = firebase_auth_sdk.EmailAlreadyExistsError("exists", None, None)
    with pytest.raises(HTTPException) as exc_info:
        fb_service.create_firebase_user("dup@example.com", "password123", "Dup User")
    assert exc_info.value.status_code == 409


@patch("app.services.firebase_auth.init_firebase", return_value=MagicMock())
@patch("app.services.firebase_auth.firebase_auth.verify_id_token")
def test_verify_id_token_success(mock_verify, _mock_init):
    mock_verify.return_value = {"uid": "abc123", "email": "test@example.com"}
    result = fb_service.verify_id_token("some_token")
    assert result["uid"] == "abc123"


@patch("app.services.firebase_auth.init_firebase", return_value=MagicMock())
@patch("app.services.firebase_auth.firebase_auth.verify_id_token")
def test_verify_id_token_expired_raises_401(mock_verify, _mock_init):
    mock_verify.side_effect = firebase_auth_sdk.ExpiredIdTokenError("expired", None)
    with pytest.raises(HTTPException) as exc_info:
        fb_service.verify_id_token("expired_token")
    assert exc_info.value.status_code == 401


def test_sign_in_missing_web_api_key_raises_503(monkeypatch):
    monkeypatch.setattr(fb_service.settings, "FIREBASE_WEB_API_KEY", "")
    import asyncio
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(fb_service.sign_in_with_password("test@example.com", "password123"))
    assert exc_info.value.status_code == 503
