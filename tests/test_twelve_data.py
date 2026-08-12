import httpx
import pytest

from app.services.twelve_data import _is_retryable, TwelveDataRateLimitError


def _http_status_error(status_code):
    request = httpx.Request("GET", "https://api.twelvedata.com/quote")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("error", request=request, response=response)


def test_429_is_retryable():
    assert _is_retryable(_http_status_error(429)) is True


def test_5xx_are_retryable():
    for code in (500, 502, 503, 504):
        assert _is_retryable(_http_status_error(code)) is True


def test_4xx_client_errors_are_not_retryable():
    # Retrying a bad API key, bad symbol, or malformed request wastes quota —
    # it will fail identically every time.
    for code in (400, 401, 403, 404):
        assert _is_retryable(_http_status_error(code)) is False


def test_rate_limit_error_is_retryable():
    assert _is_retryable(TwelveDataRateLimitError("rate limited")) is True


def test_network_errors_are_retryable():
    request = httpx.Request("GET", "https://api.twelvedata.com/quote")
    assert _is_retryable(httpx.ConnectError("boom", request=request)) is True
    assert _is_retryable(httpx.ReadTimeout("boom", request=request)) is True


def test_unrelated_exceptions_are_not_retryable():
    assert _is_retryable(ValueError("bad data")) is False


def test_rate_limit_error_carries_retry_after():
    err = TwelveDataRateLimitError("rate limited", retry_after=30.0)
    assert err.retry_after == 30.0


def test_rate_limit_error_retry_after_optional():
    err = TwelveDataRateLimitError("rate limited")
    assert err.retry_after is None
