"""
Checks whether a pair is currently inside a high-impact economic news blackout window.
Uses Twelve Data's economic calendar endpoint (available on paid Twelve Data plans).
If the calendar call fails or the plan doesn't include it, this fails OPEN (allows
signal generation) but logs a warning — trading-blocking behavior should never hinge
on a single third-party endpoint being reachable.
"""
from datetime import datetime, timedelta

from loguru import logger

from app.services.twelve_data import twelve_data_client

# Currency exposure per pair, used to match news events to relevant pairs
PAIR_CURRENCIES = {
    "EUR/USD": {"EUR", "USD"}, "GBP/USD": {"GBP", "USD"}, "USD/JPY": {"USD", "JPY"},
    "USD/CAD": {"USD", "CAD"}, "AUD/USD": {"AUD", "USD"}, "NZD/USD": {"NZD", "USD"},
    "EUR/JPY": {"EUR", "JPY"}, "GBP/JPY": {"GBP", "JPY"}, "XAU/USD": {"USD"}, "BTC/USD": {"USD"},
}

BLACKOUT_WINDOW_MINUTES = 30


async def is_news_blackout(pair: str) -> bool:
    try:
        data = await twelve_data_client._get("economic_calendar", {
            "country": "",
        })
        events = data.get("data", []) if isinstance(data, dict) else []
    except Exception as e:
        logger.warning(f"Economic calendar unavailable, failing open: {e}")
        return False

    now = datetime.utcnow()
    relevant_currencies = PAIR_CURRENCIES.get(pair, set())

    for event in events:
        try:
            if event.get("impact", "").lower() != "high":
                continue
            if event.get("currency") not in relevant_currencies:
                continue
            event_time = datetime.fromisoformat(event["date"])
            if abs((event_time - now).total_seconds()) <= BLACKOUT_WINDOW_MINUTES * 60:
                return True
        except (KeyError, ValueError):
            continue

    return False
