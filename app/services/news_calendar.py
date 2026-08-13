"""
Economic news blackout handling.

The economic calendar is fetched once per analysis cycle instead of
once for every currency pair. This significantly reduces API traffic.

If the calendar is unavailable, the system fails OPEN so that a
third-party calendar outage does not stop signal generation.
"""

from datetime import datetime, timezone

from loguru import logger

from app.services.twelve_data import (
    twelve_data_client,
    TwelveDataRateLimitError,
)


PAIR_CURRENCIES = {
    "EUR/USD": {"EUR", "USD"},
    "GBP/USD": {"GBP", "USD"},
    "USD/JPY": {"USD", "JPY"},
    "USD/CAD": {"USD", "CAD"},
    "AUD/USD": {"AUD", "USD"},
    "NZD/USD": {"NZD", "USD"},
    "EUR/JPY": {"EUR", "JPY"},
    "GBP/JPY": {"GBP", "JPY"},
    "XAU/USD": {"USD"},
    "BTC/USD": {"USD"},
}


BLACKOUT_WINDOW_MINUTES = 30


async def get_economic_events() -> list[dict]:
    """
    Fetch the economic calendar once.

    Returns an empty list if the endpoint is unavailable.
    """
    try:
        data = await twelve_data_client._get(
            "economic_calendar",
            {"country": ""},
        )

        events = data.get("data", [])

        if not isinstance(events, list):
            return []

        logger.info(
            f"Economic calendar loaded: {len(events)} events"
        )

        return events

    except TwelveDataRateLimitError:
        logger.warning(
            "Economic calendar rate-limited by Twelve Data; "
            "continuing without blackout data."
        )
        return []

    except Exception as exc:
        logger.warning(
            f"Economic calendar unavailable, failing open: {exc}"
        )
        return []


def is_pair_in_news_blackout(
    pair: str,
    events: list[dict],
) -> bool:
    """
    Check a pair against already-fetched calendar events.

    No API request is made here.
    """
    now = datetime.now(timezone.utc)
    relevant_currencies = PAIR_CURRENCIES.get(
        pair,
        set(),
    )

    for event in events:
        try:
            if str(event.get("impact", "")).lower() != "high":
                continue

            if event.get("currency") not in relevant_currencies:
                continue

            raw_date = event["date"]
            event_time = datetime.fromisoformat(
                raw_date.replace("Z", "+00:00")
            )

            if event_time.tzinfo is None:
                event_time = event_time.replace(
                    tzinfo=timezone.utc
                )

            difference = abs(
                (event_time - now).total_seconds()
            )

            if difference <= BLACKOUT_WINDOW_MINUTES * 60:
                return True

        except (KeyError, ValueError, TypeError):
            continue

    return False


async def is_news_blackout(
    pair: str,
    events: list[dict] | None = None,
) -> bool:
    """
    Backward-compatible helper.

    If events are supplied, no API request is made.
    If not supplied, it fetches the calendar once.
    """
    if events is None:
        events = await get_economic_events()

    return is_pair_in_news_blackout(pair, events)
