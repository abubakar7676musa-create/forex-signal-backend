"""
Economic news blackout handling.

IMPORTANT:
Twelve Data does not provide the economic-calendar endpoint
used by the previous implementation.

Therefore this module does NOT make any Twelve Data API request.

The signal engine remains fail-open:
- If external news events are supplied, blackout logic is used.
- If no events are available, trading continues normally.
"""

from datetime import datetime, timezone
from typing import Any

from loguru import logger


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


async def get_economic_events() -> list[dict[str, Any]]:
    """
    Return economic events for the current analysis cycle.

    Twelve Data does not provide the macroeconomic calendar endpoint
    previously used by this application.

    We intentionally return an empty list instead of making an invalid
    API request that would:
      1. produce HTTP 404;
      2. consume a Twelve Data API credit;
      3. add unnecessary latency to every analysis cycle.

    A separate economic-calendar provider can be connected here later
    without changing the scheduler or signal engine interface.
    """

    logger.debug(
        "Economic calendar provider disabled. "
        "No Twelve Data economic_calendar request will be made."
    )

    return []


def is_news_blackout(
    pair: str,
    events: list[dict[str, Any]],
) -> bool:
    """
    Check whether a pair is inside the high-impact news blackout window.

    IMPORTANT:
    This function performs NO API request.
    """

    if not events:
        return False

    now = datetime.now(timezone.utc)

    relevant_currencies = PAIR_CURRENCIES.get(
        pair,
        set(),
    )

    if not relevant_currencies:
        return False

    for event in events:
        try:
            impact = str(
                event.get("impact", "")
            ).lower()

            if impact != "high":
                continue

            currency = str(
                event.get("currency", "")
            ).upper()

            if currency not in relevant_currencies:
                continue

            date_value = event.get("date")

            if not date_value:
                continue

            event_time = datetime.fromisoformat(
                str(date_value).replace(
                    "Z",
                    "+00:00",
                )
            )

            # Normalize event time to UTC.
            if event_time.tzinfo is None:
                event_time = event_time.replace(
                    tzinfo=timezone.utc
                )
            else:
                event_time = event_time.astimezone(
                    timezone.utc
                )

            difference_seconds = abs(
                (
                    event_time - now
                ).total_seconds()
            )

            if (
                difference_seconds
                <= BLACKOUT_WINDOW_MINUTES * 60
            ):
                logger.info(
                    f"[{pair}] High-impact news blackout: "
                    f"{currency} event at {event_time.isoformat()}"
                )

                return True

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

    return False
