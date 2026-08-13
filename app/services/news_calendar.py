"""
Economic news blackout handling.

The economic calendar is fetched ONCE per analysis cycle,
then reused for every pair.

This prevents unnecessary Twelve Data API calls.
"""

from datetime import datetime
from typing import Any

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


async def get_economic_events() -> list[dict[str, Any]]:
    """
    Fetch economic calendar once.

    Failure is fail-open because news data must never stop
    the entire signal engine.
    """

    try:
        data = await twelve_data_client._get(
            "economic_calendar",
            {
                "country": "",
            },
        )

        events = (
            data.get("data", [])
            if isinstance(data, dict)
            else []
        )

        if not isinstance(events, list):
            logger.warning(
                "Economic calendar returned unexpected format."
            )
            return []

        logger.debug(
            f"Economic calendar loaded: "
            f"{len(events)} events."
        )

        return events

    except TwelveDataRateLimitError:
        logger.warning(
            "Economic calendar skipped because "
            "Twelve Data quota is exhausted."
        )
        return []

    except Exception as exc:
        logger.warning(
            "Economic calendar unavailable, "
            f"failing open: {exc}"
        )
        return []


def is_news_blackout(
    pair: str,
    events: list[dict[str, Any]],
) -> bool:
    """
    Check whether a pair is inside the blackout window.

    IMPORTANT:
    This function performs NO API request.
    """

    now = datetime.utcnow()

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

            currency = event.get("currency")

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

            # Normalize timezone-aware values to naive UTC.
            if event_time.tzinfo is not None:
                from datetime import timezone

                event_time = (
                    event_time
                    .astimezone(timezone.utc)
                    .replace(tzinfo=None)
                )

            difference_seconds = abs(
                (event_time - now).total_seconds()
            )

            if (
                difference_seconds
                <= BLACKOUT_WINDOW_MINUTES * 60
            ):
                logger.info(
                    f"[{pair}] High-impact news blackout: "
                    f"{currency} event at {event_time}"
                )

                return True

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

    return False
