import json
import asyncio
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger

from app.config import settings, SUPPORTED_PAIRS
from app.database import SessionLocal
from app.models.signal import Signal, SignalDirection
from app.models.notification import Notification
from app.services.twelve_data import (
    twelve_data_client,
    TwelveDataRateLimitError,
)
from app.services.news_calendar import (
    get_economic_events,
    is_pair_in_news_blackout,
)
from app.services.fcm import broadcast_new_signal
from app.ai_engine.signal_generator import generate_signal

scheduler = AsyncIOScheduler()


async def analyze_pair(
    pair: str,
    news_events: list[dict],
):
    db = SessionLocal()

    try:
        # Check already-fetched news events.
        # This does NOT make another API request.
        blackout = is_pair_in_news_blackout(
            pair,
            news_events,
        )

        df = await twelve_data_client.get_time_series(
            pair,
            interval="1h",
            outputsize=300,
        )

        result = generate_signal(
            pair,
            df,
            is_news_blackout=blackout,
        )

        if not result.published:
            logger.info(
                f"[{pair}] No signal published "
                f"({result.reject_reason}): "
                f"{result.explanation}"
            )
            return

        signal = Signal(
            pair=result.pair,
            direction=(
                SignalDirection.buy
                if result.direction == "BUY"
                else SignalDirection.sell
            ),
            entry_price=result.entry,
            stop_loss=result.stop_loss,
            take_profit_1=result.take_profit_1,
            take_profit_2=result.take_profit_2,
            risk_reward_ratio=result.risk_reward,
            confidence_score=result.confidence,
            timeframe="1h",
            explanation=result.explanation,
            confirmations=json.dumps(
                result.confirmations
            ),
            is_published=True,
        )

        db.add(signal)
        db.commit()
        db.refresh(signal)

        notification = Notification(
            title=(
                f"New {result.direction} Signal: "
                f"{result.pair}"
            ),
            body=(
                f"AI confidence {result.confidence}% "
                f"— RR {result.risk_reward}:1"
            ),
            signal_id=signal.id,
            is_broadcast=True,
        )

        db.add(notification)
        db.commit()

        broadcast_new_signal(
            result.pair,
            result.direction,
            result.confidence,
            str(signal.id),
        )

        logger.success(
            f"[{pair}] Published "
            f"{result.direction} signal @ "
            f"{result.confidence}% confidence"
        )

    except TwelveDataRateLimitError:
        logger.warning(
            f"[{pair}] Twelve Data rate limit reached. "
            "Skipping this pair for this cycle."
        )

    except Exception as e:
        logger.exception(
            f"[{pair}] Signal analysis failed: {e}"
        )

    finally:
        db.close()


async def analyze_all_pairs():
    logger.info(
        f"Starting signal analysis cycle for "
        f"{len(SUPPORTED_PAIRS)} pairs..."
    )

    # Fetch the economic calendar ONCE per cycle.
    # Previously this was called once for every pair.
    news_events = await get_economic_events()

    for index, pair in enumerate(SUPPORTED_PAIRS):
        await analyze_pair(
            pair,
            news_events,
        )

        # Give Twelve Data time between requests.
        # This helps prevent request bursts on lower-tier plans.
        if index < len(SUPPORTED_PAIRS) - 1:
            await asyncio.sleep(10)

    logger.info(
        "Signal analysis cycle complete."
    )


def start_scheduler():
    scheduler.add_job(
        analyze_all_pairs,
        "interval",
        seconds=settings.SIGNAL_POLL_INTERVAL_SECONDS,
        id="signal_analysis_cycle",
        next_run_time=datetime.utcnow(),
        max_instances=1,
        coalesce=True,
    )

    scheduler.start()

    logger.info(
        "Scheduler started: analyzing all pairs "
        f"every {settings.SIGNAL_POLL_INTERVAL_SECONDS}s"
    )
