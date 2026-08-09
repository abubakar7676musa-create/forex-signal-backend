import json
import asyncio
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger

from app.config import settings, SUPPORTED_PAIRS
from app.database import SessionLocal
from app.models.signal import Signal, SignalDirection
from app.models.notification import Notification
from app.services.twelve_data import twelve_data_client
from app.services.news_calendar import is_news_blackout
from app.services.fcm import broadcast_new_signal
from app.ai_engine.signal_generator import generate_signal

scheduler = AsyncIOScheduler()


async def analyze_pair(pair: str):
    db = SessionLocal()
    try:
        blackout = await is_news_blackout(pair)
        df = await twelve_data_client.get_time_series(pair, interval="1h", outputsize=300)
        result = generate_signal(pair, df, is_news_blackout=blackout)

        if not result.published:
            logger.info(f"[{pair}] No signal published ({result.reject_reason}): {result.explanation}")
            return

        signal = Signal(
            pair=result.pair,
            direction=SignalDirection.buy if result.direction == "BUY" else SignalDirection.sell,
            entry_price=result.entry,
            stop_loss=result.stop_loss,
            take_profit_1=result.take_profit_1,
            take_profit_2=result.take_profit_2,
            risk_reward_ratio=result.risk_reward,
            confidence_score=result.confidence,
            timeframe="1h",
            explanation=result.explanation,
            confirmations=json.dumps(result.confirmations),
            is_published=True,
        )
        db.add(signal)
        db.commit()
        db.refresh(signal)

        notification = Notification(
            title=f"New {result.direction} Signal: {result.pair}",
            body=f"AI confidence {result.confidence}% — RR {result.risk_reward}:1",
            signal_id=signal.id,
            is_broadcast=True,
        )
        db.add(notification)
        db.commit()

        broadcast_new_signal(result.pair, result.direction, result.confidence, str(signal.id))
        logger.success(f"[{pair}] Published {result.direction} signal @ {result.confidence}% confidence")

    except Exception as e:
        logger.exception(f"[{pair}] Signal analysis failed: {e}")
    finally:
        db.close()


async def analyze_all_pairs():
    logger.info(f"Starting signal analysis cycle for {len(SUPPORTED_PAIRS)} pairs...")
    # Stagger requests slightly to respect Twelve Data rate limits on lower-tier plans
    for pair in SUPPORTED_PAIRS:
        await analyze_pair(pair)
        await asyncio.sleep(1.5)
    logger.info("Signal analysis cycle complete.")


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
    logger.info(f"Scheduler started: analyzing all pairs every {settings.SIGNAL_POLL_INTERVAL_SECONDS}s")
