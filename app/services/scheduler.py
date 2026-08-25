import json
import asyncio
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger

from app.config import settings, SUPPORTED_PAIRS
from app.database import SessionLocal
from app.models.signal import Signal, SignalDirection
from app.models.notification import Notification

from app.services.market_data import (
    market_data_client,
)

from app.services.twelve_data import (
    TwelveDataRateLimitError,
)

from app.services.news_calendar import (
    get_economic_events,
    is_news_blackout,
)

from app.services.fcm import broadcast_new_signal
from app.ai_engine.signal_generator import generate_signal


scheduler = AsyncIOScheduler()


# ---------------------------------------------------------
# MARKET DATA SAFETY
# ---------------------------------------------------------
#
# FCS is now the PRIMARY market-data provider.
#
# Twelve Data remains available automatically as fallback
# inside market_data_client.
#
# Therefore this value is NOT a Twelve Data credit limit.
#
# It only controls how many pairs we process in one cycle
# to keep the scheduler workload reasonable.
#
MAX_PAIRS_PER_CYCLE = 10


# Continue from the next pair on the next scheduler cycle.
_pair_cursor = 0


# Prevent duplicate cycles inside the same process.
_cycle_lock = asyncio.Lock()


# =========================================================
# ANALYZE ONE PAIR
# =========================================================

async def analyze_pair(
    pair: str,
    news_events: list[dict],
):
    db = SessionLocal()

    try:
        # -----------------------------------------------------
        # NEWS BLACKOUT
        # -----------------------------------------------------
        #
        # Economic calendar has already been fetched once
        # for the entire cycle.
        #
        blackout = is_news_blackout(
            pair,
            news_events,
        )

        # -----------------------------------------------------
        # MARKET DATA
        # -----------------------------------------------------
        #
        # PRIMARY:
        #     FCS
        #
        # FALLBACK:
        #     Twelve Data
        #
        # The market_data_client handles provider switching.
        #

        df = await market_data_client.get_time_series(
            pair,
            interval="1h",
            outputsize=300,
        )

        # -----------------------------------------------------
        # GENERATE SIGNAL
        # -----------------------------------------------------

        result = generate_signal(
            pair,
            df,
            is_news_blackout=blackout,
        )

        # -----------------------------------------------------
        # NO SIGNAL
        # -----------------------------------------------------

        if not result.published:

            logger.info(
                f"[{pair}] No signal published "
                f"({result.reject_reason}): "
                f"{result.explanation}"
            )

            return True

        # -----------------------------------------------------
        # CREATE SIGNAL
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # NOTIFICATION
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # BROADCAST
        # -----------------------------------------------------

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

        return True

    # ---------------------------------------------------------
    # TWELVE DATA RATE LIMIT
    # ---------------------------------------------------------
    #
    # This should only happen if:
    #
    # FCS failed
    # AND
    # Twelve Data fallback was also quota-limited.
    #
    # ---------------------------------------------------------

    except TwelveDataRateLimitError:

        logger.warning(
            f"[{pair}] FCS unavailable and "
            "Twelve Data fallback quota is exhausted. "
            "Stopping this cycle immediately."
        )

        return False

    # ---------------------------------------------------------
    # OTHER ERRORS
    # ---------------------------------------------------------

    except Exception as exc:

        logger.exception(
            f"[{pair}] Signal analysis failed: {exc}"
        )

        return True

    finally:

        db.close()


# =========================================================
# ANALYZE ALL PAIRS
# =========================================================

async def analyze_all_pairs():

    global _pair_cursor

    # ---------------------------------------------------------
    # PREVENT OVERLAPPING CYCLES
    # ---------------------------------------------------------

    if _cycle_lock.locked():

        logger.warning(
            "Signal analysis cycle already running. "
            "Skipping duplicate scheduler invocation."
        )

        return

    async with _cycle_lock:

        total_pairs = len(
            SUPPORTED_PAIRS
        )

        if total_pairs == 0:

            logger.warning(
                "No supported pairs configured."
            )

            return

        logger.info(
            f"Starting signal analysis cycle: "
            f"{total_pairs} total pairs, "
            f"max {MAX_PAIRS_PER_CYCLE} pairs this cycle."
        )

        # -----------------------------------------------------
        # FETCH ECONOMIC NEWS ONCE
        # -----------------------------------------------------

        try:

            news_events = (
                await get_economic_events()
            )

        except Exception as exc:

            logger.warning(
                f"Economic calendar unavailable: {exc}. "
                "Continuing without news blackout events."
            )

            news_events = []

        logger.info(
            f"Economic calendar loaded with "
            f"{len(news_events)} events."
        )

        # -----------------------------------------------------
        # ROTATING PAIR WINDOW
        # -----------------------------------------------------

        start_index = _pair_cursor

        selected_pairs = []

        for offset in range(
            min(
                MAX_PAIRS_PER_CYCLE,
                total_pairs,
            )
        ):

            index = (
                start_index + offset
            ) % total_pairs

            selected_pairs.append(
                SUPPORTED_PAIRS[index]
            )

        logger.info(
            f"Selected pairs this cycle: "
            f"{selected_pairs}"
        )

        # -----------------------------------------------------
        # ANALYZE SELECTED PAIRS
        # -----------------------------------------------------

        processed = 0

        for index, pair in enumerate(
            selected_pairs,
            start=1,
        ):

            logger.info(
                f"Analyzing pair "
                f"{index}/{len(selected_pairs)}: "
                f"{pair}"
            )

            success = await analyze_pair(
                pair,
                news_events,
            )

            processed += 1

            # -------------------------------------------------
            # STOP ONLY IF BOTH PROVIDERS ARE UNAVAILABLE
            # -------------------------------------------------

            if not success:

                logger.warning(
                    "Market-data providers unavailable "
                    "for this cycle. "
                    "Stopping remaining pairs."
                )

                break

            # -------------------------------------------------
            # SMALL SPACING
            # -------------------------------------------------
            #
            # This is NOT a Twelve Data waiting mechanism.
            #
            # It simply prevents the scheduler from hammering
            # the provider APIs continuously.
            #

            await asyncio.sleep(1.0)

        # -----------------------------------------------------
        # ADVANCE ROTATING CURSOR
        # -----------------------------------------------------

        if processed > 0:

            _pair_cursor = (
                start_index + processed
            ) % total_pairs

        logger.info(
            f"Signal analysis cycle complete. "
            f"Processed={processed}, "
            f"next_pair_index={_pair_cursor}"
        )


# =========================================================
# START SCHEDULER
# =========================================================

def start_scheduler():

    # ---------------------------------------------------------
    # AVOID DUPLICATE JOBS
    # ---------------------------------------------------------

    existing_job = scheduler.get_job(
        "signal_analysis_cycle"
    )

    if existing_job is not None:

        logger.warning(
            "Signal analysis scheduler already exists. "
            "Skipping duplicate registration."
        )

        return

    # ---------------------------------------------------------
    # REGISTER JOB
    # ---------------------------------------------------------

    scheduler.add_job(
        analyze_all_pairs,
        "interval",
        seconds=settings.SIGNAL_POLL_INTERVAL_SECONDS,
        id="signal_analysis_cycle",
        next_run_time=datetime.utcnow(),
        max_instances=1,
        coalesce=True,
        misfire_grace_time=30,
    )

    scheduler.start()

    logger.info(
        "Scheduler started: analyzing up to "
        f"{MAX_PAIRS_PER_CYCLE} pairs per cycle "
        f"every "
        f"{settings.SIGNAL_POLL_INTERVAL_SECONDS}s"
    )
