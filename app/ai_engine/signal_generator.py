"""
Rule-based Forex signal engine.

Combines:
- Classic technical indicators
- Candlestick patterns
- Smart Money Concepts (SMC)
- Fibonacci
- Support / resistance

The engine publishes a signal only when directional confluence,
risk/reward and confidence requirements are satisfied.

IMPORTANT:
This is a rule-based confluence engine, not a guarantee of profitable
trades. Market risk can never be eliminated.
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from app.config import settings
from app.ai_engine.indicators import (
    compute_all_indicators,
    fibonacci_levels,
    support_resistance,
)
from app.ai_engine.patterns import detect_patterns
from app.ai_engine.smart_money import full_smc_analysis


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JPY_PAIRS = {"USD/JPY", "EUR/JPY", "GBP/JPY"}

MIN_BARS = 210

ATR_SL_MULTIPLIER = 1.5
STRUCTURE_PADDING_ATR = 0.2

TP1_R_MULTIPLE = 2.0
TP2_R_MULTIPLE = 3.5

MAX_CONFIDENCE = 97
MIN_CONFIDENCE_FLOOR = 30

MIN_DIRECTION_SCORE = 5.0


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class SignalResult:
    pair: str
    direction: Optional[str]  # "BUY" | "SELL" | None
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_reward: float
    confidence: int
    explanation: str
    confirmations: list[str] = field(default_factory=list)
    published: bool = False
    reject_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _pip_size(pair: str) -> float:
    """
    Approximate pip/price step used only as a fallback when ATR is invalid.
    """
    if pair in JPY_PAIRS:
        return 0.01

    if pair == "XAU/USD":
        return 0.1

    if pair == "BTC/USD":
        return 1.0

    return 0.0001


def _safe_float(value, default: float = 0.0) -> float:
    """
    Safely convert indicator values to float.
    """
    try:
        value = float(value)

        if pd.isna(value):
            return default

        return value
    except (TypeError, ValueError):
        return default


def _empty_result(
    pair: str,
    explanation: str,
    reject_reason: str,
    confirmations: Optional[list[str]] = None,
    confidence: int = 0,
) -> SignalResult:
    """
    Consistent rejected/empty result.
    """
    return SignalResult(
        pair=pair,
        direction=None,
        entry=0.0,
        stop_loss=0.0,
        take_profit_1=0.0,
        take_profit_2=0.0,
        risk_reward=0.0,
        confidence=confidence,
        explanation=explanation,
        confirmations=confirmations or [],
        published=False,
        reject_reason=reject_reason,
    )


# ---------------------------------------------------------------------------
# Technical scoring
# ---------------------------------------------------------------------------

def _score_technical(
    df: pd.DataFrame,
) -> tuple[float, float, list[str], list[str]]:
    """
    Returns:

        bullish_points,
        bearish_points,
        bullish_reasons,
        bearish_reasons

    Keeping bullish and bearish reasons separate is important.
    A BUY explanation must not contain bearish confirmations.
    """

    bull = 0.0
    bear = 0.0

    bullish_reasons: list[str] = []
    bearish_reasons: list[str] = []

    last = df.iloc[-1]

    # ------------------------------------------------------------------
    # EMA trend
    # ------------------------------------------------------------------

    ema20 = _safe_float(last.get("ema_20"))
    ema50 = _safe_float(last.get("ema_50"))
    ema200 = _safe_float(last.get("ema_200"))

    if ema20 > ema50 > ema200:
        bull += 2.0
        bullish_reasons.append("EMA stack bullish (20>50>200)")

    elif ema20 < ema50 < ema200:
        bear += 2.0
        bearish_reasons.append("EMA stack bearish (20<50<200)")

    # ------------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------------

    rsi = _safe_float(last.get("rsi_14"))

    if rsi < 30:
        bull += 1.0
        bullish_reasons.append(f"RSI oversold ({rsi:.1f})")

    elif rsi > 70:
        bear += 1.0
        bearish_reasons.append(f"RSI overbought ({rsi:.1f})")

    elif rsi > 50:
        bull += 0.5
        bullish_reasons.append(f"RSI bullish bias ({rsi:.1f})")

    elif rsi < 50:
        bear += 0.5
        bearish_reasons.append(f"RSI bearish bias ({rsi:.1f})")

    # ------------------------------------------------------------------
    # MACD
    # ------------------------------------------------------------------

    macd = _safe_float(last.get("macd"))
    macd_signal = _safe_float(last.get("macd_signal"))
    macd_hist = _safe_float(last.get("macd_hist"))

    if macd > macd_signal and macd_hist > 0:
        bull += 1.0
        bullish_reasons.append("MACD bullish")

    elif macd < macd_signal and macd_hist < 0:
        bear += 1.0
        bearish_reasons.append("MACD bearish")

    # ------------------------------------------------------------------
    # Bollinger Bands
    # ------------------------------------------------------------------

    close = _safe_float(last.get("close"))
    bb_lower = _safe_float(last.get("bb_lower"))
    bb_upper = _safe_float(last.get("bb_upper"))

    if close <= bb_lower:
        bull += 1.0
        bullish_reasons.append("Price at/below lower Bollinger Band")

    elif close >= bb_upper:
        bear += 1.0
        bearish_reasons.append("Price at/above upper Bollinger Band")

    # ------------------------------------------------------------------
    # ADX
    # ------------------------------------------------------------------

    adx = _safe_float(last.get("adx_14"))

    if adx >= 25:
        bullish_reasons.append(f"ADX strong trend ({adx:.1f})")
        bearish_reasons.append(f"ADX strong trend ({adx:.1f})")

    elif adx >= 20:
        bullish_reasons.append(f"ADX moderate trend ({adx:.1f})")
        bearish_reasons.append(f"ADX moderate trend ({adx:.1f})")

    else:
        bullish_reasons.append(f"ADX weak trend ({adx:.1f})")
        bearish_reasons.append(f"ADX weak trend ({adx:.1f})")

    return (
        bull,
        bear,
        bullish_reasons,
        bearish_reasons,
    )


# ---------------------------------------------------------------------------
# Candlestick pattern scoring
# ---------------------------------------------------------------------------

def _score_patterns(
    df: pd.DataFrame,
) -> tuple[float, float, list[str], list[str]]:
    """
    Score bullish and bearish candlestick patterns independently.
    """

    bull = 0.0
    bear = 0.0

    bullish_reasons: list[str] = []
    bearish_reasons: list[str] = []

    patterns = detect_patterns(df)

    bullish_patterns = {
        "bullish_engulfing",
        "hammer",
    }

    bearish_patterns = {
        "bearish_engulfing",
        "shooting_star",
    }

    for pattern in patterns:

        if pattern in bullish_patterns:
            bull += 1.0
            bullish_reasons.append(
                f"Candlestick: {pattern.replace('_', ' ')}"
            )

        elif pattern in bearish_patterns:
            bear += 1.0
            bearish_reasons.append(
                f"Candlestick: {pattern.replace('_', ' ')}"
            )

    return (
        bull,
        bear,
        bullish_reasons,
        bearish_reasons,
    )


# ---------------------------------------------------------------------------
# SMC scoring
# ---------------------------------------------------------------------------

def _score_smc(
    df: pd.DataFrame,
) -> tuple[float, float, list[str], list[str]]:
    """
    Score Smart Money Concepts independently for BUY and SELL.

    The scoring is intentionally weighted:
    - BOS = strong
    - CHoCH = strong
    - Liquidity sweep = medium/strong
    - FVG = light
    - Order block = medium
    - Supply/demand = light
    """

    bull = 0.0
    bear = 0.0

    bullish_reasons: list[str] = []
    bearish_reasons: list[str] = []

    smc = full_smc_analysis(df)

    # ------------------------------------------------------------------
    # BOS
    # ------------------------------------------------------------------

    bos = smc.get("bos")

    if bos == "bullish_bos":
        bull += 2.0
        bullish_reasons.append("Break of Structure (bullish)")

    elif bos == "bearish_bos":
        bear += 2.0
        bearish_reasons.append("Break of Structure (bearish)")

    # ------------------------------------------------------------------
    # CHoCH
    # ------------------------------------------------------------------

    choch = smc.get("choch")

    if choch == "bullish_choch":
        bull += 2.0
        bullish_reasons.append("Change of Character (bullish reversal)")

    elif choch == "bearish_choch":
        bear += 2.0
        bearish_reasons.append("Change of Character (bearish reversal)")

    # ------------------------------------------------------------------
    # Liquidity sweep
    # ------------------------------------------------------------------

    liquidity_sweep = smc.get("liquidity_sweep")

    if liquidity_sweep == "bullish_sweep":
        bull += 1.5
        bullish_reasons.append(
            "Liquidity sweep to downside, then bullish rejection"
        )

    elif liquidity_sweep == "bearish_sweep":
        bear += 1.5
        bearish_reasons.append(
            "Liquidity sweep to upside, then bearish rejection"
        )

    # ------------------------------------------------------------------
    # Fair Value Gaps
    # ------------------------------------------------------------------

    fair_value_gaps = smc.get("fair_value_gaps") or []

    for gap in fair_value_gaps[-2:]:

        gap_type = gap.get("type")

        if gap_type == "bullish_fvg":
            bull += 0.5
            bullish_reasons.append("Bullish Fair Value Gap")

        elif gap_type == "bearish_fvg":
            bear += 0.5
            bearish_reasons.append("Bearish Fair Value Gap")

    # ------------------------------------------------------------------
    # Order blocks
    # ------------------------------------------------------------------

    order_blocks = smc.get("order_blocks") or []

    if order_blocks:

        latest_ob = order_blocks[-1]
        ob_type = latest_ob.get("type")

        if ob_type == "bullish_ob":
            bull += 1.0
            bullish_reasons.append("Recent bullish order block")

        elif ob_type == "bearish_ob":
            bear += 1.0
            bearish_reasons.append("Recent bearish order block")

    # ------------------------------------------------------------------
    # Demand / supply
    # ------------------------------------------------------------------

    supply_demand = smc.get("supply_demand") or {}

    demand = supply_demand.get("demand") or []
    supply = supply_demand.get("supply") or []

    if demand:
        bull += 0.5
        bullish_reasons.append("Demand zone detected")

    if supply:
        bear += 0.5
        bearish_reasons.append("Supply zone detected")

    return (
        bull,
        bear,
        bullish_reasons,
        bearish_reasons,
    )


# ---------------------------------------------------------------------------
# Fibonacci confirmation
# ---------------------------------------------------------------------------

def _score_fibonacci(
    df: pd.DataFrame,
    direction: str,
) -> tuple[float, list[str]]:
    """
    Fibonacci is used as a light confirmation rather than a major score.

    We avoid making Fibonacci alone responsible for a trade.
    """

    try:
        fib = fibonacci_levels(df)
    except Exception:
        return 0.0, []

    if not fib:
        return 0.0, []

    last_price = _safe_float(df.iloc[-1]["close"])

    levels: list[float] = []

    if isinstance(fib, dict):
        for value in fib.values():
            try:
                value = float(value)
                if not pd.isna(value):
                    levels.append(value)
            except (TypeError, ValueError):
                continue

    elif isinstance(fib, (list, tuple)):
        for value in fib:
            try:
                value = float(value)
                if not pd.isna(value):
                    levels.append(value)
            except (TypeError, ValueError):
                continue

    if not levels:
        return 0.0, []

    # A Fibonacci level near current price acts as a small confirmation.
    atr = _safe_float(df.iloc[-1].get("atr_14"))

    if atr <= 0:
        atr = _pip_size("")

    proximity = atr * 0.5

    nearby = any(abs(last_price - level) <= proximity for level in levels)

    if not nearby:
        return 0.0, []

    if direction == "BUY":
        return 0.5, ["Price near Fibonacci support/retracement level"]

    return 0.5, ["Price near Fibonacci resistance/retracement level"]


# ---------------------------------------------------------------------------
# Direction selection
# ---------------------------------------------------------------------------

def _choose_direction(
    total_bull: float,
    total_bear: float,
) -> Optional[str]:

    if (
        total_bull >= MIN_DIRECTION_SCORE
        and total_bull > total_bear
    ):
        return "BUY"

    if (
        total_bear >= MIN_DIRECTION_SCORE
        and total_bear > total_bull
    ):
        return "SELL"

    return None


# ---------------------------------------------------------------------------
# Stop-loss / take-profit
# ---------------------------------------------------------------------------

def _build_trade_levels(
    pair: str,
    direction: str,
    df: pd.DataFrame,
) -> tuple[float, float, float, float, float]:
    """
    Returns:

        entry,
        stop_loss,
        take_profit_1,
        take_profit_2,
        risk_reward
    """

    last = df.iloc[-1]

    entry = _safe_float(last["close"])

    atr = _safe_float(last.get("atr_14"))

    if atr <= 0:
        atr = _pip_size(pair) * 20

    sl_distance = atr * ATR_SL_MULTIPLIER

    sr = support_resistance(df)

    support = sr.get("support") or []
    resistance = sr.get("resistance") or []

    # ---------------------------------------------------------------
    # BUY
    # ---------------------------------------------------------------

    if direction == "BUY":

        stop_loss = entry - sl_distance

        valid_supports = [
            _safe_float(level)
            for level in support
            if _safe_float(level) < entry
        ]

        if valid_supports:

            nearest_support = max(valid_supports)

            distance_to_support = entry - nearest_support

            if distance_to_support <= sl_distance * 1.3:
                structural_sl = (
                    nearest_support
                    - atr * STRUCTURE_PADDING_ATR
                )

                # Never allow structural SL to become invalid.
                if structural_sl < entry:
                    stop_loss = structural_sl

        risk = entry - stop_loss

        if risk <= 0:
            stop_loss = entry - sl_distance
            risk = entry - stop_loss

        take_profit_1 = entry + risk * TP1_R_MULTIPLE
        take_profit_2 = entry + risk * TP2_R_MULTIPLE

    # ---------------------------------------------------------------
    # SELL
    # ---------------------------------------------------------------

    else:

        stop_loss = entry + sl_distance

        valid_resistances = [
            _safe_float(level)
            for level in resistance
            if _safe_float(level) > entry
        ]

        if valid_resistances:

            nearest_resistance = min(valid_resistances)

            distance_to_resistance = nearest_resistance - entry

            if distance_to_resistance <= sl_distance * 1.3:
                structural_sl = (
                    nearest_resistance
                    + atr * STRUCTURE_PADDING_ATR
                )

                # Never allow structural SL to become invalid.
                if structural_sl > entry:
                    stop_loss = structural_sl

        risk = stop_loss - entry

        if risk <= 0:
            stop_loss = entry + sl_distance
            risk = stop_loss - entry

        take_profit_1 = entry - risk * TP1_R_MULTIPLE
        take_profit_2 = entry - risk * TP2_R_MULTIPLE

    risk_reward = (
        abs(take_profit_1 - entry) / abs(entry - stop_loss)
        if abs(entry - stop_loss) > 0
        else 0.0
    )

    return (
        entry,
        stop_loss,
        take_profit_1,
        take_profit_2,
        round(risk_reward, 2),
    )


# ---------------------------------------------------------------------------
# Main signal generator
# ---------------------------------------------------------------------------

def generate_signal(
    pair: str,
    df: pd.DataFrame,
    is_news_blackout: bool = False,
) -> SignalResult:

    # ------------------------------------------------------------------
    # News blackout
    # ------------------------------------------------------------------

    if is_news_blackout:
        return _empty_result(
            pair=pair,
            explanation=(
                "Signal suppressed because the pair is inside "
                "a major economic news blackout window."
            ),
            reject_reason="news_blackout",
        )

    # ------------------------------------------------------------------
    # Data validation
    # ------------------------------------------------------------------

    if df is None or len(df) < MIN_BARS:
        return _empty_result(
            pair=pair,
            explanation=(
                f"Insufficient historical data. "
                f"At least {MIN_BARS} candles are required."
            ),
            reject_reason="insufficient_data",
        )

    # ------------------------------------------------------------------
    # Indicator calculation
    # ------------------------------------------------------------------

    try:
        df = compute_all_indicators(df)
    except Exception as exc:
        return _empty_result(
            pair=pair,
            explanation=f"Indicator calculation failed: {exc}",
            reject_reason="indicator_error",
        )

    if df.empty:
        return _empty_result(
            pair=pair,
            explanation="Indicator calculation returned no data.",
            reject_reason="indicator_error",
        )

    last = df.iloc[-1]

    # ------------------------------------------------------------------
    # Score each component
    # ------------------------------------------------------------------

    (
        tech_bull,
        tech_bear,
        tech_bull_reasons,
        tech_bear_reasons,
    ) = _score_technical(df)

    (
        pat_bull,
        pat_bear,
        pat_bull_reasons,
        pat_bear_reasons,
    ) = _score_patterns(df)

    (
        smc_bull,
        smc_bear,
        smc_bull_reasons,
        smc_bear_reasons,
    ) = _score_smc(df)

    total_bull = (
        tech_bull
        + pat_bull
        + smc_bull
    )

    total_bear = (
        tech_bear
        + pat_bear
        + smc_bear
    )

    # ------------------------------------------------------------------
    # Direction
    # ------------------------------------------------------------------

    direction = _choose_direction(
        total_bull,
        total_bear,
    )

    all_confirmations = (
        tech_bull_reasons
        + tech_bear_reasons
        + pat_bull_reasons
        + pat_bear_reasons
        + smc_bull_reasons
        + smc_bear_reasons
    )

    if direction is None:

        return _empty_result(
            pair=pair,
            explanation=(
                "No high-confluence setup: "
                f"BUY score={total_bull:.1f}, "
                f"SELL score={total_bear:.1f}."
            ),
            confirmations=all_confirmations,
            reject_reason="low_confluence",
        )

    # ------------------------------------------------------------------
    # Direction-specific reasons
    # ------------------------------------------------------------------

    if direction == "BUY":

        score = total_bull

        reasons = (
            tech_bull_reasons
            + pat_bull_reasons
            + smc_bull_reasons
        )

    else:

        score = total_bear

        reasons = (
            tech_bear_reasons
            + pat_bear_reasons
            + smc_bear_reasons
        )

    # ------------------------------------------------------------------
    # Fibonacci confirmation
    # ------------------------------------------------------------------

    fib_points, fib_reasons = _score_fibonacci(
        df,
        direction,
    )

    score += fib_points
    reasons += fib_reasons

    # ------------------------------------------------------------------
    # Calculate theoretical maximum
    #
    # Technical:
    #   EMA = 2
    #   RSI = 1
    #   MACD = 1
    #   BB = 1
    #   TOTAL = 5
    #
    # Patterns:
    #   Up to 2 recognised directional patterns = 2
    #
    # SMC:
    #   BOS = 2
    #   CHoCH = 2
    #   Liquidity = 1.5
    #   Two FVG = 1
    #   OB = 1
    #   Demand/Supply = 0.5
    #   TOTAL = 8
    #
    # Fibonacci:
    #   0.5
    #
    # Maximum = 15.5
    # ------------------------------------------------------------------

    max_possible = 15.5

    confidence = round(
        (score / max_possible) * 100
    )

    confidence = min(
        MAX_CONFIDENCE,
        max(
            MIN_CONFIDENCE_FLOOR,
            confidence,
        ),
    )

    # ------------------------------------------------------------------
    # Build trade levels
    # ------------------------------------------------------------------

    try:
        (
            entry,
            stop_loss,
            take_profit_1,
            take_profit_2,
            risk_reward,
        ) = _build_trade_levels(
            pair,
            direction,
            df,
        )

    except Exception as exc:

        return SignalResult(
            pair=pair,
            direction=direction,
            entry=0.0,
            stop_loss=0.0,
            take_profit_1=0.0,
            take_profit_2=0.0,
            risk_reward=0.0,
            confidence=confidence,
            explanation=f"Failed to build trade levels: {exc}",
            confirmations=reasons,
            published=False,
            reject_reason="trade_level_error",
        )

    # ------------------------------------------------------------------
    # Risk/reward validation
    # ------------------------------------------------------------------

    if risk_reward < settings.MIN_RISK_REWARD_RATIO:

        return SignalResult(
            pair=pair,
            direction=None,
            entry=entry,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            risk_reward=risk_reward,
            confidence=confidence,
            explanation=(
                f"Setup rejected: risk:reward "
                f"{risk_reward}:1 is below the minimum "
                f"{settings.MIN_RISK_REWARD_RATIO}:1."
            ),
            confirmations=reasons,
            published=False,
            reject_reason="rr_too_low",
        )

    # ------------------------------------------------------------------
    # Confidence validation
    # ------------------------------------------------------------------

    if confidence < settings.MIN_CONFIDENCE_TO_PUBLISH:

        return SignalResult(
            pair=pair,
            direction=direction,
            entry=entry,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            risk_reward=risk_reward,
            confidence=confidence,
            explanation=(
                f"Setup rejected: confidence {confidence}% "
                f"is below publish threshold "
                f"{settings.MIN_CONFIDENCE_TO_PUBLISH}%."
            ),
            confirmations=reasons,
            published=False,
            reject_reason="low_confidence",
        )

    # ------------------------------------------------------------------
    # Final explanation
    # ------------------------------------------------------------------

    shown_reasons = reasons[:5]

    explanation = (
        f"{direction} {pair}: "
        f"{len(reasons)} directional confluences aligned "
        f"(score {score:.1f}/{max_possible:.1f}). "
        f"{', '.join(shown_reasons)}"
        f"{'...' if len(reasons) > 5 else ''}. "
        f"Entry based on current price, with ATR/structure-based stop "
        f"and TP1/TP2 at approximately "
        f"{TP1_R_MULTIPLE}R/{TP2_R_MULTIPLE}R."
    )

    # ------------------------------------------------------------------
    # Published signal
    # ------------------------------------------------------------------

    return SignalResult(
        pair=pair,
        direction=direction,
        entry=round(entry, 5),
        stop_loss=round(stop_loss, 5),
        take_profit_1=round(take_profit_1, 5),
        take_profit_2=round(take_profit_2, 5),
        risk_reward=risk_reward,
        confidence=confidence,
        explanation=explanation,
        confirmations=reasons,
        published=True,
        reject_reason=None,
    )
