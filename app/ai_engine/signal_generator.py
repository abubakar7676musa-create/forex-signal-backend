"""
Combines classic technical indicators + candlestick patterns + Smart Money Concepts
into a single weighted confluence score, then builds a full trade signal
(entry / SL / TP1 / TP2 / RR / confidence / explanation) only when the setup
is strong enough to publish.

IMPORTANT: This is a rule-based confluence engine, not a guarantee of profitable
trades. It rejects low-confluence setups and enforces a minimum risk:reward,
but market risk can never be eliminated.
"""
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from app.config import settings
from app.ai_engine.indicators import compute_all_indicators, fibonacci_levels, support_resistance
from app.ai_engine.patterns import detect_patterns
from app.ai_engine.smart_money import full_smc_analysis


# Points that a pair's price is typically quoted in — used for ATR-based SL padding.
JPY_PAIRS = {"USD/JPY", "EUR/JPY", "GBP/JPY"}


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


def _score_technical(df: pd.DataFrame) -> tuple[int, int, list[str]]:
    """Returns (bullish_points, bearish_points, reasons) from classic indicators."""
    bull, bear = 0, 0
    reasons = []
    last = df.iloc[-1]

    # Trend via EMA stack
    if last["ema_20"] > last["ema_50"] > last["ema_200"]:
        bull += 2
        reasons.append("EMA stack bullish (20>50>200)")
    elif last["ema_20"] < last["ema_50"] < last["ema_200"]:
        bear += 2
        reasons.append("EMA stack bearish (20<50<200)")

    # RSI
    if last["rsi_14"] < 30:
        bull += 1
        reasons.append(f"RSI oversold ({last['rsi_14']:.1f})")
    elif last["rsi_14"] > 70:
        bear += 1
        reasons.append(f"RSI overbought ({last['rsi_14']:.1f})")
    elif last["rsi_14"] > 50:
        bull += 0.5
    elif last["rsi_14"] < 50:
        bear += 0.5

    # MACD
    if last["macd"] > last["macd_signal"] and last["macd_hist"] > 0:
        bull += 1
        reasons.append("MACD bullish crossover")
    elif last["macd"] < last["macd_signal"] and last["macd_hist"] < 0:
        bear += 1
        reasons.append("MACD bearish crossover")

    # Bollinger Bands (mean reversion / breakout context)
    if last["close"] <= last["bb_lower"]:
        bull += 1
        reasons.append("Price at/below lower Bollinger Band")
    elif last["close"] >= last["bb_upper"]:
        bear += 1
        reasons.append("Price at/above upper Bollinger Band")

    # ADX (trend strength gate, not directional)
    trending = last["adx_14"] > 20
    if trending:
        reasons.append(f"ADX confirms trend strength ({last['adx_14']:.1f})")

    return round(bull), round(bear), reasons if trending else reasons + ["⚠ ADX below 20: weak trend"]


def _score_patterns(df: pd.DataFrame) -> tuple[int, int, list[str]]:
    bull, bear = 0, 0
    reasons = []
    patterns = detect_patterns(df)
    bullish_patterns = {"bullish_engulfing", "hammer"}
    bearish_patterns = {"bearish_engulfing", "shooting_star"}
    for p in patterns:
        if p in bullish_patterns:
            bull += 1
            reasons.append(f"Candlestick: {p.replace('_', ' ')}")
        elif p in bearish_patterns:
            bear += 1
            reasons.append(f"Candlestick: {p.replace('_', ' ')}")
    return bull, bear, reasons


def _score_smc(df: pd.DataFrame) -> tuple[int, int, list[str]]:
    bull, bear = 0, 0
    reasons = []
    smc = full_smc_analysis(df)

    if smc["bos"] == "bullish_bos":
        bull += 2
        reasons.append("Break of Structure (bullish)")
    elif smc["bos"] == "bearish_bos":
        bear += 2
        reasons.append("Break of Structure (bearish)")

    if smc["choch"] == "bullish_choch":
        bull += 2
        reasons.append("Change of Character (bullish reversal)")
    elif smc["choch"] == "bearish_choch":
        bear += 2
        reasons.append("Change of Character (bearish reversal)")

    if smc["liquidity_sweep"] == "bullish_sweep":
        bull += 1.5
        reasons.append("Liquidity sweep to the downside, reversed up")
    elif smc["liquidity_sweep"] == "bearish_sweep":
        bear += 1.5
        reasons.append("Liquidity sweep to the upside, reversed down")

    for gap in smc["fair_value_gaps"][-2:]:
        if gap["type"] == "bullish_fvg":
            bull += 0.5
            reasons.append("Unfilled bullish Fair Value Gap nearby")
        else:
            bear += 0.5
            reasons.append("Unfilled bearish Fair Value Gap nearby")

    if smc["order_blocks"]:
        last_ob = smc["order_blocks"][-1]
        if last_ob["type"] == "bullish_ob":
            bull += 1
            reasons.append("Recent bullish order block")
        else:
            bear += 1
            reasons.append("Recent bearish order block")

    if smc["supply_demand"]["demand"]:
        bull += 0.5
        reasons.append("Price near demand zone")
    if smc["supply_demand"]["supply"]:
        bear += 0.5
        reasons.append("Price near supply zone")

    return round(bull), round(bear), reasons


def _pip_size(pair: str) -> float:
    return 0.01 if pair in JPY_PAIRS else (0.1 if pair == "XAU/USD" else (1.0 if pair == "BTC/USD" else 0.0001))


def generate_signal(pair: str, df: pd.DataFrame, is_news_blackout: bool = False) -> SignalResult:
    if is_news_blackout:
        return SignalResult(
            pair=pair, direction=None, entry=0, stop_loss=0, take_profit_1=0, take_profit_2=0,
            risk_reward=0, confidence=0, explanation="Signal suppressed: major economic news window.",
            reject_reason="news_blackout",
        )

    if len(df) < 210:
        return SignalResult(
            pair=pair, direction=None, entry=0, stop_loss=0, take_profit_1=0, take_profit_2=0,
            risk_reward=0, confidence=0, explanation="Insufficient historical data for analysis.",
            reject_reason="insufficient_data",
        )

    df = compute_all_indicators(df)
    last = df.iloc[-1]

    tech_bull, tech_bear, tech_reasons = _score_technical(df)
    pat_bull, pat_bear, pat_reasons = _score_patterns(df)
    smc_bull, smc_bear, smc_reasons = _score_smc(df)

    total_bull = tech_bull + pat_bull + smc_bull
    total_bear = tech_bear + pat_bear + smc_bear
    max_possible = 12.5  # rough ceiling of the weighted point system above, used to scale confidence

    direction = None
    if total_bull > total_bear and total_bull >= 5:
        direction = "BUY"
        score, reasons = total_bull, tech_reasons + pat_reasons + smc_reasons
    elif total_bear > total_bull and total_bear >= 5:
        direction = "SELL"
        score, reasons = total_bear, tech_reasons + pat_reasons + smc_reasons
    else:
        return SignalResult(
            pair=pair, direction=None, entry=0, stop_loss=0, take_profit_1=0, take_profit_2=0,
            risk_reward=0, confidence=0,
            explanation="No high-probability setup: confluence score too low or conflicting signals.",
            confirmations=tech_reasons + pat_reasons + smc_reasons,
            reject_reason="low_confluence",
        )

    confidence = min(97, max(30, round((score / max_possible) * 100)))

    entry = float(last["close"])
    atr_val = float(last["atr_14"]) if last["atr_14"] > 0 else _pip_size(pair) * 20
    sr = support_resistance(df)
    fib = fibonacci_levels(df)

    # Stop loss: 1.5x ATR beyond entry, nudged to the nearest structural level if one is close.
    sl_distance = atr_val * 1.5

    if direction == "BUY":
        stop_loss = entry - sl_distance
        if sr["support"]:
            nearest_support = max([s for s in sr["support"] if s < entry], default=None)
            if nearest_support and (entry - nearest_support) < sl_distance * 1.3:
                stop_loss = nearest_support - atr_val * 0.2
        risk = entry - stop_loss
        take_profit_1 = entry + risk * 2.0
        take_profit_2 = entry + risk * 3.5
    else:
        stop_loss = entry + sl_distance
        if sr["resistance"]:
            nearest_resistance = min([r for r in sr["resistance"] if r > entry], default=None)
            if nearest_resistance and (nearest_resistance - entry) < sl_distance * 1.3:
                stop_loss = nearest_resistance + atr_val * 0.2
        risk = stop_loss - entry
        take_profit_1 = entry - risk * 2.0
        take_profit_2 = entry - risk * 3.5

    risk_reward = round(abs(take_profit_1 - entry) / abs(entry - stop_loss), 2) if abs(entry - stop_loss) > 0 else 0

    if risk_reward < settings.MIN_RISK_REWARD_RATIO:
        return SignalResult(
            pair=pair, direction=None, entry=entry, stop_loss=stop_loss,
            take_profit_1=take_profit_1, take_profit_2=take_profit_2,
            risk_reward=risk_reward, confidence=confidence,
            explanation=f"Setup rejected: risk:reward {risk_reward} below minimum {settings.MIN_RISK_REWARD_RATIO}.",
            confirmations=reasons, reject_reason="rr_too_low",
        )

    if confidence < settings.MIN_CONFIDENCE_TO_PUBLISH:
        return SignalResult(
            pair=pair, direction=direction, entry=entry, stop_loss=stop_loss,
            take_profit_1=take_profit_1, take_profit_2=take_profit_2,
            risk_reward=risk_reward, confidence=confidence,
            explanation=f"Setup rejected: confidence {confidence}% below publish threshold "
                        f"{settings.MIN_CONFIDENCE_TO_PUBLISH}%.",
            confirmations=reasons, reject_reason="low_confidence",
        )

    explanation = (
        f"{direction} {pair}: {len(reasons)} confluences aligned "
        f"({', '.join(reasons[:4])}{'...' if len(reasons) > 4 else ''}). "
        f"Entry near current price with stop beyond recent structure/ATR, "
        f"targeting a {risk_reward}:1 reward-to-risk."
    )

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
    )
