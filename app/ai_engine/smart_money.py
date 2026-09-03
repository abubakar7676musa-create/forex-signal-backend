"""
Smart Money Concepts (SMC) detection.

Rule-based approximations of:
- Break of Structure (BOS)
- Change of Character (CHoCH)
- Fair Value Gaps (FVG)
- Order Blocks (OB)
- Liquidity Sweeps
- Supply / Demand zones

These are heuristic confirmations, not guaranteed institutional-flow detection.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


# ============================================================
# VALIDATION / HELPERS
# ============================================================

_REQUIRED_COLUMNS = {"open", "high", "low", "close"}


def _validate_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean OHLC data."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required OHLC columns: {sorted(missing)}"
        )

    clean = df.copy()

    for column in _REQUIRED_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    clean = clean.dropna(
        subset=["open", "high", "low", "close"]
    ).copy()

    if clean.empty:
        raise ValueError("No valid OHLC data available")

    return clean


def _safe_window(window: int, minimum: int = 2) -> int:
    return max(minimum, int(window))


def _current_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Return a simple ATR estimate for proximity filtering."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_value = true_range.rolling(
        period,
        min_periods=min(period, len(true_range)),
    ).mean().iloc[-1]

    if pd.isna(atr_value) or atr_value <= 0:
        atr_value = (high - low).median()

    if pd.isna(atr_value) or atr_value <= 0:
        atr_value = max(abs(close.iloc[-1]) * 0.001, 1e-8)

    return float(atr_value)


def _is_near_price(
    top: float,
    bottom: float,
    price: float,
    atr: float,
    multiplier: float = 2.0,
) -> bool:
    """
    Check whether a zone/FVG is reasonably close to current price.

    A zone is considered relevant if:
    - current price is inside it, OR
    - the nearest edge is within multiplier * ATR.
    """
    if bottom > top:
        bottom, top = top, bottom

    if bottom <= price <= top:
        return True

    distance = min(
        abs(price - top),
        abs(price - bottom),
    )

    return distance <= atr * multiplier


# ============================================================
# SWING POINTS
# ============================================================

def find_swing_points(
    df: pd.DataFrame,
    window: int = 3,
) -> pd.DataFrame:
    """
    Tag confirmed fractal swing highs and swing lows.

    A swing high must be higher than the surrounding candles.
    A swing low must be lower than the surrounding candles.
    """
    clean = _validate_ohlc(df)

    window = _safe_window(window)

    clean["swing_high"] = False
    clean["swing_low"] = False

    if len(clean) < (window * 2 + 1):
        return clean

    highs = clean["high"].to_numpy()
    lows = clean["low"].to_numpy()

    for i in range(window, len(clean) - window):
        high_slice = highs[i - window : i + window + 1]
        low_slice = lows[i - window : i + window + 1]

        center_high = highs[i]
        center_low = lows[i]

        if center_high == high_slice.max() and (
            center_high > high_slice[:window].max()
            or center_high > high_slice[window + 1 :].max()
        ):
            clean.iloc[i, clean.columns.get_loc("swing_high")] = True

        if center_low == low_slice.min() and (
            center_low < low_slice[:window].min()
            or center_low < low_slice[window + 1 :].min()
        ):
            clean.iloc[i, clean.columns.get_loc("swing_low")] = True

    return clean


# ============================================================
# BREAK OF STRUCTURE
# ============================================================

def detect_break_of_structure(
    df: pd.DataFrame,
    window: int = 3,
) -> str | None:
    """
    Detect confirmed BOS.

    Bullish BOS:
        Latest closed candle closes above the most recent confirmed swing high.

    Bearish BOS:
        Latest closed candle closes below the most recent confirmed swing low.
    """
    clean = _validate_ohlc(df)

    if len(clean) < 10:
        return None

    tagged = find_swing_points(clean, window)

    swing_highs = tagged[tagged["swing_high"]]
    swing_lows = tagged[tagged["swing_low"]]

    last_close = float(clean["close"].iloc[-1])

    # Exclude the current candle from structure reference.
    swing_highs = swing_highs.iloc[:-1] if len(swing_highs) else swing_highs
    swing_lows = swing_lows.iloc[:-1] if len(swing_lows) else swing_lows

    if not swing_highs.empty:
        latest_high = float(swing_highs["high"].iloc[-1])

        if last_close > latest_high:
            return "bullish_bos"

    if not swing_lows.empty:
        latest_low = float(swing_lows["low"].iloc[-1])

        if last_close < latest_low:
            return "bearish_bos"

    return None


# ============================================================
# CHANGE OF CHARACTER
# ============================================================

def detect_change_of_character(
    df: pd.DataFrame,
    window: int = 3,
) -> str | None:
    """
    Detect a possible CHoCH.

    CHoCH is only returned when:
    - a clear higher-high/higher-low structure exists and price breaks
      the latest swing low, OR
    - a clear lower-high/lower-low structure exists and price breaks
      the latest swing high.
    """
    clean = _validate_ohlc(df)

    if len(clean) < 15:
        return None

    tagged = find_swing_points(clean, window)

    highs = tagged[tagged["swing_high"]]
    lows = tagged[tagged["swing_low"]]

    if len(highs) < 2 or len(lows) < 2:
        return None

    last_close = float(clean["close"].iloc[-1])

    h1 = float(highs["high"].iloc[-2])
    h2 = float(highs["high"].iloc[-1])

    l1 = float(lows["low"].iloc[-2])
    l2 = float(lows["low"].iloc[-1])

    was_uptrend = h2 > h1 and l2 > l1
    was_downtrend = h2 < h1 and l2 < l1

    if was_uptrend and last_close < l2:
        return "bearish_choch"

    if was_downtrend and last_close > h2:
        return "bullish_choch"

    return None


# ============================================================
# FAIR VALUE GAPS
# ============================================================

def detect_fair_value_gaps(
    df: pd.DataFrame,
    lookback: int = 40,
) -> list[dict]:
    """
    Detect 3-candle Fair Value Gaps.

    Bullish:
        candle[i+1].low > candle[i-1].high

    Bearish:
        candle[i+1].high < candle[i-1].low

    Only unfilled / still-relevant gaps near current price are returned.
    """
    clean = _validate_ohlc(df)

    lookback = max(10, int(lookback))
    window = clean.tail(lookback).reset_index(drop=True)

    if len(window) < 3:
        return []

    current_price = float(clean["close"].iloc[-1])
    atr = _current_atr(clean)

    gaps: list[dict[str, Any]] = []

    for i in range(1, len(window) - 1):
        c1 = window.iloc[i - 1]
        c2 = window.iloc[i]
        c3 = window.iloc[i + 1]

        # ----------------------------------------------------
        # Bullish FVG
        # ----------------------------------------------------
        if float(c3["low"]) > float(c1["high"]):
            top = float(c3["low"])
            bottom = float(c1["high"])

            # Check whether the middle/future candles filled the gap.
            future = window.iloc[i + 1 :]

            filled = (
                not future.empty
                and float(future["low"].min()) <= bottom
            )

            if not filled and _is_near_price(
                top,
                bottom,
                current_price,
                atr,
                multiplier=2.5,
            ):
                gaps.append(
                    {
                        "type": "bullish_fvg",
                        "top": top,
                        "bottom": bottom,
                        "index": i,
                        "size": top - bottom,
                    }
                )

        # ----------------------------------------------------
        # Bearish FVG
        # ----------------------------------------------------
        elif float(c3["high"]) < float(c1["low"]):
            top = float(c1["low"])
            bottom = float(c3["high"])

            future = window.iloc[i + 1 :]

            filled = (
                not future.empty
                and float(future["high"].max()) >= top
            )

            if not filled and _is_near_price(
                top,
                bottom,
                current_price,
                atr,
                multiplier=2.5,
            ):
                gaps.append(
                    {
                        "type": "bearish_fvg",
                        "top": top,
                        "bottom": bottom,
                        "index": i,
                        "size": top - bottom,
                    }
                )

    # Most recent first.
    return gaps[-5:]


# ============================================================
# ORDER BLOCKS
# ============================================================

def detect_order_blocks(
    df: pd.DataFrame,
    lookback: int = 60,
) -> list[dict]:
    """
    Detect approximate Order Blocks.

    Bullish OB:
        A bearish candle followed by a strong bullish impulse.

    Bearish OB:
        A bullish candle followed by a strong bearish impulse.

    Blocks are kept only when they remain reasonably close
    to current price.
    """
    clean = _validate_ohlc(df)

    lookback = max(15, int(lookback))
    window = clean.tail(lookback).reset_index(drop=True)

    if len(window) < 5:
        return []

    current_price = float(clean["close"].iloc[-1])
    atr = _current_atr(clean)

    ranges = window["high"] - window["low"]
    avg_range = float(ranges.mean())

    if avg_range <= 0:
        return []

    blocks: list[dict[str, Any]] = []

    for i in range(1, len(window) - 1):
        candle = window.iloc[i]
        impulse = window.iloc[i + 1]

        candle_open = float(candle["open"])
        candle_close = float(candle["close"])
        candle_high = float(candle["high"])
        candle_low = float(candle["low"])

        impulse_open = float(impulse["open"])
        impulse_close = float(impulse["close"])

        impulse_range = float(
            impulse["high"] - impulse["low"]
        )

        is_impulsive = impulse_range >= avg_range * 1.5

        if not is_impulsive:
            continue

        # ----------------------------------------------------
        # Bullish Order Block
        # ----------------------------------------------------
        if (
            candle_close < candle_open
            and impulse_close > impulse_open
            and impulse_close > candle_high
        ):
            top = candle_high
            bottom = candle_low

            if _is_near_price(
                top,
                bottom,
                current_price,
                atr,
                multiplier=3.0,
            ):
                blocks.append(
                    {
                        "type": "bullish_ob",
                        "top": top,
                        "bottom": bottom,
                        "index": i,
                        "strength": impulse_range / avg_range,
                    }
                )

        # ----------------------------------------------------
        # Bearish Order Block
        # ----------------------------------------------------
        elif (
            candle_close > candle_open
            and impulse_close < impulse_open
            and impulse_close < candle_low
        ):
            top = candle_high
            bottom = candle_low

            if _is_near_price(
                top,
                bottom,
                current_price,
                atr,
                multiplier=3.0,
            ):
                blocks.append(
                    {
                        "type": "bearish_ob",
                        "top": top,
                        "bottom": bottom,
                        "index": i,
                        "strength": impulse_range / avg_range,
                    }
                )

    return blocks[-5:]


# ============================================================
# LIQUIDITY SWEEP
# ============================================================

def detect_liquidity_sweep(
    df: pd.DataFrame,
    window: int = 3,
    lookback: int = 40,
) -> str | None:
    """
    Detect liquidity sweeps.

    Bearish sweep:
        Price trades above a confirmed swing high,
        then closes back below it.

    Bullish sweep:
        Price trades below a confirmed swing low,
        then closes back above it.
    """
    clean = _validate_ohlc(df)

    if len(clean) < 15:
        return None

    lookback = max(15, int(lookback))

    recent = clean.tail(lookback).copy()
    tagged = find_swing_points(recent, window)

    swing_highs = tagged[tagged["swing_high"]]
    swing_lows = tagged[tagged["swing_low"]]

    if swing_highs.empty and swing_lows.empty:
        return None

    last = clean.iloc[-1]

    last_high = float(last["high"])
    last_low = float(last["low"])
    last_close = float(last["close"])

    # --------------------------------------------------------
    # Bearish liquidity sweep
    # --------------------------------------------------------
    if not swing_highs.empty:
        swing_high = float(swing_highs["high"].iloc[-1])

        if (
            last_high > swing_high
            and last_close < swing_high
        ):
            return "bearish_sweep"

    # --------------------------------------------------------
    # Bullish liquidity sweep
    # --------------------------------------------------------
    if not swing_lows.empty:
        swing_low = float(swing_lows["low"].iloc[-1])

        if (
            last_low < swing_low
            and last_close > swing_low
        ):
            return "bullish_sweep"

    return None


# ============================================================
# SUPPLY / DEMAND
# ============================================================

def detect_supply_demand_zones(
    df: pd.DataFrame,
    lookback: int = 100,
) -> dict:
    """
    Detect approximate supply/demand zones.

    Demand:
        Tight consolidation followed by a strong bullish impulse.

    Supply:
        Tight consolidation followed by a strong bearish impulse.

    Only zones near current price are returned.
    """
    clean = _validate_ohlc(df)

    lookback = max(20, int(lookback))
    window = clean.tail(lookback).reset_index(drop=True)

    if len(window) < 8:
        return {
            "demand": [],
            "supply": [],
        }

    current_price = float(clean["close"].iloc[-1])
    atr = _current_atr(clean)

    ranges = window["high"] - window["low"]
    avg_range = float(ranges.mean())

    if avg_range <= 0:
        return {
            "demand": [],
            "supply": [],
        }

    demand_zones: list[dict[str, Any]] = []
    supply_zones: list[dict[str, Any]] = []

    for i in range(3, len(window) - 1):
        cluster = window.iloc[i - 3 : i]
        impulse = window.iloc[i]

        cluster_high = float(cluster["high"].max())
        cluster_low = float(cluster["low"].min())

        cluster_range = cluster_high - cluster_low

        impulse_open = float(impulse["open"])
        impulse_close = float(impulse["close"])
        impulse_range = float(
            impulse["high"] - impulse["low"]
        )

        # Tight consolidation.
        if cluster_range > avg_range * 1.5:
            continue

        is_strong = impulse_range >= avg_range * 1.2

        if not is_strong:
            continue

        # ----------------------------------------------------
        # Demand
        # ----------------------------------------------------
        if (
            impulse_close > impulse_open
            and (impulse_close - impulse_open) >= avg_range
        ):
            if _is_near_price(
                cluster_high,
                cluster_low,
                current_price,
                atr,
                multiplier=3.0,
            ):
                demand_zones.append(
                    {
                        "top": cluster_high,
                        "bottom": cluster_low,
                        "strength": impulse_range / avg_range,
                    }
                )

        # ----------------------------------------------------
        # Supply
        # ----------------------------------------------------
        elif (
            impulse_close < impulse_open
            and (impulse_open - impulse_close) >= avg_range
        ):
            if _is_near_price(
                cluster_high,
                cluster_low,
                current_price,
                atr,
                multiplier=3.0,
            ):
                supply_zones.append(
                    {
                        "top": cluster_high,
                        "bottom": cluster_low,
                        "strength": impulse_range / avg_range,
                    }
                )

    return {
        "demand": demand_zones[-3:],
        "supply": supply_zones[-3:],
    }


# ============================================================
# FULL SMC ANALYSIS
# ============================================================

def full_smc_analysis(df: pd.DataFrame) -> dict:
    """
    Run all SMC detectors.

    Output keys are intentionally kept compatible with
    signal_generator.py.
   
