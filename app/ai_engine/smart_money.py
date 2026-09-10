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

from typing import Any

import pandas as pd


# ============================================================
# CONSTANTS
# ============================================================

_REQUIRED_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
}


# ============================================================
# VALIDATION / HELPERS
# ============================================================

def _validate_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean OHLC data.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    missing = _REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required OHLC columns: {sorted(missing)}"
        )

    clean = df.copy()

    for column in _REQUIRED_COLUMNS:
        clean[column] = pd.to_numeric(
            clean[column],
            errors="coerce",
        )

    clean = clean.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    ).copy()

    if clean.empty:
        raise ValueError(
            "No valid OHLC data available"
        )

    # Remove impossible candles.
    clean = clean[
        (clean["high"] >= clean["low"])
        & (clean["high"] >= clean["open"])
        & (clean["high"] >= clean["close"])
        & (clean["low"] <= clean["open"])
        & (clean["low"] <= clean["close"])
    ].copy()

    if clean.empty:
        raise ValueError(
            "No valid OHLC candles remain after cleaning"
        )

    return clean


def _safe_window(
    window: int,
    minimum: int = 2,
) -> int:
    """
    Ensure window is a valid positive integer.
    """

    try:
        value = int(window)
    except (TypeError, ValueError):
        value = minimum

    return max(
        minimum,
        value,
    )


def _current_atr(
    df: pd.DataFrame,
    period: int = 14,
) -> float:
    """
    Calculate a robust ATR estimate used for
    proximity filtering.
    """

    clean = _validate_ohlc(df)

    period = max(
        2,
        int(period),
    )

    high = clean["high"]
    low = clean["low"]
    close = clean["close"]

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    if len(true_range) >= period:
        atr_value = (
            true_range
            .rolling(
                period,
                min_periods=period,
            )
            .mean()
            .iloc[-1]
        )
    else:
        atr_value = true_range.mean()

    if pd.isna(atr_value) or atr_value <= 0:
        atr_value = (
            high - low
        ).median()

    if pd.isna(atr_value) or atr_value <= 0:
        last_close = abs(
            float(close.iloc[-1])
        )

        atr_value = max(
            last_close * 0.001,
            1e-8,
        )

    return float(atr_value)


def _is_near_price(
    top: float,
    bottom: float,
    price: float,
    atr: float,
    multiplier: float = 2.0,
) -> bool:
    """
    Check whether a zone is close enough to
    current price to be relevant.

    True when:
    - price is inside the zone, OR
    - nearest zone edge is within ATR distance.
    """

    top = float(top)
    bottom = float(bottom)
    price = float(price)
    atr = max(
        float(atr),
        1e-12,
    )

    if bottom > top:
        bottom, top = top, bottom

    if bottom <= price <= top:
        return True

    distance = min(
        abs(price - top),
        abs(price - bottom),
    )

    return distance <= (
        atr * float(multiplier)
    )


def _candle_body(
    candle: pd.Series,
) -> float:
    return abs(
        float(candle["close"])
        - float(candle["open"])
    )


def _candle_range(
    candle: pd.Series,
) -> float:
    return max(
        float(candle["high"])
        - float(candle["low"]),
        0.0,
    )


def _is_bullish_candle(
    candle: pd.Series,
) -> bool:
    return float(candle["close"]) > float(
        candle["open"]
    )


def _is_bearish_candle(
    candle: pd.Series,
) -> bool:
    return float(candle["close"]) < float(
        candle["open"]
    )


# ============================================================
# SWING POINTS
# ============================================================

def find_swing_points(
    df: pd.DataFrame,
    window: int = 3,
) -> pd.DataFrame:
    """
    Detect confirmed fractal swing highs and lows.

    A swing high must be greater than the surrounding
    candles.

    A swing low must be lower than the surrounding
    candles.
    """

    clean = _validate_ohlc(df)

    window = _safe_window(
        window,
        minimum=2,
    )

    clean["swing_high"] = False
    clean["swing_low"] = False

    if len(clean) < (
        window * 2 + 1
    ):
        return clean

    highs = clean["high"].to_numpy()
    lows = clean["low"].to_numpy()

    high_column = clean.columns.get_loc(
        "swing_high"
    )

    low_column = clean.columns.get_loc(
        "swing_low"
    )

    for i in range(
        window,
        len(clean) - window,
    ):
        left_highs = highs[
            i - window : i
        ]

        right_highs = highs[
            i + 1 : i + window + 1
        ]

        left_lows = lows[
            i - window : i
        ]

        right_lows = lows[
            i + 1 : i + window + 1
        ]

        center_high = highs[i]
        center_low = lows[i]

        # Strict fractal high.
        if (
            center_high > left_highs.max()
            and center_high > right_highs.max()
        ):
            clean.iloc[
                i,
                high_column,
            ] = True

        # Strict fractal low.
        if (
            center_low < left_lows.min()
            and center_low < right_lows.min()
        ):
            clean.iloc[
                i,
                low_column,
            ] = True

    return clean


# ============================================================
# BREAK OF STRUCTURE
# ============================================================

def detect_break_of_structure(
    df: pd.DataFrame,
    window: int = 3,
) -> str | None:
    """
    Detect confirmed Break of Structure.

    Bullish BOS:
        Latest close breaks above the latest
        confirmed swing high.

    Bearish BOS:
        Latest close breaks below the latest
        confirmed swing low.
    """

    clean = _validate_ohlc(df)

    if len(clean) < 15:
        return None

    tagged = find_swing_points(
        clean,
        window,
    )

    swing_highs = tagged[
        tagged["swing_high"]
    ]

    swing_lows = tagged[
        tagged["swing_low"]
    ]

    if (
        swing_highs.empty
        and swing_lows.empty
    ):
        return None

    last_close = float(
        clean["close"].iloc[-1]
    )

    # A confirmed fractal cannot be the final
    # candle because it requires candles after it.
    # Therefore the latest swing is safe to use.
    latest_high = None
    latest_low = None

    if not swing_highs.empty:
        latest_high = float(
            swing_highs["high"].iloc[-1]
        )

    if not swing_lows.empty:
        latest_low = float(
            swing_lows["low"].iloc[-1]
        )

    bullish_break = (
        latest_high is not None
        and last_close > latest_high
    )

    bearish_break = (
        latest_low is not None
        and last_close < latest_low
    )

    # If both somehow happen due to abnormal
    # data, don't force a direction.
    if bullish_break and bearish_break:
        return None

    if bullish_break:
        return "bullish_bos"

    if bearish_break:
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
    Detect possible Change of Character.

    Bullish CHoCH:
        Previous structure was bearish and price
        breaks the latest swing high.

    Bearish CHoCH:
        Previous structure was bullish and price
        breaks the latest swing low.
    """

    clean = _validate_ohlc(df)

    if len(clean) < 20:
        return None

    tagged = find_swing_points(
        clean,
        window,
    )

    highs = tagged[
        tagged["swing_high"]
    ]

    lows = tagged[
        tagged["swing_low"]
    ]

    if len(highs) < 2 or len(lows) < 2:
        return None

    last_close = float(
        clean["close"].iloc[-1]
    )

    previous_high = float(
        highs["high"].iloc[-2]
    )

    latest_high = float(
        highs["high"].iloc[-1]
    )

    previous_low = float(
        lows["low"].iloc[-2]
    )

    latest_low = float(
        lows["low"].iloc[-1]
    )

    was_uptrend = (
        latest_high > previous_high
        and latest_low > previous_low
    )

    was_downtrend = (
        latest_high < previous_high
        and latest_low < previous_low
    )

    # Bullish CHoCH = bearish structure breaks upward.
    if (
        was_downtrend
        and last_close > latest_high
    ):
        return "bullish_choch"

    # Bearish CHoCH = bullish structure breaks downward.
    if (
        was_uptrend
        and last_close < latest_low
    ):
        return "bearish_choch"

    return None


# ============================================================
# FAIR VALUE GAPS
# ============================================================

def detect_fair_value_gaps(
    df: pd.DataFrame,
    lookback: int = 60,
) -> list[dict]:
    """
    Detect 3-candle Fair Value Gaps.

    Bullish FVG:
        Candle 3 low > Candle 1 high.

    Bearish FVG:
        Candle 3 high < Candle 1 low.

    Only unfilled and price-relevant FVGs are returned.
    """

    clean = _validate_ohlc(df)

    lookback = max(
        10,
        int(lookback),
    )

    window = (
        clean
        .tail(lookback)
        .reset_index(drop=True)
    )

    if len(window) < 3:
        return []

    current_price = float(
        clean["close"].iloc[-1]
    )

    atr = _current_atr(clean)

    gaps: list[
        dict[str, Any]
    ] = []

    for i in range(
        1,
        len(window) - 1,
    ):
        first = window.iloc[i - 1]
        middle = window.iloc[i]
        third = window.iloc[i + 1]

        first_high = float(
            first["high"]
        )

        first_low = float(
            first["low"]
        )

        third_high = float(
            third["high"]
        )

        third_low = float(
            third["low"]
        )

        # ====================================================
        # BULLISH FVG
        # ====================================================

        if third_low > first_high:

            bottom = first_high
            top = third_low

            # Check candles after the FVG formation.
            future = window.iloc[
                i + 2 :
            ]

            filled = False

            if not future.empty:
                future_low = float(
                    future["low"].min()
                )

                if future_low <= bottom:
                    filled = True

            if (
                not filled
                and _is_near_price(
                    top,
                    bottom,
                    current_price,
                    atr,
                    multiplier=2.5,
                )
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

        # ====================================================
        # BEARISH FVG
        # ====================================================

        elif third_high < first_low:

            top = first_low
            bottom = third_high

            future = window.iloc[
                i + 2 :
            ]

            filled = False

            if not future.empty:
                future_high = float(
                    future["high"].max()
                )

                if future_high >= top:
                    filled = True

            if (
                not filled
                and _is_near_price(
                    top,
                    bottom,
                    current_price,
                    atr,
                    multiplier=2.5,
                )
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

    # Most recent gaps first.
    gaps = gaps[-5:]

    gaps.reverse()

    return gaps


# ============================================================
# ORDER BLOCKS
# ============================================================

def detect_order_blocks(
    df: pd.DataFrame,
    lookback: int = 80,
) -> list[dict]:
    """
    Detect approximate Order Blocks.

    Bullish OB:
        Last bearish candle before a strong bullish
        displacement.

    Bearish OB:
        Last bullish candle before a strong bearish
        displacement.

    Only price-relevant blocks are returned.
    """

    clean = _validate_ohlc(df)

    lookback = max(
        20,
        int(lookback),
    )

    window = (
        clean
        .tail(lookback)
        .reset_index(drop=True)
    )

    if len(window) < 5:
        return []

    current_price = float(
        clean["close"].iloc[-1]
    )

    atr = _current_atr(clean)

    ranges = (
        window["high"]
        - window["low"]
    )

    avg_range = float(
        ranges.mean()
    )

    if avg_range <= 0:
        return []

    blocks: list[
        dict[str, Any]
    ] = []

    for i in range(
        1,
        len(window) - 1,
    ):
        candle = window.iloc[i]
        impulse = window.iloc[i + 1]

        candle_open = float(
            candle["open"]
        )

        candle_close = float(
            candle["close"]
        )

        candle_high = float(
            candle["high"]
        )

        candle_low = float(
            candle["low"]
        )

        impulse_open = float(
            impulse["open"]
        )

        impulse_close = float(
            impulse["close"]
        )

        impulse_high = float(
            impulse["high"]
        )

        impulse_low = float(
            impulse["low"]
        )

        impulse_range = (
            impulse_high
            - impulse_low
        )

        impulse_body = abs(
            impulse_close
            - impulse_open
        )

        if impulse_range <= 0:
            continue

        # Strong displacement.
        is_strong_impulse = (
            impulse_range
            >= avg_range * 1.5
            and impulse_body
            >= avg_range * 0.8
        )

        if not is_strong_impulse:
            continue

        # ====================================================
        # BULLISH ORDER BLOCK
        # ====================================================

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
                        "strength": round(
                            impulse_range
                            / avg_range,
                            2,
                        ),
                    }
                )

        # ====================================================
        # BEARISH ORDER BLOCK
        # ====================================================

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
                        "strength": round(
                            impulse_range
                            / avg_range,
                            2,
                        ),
                    }
                )

    blocks = blocks[-5:]

    blocks.reverse()

    return blocks


# ============================================================
# LIQUIDITY SWEEP
# ============================================================

def detect_liquidity_sweep(
    df: pd.DataFrame,
    window: int = 3,
    lookback: int = 50,
) -> str | None:
    """
    Detect liquidity sweeps.

    Bearish sweep:
        Price takes a previous swing high but closes
        back below that level.

    Bullish sweep:
        Price takes a previous swing low but closes
        back above that level.
    """

    clean = _validate_ohlc(df)

    if len(clean) < 20:
        return None

    window = _safe_window(
        window,
        minimum=2,
    )

    lookback = max(
        15,
        int(lookback),
    )

    recent = (
        clean
        .tail(lookback)
        .copy()
    )

    tagged = find_swing_points(
        recent,
        window,
    )

    swing_highs = tagged[
        tagged["swing_high"]
    ]

    swing_lows = tagged[
        tagged["swing_low"]
    ]

    if (
        swing_highs.empty
        and swing_lows.empty
    ):
        return None

    last = clean.iloc[-1]

    last_high = float(
        last["high"]
    )

    last_low = float(
        last["low"]
    )

    last_close = float(
        last["close"]
    )

    # ========================================================
    # BEARISH LIQUIDITY SWEEP
    # ========================================================

    if not swing_highs.empty:

        # Avoid using a swing that is effectively
        # the same final candle.
        previous_highs = swing_highs[
            swing_highs.index
            < recent.index[-1]
        ]

        if not previous_highs.empty:
            swing_high = float(
                previous_highs["high"].iloc[-1]
            )

            if (
                last_high > swing_high
                and last_close < swing_high
            ):
                return "bearish_sweep"

    # ========================================================
    # BULLISH LIQUIDITY SWEEP
    # ========================================================

    if not swing_lows.empty:

        previous_lows = swing_lows[
            swing_lows.index
            < recent.index[-1]
        ]

        if not previous_lows.empty:
            swing_low = float(
                previous_lows["low"].iloc[-1]
            )

            if (
                last_low < swing_low
                and last_close > swing_low
            ):
                return "bullish_sweep"

    return None


# ============================================================
# SUPPLY / DEMAND ZONES
# ============================================================

def detect_supply_demand_zones(
    df: pd.DataFrame,
    lookback: int = 120,
) -> dict:
    """
    Detect approximate Supply and Demand zones.

    Demand:
        Consolidation followed by strong bullish
        displacement.

    Supply:
        Consolidation followed by strong bearish
        displacement.

    Only zones reasonably close to current price
    are returned.
    """

    clean = _validate_ohlc(df)

    lookback = max(
        30,
        int(lookback),
    )

    window = (
        clean
        .tail(lookback)
        .reset_index(drop=True)
    )

    if len(window) < 10:
        return {
            "demand": [],
            "supply": [],
        }

    current_price = float(
        clean["close"].iloc[-1]
    )

    atr = _current_atr(clean)

    ranges = (
        window["high"]
        - window["low"]
    )

    avg_range = float(
        ranges.mean()
    )

    if avg_range <= 0:
        return {
            "demand": [],
            "supply": [],
        }

    demand_zones: list[
        dict[str, Any]
    ] = []

    supply_zones: list[
        dict[str, Any]
    ] = []

    # Use 3-candle consolidation before displacement.
    for i in range(
        3,
        len(window) - 1,
    ):
        cluster = window.iloc[
            i - 3 : i
        ]

        impulse = window.iloc[i]

        cluster_high = float(
            cluster["high"].max()
        )

        cluster_low = float(
            cluster["low"].min()
        )

        cluster_range = (
            cluster_high
            - cluster_low
        )

        impulse_open = float(
            impulse["open"]
        )

        impulse_close = float(
            impulse["close"]
        )

        impulse_high = float(
            impulse["high"]
        )

        impulse_low = float(
            impulse["low"]
        )

        impulse_range = (
            impulse_high
            - impulse_low
        )

        impulse_body = abs(
            impulse_close
            - impulse_open
        )

        # Consolidation should be relatively tight.
        if cluster_range > (
            avg_range * 1.5
        ):
            continue

        # Strong displacement.
        if impulse_range < (
            avg_range * 1.2
        ):
            continue

        if impulse_body < (
            avg_range * 0.8
        ):
            continue

        # ====================================================
        # DEMAND
        # ====================================================

        if (
            impulse_close > impulse_open
            and impulse_close > cluster_high
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
                        "strength": round(
                            impulse_range
                            / avg_range,
                            2,
                        ),
                    }
                )

        # ====================================================
        # SUPPLY
        # ====================================================

        elif (
            impulse_close < impulse_open
            and impulse_close < cluster_low
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
                        "strength": round(
                            impulse_range
                            / avg_range,
                            2,
                        ),
                    }
                )

    return {
        "demand": demand_zones[-3:],
        "supply": supply_zones[-3:],
    }


# ============================================================
# SMC CONFLUENCE
# ============================================================

def _calculate_smc_confluence(
    bos: str | None,
    choch: str | None,
    liquidity_sweep: str | None,
    fair_value_gaps: list[dict],
    order_blocks: list[dict],
    supply_demand: dict,
) -> dict:
    """
    Calculate directional SMC confirmation.

    This is intentionally separate from signal_generator.py
    so that the SMC engine remains responsible only for
    structural information.
    """

    bullish_score = 0.0
    bearish_score = 0.0

    bullish_reasons: list[str] = []
    bearish_reasons: list[str] = []

    # ========================================================
    # BOS
    # ========================================================

    if bos == "bullish_bos":
        bullish_score += 2.0
        bullish_reasons.append(
            "Bullish BOS"
        )

    elif bos == "bearish_bos":
        bearish_score += 2.0
        bearish_reasons.append(
            "Bearish BOS"
        )

    # ========================================================
    # CHOCH
    # ========================================================

    if choch == "bullish_choch":
        bullish_score += 2.0
        bullish_reasons.append(
            "Bullish CHoCH"
        )

    elif choch == "bearish_choch":
        bearish_score += 2.0
        bearish_reasons.append(
            "Bearish CHoCH"
        )

    # ========================================================
    # LIQUIDITY SWEEP
    # ========================================================

    if liquidity_sweep == "bullish_sweep":
        bullish_score += 1.5
        bullish_reasons.append(
            "Bullish liquidity sweep"
        )

    elif liquidity_sweep == "bearish_sweep":
        bearish_score += 1.5
        bearish_reasons.append(
            "Bearish liquidity sweep"
        )

    # ========================================================
    # FVG
    # ========================================================

    for fvg in fair_value_gaps[-2:]:

        fvg_type = fvg.get(
            "type"
        )

        if fvg_type == "bullish_fvg":
            bullish_score += 0.5
            bullish_reasons.append(
                "Bullish FVG nearby"
            )

        elif fvg_type == "bearish_fvg":
            bearish_score += 0.5
            bearish_reasons.append(
                "Bearish FVG nearby"
            )

    # ========================================================
    # ORDER BLOCK
    # ========================================================

    if order_blocks:

        latest_ob = order_blocks[0]

        ob_type = latest_ob.get(
            "type"
        )

        if ob_type == "bullish_ob":
            bullish_score += 1.0
            bullish_reasons.append(
                "Bullish order block nearby"
            )

        elif ob_type == "bearish_ob":
            bearish_score += 1.0
            bearish_reasons.append(
                "Bearish order block nearby"
            )

    # ========================================================
    # DEMAND / SUPPLY
    # ========================================================

    demand = supply_demand.get(
        "demand",
        [],
    )

    supply = supply_demand.get(
        "supply",
        [],
    )

    if demand:
        bullish_score += 0.5
        bullish_reasons.append(
            "Demand zone nearby"
        )

    if supply:
        bearish_score += 0.5
        bearish_reasons.append(
            "Supply zone nearby"
        )

    return {
        "bullish_score": round(
            bullish_score,
            2,
        ),
        "bearish_score": round(
            bearish_score,
            2,
        ),
        "bullish_reasons": bullish_reasons,
        "bearish_reasons": bearish_reasons,
    }


# ============================================================
# FULL SMC ANALYSIS
# ============================================================

def full_smc_analysis(
    df: pd.DataFrame,
) -> dict:
    """
    Run the complete SMC analysis.

    Output keys are intentionally compatible with
    signal_generator.py.
    """

    clean = _validate_ohlc(df)

    # --------------------------------------------------------
    # Minimum data protection
    # --------------------------------------------------------

    if len(clean) < 20:
        return {
            "bos": None,
            "choch": None,
            "liquidity_sweep": None,
            "fair_value_gaps": [],
            "order_blocks": [],
            "supply_demand": {
                "demand": [],
                "supply": [],
            },
            "bullish_score": 0.0,
            "bearish_score": 0.0,
            "bullish_reasons": [],
            "bearish_reasons": [],
        }

    # --------------------------------------------------------
    # Run detectors
    # --------------------------------------------------------

    bos = detect_break_of_structure(
        clean,
        window=3,
    )

    choch = detect_change_of_character(
        clean,
        window=3,
    )

    liquidity_sweep = detect_liquidity_sweep(
        clean,
        window=3,
        lookback=50,
    )

    fair_value_gaps = detect_fair_value_gaps(
        clean,
        lookback=60,
    )

    order_blocks = detect_order_blocks(
        clean,
        lookback=80,
    )

    supply_demand = detect_supply_demand_zones(
        clean,
        lookback=120,
    )

    # --------------------------------------------------------
    # Calculate SMC confluence
    # --------------------------------------------------------

    confluence = _calculate_smc_confluence(
        bos=bos,
        choch=choch,
        liquidity_sweep=liquidity_sweep,
        fair_value_gaps=fair_value_gaps,
        order_blocks=order_blocks,
        supply_demand=supply_demand,
    )

    # --------------------------------------------------------
    # Final compatible result
    # --------------------------------------------------------

    return {
        "bos": bos,
        "choch": choch,
        "liquidity_sweep": liquidity_sweep,
        "fair_value_gaps": fair_value_gaps,
        "order_blocks": order_blocks,
        "supply_demand": supply_demand,

        # Additional confluence information.
        "bullish_score": confluence[
            "bullish_score"
        ],
        "bearish_score": confluence[
            "bearish_score"
        ],
        "bullish_reasons": confluence[
            "bullish_reasons"
        ],
        "bearish_reasons": confluence[
            "bearish_reasons"
        ],
    }


# ============================================================
# PUBLIC EXPORTS
# ============================================================

__all__ = [
    "find_swing_points",
    "detect_break_of_structure",
    "detect_change_of_character",
    "detect_fair_value_gaps",
    "detect_order_blocks",
    "detect_liquidity_sweep",
    "detect_supply_demand_zones",
    "full_smc_analysis",
]
