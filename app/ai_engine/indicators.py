"""
Classic technical indicators computed with pandas/numpy.

No TA-Lib dependency, so this installs cleanly on cloud hosts.

Expected DataFrame columns:
    open, high, low, close, volume

The module provides:
    - EMA
    - RSI
    - MACD
    - Bollinger Bands
    - ATR
    - ADX
    - Fibonacci levels
    - Fractal-based support/resistance
    - Combined indicator calculation
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_ohlc(df: pd.DataFrame) -> None:
    """
    Validate that the input DataFrame contains the required market columns.
    """

    if df is None or df.empty:
        raise ValueError("Market DataFrame is empty.")

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required market columns: {sorted(missing)}"
        )


def _clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert OHLCV columns to numeric and remove invalid rows.
    """

    df = df.copy()

    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    )

    return df


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

def ema(
    series: pd.Series,
    period: int,
) -> pd.Series:

    if period <= 0:
        raise ValueError("EMA period must be greater than zero.")

    return series.ewm(
        span=period,
        adjust=False,
        min_periods=period,
    ).mean()


def add_emas(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["ema_20"] = ema(
        df["close"],
        20,
    )

    df["ema_50"] = ema(
        df["close"],
        50,
    )

    df["ema_200"] = ema(
        df["close"],
        200,
    )

    return df


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def rsi(
    series: pd.Series,
    period: int = 14,
) -> pd.Series:

    if period <= 0:
        raise ValueError("RSI period must be greater than zero.")

    delta = series.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan,
    )

    result = 100 - (
        100 / (1 + rs)
    )

    # Handle strong one-sided moves.
    result = result.where(
        avg_loss != 0,
        100,
    )

    result = result.where(
        avg_gain != 0,
        0,
    )

    # Neutral fallback during warm-up.
    return result.fillna(50)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
):
    if fast <= 0 or slow <= 0 or signal <= 0:
        raise ValueError(
            "MACD periods must be greater than zero."
        )

    if fast >= slow:
        raise ValueError(
            "MACD fast period must be smaller than slow period."
        )

    ema_fast = ema(
        series,
        fast,
    )

    ema_slow = ema(
        series,
        slow,
    )

    macd_line = ema_fast - ema_slow

    signal_line = ema(
        macd_line,
        signal,
    )

    histogram = (
        macd_line - signal_line
    )

    return (
        macd_line,
        signal_line,
        histogram,
    )


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
):

    if period <= 0:
        raise ValueError(
            "Bollinger period must be greater than zero."
        )

    if std_dev <= 0:
        raise ValueError(
            "Bollinger standard deviation must be greater than zero."
        )

    sma = series.rolling(
        period,
        min_periods=period,
    ).mean()

    std = series.rolling(
        period,
        min_periods=period,
    ).std()

    upper = sma + (
        std_dev * std
    )

    lower = sma - (
        std_dev * std
    )

    return (
        upper,
        sma,
        lower,
    )


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

def atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:

    if period <= 0:
        raise ValueError(
            "ATR period must be greater than zero."
        )

    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


# ---------------------------------------------------------------------------
# ADX
# ---------------------------------------------------------------------------

def adx(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:

    if period <= 0:
        raise ValueError(
            "ADX period must be greater than zero."
        )

    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move)
            & (up_move > 0),
            up_move,
            0.0,
        ),
        index=df.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move)
            & (down_move > 0),
            down_move,
            0.0,
        ),
        index=df.index,
    )

    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_smooth = true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    plus_dm_smooth = plus_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    minus_dm_smooth = minus_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    plus_di = (
        100
        * plus_dm_smooth
        / atr_smooth.replace(
            0,
            np.nan,
        )
    )

    minus_di = (
        100
        * minus_dm_smooth
        / atr_smooth.replace(
            0,
            np.nan,
        )
    )

    denominator = (
        plus_di + minus_di
    ).replace(
        0,
        np.nan,
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / denominator
    )

    return dx.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean().fillna(0)


# ---------------------------------------------------------------------------
# Fibonacci
# ---------------------------------------------------------------------------

def fibonacci_levels(
    df: pd.DataFrame,
    lookback: int = 100,
) -> dict:

    if lookback <= 0:
        raise ValueError(
            "Fibonacci lookback must be greater than zero."
        )

    if df.empty:
        return {}

    window = df.tail(
        lookback
    )

    swing_high = float(
        window["high"].max()
    )

    swing_low = float(
        window["low"].min()
    )

    if not np.isfinite(swing_high):
        return {}

    if not np.isfinite(swing_low):
        return {}

    difference = (
        swing_high - swing_low
    )

    if difference <= 0:
        return {}

    return {
        "0.0": swing_high,
        "0.236": swing_high - (
            0.236 * difference
        ),
        "0.382": swing_high - (
            0.382 * difference
        ),
        "0.5": swing_high - (
            0.500 * difference
        ),
        "0.618": swing_high - (
            0.618 * difference
        ),
        "0.786": swing_high - (
            0.786 * difference
        ),
        "1.0": swing_low,
    }


# ---------------------------------------------------------------------------
# Support / Resistance
# ---------------------------------------------------------------------------

def _deduplicate_levels(
    levels: list[float],
    tolerance: float,
) -> list[float]:
    """
    Remove levels that are extremely close to one another.

    This prevents many almost-identical fractal levels from
    flooding the S/R list.
    """

    if not levels:
        return []

    sorted_levels = sorted(
        float(level)
        for level in levels
        if np.isfinite(level)
    )

    result: list[float] = []

    for level in sorted_levels:

        if not result:
            result.append(level)
            continue

        previous = result[-1]

        if abs(level - previous) > tolerance:
            result.append(level)

    return result


def support_resistance(
    df: pd.DataFrame,
    lookback: int = 100,
    window: int = 5,
) -> dict:
    """
    Fractal-based support/resistance detector.

    Unlike the old implementation, this does NOT simply return the
    highest three resistance levels and lowest three support levels.

    It returns multiple nearby structural levels so the signal engine
    can select the nearest valid level relative to current price.
    """

    if df.empty:
        return {
            "support": [],
            "resistance": [],
        }

    if lookback <= 0:
        raise ValueError(
            "S/R lookback must be greater than zero."
        )

    if window <= 0:
        raise ValueError(
            "S/R window must be greater than zero."
        )

    data = df.tail(
        lookback
    ).reset_index(
        drop=True
    )

    if len(data) < (
        window * 2 + 1
    ):
        return {
            "support": [],
            "resistance": [],
        }

    highs: list[float] = []
    lows: list[float] = []

    for i in range(
        window,
        len(data) - window,
    ):

        high = float(
            data["high"].iloc[i]
        )

        low = float(
            data["low"].iloc[i]
        )

        surrounding_highs = data[
            "high"
        ].iloc[
            i - window:i + window + 1
        ]

        surrounding_lows = data[
            "low"
        ].iloc[
            i - window:i + window + 1
        ]

        # Swing high
        if high >= float(
            surrounding_highs.max()
        ):
            highs.append(high)

        # Swing low
        if low <= float(
            surrounding_lows.min()
        ):
            lows.append(low)

    if not highs and not lows:
        return {
            "support": [],
            "resistance": [],
        }

    # Estimate a tolerance from current ATR if available.
    if "atr_14" in data.columns:

        atr_values = pd.to_numeric(
            data["atr_14"],
            errors="coerce",
        ).dropna()

        if not atr_values.empty:
            tolerance = max(
                float(
                    atr_values.iloc[-1]
                ) * 0.15,
                1e-10,
            )
        else:
            tolerance = 1e-10

    else:
        current_price = float(
            data["close"].iloc[-1]
        )

        tolerance = max(
            abs(current_price) * 0.0001,
            1e-10,
        )

    resistance = _deduplicate_levels(
        highs,
        tolerance,
    )

    support = _deduplicate_levels(
        lows,
        tolerance,
    )

    # Keep the most recent/relevant levels rather than only the
    # absolute highest/lowest values.
    current_price = float(
        data["close"].iloc[-1]
    )

    resistance = sorted(
        [
            level
            for level in resistance
            if level > current_price
        ],
        key=lambda level: level - current_price,
    )[:10]

    support = sorted(
        [
            level
            for level in support
            if level < current_price
        ],
        key=lambda level: current_price - level,
    )[:10]

    return {
        "support": support,
        "resistance": resistance,
    }


# ---------------------------------------------------------------------------
# Combined indicators
# ---------------------------------------------------------------------------

def compute_all_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate all indicators required by signal_generator.py.
    """

    _validate_ohlc(df)

    df = _clean_numeric_columns(
        df
    )

    if df.empty:
        raise ValueError(
            "No valid OHLC data remains after cleaning."
        )

    # ---------------------------------------------------------------
    # Trend
    # ---------------------------------------------------------------

    df = add_emas(
        df
    )

    # ---------------------------------------------------------------
    # Momentum
    # ---------------------------------------------------------------

    df["rsi_14"] = rsi(
        df["close"],
        14,
    )

    (
        macd_line,
        signal_line,
        histogram,
    ) = macd(
        df["close"]
    )

    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = histogram

    # ---------------------------------------------------------------
    # Volatility
    # ---------------------------------------------------------------

    (
        upper,
        middle,
        lower,
    ) = bollinger_bands(
        df["close"]
    )

    df["bb_upper"] = upper
    df["bb_mid"] = middle
    df["bb_lower"] = lower

    df["atr_14"] = atr(
        df,
        14,
    )

    # ---------------------------------------------------------------
    # Trend strength
    # ---------------------------------------------------------------

    df["adx_14"] = adx(
        df,
        14,
    )

    return df
