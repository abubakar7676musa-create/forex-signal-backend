"""
Lightweight candlestick pattern recognition.

No TA-Lib dependency.

The detectors focus on the most recent candles and use simple price-action
rules. Patterns are confirmations only; they should not be treated as
guaranteed reversal signals.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
}


def _validate_dataframe(df: pd.DataFrame) -> bool:
    """Return True when enough valid OHLC data exists."""

    if df is None or len(df) < 3:
        return False

    return REQUIRED_COLUMNS.issubset(df.columns)


def _safe_value(value, default: float = 0.0) -> float:
    """Safely convert a value to float."""

    try:
        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except (TypeError, ValueError):
        return default


def _body(row) -> float:
    return abs(
        _safe_value(row["close"])
        - _safe_value(row["open"])
    )


def _range(row) -> float:
    high = _safe_value(row["high"])
    low = _safe_value(row["low"])

    return max(
        high - low,
        1e-9,
    )


def _upper_wick(row) -> float:
    high = _safe_value(row["high"])
    close = _safe_value(row["close"])
    open_price = _safe_value(row["open"])

    return max(
        0.0,
        high - max(close, open_price),
    )


def _lower_wick(row) -> float:
    low = _safe_value(row["low"])
    close = _safe_value(row["close"])
    open_price = _safe_value(row["open"])

    return max(
        0.0,
        min(close, open_price) - low,
    )


def _is_bullish(row) -> bool:
    return _safe_value(row["close"]) > _safe_value(row["open"])


def _is_bearish(row) -> bool:
    return _safe_value(row["close"]) < _safe_value(row["open"])


# ---------------------------------------------------------------------------
# Candle context
# ---------------------------------------------------------------------------

def _recent_bearish_context(
    df: pd.DataFrame,
    candles: int = 3,
) -> bool:
    """
    Checks whether the recent candles show bearish pressure.

    Used as context for bullish reversal patterns.
    """

    if len(df) < candles + 1:
        return False

    recent = df.iloc[
        -(candles + 1):-1
    ]

    closes = pd.to_numeric(
        recent["close"],
        errors="coerce",
    )

    return (
        closes.notna().all()
        and closes.iloc[-1] < closes.iloc[0]
    )


def _recent_bullish_context(
    df: pd.DataFrame,
    candles: int = 3,
) -> bool:
    """
    Checks whether the recent candles show bullish pressure.

    Used as context for bearish reversal patterns.
    """

    if len(df) < candles + 1:
        return False

    recent = df.iloc[
        -(candles + 1):-1
    ]

    closes = pd.to_numeric(
        recent["close"],
        errors="coerce",
    )

    return (
        closes.notna().all()
        and closes.iloc[-1] > closes.iloc[0]
    )


# ---------------------------------------------------------------------------
# Engulfing
# ---------------------------------------------------------------------------

def is_bullish_engulfing(
    df: pd.DataFrame,
) -> bool:
    """
    Bullish engulfing:

    Previous candle bearish.
    Current candle bullish.
    Current real body covers previous real body.
    """

    if not _validate_dataframe(df):
        return False

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    prev_open = _safe_value(prev["open"])
    prev_close = _safe_value(prev["close"])
    curr_open = _safe_value(curr["open"])
    curr_close = _safe_value(curr["close"])

    return (
        prev_close < prev_open
        and curr_close > curr_open
        and curr_open <= prev_close
        and curr_close >= prev_open
    )


def is_bearish_engulfing(
    df: pd.DataFrame,
) -> bool:
    """
    Bearish engulfing:

    Previous candle bullish.
    Current candle bearish.
    Current real body covers previous real body.
    """

    if not _validate_dataframe(df):
        return False

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    prev_open = _safe_value(prev["open"])
    prev_close = _safe_value(prev["close"])
    curr_open = _safe_value(curr["open"])
    curr_close = _safe_value(curr["close"])

    return (
        prev_close > prev_open
        and curr_close < curr_open
        and curr_open >= prev_close
        and curr_close <= prev_open
    )


# ---------------------------------------------------------------------------
# Hammer
# ---------------------------------------------------------------------------

def is_hammer(
    df: pd.DataFrame,
) -> bool:
    """
    Hammer-like bullish reversal candle.

    Requirements:
    - Recent bearish context
    - Small body relative to range
    - Lower wick >= 2x body
    - Small upper wick
    """

    if not _validate_dataframe(df):
        return False

    if not _recent_bearish_context(df):
        return False

    row = df.iloc[-1]

    body = _body(row)
    candle_range = _range(row)
    upper_wick = _upper_wick(row)
    lower_wick = _lower_wick(row)

    body_ratio = body / candle_range

    # Avoid treating a completely flat candle as a hammer.
    if body <= 0:
        return False

    return (
        lower_wick >= 2.0 * body
        and upper_wick <= 0.75 * body
        and body_ratio <= 0.40
    )


# ---------------------------------------------------------------------------
# Shooting star
# ---------------------------------------------------------------------------

def is_shooting_star(
    df: pd.DataFrame,
) -> bool:
    """
    Shooting-star-like bearish reversal candle.

    Requirements:
    - Recent bullish context
    - Small body relative to range
    - Upper wick >= 2x body
    - Small lower wick
    """

    if not _validate_dataframe(df):
        return False

    if not _recent_bullish_context(df):
        return False

    row = df.iloc[-1]

    body = _body(row)
    candle_range = _range(row)
    upper_wick = _upper_wick(row)
    lower_wick = _lower_wick(row)

    body_ratio = body / candle_range

    if body <= 0:
        return False

    return (
        upper_wick >= 2.0 * body
        and lower_wick <= 0.75 * body
        and body_ratio <= 0.40
    )


# ---------------------------------------------------------------------------
# Doji
# ---------------------------------------------------------------------------

def is_doji(
    df: pd.DataFrame,
) -> bool:
    """
    Detect a small-body indecision candle.

    Doji is informational only and is not scored as bullish or bearish.
    """

    if not _validate_dataframe(df):
        return False

    row = df.iloc[-1]

    body = _body(row)
    candle_range = _range(row)

    return (
        body / candle_range
        <= 0.10
    )


# ---------------------------------------------------------------------------
# Pin bars
# ---------------------------------------------------------------------------

def is_pin_bar_bullish(
    df: pd.DataFrame,
) -> bool:
    """
    Compatibility alias for bullish pin-bar detection.
    """

    return is_hammer(df)


def is_pin_bar_bearish(
    df: pd.DataFrame,
) -> bool:
    """
    Compatibility alias for bearish pin-bar detection.
    """

    return is_shooting_star(df)


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------

def detect_patterns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Return pattern names detected on the latest candle(s).

    Possible values:

        bullish_engulfing
        bearish_engulfing
        hammer
        shooting_star
        doji
    """

    if not _validate_dataframe(df):
        return []

    found: list[str] = []

    checks = (
        (
            "bullish_engulfing",
            is_bullish_engulfing,
        ),
        (
            "bearish_engulfing",
            is_bearish_engulfing,
        ),
        (
            "hammer",
            is_hammer,
        ),
        (
            "shooting_star",
            is_shooting_star,
        ),
        (
            "doji",
            is_doji,
        ),
    )

    for name, detector in checks:

        try:
            if detector(df):
                found.append(name)

        except (
            KeyError,
            TypeError,
            ValueError,
            IndexError,
        ):
            # Invalid candle data should not crash the whole signal engine.
            continue

    return found
