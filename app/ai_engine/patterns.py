"""
Lightweight candlestick pattern recognition (no external TA-Lib dependency).
Each detector looks at the most recent 1-3 candles and returns a pattern name or None.
"""
import pandas as pd


def _body(row) -> float:
    return abs(row["close"] - row["open"])


def _range(row) -> float:
    return max(row["high"] - row["low"], 1e-9)


def _upper_wick(row) -> float:
    return row["high"] - max(row["close"], row["open"])


def _lower_wick(row) -> float:
    return min(row["close"], row["open"]) - row["low"]


def is_bullish_engulfing(df: pd.DataFrame) -> bool:
    prev, curr = df.iloc[-2], df.iloc[-1]
    return (
        prev["close"] < prev["open"]
        and curr["close"] > curr["open"]
        and curr["close"] >= prev["open"]
        and curr["open"] <= prev["close"]
    )


def is_bearish_engulfing(df: pd.DataFrame) -> bool:
    prev, curr = df.iloc[-2], df.iloc[-1]
    return (
        prev["close"] > prev["open"]
        and curr["close"] < curr["open"]
        and curr["open"] >= prev["close"]
        and curr["close"] <= prev["open"]
    )


def is_hammer(df: pd.DataFrame) -> bool:
    row = df.iloc[-1]
    body, rng = _body(row), _range(row)
    return _lower_wick(row) >= 2 * body and _upper_wick(row) <= body * 0.5 and body / rng < 0.4


def is_shooting_star(df: pd.DataFrame) -> bool:
    row = df.iloc[-1]
    body, rng = _body(row), _range(row)
    return _upper_wick(row) >= 2 * body and _lower_wick(row) <= body * 0.5 and body / rng < 0.4


def is_doji(df: pd.DataFrame) -> bool:
    row = df.iloc[-1]
    return _body(row) / _range(row) < 0.1


def is_pin_bar_bullish(df: pd.DataFrame) -> bool:
    return is_hammer(df)


def is_pin_bar_bearish(df: pd.DataFrame) -> bool:
    return is_shooting_star(df)


def detect_patterns(df: pd.DataFrame) -> list[str]:
    """Returns a list of pattern names detected on the latest candle(s)."""
    if len(df) < 3:
        return []
    found = []
    checks = {
        "bullish_engulfing": is_bullish_engulfing,
        "bearish_engulfing": is_bearish_engulfing,
        "hammer": is_hammer,
        "shooting_star": is_shooting_star,
        "doji": is_doji,
    }
    for name, fn in checks.items():
        try:
            if fn(df):
                found.append(name)
        except Exception:
            continue
    return found
