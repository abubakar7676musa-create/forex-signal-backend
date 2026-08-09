import pandas as pd

from app.ai_engine.patterns import detect_patterns, is_bullish_engulfing, is_bearish_engulfing, is_doji


def _df_from_candles(candles):
    rows = []
    for i, (o, h, l, c) in enumerate(candles):
        rows.append({"open": o, "high": h, "low": l, "close": c, "volume": 100})
    return pd.DataFrame(rows)


def test_bullish_engulfing_detected():
    df = _df_from_candles([
        (1.10, 1.101, 1.095, 1.096),   # bearish candle
        (1.094, 1.112, 1.093, 1.111),  # bullish candle fully engulfing the previous body
    ])
    assert is_bullish_engulfing(df) is True


def test_bearish_engulfing_detected():
    df = _df_from_candles([
        (1.096, 1.101, 1.095, 1.10),   # bullish candle
        (1.111, 1.112, 1.093, 1.094),  # bearish candle fully engulfing the previous body
    ])
    assert is_bearish_engulfing(df) is True


def test_doji_detected():
    df = _df_from_candles([
        (1.10, 1.105, 1.095, 1.101),
        (1.100, 1.110, 1.090, 1.1002),  # open ~= close, small body vs range
    ])
    assert is_doji(df) is True


def test_detect_patterns_returns_list_for_short_df():
    df = _df_from_candles([(1.1, 1.11, 1.09, 1.1)])
    assert detect_patterns(df) == []
