import pandas as pd

from app.ai_engine.indicators import (
    compute_all_indicators, fibonacci_levels, support_resistance, rsi, macd, bollinger_bands, atr, adx,
)


def test_ema_stack_reflects_uptrend(uptrend_df):
    df = compute_all_indicators(uptrend_df)
    last = df.iloc[-1]
    assert last["ema_20"] > last["ema_50"] > last["ema_200"]


def test_ema_stack_reflects_downtrend(downtrend_df):
    df = compute_all_indicators(downtrend_df)
    last = df.iloc[-1]
    assert last["ema_20"] < last["ema_50"] < last["ema_200"]


def test_rsi_bounds(uptrend_df):
    result = rsi(uptrend_df["close"])
    assert result.between(0, 100).all()


def test_rsi_higher_in_uptrend_than_downtrend(uptrend_df, downtrend_df):
    up_rsi = rsi(uptrend_df["close"]).iloc[-1]
    down_rsi = rsi(downtrend_df["close"]).iloc[-1]
    assert up_rsi > down_rsi


def test_macd_returns_three_series(uptrend_df):
    macd_line, signal_line, hist = macd(uptrend_df["close"])
    assert len(macd_line) == len(uptrend_df)
    assert len(signal_line) == len(uptrend_df)
    assert len(hist) == len(uptrend_df)


def test_bollinger_bands_ordering(uptrend_df):
    upper, mid, lower = bollinger_bands(uptrend_df["close"])
    valid = mid.notna()
    assert (upper[valid] >= mid[valid]).all()
    assert (mid[valid] >= lower[valid]).all()


def test_atr_non_negative(uptrend_df):
    result = atr(uptrend_df)
    assert (result.dropna() >= 0).all()


def test_adx_bounds(uptrend_df):
    result = adx(uptrend_df)
    assert result.between(0, 100).all()


def test_fibonacci_levels_ordering(uptrend_df):
    levels = fibonacci_levels(uptrend_df)
    ordered = [levels["0.0"], levels["0.236"], levels["0.382"], levels["0.5"],
               levels["0.618"], levels["0.786"], levels["1.0"]]
    assert ordered == sorted(ordered, reverse=True)


def test_support_resistance_returns_lists(uptrend_df):
    result = support_resistance(uptrend_df)
    assert "support" in result and "resistance" in result
    assert isinstance(result["support"], list)
    assert isinstance(result["resistance"], list)
