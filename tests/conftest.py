import os

# Test-only environment values so app.config.Settings() can be instantiated
# without requiring real secrets. Set BEFORE any `app.*` module is imported.
os.environ.setdefault("TWELVE_DATA_API_KEY", "test_key")
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test_forex")

import numpy as np
import pandas as pd
import pytest


def _make_ohlc(prices: np.ndarray, start="2024-01-01", freq="h") -> pd.DataFrame:
    n = len(prices)
    idx = pd.date_range(start=start, periods=n, freq=freq)
    highs = prices * 1.0015
    lows = prices * 0.9985
    opens = np.roll(prices, 1)
    opens[0] = prices[0]
    df = pd.DataFrame({
        "datetime": idx,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": np.random.randint(100, 1000, size=n).astype(float),
    })
    return df


@pytest.fixture
def uptrend_df() -> pd.DataFrame:
    """300 candles of a steady uptrend with mild noise — should trigger bullish confluence."""
    rng = np.random.default_rng(42)
    base = np.linspace(1.0500, 1.1200, 300)
    noise = rng.normal(0, 0.0006, 300)
    prices = base + noise
    return _make_ohlc(prices)


@pytest.fixture
def downtrend_df() -> pd.DataFrame:
    """300 candles of a steady downtrend with mild noise — should trigger bearish confluence."""
    rng = np.random.default_rng(7)
    base = np.linspace(1.1200, 1.0500, 300)
    noise = rng.normal(0, 0.0006, 300)
    prices = base + noise
    return _make_ohlc(prices)


@pytest.fixture
def choppy_df() -> pd.DataFrame:
    """300 candles of sideways noise with no clear trend — should be rejected (low confluence)."""
    rng = np.random.default_rng(1)
    prices = 1.0800 + rng.normal(0, 0.0015, 300).cumsum() * 0.02
    prices = np.clip(prices, 1.075, 1.085)
    return _make_ohlc(prices)


@pytest.fixture
def short_df() -> pd.DataFrame:
    """Too few candles — should be rejected as insufficient data."""
    rng = np.random.default_rng(3)
    prices = 1.08 + rng.normal(0, 0.0005, 50)
    return _make_ohlc(prices)
