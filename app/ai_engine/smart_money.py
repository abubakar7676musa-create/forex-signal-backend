"""
Smart Money Concepts (SMC) detection: Order Blocks, Fair Value Gaps, Liquidity Sweeps,
Break of Structure (BOS), Change of Character (CHoCH), and Supply/Demand zones.

These are heuristic, rule-based approximations of SMC concepts (there is no single
universally-agreed formal definition), built for confluence scoring rather than as a
standalone trading system.
"""
import pandas as pd


def find_swing_points(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Tags each row as a swing high / swing low using a simple fractal rule."""
    df = df.copy()
    df["swing_high"] = False
    df["swing_low"] = False
    for i in range(window, len(df) - window):
        seg = df.iloc[i - window:i + window + 1]
        if df["high"].iloc[i] == seg["high"].max():
            df.loc[df.index[i], "swing_high"] = True
        if df["low"].iloc[i] == seg["low"].min():
            df.loc[df.index[i], "swing_low"] = True
    return df


def detect_break_of_structure(df: pd.DataFrame, window: int = 3) -> str | None:
    """
    BOS: price closes beyond the most recent confirmed swing high (bullish BOS)
    or swing low (bearish BOS), signaling continuation of the prevailing trend.
    """
    tagged = find_swing_points(df, window)
    swing_highs = tagged[tagged["swing_high"]]["high"]
    swing_lows = tagged[tagged["swing_low"]]["low"]
    if swing_highs.empty or swing_lows.empty:
        return None

    last_close = df["close"].iloc[-1]
    last_swing_high = swing_highs.iloc[-1]
    last_swing_low = swing_lows.iloc[-1]

    if last_close > last_swing_high:
        return "bullish_bos"
    if last_close < last_swing_low:
        return "bearish_bos"
    return None


def detect_change_of_character(df: pd.DataFrame, window: int = 3) -> str | None:
    """
    CHoCH: the first break of structure *against* the prevailing trend direction,
    often the earliest signal of a potential trend reversal.
    """
    tagged = find_swing_points(df, window)
    highs = tagged[tagged["swing_high"]]
    lows = tagged[tagged["swing_low"]]
    if len(highs) < 2 or len(lows) < 2:
        return None

    was_uptrend = highs["high"].iloc[-1] > highs["high"].iloc[-2] and lows["low"].iloc[-1] > lows["low"].iloc[-2]
    was_downtrend = highs["high"].iloc[-1] < highs["high"].iloc[-2] and lows["low"].iloc[-1] < lows["low"].iloc[-2]

    last_close = df["close"].iloc[-1]
    if was_uptrend and last_close < lows["low"].iloc[-1]:
        return "bearish_choch"
    if was_downtrend and last_close > highs["high"].iloc[-1]:
        return "bullish_choch"
    return None


def detect_fair_value_gaps(df: pd.DataFrame, lookback: int = 30) -> list[dict]:
    """
    FVG (3-candle imbalance): gap between candle[i-1].high and candle[i+1].low (bullish)
    or candle[i-1].low and candle[i+1].high (bearish), left unfilled by candle[i].
    """
    gaps = []
    window = df.tail(lookback).reset_index(drop=True)
    for i in range(1, len(window) - 1):
        c1, c3 = window.iloc[i - 1], window.iloc[i + 1]
        if c3["low"] > c1["high"]:
            gaps.append({"type": "bullish_fvg", "top": c3["low"], "bottom": c1["high"], "index": i})
        elif c3["high"] < c1["low"]:
            gaps.append({"type": "bearish_fvg", "top": c1["low"], "bottom": c3["high"], "index": i})
    return gaps[-5:]  # most recent 5


def detect_order_blocks(df: pd.DataFrame, lookback: int = 50) -> list[dict]:
    """
    Order Block: the last opposite-colored candle before a strong impulsive move
    that breaks structure — approximates institutional accumulation/distribution.
    """
    window = df.tail(lookback).reset_index(drop=True)
    blocks = []
    avg_range = (window["high"] - window["low"]).mean()

    for i in range(1, len(window) - 1):
        candle = window.iloc[i]
        next_candle = window.iloc[i + 1]
        candle_range = candle["high"] - candle["low"]
        next_range = next_candle["high"] - next_candle["low"]
        is_impulsive = next_range > avg_range * 1.5

        # Bullish OB: last down-candle before a strong up impulse
        if candle["close"] < candle["open"] and next_candle["close"] > next_candle["open"] and is_impulsive:
            blocks.append({
                "type": "bullish_ob", "top": candle["high"], "bottom": candle["low"], "index": i,
            })
        # Bearish OB: last up-candle before a strong down impulse
        if candle["close"] > candle["open"] and next_candle["close"] < next_candle["open"] and is_impulsive:
            blocks.append({
                "type": "bearish_ob", "top": candle["high"], "bottom": candle["low"], "index": i,
            })
    return blocks[-5:]


def detect_liquidity_sweep(df: pd.DataFrame, window: int = 3, lookback: int = 30) -> str | None:
    """
    Liquidity sweep: price briefly wicks beyond a recent swing high/low (grabbing stop-loss
    liquidity) then closes back inside range — a classic stop-hunt / SMC reversal signal.
    """
    tagged = find_swing_points(df.tail(lookback), window)
    swing_highs = tagged[tagged["swing_high"]]["high"]
    swing_lows = tagged[tagged["swing_low"]]["low"]
    if swing_highs.empty or swing_lows.empty:
        return None

    last = df.iloc[-1]
    recent_high = swing_highs.iloc[-1]
    recent_low = swing_lows.iloc[-1]

    if last["high"] > recent_high and last["close"] < recent_high:
        return "bearish_sweep"  # swept buy-side liquidity, closed back down
    if last["low"] < recent_low and last["close"] > recent_low:
        return "bullish_sweep"  # swept sell-side liquidity, closed back up
    return None


def detect_supply_demand_zones(df: pd.DataFrame, lookback: int = 80) -> dict:
    """
    Approximates supply/demand zones using consolidation-before-impulse logic:
    a tight-range cluster of candles immediately followed by a strong directional move.
    """
    window = df.tail(lookback).reset_index(drop=True)
    avg_range = (window["high"] - window["low"]).mean()
    demand_zones, supply_zones = [], []

    for i in range(3, len(window) - 1):
        cluster = window.iloc[i - 3:i]
        cluster_range = cluster["high"].max() - cluster["low"].min()
        next_candle = window.iloc[i]
        if cluster_range < avg_range * 1.2:
            if next_candle["close"] > next_candle["open"] and (next_candle["close"] - next_candle["open"]) > avg_range:
                demand_zones.append({"top": cluster["high"].max(), "bottom": cluster["low"].min()})
            elif next_candle["close"] < next_candle["open"] and (next_candle["open"] - next_candle["close"]) > avg_range:
                supply_zones.append({"top": cluster["high"].max(), "bottom": cluster["low"].min()})

    return {"demand": demand_zones[-3:], "supply": supply_zones[-3:]}


def full_smc_analysis(df: pd.DataFrame) -> dict:
    return {
        "bos": detect_break_of_structure(df),
        "choch": detect_change_of_character(df),
        "fair_value_gaps": detect_fair_value_gaps(df),
        "order_blocks": detect_order_blocks(df),
        "liquidity_sweep": detect_liquidity_sweep(df),
        "supply_demand": detect_supply_demand_zones(df),
    }
