from app.ai_engine.smart_money import (
    find_swing_points, detect_break_of_structure, detect_fair_value_gaps,
    detect_order_blocks, detect_liquidity_sweep, detect_supply_demand_zones, full_smc_analysis,
)


def test_swing_points_tagged(uptrend_df):
    tagged = find_swing_points(uptrend_df)
    assert "swing_high" in tagged.columns
    assert "swing_low" in tagged.columns
    assert tagged["swing_high"].any() or tagged["swing_low"].any()


def test_bos_detected_in_uptrend(uptrend_df):
    result = detect_break_of_structure(uptrend_df)
    assert result in (None, "bullish_bos", "bearish_bos")


def test_bos_detected_in_downtrend(downtrend_df):
    result = detect_break_of_structure(downtrend_df)
    assert result in (None, "bullish_bos", "bearish_bos")


def test_fair_value_gaps_shape(uptrend_df):
    gaps = detect_fair_value_gaps(uptrend_df)
    assert isinstance(gaps, list)
    for gap in gaps:
        assert gap["type"] in ("bullish_fvg", "bearish_fvg")
        assert gap["top"] >= gap["bottom"]


def test_order_blocks_shape(uptrend_df):
    blocks = detect_order_blocks(uptrend_df)
    assert isinstance(blocks, list)
    for ob in blocks:
        assert ob["type"] in ("bullish_ob", "bearish_ob")
        assert ob["top"] >= ob["bottom"]


def test_liquidity_sweep_returns_valid_value(uptrend_df):
    result = detect_liquidity_sweep(uptrend_df)
    assert result in (None, "bullish_sweep", "bearish_sweep")


def test_supply_demand_zones_shape(uptrend_df):
    zones = detect_supply_demand_zones(uptrend_df)
    assert "demand" in zones and "supply" in zones


def test_full_smc_analysis_keys(uptrend_df):
    result = full_smc_analysis(uptrend_df)
    for key in ("bos", "choch", "fair_value_gaps", "order_blocks", "liquidity_sweep", "supply_demand"):
        assert key in result
