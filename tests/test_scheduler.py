from unittest.mock import patch, AsyncMock

import pytest

from app.services import scheduler as scheduler_module
from app.config import SUPPORTED_PAIRS


@pytest.mark.asyncio
async def test_cycle_stops_early_on_rate_limit():
    """Once a pair comes back rate_limited, the remaining pairs in that cycle
    should be skipped entirely rather than each triggering their own (futile)
    request against an already-exhausted rate limit."""
    call_order = []

    async def fake_analyze_pair(pair, events):
        call_order.append(pair)
        if pair == SUPPORTED_PAIRS[2]:
            return "rate_limited"
        return "rejected"

    with patch.object(scheduler_module, "fetch_calendar_events", AsyncMock(return_value=[])), \
         patch.object(scheduler_module, "analyze_pair", fake_analyze_pair), \
         patch("asyncio.sleep", AsyncMock()):
        await scheduler_module.analyze_all_pairs()

    # Should have stopped right after the 3rd pair triggered rate_limited —
    # never even attempted pairs 4 through 10.
    assert call_order == SUPPORTED_PAIRS[:3]


@pytest.mark.asyncio
async def test_cycle_completes_all_pairs_when_no_rate_limit():
    call_order = []

    async def fake_analyze_pair(pair, events):
        call_order.append(pair)
        return "published"

    with patch.object(scheduler_module, "fetch_calendar_events", AsyncMock(return_value=[])), \
         patch.object(scheduler_module, "analyze_pair", fake_analyze_pair), \
         patch("asyncio.sleep", AsyncMock()):
        await scheduler_module.analyze_all_pairs()

    assert call_order == SUPPORTED_PAIRS


@pytest.mark.asyncio
async def test_calendar_fetched_exactly_once_per_cycle():
    """The whole point of the fix: fetch_calendar_events must be called ONCE per
    cycle, never once per pair."""
    fetch_mock = AsyncMock(return_value=[])

    async def fake_analyze_pair(pair, events):
        return "rejected"

    with patch.object(scheduler_module, "fetch_calendar_events", fetch_mock), \
         patch.object(scheduler_module, "analyze_pair", fake_analyze_pair), \
         patch("asyncio.sleep", AsyncMock()):
        await scheduler_module.analyze_all_pairs()

    fetch_mock.assert_awaited_once()
