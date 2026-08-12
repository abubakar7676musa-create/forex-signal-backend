from datetime import datetime, timedelta

from app.services.news_calendar import is_pair_in_blackout


def _event(currency, impact, minutes_from_now):
    event_time = datetime.utcnow() + timedelta(minutes=minutes_from_now)
    return {"currency": currency, "impact": impact, "date": event_time.isoformat()}


def test_no_events_never_blacks_out():
    assert is_pair_in_blackout("EUR/USD", []) is False


def test_high_impact_event_within_window_triggers_blackout():
    events = [_event("USD", "High", 10)]
    assert is_pair_in_blackout("EUR/USD", events) is True


def test_high_impact_event_outside_window_does_not_trigger():
    events = [_event("USD", "High", 90)]
    assert is_pair_in_blackout("EUR/USD", events) is False


def test_low_impact_event_never_triggers_even_if_imminent():
    events = [_event("USD", "Low", 1)]
    assert is_pair_in_blackout("EUR/USD", events) is False


def test_event_for_unrelated_currency_does_not_trigger():
    events = [_event("JPY", "High", 5)]
    assert is_pair_in_blackout("EUR/USD", events) is False


def test_event_for_related_currency_triggers():
    # GBP/JPY is exposed to both GBP and JPY
    events = [_event("JPY", "High", 5)]
    assert is_pair_in_blackout("GBP/JPY", events) is True


def test_malformed_event_is_skipped_not_crashed_on():
    events = [{"currency": "USD", "impact": "High", "date": "not-a-real-date"}]
    assert is_pair_in_blackout("EUR/USD", events) is False


def test_missing_fields_are_skipped_gracefully():
    events = [{"impact": "High"}]  # no currency, no date
    assert is_pair_in_blackout("EUR/USD", events) is False
