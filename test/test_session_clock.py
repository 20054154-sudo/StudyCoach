from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from backend.session_clock import elapsed_seconds, pause_clock, resume_clock


def test_clock_accumulates_only_active_time_across_pause_and_resume():
    start = datetime(2026, 10, 13, 10, 0, tzinfo=timezone.utc)
    session = SimpleNamespace(status="in_corso", duration_seconds=0.0,
                              active_started_at=start.replace(tzinfo=None))
    pause_clock(session, start + timedelta(seconds=65))
    session.status = "interrotta"
    assert elapsed_seconds(session, start + timedelta(hours=2)) == 65

    resume_clock(session, start + timedelta(hours=2))
    session.status = "in_corso"
    assert elapsed_seconds(session, start + timedelta(hours=2, seconds=35)) == 100
    pause_clock(session, start + timedelta(hours=2, seconds=35))
    session.status = "completata"
    assert session.duration_seconds == 100
    assert session.active_started_at is None
