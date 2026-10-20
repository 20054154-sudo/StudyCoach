"""Persist elapsed study time across interruptions and resumptions."""

from datetime import datetime, timezone


def _utc(value: datetime) -> datetime:
    # SQLite returns naive datetimes even for timezone-aware columns.
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def elapsed_seconds(session, now: datetime | None = None) -> float:
    elapsed = float(session.duration_seconds or 0)
    if session.active_started_at is not None and session.status == "in_corso":
        current = now or datetime.now(timezone.utc)
        elapsed += max(0.0, (_utc(current) - _utc(session.active_started_at)).total_seconds())
    return elapsed


def pause_clock(session, now: datetime | None = None) -> None:
    current = now or datetime.now(timezone.utc)
    session.duration_seconds = elapsed_seconds(session, current)
    session.active_started_at = None


def resume_clock(session, now: datetime | None = None) -> None:
    if session.active_started_at is None:
        session.active_started_at = now or datetime.now(timezone.utc)
