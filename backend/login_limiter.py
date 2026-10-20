"""Persistent login attempt limits shared by API and browser login."""

import hashlib
import time

from sqlalchemy import and_, case
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from backend.db.models import LoginAttempt, SessionLocal

WINDOW_SECONDS = 15 * 60
ACCOUNT_LIMIT = 5
ORIGIN_LIMIT = 20


def _keys(username: str, origin: str) -> tuple[str, str]:
    account_origin = f"{username.strip().casefold()}\0{origin}"
    return tuple(hashlib.sha256(value.encode("utf-8")).hexdigest() for value in
                 (f"account-origin:{account_origin}", f"origin:{origin}"))


def retry_after(username: str, origin: str) -> int:
    now = time.time()
    with SessionLocal() as db:
        rows = [db.get(LoginAttempt, key) for key in _keys(username, origin)]
    return max((max(1, int(row.blocked_until - now + 0.999)) for row in rows
                if row is not None and row.blocked_until > now), default=0)


def record_failure(username: str, origin: str) -> None:
    now = time.time()
    with SessionLocal.begin() as db:
        for key, limit in zip(_keys(username, origin), (ACCOUNT_LIMIT, ORIGIN_LIMIT)):
            statement = sqlite_insert(LoginAttempt).values(
                scope_key=key, failures=1, window_started=now, blocked_until=0.0,
            )
            expired = and_(LoginAttempt.window_started <= now - WINDOW_SECONDS,
                           LoginAttempt.blocked_until <= now)
            statement = statement.on_conflict_do_update(
                index_elements=[LoginAttempt.scope_key],
                set_={
                    "failures": case((expired, 1), else_=LoginAttempt.failures + 1),
                    "window_started": case((expired, now), else_=LoginAttempt.window_started),
                    "blocked_until": case(
                        (expired, 0.0),
                        (LoginAttempt.failures >= limit - 1, now + WINDOW_SECONDS),
                        else_=LoginAttempt.blocked_until,
                    ),
                },
            )
            db.execute(statement)


def clear_account(username: str, origin: str) -> None:
    account_key, _ = _keys(username, origin)
    with SessionLocal.begin() as db:
        db.query(LoginAttempt).filter(LoginAttempt.scope_key == account_key).delete()
