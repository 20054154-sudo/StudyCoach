"""Persistent login: server validation and browser cookie synchronization."""

from datetime import datetime, timezone
import hashlib
import hmac
from typing import Optional

from backend.db.models import SessionLocal, Student
from backend.security import TOKEN_TTL, new_token, verify_password

COOKIE_NAME = "sc_token"
COOKIE_MAX_AGE = int(TOKEN_TTL.total_seconds())


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def get_or_create_token(student_id: int, existing_token: Optional[str] = None) -> Optional[str]:
    """Reuse a presented valid token, or rotate it when none can be presented.

    Only its hash is persisted; callers retain the raw token in the browser
    cookie to reuse it.
    """
    with SessionLocal() as db:
        student = db.query(Student).filter(Student.id == student_id).first()
        if not student:
            return None
        expires = student.token_expires_at
        if expires and expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        stored = student.session_token_hash or ""
        valid = (hmac.compare_digest(_digest(existing_token), stored)
                 if isinstance(existing_token, str) and stored and not stored.startswith("$2") else
                 bool(existing_token and stored and verify_password(existing_token, stored)))
        if (valid and expires and expires > datetime.now(timezone.utc)):
            if stored.startswith("$2"):
                student.session_token_hash = _digest(existing_token)
                db.commit()
            return existing_token
        token, token_hash, expires = new_token()
        student.session_token = None
        student.session_token_hash = token_hash
        student.token_expires_at = expires
        db.commit()
        db.refresh(student)
        return token


def verify_token(token: str) -> Optional[Student]:
    if not isinstance(token, str) or not token:
        return None
    with SessionLocal() as db:
        now = datetime.now(timezone.utc)
        student = db.query(Student).filter(
            Student.session_token_hash == _digest(token), Student.token_expires_at > now,
        ).first()
        if student is not None:
            return student
        # Only legacy bcrypt hashes require a scan. Upgrade on first use.
        legacy = db.query(Student).filter(
            Student.session_token_hash.like("$2%"), Student.token_expires_at > now,
        ).all()
        for student in legacy:
            if verify_password(token, student.session_token_hash):
                student.session_token_hash = _digest(token)
                db.commit()
                db.refresh(student)
                return student
        return None


def invalidate_token(student_id: int) -> None:
    """Explicit logout revokes the credential even if cookie deletion fails."""
    if not student_id:
        return
    with SessionLocal() as db:
        student = db.query(Student).filter(Student.id == student_id).first()
        if student:
            student.session_token = None
            student.session_token_hash = None
            student.token_expires_at = None
            db.commit()
