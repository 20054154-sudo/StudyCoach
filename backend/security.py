"""Security primitives shared by API and UI."""

import re
import secrets
import hashlib
from datetime import datetime, timedelta, timezone

import bcrypt

TOKEN_TTL = timedelta(days=7)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, encoded: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), encoded.encode("ascii"))
    except (ValueError, TypeError):
        return False


def new_token() -> tuple[str, str, datetime]:
    token = secrets.token_urlsafe(32)
    return token, hashlib.sha256(token.encode("utf-8")).hexdigest(), datetime.now(timezone.utc) + TOKEN_TTL


def safe_name(value: str, fallback: str = "file") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value or "").strip("._")
    return cleaned[:120] or fallback
