"""Unit/integration test — backend/session_cookie.py (token persistente)."""

from backend.db.models import SessionLocal, Student
from backend.session_cookie import get_or_create_token, invalidate_token, verify_token


def _create_student(username="mario") -> int:
    db = SessionLocal()
    try:
        student = Student(username=username, password_hash="hash")
        db.add(student)
        db.commit()
        db.refresh(student)
        return student.id
    finally:
        db.close()


def test_get_or_create_token_generates_new_token():
    student_id = _create_student()
    token = get_or_create_token(student_id)
    assert token
    assert len(token) > 20


def test_get_or_create_token_is_stable_across_calls():
    student_id = _create_student()
    token1 = get_or_create_token(student_id)
    token2 = get_or_create_token(student_id, token1)
    assert token1 == token2


def test_get_or_create_token_replaces_invalid_presented_token():
    student_id = _create_student()
    token1 = get_or_create_token(student_id)
    token2 = get_or_create_token(student_id, "wrong-token")
    assert token2 != token1
    assert verify_token(token1) is None
    assert verify_token(token2) is not None


def test_get_or_create_token_returns_none_for_unknown_student():
    assert get_or_create_token(999999) is None


def test_verify_token_returns_matching_student():
    student_id = _create_student("giulia")
    token = get_or_create_token(student_id)

    student = verify_token(token)
    assert student is not None
    assert student.id == student_id
    assert student.username == "giulia"


def test_verify_token_uses_indexed_digest_without_scanning_other_hashes(monkeypatch):
    from backend import session_cookie
    first = _create_student("first")
    second = _create_student("second")
    get_or_create_token(first)
    token = get_or_create_token(second)
    monkeypatch.setattr(session_cookie, "verify_password", lambda *_: (_ for _ in ()).throw(AssertionError("bcrypt scan")))
    assert verify_token(token).id == second


def test_legacy_bcrypt_token_is_upgraded_on_use():
    from datetime import datetime, timedelta, timezone
    from backend.security import hash_password
    student_id = _create_student("legacy")
    token = "legacy-token"
    with SessionLocal() as db:
        student = db.get(Student, student_id)
        student.session_token_hash = hash_password(token)
        student.token_expires_at = datetime.now(timezone.utc) + timedelta(days=1)
        db.commit()
    assert verify_token(token).id == student_id
    with SessionLocal() as db:
        assert len(db.get(Student, student_id).session_token_hash) == 64


def test_verify_token_returns_none_for_invalid_token():
    assert verify_token("token-inesistente") is None


def test_verify_token_returns_none_for_empty_token():
    assert verify_token("") is None


def test_invalidate_token_clears_it():
    student_id = _create_student()
    token = get_or_create_token(student_id)
    assert verify_token(token) is not None

    invalidate_token(student_id)

    assert verify_token(token) is None


def test_invalidate_token_noop_for_falsy_id():
    invalidate_token(None)
