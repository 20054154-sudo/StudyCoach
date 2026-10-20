"""Integration test — backend/api/auth.py (via FastAPI TestClient + DB isolato)."""


def test_register_creates_student(api_client):
    resp = api_client.post("/auth/register", json={"username": "mario_rossi", "email": "mario_rossi@example.com", "password": "secret123", "password_confirm": "secret123"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["username"] == "mario_rossi"
    assert isinstance(body["student_id"], int)


def test_register_duplicate_username_conflicts(api_client):
    api_client.post("/auth/register", json={"username": "mario_rossi", "email": "mario_rossi@example.com", "password": "secret123", "password_confirm": "secret123"})
    resp = api_client.post("/auth/register", json={"username": "mario_rossi", "email": "mario_rossi@example.com", "password": "another123", "password_confirm": "another123"})
    assert resp.status_code == 409


def test_register_duplicate_email_conflicts_even_with_different_case(api_client):
    first = api_client.post("/auth/register", json={"username": "first", "email": "first@example.com", "password": "secret123", "password_confirm": "secret123"})
    second = api_client.post("/auth/register", json={"username": "second", "email": "FIRST@EXAMPLE.COM", "password": "secret123", "password_confirm": "secret123"})
    assert first.status_code == 201
    assert second.status_code == 409
    assert "email" in second.json()["detail"]


def test_register_duplicate_email_race_returns_conflict(api_client, monkeypatch):
    from backend.api import auth

    api_client.post("/auth/register", json={"username": "first", "email": "first@example.com", "password": "secret123", "password_confirm": "secret123"})
    monkeypatch.setattr(auth, "_get_student_by_email", lambda _email: None)
    response = api_client.post("/auth/register", json={"username": "second", "email": "FIRST@EXAMPLE.COM", "password": "secret123", "password_confirm": "secret123"})
    assert response.status_code == 409
    assert "email" in response.json()["detail"]


def test_register_requires_email_and_matching_confirmation(api_client):
    missing = api_client.post("/auth/register", json={"username": "first", "password": "secret123", "password_confirm": "secret123"})
    mismatch = api_client.post("/auth/register", json={"username": "first", "email": "first@example.com", "password": "secret123", "password_confirm": "other123"})
    assert missing.status_code == 422
    assert mismatch.status_code == 422


def test_register_rejects_short_password(api_client):
    resp = api_client.post("/auth/register", json={"username": "mario_rossi", "email": "mario_rossi@example.com", "password": "123", "password_confirm": "123"})
    assert resp.status_code == 422


def test_register_rejects_password_over_bcrypt_byte_limit(api_client):
    for password in ("a" * 73, "é" * 37):
        response = api_client.post("/auth/register", json={
            "username": "mario_rossi", "email": "mario@example.com",
            "password": password, "password_confirm": password,
        })
        assert response.status_code == 422
        assert "72 byte" in str(response.json())


def test_register_rejects_short_username(api_client):
    resp = api_client.post("/auth/register", json={"username": "ab", "email": "ab@example.com", "password": "secret123", "password_confirm": "secret123"})
    assert resp.status_code == 422


def test_login_with_correct_credentials(api_client):
    api_client.post("/auth/register", json={"username": "mario_rossi", "email": "mario_rossi@example.com", "password": "secret123", "password_confirm": "secret123"})
    resp = api_client.post("/auth/login", json={"username": "mario_rossi", "email": "mario_rossi@example.com", "password": "secret123", "password_confirm": "secret123"})
    assert resp.status_code == 200
    assert resp.json()["username"] == "mario_rossi"
    assert "httponly" in resp.headers["set-cookie"].lower()


def test_login_with_wrong_password_unauthorized(api_client):
    api_client.post("/auth/register", json={"username": "mario_rossi", "email": "mario_rossi@example.com", "password": "secret123", "password_confirm": "secret123"})
    resp = api_client.post("/auth/login", json={"username": "mario_rossi", "email": "mario_rossi@example.com", "password": "wrongpass", "password_confirm": "wrongpass"})
    assert resp.status_code == 401


def test_login_locks_repeated_failures_and_reports_retry_time(api_client):
    api_client.post("/auth/register", json={"username": "mario_rossi", "email": "mario@example.com",
                                           "password": "secret123", "password_confirm": "secret123"})
    for _ in range(5):
        failed = api_client.post("/auth/login", json={"username": "mario_rossi", "password": "wrongpass"})
        assert failed.status_code == 401
    blocked = api_client.post("/auth/login", json={"username": "mario_rossi", "password": "secret123"})
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) > 0


def test_successful_login_clears_account_failures(api_client):
    api_client.post("/auth/register", json={"username": "mario_rossi", "email": "mario@example.com",
                                           "password": "secret123", "password_confirm": "secret123"})
    for _ in range(4):
        assert api_client.post("/auth/login", json={"username": "mario_rossi", "password": "wrongpass"}).status_code == 401
    assert api_client.post("/auth/login", json={"username": "mario_rossi", "password": "secret123"}).status_code == 200
    assert api_client.post("/auth/login", json={"username": "mario_rossi", "password": "wrongpass"}).status_code == 401


def test_login_limits_origin_across_usernames(api_client):
    api_client.post("/auth/register", json={"username": "mario_rossi", "email": "mario@example.com",
                                           "password": "secret123", "password_confirm": "secret123"})
    for number in range(20):
        response = api_client.post("/auth/login", json={"username": f"unknown_{number}", "password": "wrongpass"})
        assert response.status_code == 401
    blocked = api_client.post("/auth/login", json={"username": "mario_rossi", "password": "secret123"})
    assert blocked.status_code == 429


def test_login_lock_expires(api_client):
    import time
    from backend.db.models import LoginAttempt, SessionLocal

    api_client.post("/auth/register", json={"username": "mario_rossi", "email": "mario@example.com",
                                           "password": "secret123", "password_confirm": "secret123"})
    for _ in range(5):
        api_client.post("/auth/login", json={"username": "mario_rossi", "password": "wrongpass"})
    with SessionLocal.begin() as db:
        for row in db.query(LoginAttempt):
            row.window_started = time.time() - 1800
            row.blocked_until = time.time() - 1
    assert api_client.post("/auth/login", json={"username": "mario_rossi", "password": "secret123"}).status_code == 200


def test_login_unknown_username_unauthorized(api_client):
    resp = api_client.post("/auth/login", json={"username": "nessuno", "email": "nessuno@example.com", "password": "whatever", "password_confirm": "whatever"})
    assert resp.status_code == 401


def test_logout_returns_message(api_client):
    resp = api_client.post("/auth/logout")
    assert resp.status_code == 200
    assert "message" in resp.json()


def test_logout_revokes_api_cookie(api_client):
    api_client.post("/auth/register", json={"username": "mario_rossi", "email": "mario_rossi@example.com", "password": "secret123", "password_confirm": "secret123"})
    assert api_client.get("/exams", params={"student_id": 1}).status_code == 200
    api_client.post("/auth/logout")
    assert api_client.get("/exams", params={"student_id": 1}).status_code == 401


def test_register_duplicate_username_after_precheck_returns_conflict(api_client, monkeypatch):
    from backend.api import auth

    first = api_client.post("/auth/register", json={"username": "same_user", "email": "same_user@example.com", "password": "secret123", "password_confirm": "secret123"})
    assert first.status_code == 201
    monkeypatch.setattr(auth, "_get_student_by_username", lambda username: None)
    second = api_client.post("/auth/register", json={"username": "same_user", "email": "same_user@example.com", "password": "secret123", "password_confirm": "secret123"})
    assert second.status_code == 409
