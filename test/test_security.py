from backend.security import hash_password, safe_name, verify_password


def test_password_hash_is_salted_and_verifiable():
    first = hash_password("segreta")
    second = hash_password("segreta")
    assert first != second
    assert verify_password("segreta", first)
    assert not verify_password("errata", first)


def test_client_filename_is_sanitized():
    sanitized = safe_name("../../<script>.pdf")
    assert "/" not in sanitized and "<" not in sanitized and ">" not in sanitized
