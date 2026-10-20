"""Unit test — backend/models/schemas.py (validatori Pydantic)."""

import pytest
from pydantic import ValidationError

from backend.models.schemas import (
    Answer1Request,
    ContestRequest,
    ExamCreate,
    LoginRequest,
    QuestionOut,
    QuizGenerateRequest,
    RegisterRequest,
)


def test_register_request_strips_username_whitespace():
    req = RegisterRequest(username="  mario  ", email=" Mario@Example.com ", password="secret123", password_confirm="secret123")
    assert req.username == "mario"
    assert req.email == "mario@example.com"


def test_register_request_rejects_short_username():
    with pytest.raises(ValidationError):
        RegisterRequest(username="ab", email="mario@example.com", password="secret123", password_confirm="secret123")


def test_register_request_rejects_short_password():
    with pytest.raises(ValidationError):
        RegisterRequest(username="mario", email="mario@example.com", password="123", password_confirm="123")


def test_register_request_rejects_invalid_email_and_mismatched_password():
    with pytest.raises(ValidationError):
        RegisterRequest(username="mario", email="invalid", password="secret123", password_confirm="secret123")
    with pytest.raises(ValidationError):
        RegisterRequest(username="mario", email="mario@example.com", password="secret123", password_confirm="different")


def test_exam_create_strips_name():
    req = ExamCreate(name="  Analisi 1  ")
    assert req.name == "Analisi 1"


def test_exam_create_rejects_empty_name():
    with pytest.raises(ValidationError):
        ExamCreate(name="")


def test_login_request_accepts_any_nonempty_credentials():
    req = LoginRequest(username="mario", password="whatever")
    assert req.username == "mario"


def test_answer1_request_rejects_empty_answer():
    with pytest.raises(ValidationError):
        Answer1Request(session_id=1, question_id=1, answer="")


def test_contest_request_requires_min_length_argumentation():
    with pytest.raises(ValidationError):
        ContestRequest(session_id=1, question_id=1, argumentation="corto")


def test_contest_request_accepts_valid_argumentation():
    req = ContestRequest(session_id=1, question_id=1, argumentation="la mia risposta era corretta perche'...")
    assert len(req.argumentation) >= 10


def test_quiz_generate_request_defaults():
    req = QuizGenerateRequest()
    assert req.n_questions == 5
    assert req.topic_query == "concetti principali del corso"


def test_quiz_generate_request_rejects_out_of_range_n_questions():
    with pytest.raises(ValidationError):
        QuizGenerateRequest(n_questions=0)
    with pytest.raises(ValidationError):
        QuizGenerateRequest(n_questions=21)


def test_question_out_roundtrip():
    q = QuestionOut(id=1, text="domanda?", difficulty="base", topic_hint="x")
    assert q.model_dump() == {"id": 1, "text": "domanda?", "difficulty": "base", "topic_hint": "x"}
