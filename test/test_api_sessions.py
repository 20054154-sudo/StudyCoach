"""
Integration test — backend/api/sessions.py (via FastAPI TestClient).
"""

import json

import pytest

from backend.agents import quiz_generator as qg_mod
from backend.agents import socratic_tutor as st_mod
from backend.api.sessions import _engine_from_session
from backend.db.models import Session as DbSession, SessionLocal

QUIZ_JSON = json.dumps({
    "source_name": "Analisi 1",
    "questions": [
        {"id": 1, "text": "Cos'e' il machine learning?", "difficulty": "base", "topic_hint": "ml", "source_quote": "Il machine learning e' una branca"},
        {"id": 2, "text": "Come apprendono dai dati gli algoritmi descritti?", "difficulty": "intermedia", "topic_hint": "algoritmi", "source_quote": "algoritmi in grado di apprendere dai dati"},
    ],
})

GUIDE_JSON = json.dumps({
    "guiding_question": "Cosa intendi con 'apprendere dai dati'?",
    "positive_feedback": "Buon punto di partenza.",
    "focus_area": "definizione formale",
})

EVAL_JSON = json.dumps({
    "score": 7,
    "motivation": "Buona comprensione con qualche imprecisione.",
    "correct_points": ["Definizione generale corretta"],
    "missing_points": ["Manca un esempio pratico"],
    "improvement_after_guidance": "Migliorata rispetto al primo tentativo.",
})

CONTEST_JSON = json.dumps({
    "final_score": 8,
    "score_changed": True,
    "score_delta": 1,
    "contestation_assessment": "parzialmente_valido",
    "final_motivation": "L'argomentazione aggiunge un punto valido.",
    "response_to_student": "Accolgo parzialmente.",
})


def _setup_exam_with_material(api_client, sample_pdf_path) -> tuple[int, int]:
    student_id = api_client.post(
        "/auth/register", json={"username": "mario_rossi", "email": "mario_rossi@example.com", "password": "secret123", "password_confirm": "secret123"}
    ).json()["student_id"]
    exam_id = api_client.post(
        "/exams", params={"student_id": student_id}, json={"name": "Analisi 1"}
    ).json()["id"]
    with open(sample_pdf_path, "rb") as f:
        api_client.post(
            f"/exams/{exam_id}/ingest",
            params={"student_id": student_id},
            files={"file": (sample_pdf_path.name, f, "application/pdf")},
        )
    return student_id, exam_id


def test_create_session_initializes_adaptive_engine(api_client, sample_pdf_path):
    student_id, exam_id = _setup_exam_with_material(api_client, sample_pdf_path)
    resp = api_client.post("/sessions", params={"student_id": student_id}, json={"exam_id": exam_id})
    assert resp.status_code == 201
    body = resp.json()
    assert body["exam_id"] == exam_id
    assert body["ended_at"] is None


def test_session_owner_required_for_reads_and_writes(api_client):
    from backend.db.models import Exam

    owner_id = api_client.post("/auth/register", json={"username": "owner_user", "email": "owner_user@example.com", "password": "secret123", "password_confirm": "secret123"}).json()["student_id"]
    with SessionLocal() as db:
        exam = Exam(student_id=owner_id, name="Privato")
        db.add(exam)
        db.commit()
        session = DbSession(exam_id=exam.id, student_id=owner_id, status="in_corso")
        db.add(session)
        db.commit()
        exam_id, session_id = exam.id, session.id

    other_id = api_client.post("/auth/register", json={"username": "other_user", "email": "other_user@example.com", "password": "secret123", "password_confirm": "secret123"}).json()["student_id"]
    assert api_client.get(f"/sessions/{session_id}").status_code == 404
    assert api_client.get(f"/sessions/{session_id}/resume").status_code == 404
    assert api_client.get("/sessions", params={"exam_id": exam_id}).status_code == 404
    assert api_client.post("/sessions", params={"student_id": other_id}, json={"exam_id": exam_id}).status_code == 404
    assert api_client.post(f"/sessions/{session_id}/close", json={"session_id": session_id}).status_code == 404


def test_generate_session_quiz_returns_questions(api_client, sample_pdf_path, monkeypatch, fake_agent_factory):
    student_id, exam_id = _setup_exam_with_material(api_client, sample_pdf_path)
    session_id = api_client.post(
        "/sessions", params={"student_id": student_id}, json={"exam_id": exam_id}
    ).json()["id"]

    monkeypatch.setattr(qg_mod, "_make_quiz_agent", fake_agent_factory(QUIZ_JSON))

    resp = api_client.post(
        f"/sessions/{session_id}/quiz",
        params={"student_id": student_id},
        json={"topic_query": "concetti principali", "n_questions": 2, "source_name": "Analisi 1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_questions"] == 2
    assert body["questions"][0]["text"] == "Cos'e' il machine learning?"


def test_quiz_with_fewer_questions_than_requested_is_not_saved(api_client, monkeypatch):
    from backend.agents.quiz_generator import Question, QuizResult
    from backend.api import sessions as session_api
    from backend.db.models import Exam

    student_id = api_client.post(
        "/auth/register", json={"username": "partial_quiz", "email": "partial_quiz@example.com", "password": "secret123", "password_confirm": "secret123"}
    ).json()["student_id"]
    with SessionLocal() as db:
        exam = Exam(student_id=student_id, name="Letteratura")
        db.add(exam)
        db.flush()
        session = DbSession(exam_id=exam.id, student_id=student_id, status="in_corso")
        db.add(session)
        db.commit()
        session_id, exam_id = session.id, exam.id

    monkeypatch.setattr(session_api, "verify_exam_material", lambda *args, **kwargs: {"ok": True, "errors": []})
    monkeypatch.setattr(session_api, "generate_quiz", lambda **kwargs: QuizResult(
        questions=[Question(id=1, text="Prima domanda?", difficulty="base", topic_hint="tema"),
                   Question(id=2, text="Seconda domanda?", difficulty="intermedia", topic_hint="tema")],
        source_name="Letteratura", n_questions=2, prompt_version="test", exam_id=exam_id,
        student_id=student_id,
    ))
    response = api_client.post(
        f"/sessions/{session_id}/quiz", params={"student_id": student_id},
        json={"n_questions": 3},
    )
    assert response.status_code == 422
    assert "2 domande verificabili su 3 richieste" in response.json()["detail"]
    with SessionLocal() as db:
        assert db.get(DbSession, session_id).quiz_json is None


@pytest.mark.parametrize("stage", ["answer1", "answer2"])
def test_tutor_keeps_submitted_text_when_model_fails(api_client, monkeypatch, stage):
    from backend.api import sessions as session_api
    from backend.db.models import Exam

    student_id = api_client.post(
        "/auth/register", json={"username": f"save_{stage}", "email": f"save_{stage}@example.com", "password": "secret123", "password_confirm": "secret123"}
    ).json()["student_id"]
    state = "WAITING_ANSWER_1" if stage == "answer1" else "WAITING_ANSWER_2"
    turn = {"question_id": 1, "question_text": "Che cosa dice la fonte?", "difficulty": "base",
            "topic_hint": "tema", "state": state, "student_answer_1": "Prima risposta",
            "guiding_question": "Quale passo lo dimostra?"}
    with SessionLocal() as db:
        exam = Exam(student_id=student_id, name="Letteratura")
        db.add(exam)
        db.flush()
        session = DbSession(exam_id=exam.id, student_id=student_id, status="in_corso",
                            quiz_json=json.dumps({"questions": [{"id": 1, "text": turn["question_text"],
                                                                    "difficulty": "base", "topic_hint": "tema"}]}),
                            current_turn_json=json.dumps(turn))
        db.add(session)
        db.commit()
        session_id = session.id

    monkeypatch.setattr(session_api, "search_passages", lambda *args, **kwargs: [])

    def unavailable(*args, **kwargs):
        raise ValueError("model unavailable")

    monkeypatch.setattr(st_mod.SocraticTutor, f"submit_answer_{1 if stage == 'answer1' else 2}", unavailable)
    response = api_client.post(f"/sessions/{session_id}/tutor/{stage}", json={
        "session_id": session_id, "question_id": 1, "answer": "Risposta da conservare",
    })
    assert response.status_code == 500
    key = "student_answer_1" if stage == "answer1" else "student_answer_2"
    with SessionLocal() as db:
        saved = json.loads(db.get(DbSession, session_id).current_turn_json)
    assert saved[key] == "Risposta da conservare"
    assert saved["state"] == state
    assert "Risposta da conservare</textarea>" in api_client.get(f"/web/study/{session_id}").text


def test_unverifiable_quiz_returns_actionable_error(api_client, sample_pdf_path, monkeypatch):
    from backend.api import sessions as session_api

    student_id, exam_id = _setup_exam_with_material(api_client, sample_pdf_path)
    session_id = api_client.post(
        "/sessions", params={"student_id": student_id}, json={"exam_id": exam_id}
    ).json()["id"]
    def fail_generation(**_kwargs):
        raise qg_mod.UnverifiableQuizError("Citazioni non verificabili")

    monkeypatch.setattr(session_api, "generate_quiz", fail_generation)
    response = api_client.post(
        f"/sessions/{session_id}/quiz", params={"student_id": student_id},
        json={"n_questions": 2},
    )
    assert response.status_code == 422
    assert "citazioni" in response.json()["detail"].lower()


def test_quiz_rate_limit_returns_http_429(api_client, sample_pdf_path, monkeypatch):
    from backend.api import sessions as session_api
    from backend.agents.llm_errors import RetryableLLMError

    student_id, exam_id = _setup_exam_with_material(api_client, sample_pdf_path)
    session_id = api_client.post(
        "/sessions", params={"student_id": student_id}, json={"exam_id": exam_id}
    ).json()["id"]

    def fail_generation(**_kwargs):
        error = RetryableLLMError("Quota giornaliera di token Groq insufficiente per questa richiesta.")
        error.status_code = 429
        raise error

    monkeypatch.setattr(session_api, "generate_quiz", fail_generation)
    response = api_client.post(
        f"/sessions/{session_id}/quiz", params={"student_id": student_id}, json={"n_questions": 2},
    )
    assert response.status_code == 429
    assert "quota giornaliera" in response.json()["detail"].lower()
    assert "citazioni" not in response.json()["detail"].lower()


def test_generate_session_quiz_limits_and_records_selected_materials(
    api_client, sample_pdf_path, monkeypatch, fake_agent_factory,
):
    from backend.db.models import Material

    student_id, exam_id = _setup_exam_with_material(api_client, sample_pdf_path)
    with SessionLocal() as db:
        material_id = db.query(Material).filter(Material.exam_id == exam_id).one().id
    session_id = api_client.post(
        "/sessions", params={"student_id": student_id}, json={"exam_id": exam_id}
    ).json()["id"]
    monkeypatch.setattr(qg_mod, "_make_quiz_agent", fake_agent_factory(QUIZ_JSON))

    response = api_client.post(
        f"/sessions/{session_id}/quiz", params={"student_id": student_id},
        json={"n_questions": 2, "material_ids": [material_id]},
    )
    assert response.status_code == 200
    with SessionLocal() as db:
        assert json.loads(db.get(DbSession, session_id).quiz_json)["material_ids"] == [material_id]

    invalid = api_client.post(
        f"/sessions/{session_id}/quiz", params={"student_id": student_id},
        json={"material_ids": [999999]},
    )
    assert invalid.status_code == 409


def test_generate_quiz_for_unknown_session_returns_404(api_client):
    student_id = api_client.post("/auth/register", json={"username": "missing_user", "email": "missing_user@example.com", "password": "secret123", "password_confirm": "secret123"}).json()["student_id"]
    resp = api_client.post("/sessions/999999/quiz", params={"student_id": student_id}, json={})
    assert resp.status_code == 404


def _full_tutor_loop(api_client, session_id, student_id, question, monkeypatch, fake_agent_factory):
    monkeypatch.setattr(st_mod, "_guide_agent", fake_agent_factory(GUIDE_JSON))
    monkeypatch.setattr(st_mod, "_eval_agent", fake_agent_factory(EVAL_JSON))
    monkeypatch.setattr(st_mod, "_contest_agent", fake_agent_factory(CONTEST_JSON))

    resp = api_client.post(
        f"/sessions/{session_id}/tutor/start",
        params={"student_id": student_id},
        json={"session_id": session_id, "question": question},
    )
    assert resp.status_code == 200
    assert resp.json()["question_text"] == question["text"]

    resp = api_client.post(
        f"/sessions/{session_id}/tutor/answer1",
        json={"session_id": session_id, "question_id": question["id"], "answer": "forse riguarda i dati"},
    )
    assert resp.status_code == 200
    assert resp.json()["guiding_question"] == "Cosa intendi con 'apprendere dai dati'?"

    resp = api_client.post(
        f"/sessions/{session_id}/tutor/answer2",
        json={"session_id": session_id, "question_id": question["id"], "answer": "e' un modello che apprende pattern"},
    )
    assert resp.status_code == 200
    eval_body = resp.json()
    assert eval_body["score"] == 7
    assert eval_body["reliable"] is True
    assert eval_body["can_contest"] is True

    resp = api_client.post(
        f"/sessions/{session_id}/tutor/contest",
        json={"session_id": session_id, "question_id": question["id"], "argumentation": "meritavo di piu' perche'..."},
    )
    assert resp.status_code == 200
    assert resp.json()["final_score"] == 8

    resp = api_client.post(
        f"/sessions/{session_id}/tutor/close",
        json={"session_id": session_id, "question_id": question["id"]},
    )
    assert resp.status_code == 200
    close_body = resp.json()
    assert close_body["final_score"] == 8
    assert close_body["topic_hint"] == question["topic_hint"]
    with SessionLocal() as db:
        saved = db.get(DbSession, session_id)
        assert saved.current_turn_json is None
        assert len(json.loads(saved.qa_log)) == 1
    return close_body


def test_full_session_flow_quiz_tutor_report(api_client, sample_pdf_path, monkeypatch, fake_agent_factory):
    student_id, exam_id = _setup_exam_with_material(api_client, sample_pdf_path)
    session_id = api_client.post(
        "/sessions", params={"student_id": student_id}, json={"exam_id": exam_id}
    ).json()["id"]

    monkeypatch.setattr(qg_mod, "_make_quiz_agent", fake_agent_factory(QUIZ_JSON))
    quiz = api_client.post(
        f"/sessions/{session_id}/quiz",
        params={"student_id": student_id},
        json={"n_questions": 2},
    ).json()

    first_question = quiz["questions"][0]
    _full_tutor_loop(api_client, session_id, student_id, first_question, monkeypatch, fake_agent_factory)

    # Prossima domanda suggerita dall'AdaptiveEngine tra quelle rimanenti.
    remaining = quiz["questions"][1:]
    resp = api_client.post(f"/sessions/{session_id}/next-question", json={
        "session_id": session_id,
        "remaining_questions": remaining,
    })
    assert resp.status_code == 200
    next_body = resp.json()
    assert next_body["question"]["id"] == remaining[0]["id"]
    assert next_body["is_last"] is True

    # Chiusura sessione + report finale.
    resp = api_client.post(f"/sessions/{session_id}/close", json={
        "session_id": session_id,
        "exam_name": "Analisi 1",
    })
    assert resp.status_code == 200
    report = resp.json()
    assert report["session_id"] == session_id
    assert report["n_questions_answered"] == 1
    assert report["overall_avg"] == 8.0
    assert report["report_md_path"] is not None
    assert report["report_pdf_path"] is not None

    # Dettaglio e lista sessioni riflettono lo stato finale.
    resp = api_client.get(f"/sessions/{session_id}")
    assert resp.status_code == 200
    assert resp.json()["score_total"] == 8.0

    with SessionLocal() as db:
        before = db.query(DbSession).filter(DbSession.exam_id == exam_id).count()
    resp = api_client.get("/sessions", params={"exam_id": exam_id})
    assert resp.status_code == 200
    assert resp.json()["total"] == 1
    with SessionLocal() as db:
        after = db.query(DbSession).filter(DbSession.exam_id == exam_id).count()
    assert after == before
    assert resp.json()["sessions"][0]["status"] == "completata"


def test_tutor_answer1_without_start_returns_404(api_client):
    api_client.post("/auth/register", json={"username": "missing_user", "email": "missing_user@example.com", "password": "secret123", "password_confirm": "secret123"})
    resp = api_client.post(
        "/sessions/1/tutor/answer1",
        json={"session_id": 1, "question_id": 1, "answer": "qualcosa"},
    )
    assert resp.status_code == 404


@pytest.mark.parametrize("session_status", ["interrotta", "completata"])
def test_tutor_start_requires_active_session(api_client, session_status):
    from backend.db.models import Exam

    student_id = api_client.post(
        "/auth/register", json={"username": "inactive_user", "email": "inactive@example.com",
                                "password": "secret123", "password_confirm": "secret123"},
    ).json()["student_id"]
    question = {"id": 1, "text": "Domanda?", "difficulty": "base", "topic_hint": "tema"}
    with SessionLocal() as db:
        exam = Exam(student_id=student_id, name="Fisica")
        db.add(exam)
        db.flush()
        session = DbSession(exam_id=exam.id, student_id=student_id, status=session_status,
                            quiz_json=json.dumps({"questions": [question]}))
        db.add(session)
        db.commit()
        session_id = session.id

    response = api_client.post(
        f"/sessions/{session_id}/tutor/start", params={"student_id": student_id},
        json={"session_id": session_id, "question": question},
    )
    assert response.status_code == 409
    with SessionLocal() as db:
        assert db.get(DbSession, session_id).current_turn_json is None


def test_tutor_answer2_before_answer1_returns_409(api_client, sample_pdf_path, monkeypatch, fake_agent_factory):
    student_id, exam_id = _setup_exam_with_material(api_client, sample_pdf_path)
    session_id = api_client.post(
        "/sessions", params={"student_id": student_id}, json={"exam_id": exam_id}
    ).json()["id"]

    monkeypatch.setattr(st_mod, "_guide_agent", fake_agent_factory(GUIDE_JSON))
    monkeypatch.setattr(st_mod, "_eval_agent", fake_agent_factory(EVAL_JSON))

    question = {"id": 1, "text": "Domanda?", "difficulty": "base", "topic_hint": "x"}
    with SessionLocal() as db:
        db.get(DbSession, session_id).quiz_json = json.dumps({"questions": [question]})
        db.commit()
    api_client.post(
        f"/sessions/{session_id}/tutor/start",
        params={"student_id": student_id},
        json={"session_id": session_id, "question": question},
    )

    resp = api_client.post(
        f"/sessions/{session_id}/tutor/answer2",
        json={"session_id": session_id, "question_id": 1, "answer": "risposta prematura"},
    )
    assert resp.status_code == 409


def test_tutor_rejects_forged_question_and_next_ignores_forged_remaining(api_client, sample_pdf_path):
    student_id, exam_id = _setup_exam_with_material(api_client, sample_pdf_path)
    session_id = api_client.post("/sessions", params={"student_id": student_id},
                                 json={"exam_id": exam_id}).json()["id"]
    canonical = {"id": 1, "text": "Domanda dal quiz", "difficulty": "base", "topic_hint": "tema"}
    forged = {**canonical, "text": "Domanda inventata dal client"}
    with SessionLocal() as db:
        db.get(DbSession, session_id).quiz_json = json.dumps({"questions": [canonical]})
        db.commit()
    response = api_client.post(f"/sessions/{session_id}/tutor/start",
                               params={"student_id": student_id},
                               json={"session_id": session_id, "question": forged})
    assert response.status_code == 422
    with SessionLocal() as db:
        assert db.get(DbSession, session_id).current_turn_json is None
    selected = api_client.post(f"/sessions/{session_id}/next-question", json={
        "session_id": session_id, "remaining_questions": [forged],
    })
    assert selected.status_code == 200
    assert selected.json()["question"]["text"] == canonical["text"]


def test_turn_close_commit_failure_preserves_log_and_active_turn(api_client, monkeypatch):
    from backend.api import sessions as session_api
    from backend.db.models import Exam
    student_id = api_client.post("/auth/register", json={"username": "atomic", "email": "atomic@example.com", "password": "secret123", "password_confirm": "secret123"}).json()["student_id"]
    with SessionLocal() as db:
        exam = Exam(student_id=student_id, name="Fisica")
        db.add(exam)
        db.flush()
        session = DbSession(exam_id=exam.id, student_id=student_id, status="in_corso",
                            qa_log="[]", current_turn_json='{"question_id": 1}')
        db.add(session)
        db.commit()
        session_id = session.id
    original_factory = session_api.SessionLocal

    def failing_factory():
        db = original_factory()
        db.commit = lambda: (_ for _ in ()).throw(OSError("disk failure"))
        return db

    monkeypatch.setattr(session_api, "SessionLocal", failing_factory)
    with pytest.raises(RuntimeError):
        session_api._append_attempt_to_log(session_id, {"question_id": 1, "final_score": 7})
    with original_factory() as db:
        saved = db.get(DbSession, session_id)
        assert saved.qa_log == "[]"
        assert saved.current_turn_json == '{"question_id": 1}'


def test_close_session_without_engine_returns_404(api_client):
    api_client.post("/auth/register", json={"username": "missing_user", "email": "missing_user@example.com", "password": "secret123", "password_confirm": "secret123"})
    resp = api_client.post("/sessions/999999/close", json={"session_id": 999999})
    assert resp.status_code == 404


def test_close_session_without_answers_does_not_store_zero_score(api_client):
    from backend.db.models import Exam

    student_id = api_client.post("/auth/register", json={
        "username": "empty_user", "email": "empty@example.com",
        "password": "secret123", "password_confirm": "secret123",
    }).json()["student_id"]
    with SessionLocal() as db:
        exam = Exam(student_id=student_id, name="Esame")
        db.add(exam)
        db.flush()
        session = DbSession(exam_id=exam.id, student_id=student_id, status="in_corso", qa_log="[]")
        db.add(session)
        db.commit()
        exam_id, session_id = exam.id, session.id

    response = api_client.post(f"/sessions/{session_id}/close", json={
        "session_id": session_id, "exam_name": "Esame",
    })
    assert response.status_code == 409
    with SessionLocal() as db:
        stored = db.get(DbSession, session_id)
        assert stored.score_total is None
        assert stored.status == "in_corso"
    exams = api_client.get("/exams", params={"student_id": student_id}).json()["exams"]
    assert next(exam for exam in exams if exam["id"] == exam_id)["avg_score"] is None


def test_next_question_without_engine_returns_404(api_client):
    api_client.post("/auth/register", json={"username": "missing_user", "email": "missing_user@example.com", "password": "secret123", "password_confirm": "secret123"})
    resp = api_client.post("/sessions/999999/next-question", json={
        "session_id": 999999,
        "remaining_questions": [],
    })
    assert resp.status_code == 404


def test_get_unknown_session_returns_404(api_client):
    api_client.post("/auth/register", json={"username": "missing_user", "email": "missing_user@example.com", "password": "secret123", "password_confirm": "secret123"})
    resp = api_client.get("/sessions/999999")
    assert resp.status_code == 404


def test_close_session_db_failure_removes_new_reports(api_client, sample_pdf_path, isolated_reports_dir, monkeypatch):
    from types import SimpleNamespace
    from backend.api import sessions as sessions_api

    student_id, exam_id = _setup_exam_with_material(api_client, sample_pdf_path)
    session_id = api_client.post(
        "/sessions", params={"student_id": student_id}, json={"exam_id": exam_id}
    ).json()["id"]
    report = {
        "student_id": student_id, "exam_id": exam_id, "overall_avg": 7.0,
        "n_questions_answered": 1, "weak_topics": [], "strong_topics": [],
        "topic_breakdown": [], "summary": "Sessione conclusa.",
    }
    monkeypatch.setattr(sessions_api, "_engine_from_session", lambda sess, **_kwargs: SimpleNamespace(generate_session_report=lambda: report))
    original_session_local = sessions_api.SessionLocal

    def failing_session():
        db = original_session_local()
        def fail_commit():
            raise OSError("SECRET database")
        monkeypatch.setattr(db, "commit", fail_commit)
        return db

    monkeypatch.setattr(sessions_api, "SessionLocal", failing_session)
    response = api_client.post(
        f"/sessions/{session_id}/close", json={"session_id": session_id, "exam_name": "Analisi"}
    )
    assert response.status_code == 500
    assert "SECRET" not in response.text
    assert not (isolated_reports_dir / f"{student_id}_{exam_id}_{session_id}.md").exists()
    assert not (isolated_reports_dir / f"{student_id}_{exam_id}_{session_id}.pdf").exists()


def test_adaptive_engine_rebuilt_from_persisted_qa_log(isolated_db):
    db = SessionLocal()
    sess = DbSession(exam_id=7, student_id=42, qa_log=json.dumps([{
        "question_id": 3, "topic_hint": "limiti",
        "evaluation": {"score": 4},
    }]))
    db.add(sess)
    db.commit()
    db.refresh(sess)
    rebuilt = _engine_from_session(sess)
    assert rebuilt.stats.student_id == 42
    assert rebuilt.stats.topic_stats["limiti"].scores == [4]
    db.close()


def test_unreliable_double_judgment_is_excluded_from_adaptive_engine():
    from types import SimpleNamespace

    sess = SimpleNamespace(student_id=42, exam_id=7, qa_log=json.dumps([{
        "question_id": 3, "topic_hint": "limiti",
        "evaluation": {"score": 1, "reliable": False},
    }]))
    rebuilt = _engine_from_session(sess)
    assert rebuilt.stats.answered_question_ids == []
    report_engine = _engine_from_session(sess, include_unreliable=True)
    assert report_engine.stats.overall_avg == 1
