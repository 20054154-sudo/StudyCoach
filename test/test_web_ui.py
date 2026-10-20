"""Browser form and page smoke tests for the Python served interface."""

import base64
import json
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from fastapi import HTTPException

from backend.api import exams as exams_api, frontend_extras, session_extras, sessions as sessions_api
from backend.db.models import Material, Session as DbSession, SessionLocal
from backend.models.schemas import NextQuestionResponse, QuestionOut


def _register(client):
    response = client.post("/web/register", data={"username": "webuser", "email": "webuser@example.com", "password": "secret123", "password_confirm": "secret123"}, follow_redirects=False)
    assert response.status_code == 303
    return client.get("/auth/me").json()["student_id"]


def test_study_rate_limit_is_shown_as_quota_error(api_client, monkeypatch):
    _register(api_client)
    exam_id = int(api_client.post("/web/exams", data={"name": "Fisica"}, follow_redirects=False)
                  .headers["location"].rsplit("/", 1)[1])
    monkeypatch.setattr(frontend_extras, "_selected_materials", lambda _exam_id, ids: [SimpleNamespace(id=7)] if ids is not None else [])
    monkeypatch.setattr(sessions_api, "create_session", lambda *_args: SimpleNamespace(id=123))
    monkeypatch.setattr(sessions_api, "generate_session_quiz", lambda *_args: (
        _ for _ in ()).throw(HTTPException(429, "Quota giornaliera di token Groq insufficiente per questa richiesta."))
    )
    deleted = []
    monkeypatch.setattr(session_extras, "delete_incomplete_session", deleted.append)

    response = api_client.post(
        f"/web/exams/{exam_id}/study", data={"material_ids": "7", "n_questions": "3"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert "Quota giornaliera di token Groq" in response.text
    assert "citazioni verificabili" not in response.text
    assert deleted == [123]


def test_study_stream_reports_generation_failure(api_client, monkeypatch):
    _register(api_client)
    exam_id = int(api_client.post("/web/exams", data={"name": "Fisica"}, follow_redirects=False)
                  .headers["location"].rsplit("/", 1)[1])
    monkeypatch.setattr(frontend_extras, "_selected_materials", lambda *_args: [SimpleNamespace(id=7)])
    monkeypatch.setattr(sessions_api, "create_session", lambda *_args: SimpleNamespace(id=123))
    monkeypatch.setattr(sessions_api, "generate_session_quiz", lambda *_args: (
        _ for _ in ()).throw(HTTPException(429, "Quota del modello esaurita."))
    )
    deleted = []
    monkeypatch.setattr(session_extras, "delete_incomplete_session", deleted.append)
    response = api_client.post(f"/web/exams/{exam_id}/study",
                               data={"material_ids": "7", "n_questions": "3"},
                               headers={"Accept": "application/x-ndjson"})
    events = [json.loads(line) for line in response.text.splitlines()]
    assert [event["type"] for event in events] == ["progress"] * 3 + ["error"]
    assert "Genero" in events[2]["message"]
    assert events[-1]["message"] == "Quota del modello esaurita."
    assert events[-1]["code"] == "api_limit"
    assert deleted == [123]


def test_root_login_library_exam_and_logout(api_client):
    assert api_client.get("/", follow_redirects=False).headers["location"] == "/web/login"
    assert api_client.get("/favicon.ico", follow_redirects=False).headers["location"] == "/static/logo.svg"
    assert api_client.get("/static/logo.svg").status_code == 200
    assert api_client.get("/web/login").status_code == 200
    login_html = api_client.get("/web/login").text
    register_html = api_client.get("/web/register").text
    assert 'action="/web/login"' in login_html and 'action="/web/register"' not in login_html
    assert 'action="/web/register"' in register_html and 'action="/web/login"' not in register_html
    assert 'name="email"' in register_html and 'name="password_confirm"' in register_html
    assert api_client.get("/web/exams", follow_redirects=False).status_code == 303
    student_id = _register(api_client)
    assert api_client.get("/", follow_redirects=False).headers["location"] == "/web/exams"
    created = api_client.post("/web/exams", data={"name": "Fisica"}, follow_redirects=False)
    assert created.status_code == 303
    exam_url = created.headers["location"]
    exam_id = int(exam_url.rsplit("/", 1)[1])
    assert "Fisica" in api_client.get(exam_url).text
    assert "Fisica" in api_client.get("/web/exams").text
    assert api_client.get(f"/web/history/{exam_id}").status_code == 200
    assert api_client.get(f"/web/chat/{exam_id}").status_code == 200
    assert api_client.post(f"/web/exams/{exam_id}/rename", data={"name": "Fisica II"}, follow_redirects=False).status_code == 303
    assert "Fisica II" in api_client.get(exam_url).text
    assert api_client.post("/web/logout", follow_redirects=False).status_code == 303
    assert api_client.get("/web/exams", follow_redirects=False).status_code == 303
    assert api_client.post("/web/login", data={"username": "webuser", "password": "secret123"}, follow_redirects=False).status_code == 303
    assert api_client.get("/auth/me").json()["student_id"] == student_id


def test_chat_uses_current_operation_script(api_client):
    from pathlib import Path
    from frontend import web

    _register(api_client)
    exam_id = int(api_client.post("/web/exams", data={"name": "Fisica"}, follow_redirects=False)
                  .headers["location"].rsplit("/", 1)[1])
    version = (Path(web.__file__).parent / "static" / "operation-progress.js").stat().st_mtime_ns
    html = api_client.get(f"/web/chat/{exam_id}").text
    assert f"/static/operation-progress.js?v={version}" in html
    script = api_client.get(f"/static/operation-progress.js?v={version}")
    assert script.status_code == 200
    assert 'fetch(form.getAttribute("action")' in script.text


def test_web_register_rejects_mismatch_and_duplicate_email(api_client):
    mismatch = api_client.post("/web/register", data={"username": "first", "email": "first@example.com",
                                                      "password": "secret123", "password_confirm": "other123"},
                               follow_redirects=False)
    assert mismatch.status_code == 303 and "password" in mismatch.headers["location"].lower()
    first = api_client.post("/web/register", data={"username": "first", "email": "first@example.com",
                                                   "password": "secret123", "password_confirm": "secret123"},
                            follow_redirects=False)
    assert first.status_code == 303
    api_client.post("/web/logout")
    duplicate = api_client.post("/web/register", data={"username": "second", "email": "FIRST@EXAMPLE.COM",
                                                       "password": "secret123", "password_confirm": "secret123"},
                                follow_redirects=False)
    assert duplicate.status_code == 303 and "email" in duplicate.headers["location"].lower()


def test_source_selection_and_study_history_pages(api_client, monkeypatch, tmp_path):
    student_id = _register(api_client)
    exam_id = int(api_client.post("/web/exams", data={"name": "Fisica"}, follow_redirects=False).headers["location"].rsplit("/", 1)[1])
    pdf = tmp_path / "fonte.pdf"
    pdf.write_bytes(b"%PDF-test")
    with SessionLocal() as db:
        material = Material(exam_id=exam_id, filename="fonte.pdf", stored_path=str(pdf), pages_count=1, chunks_count=1)
        db.add(material)
        db.commit()
        material_id = material.id
    monkeypatch.setattr(frontend_extras, "verify_exam_material", lambda *args, **kwargs: {"ok": True, "errors": []})
    seen = []
    monkeypatch.setattr(frontend_extras, "generate_notes_pdf_from_materials", lambda *args, **kwargs: seen.extend(m.id for m in args[2]) or b"%PDF-notes")
    page = api_client.get(f"/web/exams/{exam_id}")
    assert page.status_code == 200 and "fonte.pdf" in page.text
    notes = api_client.post(f"/web/exams/{exam_id}/sources", data={"material_ids": str(material_id), "action": "notes"})
    assert notes.content == b"%PDF-notes" and seen == [material_id]
    setup = api_client.post(f"/web/exams/{exam_id}/sources", data={"material_ids": str(material_id), "action": "study"})
    assert setup.status_code == 200 and "Prepara la sessione" in setup.text
    with SessionLocal() as db:
        session = DbSession(exam_id=exam_id, student_id=student_id, quiz_json=json.dumps({"questions": [{"id": 1, "text": "Forza?", "difficulty": "base", "topic_hint": "fisica"}], "material_ids": [material_id]}), status="in_corso")
        db.add(session)
        db.commit()
        session_id = session.id
    assert "Continua" in api_client.get(f"/web/study/{session_id}").text
    assert "Sessione #" in api_client.get(f"/web/history/{exam_id}").text
    assert api_client.post(f"/web/history/{exam_id}/{session_id}/resume", follow_redirects=False).headers["location"] == f"/web/study/{session_id}"
    assert api_client.post(f"/web/study/{session_id}/leave", follow_redirects=False).headers["location"] == f"/web/history/{exam_id}"
    assert "Sessione interrotta" in api_client.get(f"/web/study/{session_id}").text


def test_notes_stream_shows_real_progress_and_selected_source(api_client, monkeypatch, tmp_path):
    _register(api_client)
    exam_id = int(api_client.post("/web/exams", data={"name": "Letteratura"}, follow_redirects=False)
                  .headers["location"].rsplit("/", 1)[1])
    selected_ids = []
    with SessionLocal() as db:
        for filename in ("scelta.pdf", "esclusa.pdf"):
            path = tmp_path / filename
            path.write_bytes(b"%PDF-test")
            item = Material(exam_id=exam_id, filename=filename, stored_path=str(path), chunks_count=1)
            db.add(item)
            db.flush()
            selected_ids.append(item.id)
        db.commit()

    monkeypatch.setattr(frontend_extras, "verify_exam_material", lambda *args, **kwargs: {"ok": True, "errors": []})
    observed = []

    def make_pdf(_student_id, _exam_id, materials, *, progress, **_kwargs):
        observed.extend(material.filename for material in materials)
        progress({"phase": "reading", "source": materials[0].filename, "source_index": 1, "source_total": 1})
        progress({"phase": "reviewing", "source": materials[0].filename, "part": 1, "parts": 2})
        return b"%PDF-progress", "# Fonte: scelta.pdf\n\nAppunti di prova\n"

    monkeypatch.setattr(frontend_extras, "generate_notes_pdf_from_materials", make_pdf)
    response = api_client.post(
        f"/web/exams/{exam_id}/sources",
        data={"material_ids": str(selected_ids[0]), "action": "notes"},
        headers={"Accept": "application/x-ndjson"},
    )
    events = [json.loads(line) for line in response.text.splitlines()]
    assert response.status_code == 200 and response.headers["content-type"].startswith("application/x-ndjson")
    assert observed == ["scelta.pdf"]
    assert [event["phase"] for event in events[:-1]] == ["starting", "reading", "reviewing"]
    assert base64.b64decode(events[-1]["pdf_base64"]) == b"%PDF-progress"
    assert "scelta.pdf" in events[-1]["markdown"]
    assert "notes-progress.js" in api_client.get(f"/web/exams/{exam_id}").text


def test_notes_stream_reports_review_failure_without_pdf(api_client, monkeypatch, tmp_path):
    from backend.agents.llm_errors import NotesReviewUnavailable

    _register(api_client)
    exam_id = int(api_client.post("/web/exams", data={"name": "Letteratura"}, follow_redirects=False)
                  .headers["location"].rsplit("/", 1)[1])
    path = tmp_path / "fonte.pdf"
    path.write_bytes(b"%PDF-test")
    with SessionLocal() as db:
        item = Material(exam_id=exam_id, filename="fonte.pdf", stored_path=str(path), chunks_count=1)
        db.add(item)
        db.commit()
        material_id = item.id
    monkeypatch.setattr(frontend_extras, "verify_exam_material", lambda *args, **kwargs: {"ok": True, "errors": []})

    def fail_pdf(*_args, **_kwargs):
        raise NotesReviewUnavailable("Il modello non ha completato la verifica degli appunti. Riprova la generazione.")

    monkeypatch.setattr(frontend_extras, "generate_notes_pdf_from_materials", fail_pdf)
    response = api_client.post(
        f"/web/exams/{exam_id}/sources",
        data={"material_ids": str(material_id), "action": "notes"},
        headers={"Accept": "application/x-ndjson"},
    )
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[-1]["type"] == "error"
    assert "verifica" in events[-1]["message"]
    assert all("pdf_base64" not in event for event in events)


def test_browser_origin_and_ownership(api_client):
    _register(api_client)
    blocked = api_client.post("/web/exams", data={"name": "Cross site"}, headers={"Origin": "https://evil.example"})
    assert blocked.status_code == 403
    assert api_client.get("/web/exams/999").status_code == 404
    assert api_client.get("/web/study/999").status_code == 404


def test_pdf_upload_form_passes_file_to_existing_ingestor(api_client, monkeypatch):
    student_id = _register(api_client)
    exam_id = int(api_client.post("/web/exams", data={"name": "Fisica"}, follow_redirects=False).headers["location"].rsplit("/", 1)[1])
    received = []

    async def fake_ingest(id_value, owner_id, file):
        received.append((id_value, owner_id, file.filename, await file.read()))

    monkeypatch.setattr(exams_api, "ingest_pdf_endpoint", fake_ingest)
    response = api_client.post(f"/web/exams/{exam_id}/upload", files={"files": ("fonte.pdf", b"%PDF-1.4 test", "application/pdf")}, follow_redirects=False)
    assert response.status_code == 303
    assert received == [(exam_id, student_id, "fonte.pdf", b"%PDF-1.4 test")]


def test_multi_upload_reports_success_and_failure(api_client, monkeypatch):
    _register(api_client)
    exam_id = int(api_client.post("/web/exams", data={"name": "Fisica"}, follow_redirects=False).headers["location"].rsplit("/", 1)[1])

    async def fake_ingest(_exam_id, _student_id, file):
        if file.filename == "bad.pdf":
            from fastapi import HTTPException
            raise HTTPException(422, "PDF non utilizzabile")

    monkeypatch.setattr(exams_api, "ingest_pdf_endpoint", fake_ingest)
    response = api_client.post(f"/web/exams/{exam_id}/upload", files=[
        ("files", ("good.pdf", b"%PDF-ok", "application/pdf")),
        ("files", ("bad.pdf", b"%PDF-bad", "application/pdf")),
    ])
    assert response.status_code == 200
    assert "1 PDF caricati" in response.text
    assert "bad.pdf: PDF non utilizzabile" in response.text


def test_upload_stream_reports_each_file_and_destination(api_client, monkeypatch):
    _register(api_client)
    exam_id = int(api_client.post("/web/exams", data={"name": "Fisica"}, follow_redirects=False)
                  .headers["location"].rsplit("/", 1)[1])

    async def fake_ingest(_exam_id, _student_id, file):
        await file.read()

    monkeypatch.setattr(exams_api, "ingest_pdf_endpoint", fake_ingest)
    response = api_client.post(f"/web/exams/{exam_id}/upload", files=[
        ("files", ("uno.pdf", b"%PDF-one", "application/pdf")),
        ("files", ("due.pdf", b"%PDF-two", "application/pdf")),
    ], headers={"Accept": "application/x-ndjson"})
    events = [json.loads(line) for line in response.text.splitlines()]
    assert response.status_code == 200
    assert [event["type"] for event in events] == ["progress"] * 4 + ["complete"]
    assert "uno.pdf" in events[0]["message"] and "due.pdf" in events[2]["message"]
    assert events[-1]["url"].startswith(f"/web/exams/{exam_id}?ok=")


def test_exam_remains_accessible_when_material_verification_fails(api_client, monkeypatch, tmp_path):
    _register(api_client)
    exam_id = int(api_client.post("/web/exams", data={"name": "Fisica"}, follow_redirects=False).headers["location"].rsplit("/", 1)[1])
    pdf = tmp_path / "fonte.pdf"
    pdf.write_bytes(b"%PDF-test")
    with SessionLocal() as db:
        material = Material(exam_id=exam_id, filename="fonte.pdf", stored_path=str(pdf), pages_count=1)
        db.add(material)
        db.commit()
        material_id = material.id

    def unavailable(*args, **kwargs):
        raise RuntimeError("internal path must not appear in the UI")

    monkeypatch.setattr(frontend_extras, "verify_exam_material", unavailable)
    page = api_client.get(f"/web/exams/{exam_id}")
    assert page.status_code == 200
    assert "fonte.pdf" in page.text
    assert "Verifica delle fonti non disponibile" in page.text
    assert "internal path" not in page.text
    action = api_client.post(f"/web/exams/{exam_id}/sources", data={"material_ids": str(material_id), "action": "study"})
    assert "Verifica delle fonti non disponibile" in action.text


def test_study_starts_first_question_and_completes_after_last_answer(api_client, monkeypatch, tmp_path):
    student_id = _register(api_client)
    exam_id = int(api_client.post("/web/exams", data={"name": "Fisica"}, follow_redirects=False).headers["location"].rsplit("/", 1)[1])
    pdf = tmp_path / "fonte.pdf"
    pdf.write_bytes(b"%PDF-test")
    with SessionLocal() as db:
        material = Material(exam_id=exam_id, filename="fonte.pdf", stored_path=str(pdf), pages_count=1, chunks_count=1)
        db.add(material)
        db.commit()
        material_id = material.id
    monkeypatch.setattr(sessions_api, "verify_exam_material", lambda *args, **kwargs: {"ok": True, "errors": []})
    question = {"id": 1, "text": "Che cos'è la forza?", "difficulty": "base", "topic_hint": "fisica"}

    def fake_quiz(session_id, payload, student_id):
        assert payload.material_ids == [material_id]
        with SessionLocal() as db:
            db.get(DbSession, session_id).quiz_json = json.dumps({"questions": [question], "material_ids": [material_id]})
            db.commit()

    def fake_tutor_start(session_id, payload, student_id):
        with SessionLocal() as db:
            db.get(DbSession, session_id).current_turn_json = json.dumps({
                "question_id": 1, "question_text": question["text"], "difficulty": "base",
                "topic_hint": "fisica", "state": "WAITING_ANSWER_1",
            })
            db.commit()

    monkeypatch.setattr(sessions_api, "generate_session_quiz", fake_quiz)
    monkeypatch.setattr(sessions_api, "next_question", lambda *args: NextQuestionResponse(question=QuestionOut(**question)))
    monkeypatch.setattr(sessions_api, "tutor_start", fake_tutor_start)
    started = api_client.post(f"/web/exams/{exam_id}/study", data={"material_ids": str(material_id), "n_questions": "3"}, follow_redirects=False)
    assert started.status_code == 303
    study_url = started.headers["location"]
    assert "Che cos&#39;è la forza?" in api_client.get(study_url).text

    def fake_close_question(session_id, payload):
        with SessionLocal() as db:
            session = db.get(DbSession, session_id)
            session.current_turn_json = None
            session.qa_log = json.dumps([{"question_id": 1, "question_text": question["text"], "final_score": 8}])
            db.commit()

    completed = []

    def fake_close_session(session_id, payload):
        completed.append(session_id)
        with SessionLocal() as db:
            session = db.get(DbSession, session_id)
            session.status = "completata"
            session.score_total = 8
            db.commit()

    monkeypatch.setattr(sessions_api, "tutor_close", fake_close_question)
    monkeypatch.setattr(sessions_api, "close_session", fake_close_session)
    closed = api_client.post(f"{study_url}/close", follow_redirects=False)
    assert closed.status_code == 303
    assert completed == [int(study_url.rsplit("/", 1)[1])]
    assert "Sessione completata" in api_client.get(study_url).text


def test_study_counts_unreliable_scores(api_client):
    student_id = _register(api_client)
    exam_id = int(api_client.post("/web/exams", data={"name": "Letteratura"}, follow_redirects=False)
                  .headers["location"].rsplit("/", 1)[1])
    attempts = [
        {"question_id": number, "question_text": f"Domanda {number}?", "final_score": score,
         "reliable": reliable, "evaluation": {"reliable": reliable,
         "motivation": "Risposta completa" if reliable else
         "Valutazione non affidabile: voto alto incoerente con molte lacune"}}
        for number, score, reliable in ((1, 8, False), (2, 8, False), (3, 10, True))
    ]
    with SessionLocal() as db:
        session = DbSession(exam_id=exam_id, student_id=student_id, status="completata",
                            score_total=8.7, quiz_json=json.dumps({"questions": [
                                {"id": number, "text": f"Domanda {number}?"} for number in (1, 2, 3)]}),
                            qa_log=json.dumps(attempts))
        db.add(session)
        db.commit()
        session_id = session.id
    page = api_client.get(f"/web/study/{session_id}")
    assert page.status_code == 200
    assert "Media calcolata su 3 valutazioni su 3" in page.text
    assert "8.7<small>/10</small>" in page.text
    assert page.text.count("giudizio incoerente") >= 2
    assert "voto alto incoerente con molte lacune" not in page.text


def test_interrupted_session_shows_saved_duration(api_client):
    student_id = _register(api_client)
    exam_id = int(api_client.post("/web/exams", data={"name": "Fisica"}, follow_redirects=False)
                  .headers["location"].rsplit("/", 1)[1])
    with SessionLocal() as db:
        session = DbSession(exam_id=exam_id, student_id=student_id, status="interrotta",
                            duration_seconds=90, quiz_json=json.dumps({"questions": []}))
        db.add(session)
        db.commit()
        session_id = session.id
    page = api_client.get(f"/web/study/{session_id}")
    assert page.status_code == 200
    assert 'data-running="false"' in page.text
    assert "00:01:30" in page.text
    history = api_client.get(f"/web/history/{exam_id}")
    assert "Durata 00:01:30" in history.text


def test_live_timer_checkpoint_is_saved_to_db(api_client):
    student_id = _register(api_client)
    exam_id = int(api_client.post("/web/exams", data={"name": "Fisica"}, follow_redirects=False)
                  .headers["location"].rsplit("/", 1)[1])
    with SessionLocal() as db:
        session = DbSession(exam_id=exam_id, student_id=student_id, status="in_corso",
                            active_started_at=datetime.now(timezone.utc) - timedelta(seconds=75),
                            duration_seconds=0)
        db.add(session)
        db.commit()
        session_id = session.id
    response = api_client.post(f"/sessions/{session_id}/timer")
    assert response.status_code == 200
    assert 74 <= response.json()["duration_seconds"] < 90
    with SessionLocal() as db:
        saved = db.get(DbSession, session_id)
        assert saved.duration_seconds >= 74
        assert saved.active_started_at is not None


def test_evaluated_second_question_can_continue_to_third(api_client, monkeypatch):
    student_id = _register(api_client)
    exam_id = int(api_client.post("/web/exams", data={"name": "Letteratura"}, follow_redirects=False)
                  .headers["location"].rsplit("/", 1)[1])
    questions = [{"id": i, "text": f"Domanda {i}?", "difficulty": "base", "topic_hint": "tema"}
                 for i in (1, 2, 3)]
    turn = {"question_id": 2, "question_text": "Domanda 2?", "difficulty": "base",
            "topic_hint": "tema", "state": "EVALUATED", "student_answer_1": "Prima risposta",
            "student_answer_2": "Seconda risposta", "guiding_question": "Perché?",
            "evaluation": {"score": 0, "motivation": "Risposta non sufficiente.",
                           "correct_points": [], "missing_points": ["Un punto da approfondire"],
                           "improvement_after_guidance": "Nessuno"}}
    with SessionLocal() as db:
        session = DbSession(exam_id=exam_id, student_id=student_id, status="in_corso",
                            quiz_json=json.dumps({"questions": questions, "n_questions": 3}),
                            qa_log=json.dumps([{"question_id": 1, "question_text": "Domanda 1?",
                                                "final_score": 0}]),
                            current_turn_json=json.dumps(turn))
        db.add(session)
        db.commit()
        session_id = session.id
    page = api_client.get(f"/web/study/{session_id}")
    assert page.status_code == 200
    assert "DOMANDA 2 DI 3" in page.text
    assert "Prossima domanda →" in page.text
    assert "registrare la risposta e continuare" in page.text

    def fake_close(_session_id, _payload):
        with SessionLocal() as db:
            stored = db.get(DbSession, session_id)
            attempts = json.loads(stored.qa_log)
            attempts.append({"question_id": 2, "question_text": "Domanda 2?", "final_score": 0})
            stored.qa_log = json.dumps(attempts)
            stored.current_turn_json = None
            db.commit()

    def fake_start(_session_id, payload, _student_id):
        with SessionLocal() as db:
            stored = db.get(DbSession, session_id)
            stored.current_turn_json = json.dumps({"question_id": 3, "question_text": payload.question.text,
                                                   "difficulty": "base", "topic_hint": "tema",
                                                   "state": "WAITING_ANSWER_1"})
            db.commit()

    monkeypatch.setattr(sessions_api, "tutor_close", fake_close)
    monkeypatch.setattr(sessions_api, "next_question", lambda *args: NextQuestionResponse(
        question=QuestionOut(**questions[2])))
    monkeypatch.setattr(sessions_api, "tutor_start", fake_start)
    response = api_client.post(f"/web/study/{session_id}/close", follow_redirects=True)
    assert response.status_code == 200
    assert "DOMANDA 3 DI 3" in response.text
    assert "Domanda 3?" in response.text


def test_history_notes_render_inline_and_escape_html(api_client, monkeypatch):
    student_id = _register(api_client)
    exam_id = int(api_client.post("/web/exams", data={"name": "Fisica"}, follow_redirects=False).headers["location"].rsplit("/", 1)[1])
    with SessionLocal() as db:
        session = DbSession(exam_id=exam_id, student_id=student_id,
                            qa_log=json.dumps([{"question_id": 1, "question_text": "Forza?", "final_score": 8}]))
        db.add(session)
        db.commit()
        session_id = session.id
    monkeypatch.setattr(session_extras, "generate_notes_from_log", lambda *args, **kwargs: "# Ripasso\n<script>alert(1)</script>")
    page = api_client.post(f"/web/history/{exam_id}/{session_id}/notes")
    assert page.status_code == 200
    assert "<h1>Ripasso</h1>" in page.text
    assert "&lt;script&gt;" in page.text
    assert "<script>" not in page.text


def test_chat_transcript_persists(api_client, monkeypatch):
    _register(api_client)
    exam_id = int(api_client.post("/web/exams", data={"name": "Fisica"}, follow_redirects=False).headers["location"].rsplit("/", 1)[1])

    class FakeDoubtTutor:
        n_chunks = 1

        def __init__(self, **kwargs):
            self.attempt = None

        def start(self, doubt):
            return {"guiding_question": "Quale legge?", "topic_hint": "fisica"}

        def submit_answer(self, answer):
            return {"correct_answer": "F = ma"}

    monkeypatch.setattr(frontend_extras, "DoubtTutor", FakeDoubtTutor)
    monkeypatch.setattr(frontend_extras, "search_passages", lambda *args, **kwargs: [{"text": "F = ma"}])
    url = f"/web/chat/{exam_id}"
    assert api_client.post(url, data={"action": "start", "text": "La forza?"}, follow_redirects=False).status_code == 303
    assert "Quale legge?" in api_client.get(url).text
    assert api_client.post(url, data={"action": "start", "text": "La forza?"}, follow_redirects=False).status_code == 303
    with SessionLocal() as db:
        from backend.db.models import DoubtConversation
        conversation_id = db.query(DoubtConversation).one().id
    assert api_client.get(url).text.count("Quale legge?") == 1
    assert api_client.post(url, data={"action": "answer", "conversation_id": conversation_id, "text": "Una legge"}, follow_redirects=False).status_code == 303
    assert "F = ma" in api_client.get(url).text
    created = api_client.post(url, data={"action": "new"}, follow_redirects=False)
    assert created.status_code == 303
    assert "Quale legge?" not in api_client.get(created.headers["location"]).text
    assert "La forza?" in api_client.get(url + "/history").text


def test_chat_archive_reopen_and_delete_only_selected_thread(api_client, monkeypatch):
    student_id = _register(api_client)
    exam_id = int(api_client.post("/web/exams", data={"name": "Fisica"}, follow_redirects=False)
                  .headers["location"].rsplit("/", 1)[1])
    from backend.db.models import DoubtConversation, DoubtThread

    with SessionLocal() as db:
        first = DoubtThread(id=str(uuid.uuid4()), student_id=student_id, exam_id=exam_id, title="Meccanica")
        second = DoubtThread(id=str(uuid.uuid4()), student_id=student_id, exam_id=exam_id, title="Ottica")
        db.add_all([first, second])
        db.flush()
        db.add_all([
            DoubtConversation(id=str(uuid.uuid4()), student_id=student_id, exam_id=exam_id,
                              thread_id=first.id, doubt_text="Che cos'è la forza?", guiding_question="Quale legge?"),
            DoubtConversation(id=str(uuid.uuid4()), student_id=student_id, exam_id=exam_id,
                              thread_id=second.id, doubt_text="Che cos'è la luce?", guiding_question="Come si propaga?"),
        ])
        db.commit()
        first_id, second_id = first.id, second.id

    url = f"/web/chat/{exam_id}"
    board = api_client.get(url + "/history")
    assert board.status_code == 200 and "Meccanica" in board.text and "Ottica" in board.text
    first_page = api_client.get(f"{url}?thread={first_id}")
    assert "Che cos&#39;è la forza?" in first_page.text
    assert "Come si propaga?" not in first_page.text
    assert api_client.post(url, data={"action": "delete", "thread_id": first_id},
                           follow_redirects=False).status_code == 303
    assert "Meccanica" in api_client.get(url + "/history").text
    assert api_client.post(url, data={"action": "delete", "thread_id": first_id,
                                      "confirm_delete": "yes"}, follow_redirects=False).status_code == 303
    assert api_client.get(f"{url}?thread={first_id}").status_code == 404
    assert "Come si propaga?" in api_client.get(f"{url}?thread={second_id}").text
    with SessionLocal() as db:
        assert db.query(DoubtConversation).filter_by(thread_id=first_id).count() == 0
        assert db.query(DoubtConversation).filter_by(thread_id=second_id).count() == 1
    api_client.post("/web/register", data={"username": "secondo", "email": "secondo@example.com", "password": "secret123", "password_confirm": "secret123"},
                    follow_redirects=False)
    assert api_client.get(f"{url}?thread={second_id}").status_code == 404
    assert api_client.get(url + "/history").status_code == 404
    assert api_client.post(url, data={"action": "delete", "thread_id": second_id,
                                      "confirm_delete": "yes"}, follow_redirects=False).status_code == 404


def test_existing_duplicate_chat_is_displayed_once_and_does_not_block_new_doubts(api_client, monkeypatch):
    student_id = _register(api_client)
    exam_id = int(api_client.post("/web/exams", data={"name": "Fisica"}, follow_redirects=False)
                  .headers["location"].rsplit("/", 1)[1])

    class FakeDoubtTutor:
        n_chunks = 1

        def __init__(self, **kwargs):
            self.attempt = None

        def start(self, doubt):
            return {"guiding_question": "Quale legge?", "topic_hint": "fisica"}

        def submit_answer(self, answer):
            return {"correct_answer": "F = ma"}

    monkeypatch.setattr(frontend_extras, "DoubtTutor", FakeDoubtTutor)
    monkeypatch.setattr(frontend_extras, "search_passages", lambda *args, **kwargs: [{"text": "F = ma"}])
    url = f"/web/chat/{exam_id}"
    assert api_client.post(url, data={"action": "start", "text": "La forza?"}, follow_redirects=False).status_code == 303
    with SessionLocal() as db:
        from backend.db.models import DoubtConversation
        latest = DoubtConversation(id=str(uuid.uuid4()), student_id=student_id, exam_id=exam_id,
                                   thread_id=db.query(DoubtConversation).first().thread_id,
                                   doubt_text="La forza?", guiding_question="Quale principio?",
                                   topic_hint="fisica", answered=0)
        db.add(latest)
        db.commit()
        latest_id = latest.id
    page = api_client.get(url).text
    assert page.count("Quale principio?") == 1
    assert "Quale principio?" in page and "Quale legge?" not in page
    assert api_client.post(url, data={"action": "answer", "conversation_id": latest_id,
                                      "text": "Una legge"}, follow_redirects=False).status_code == 303
    assert api_client.post(url, data={"action": "start", "text": "Un nuovo dubbio"},
                           follow_redirects=False).status_code == 303
    assert "Un nuovo dubbio" in api_client.get(url).text
