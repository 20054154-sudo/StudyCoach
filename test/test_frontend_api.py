"""Regression coverage for the HTTP flows replacing Streamlit's Python calls."""

import json

from backend.api import frontend_extras, session_extras, sessions as sessions_api
from backend.db.models import DoubtConversation, Exam, Material, Session as DbSession, SessionLocal


def _account_and_exam(client, username="studente"):
    student_id = client.post("/auth/register", json={"username": username, "email": f"{username}@example.com", "password": "secret123", "password_confirm": "secret123"}).json()["student_id"]
    exam_id = client.post("/exams", params={"student_id": student_id}, json={"name": "Fisica"}).json()["id"]
    return student_id, exam_id


def test_auth_cookie_exam_materials_and_rename(api_client):
    student_id, exam_id = _account_and_exam(api_client)
    assert api_client.get("/auth/me").json()["student_id"] == student_id

    with SessionLocal() as db:
        db.add(Material(exam_id=exam_id, filename="dispensa.pdf", stored_path="missing.pdf",
                        pages_count=4, pages_with_text=3, pages_skipped=1, chunks_count=2))
        db.commit()

    materials = api_client.get(f"/exams/{exam_id}/materials", params={"student_id": student_id})
    assert materials.status_code == 200
    assert materials.json()["materials"][0]["pages_with_text"] == 3

    renamed = api_client.patch(f"/exams/{exam_id}", params={"student_id": student_id}, json={"name": "Fisica II"})
    assert renamed.status_code == 200
    assert renamed.json()["name"] == "Fisica II"

    check = api_client.post(f"/exams/{exam_id}/verify", params={"student_id": student_id}, json={"material_ids": []})
    assert check.status_code == 200
    assert check.json()["ok"] is True

    api_client.post("/auth/logout")
    assert api_client.get("/auth/me").status_code == 401
    assert api_client.get(f"/exams/{exam_id}/materials", params={"student_id": student_id}).status_code == 401


def test_doubt_turn_persists_between_requests_and_is_owned(api_client, monkeypatch):
    student_id, exam_id = _account_and_exam(api_client)
    starts = []

    class FakeDoubtTutor:
        n_chunks = 5

        def __init__(self, student_id, exam_id):
            self.student_id = student_id
            self.exam_id = exam_id
            self.attempt = None

        def start(self, doubt):
            starts.append(doubt)
            return {"guiding_question": "Qual è il primo principio?", "topic_hint": "meccanica"}

        def submit_answer(self, answer):
            assert self.attempt.doubt_text == "Non capisco la forza"
            assert self.attempt.guiding_question == "Qual è il primo principio?"
            return {"correct_points": [answer], "missing_points": [], "correct_answer": "F = ma"}

    monkeypatch.setattr(frontend_extras, "DoubtTutor", FakeDoubtTutor)
    monkeypatch.setattr(frontend_extras, "search_passages", lambda *args, **kwargs: [{"text": "F = ma"}])

    started = api_client.post(f"/exams/{exam_id}/doubts", params={"student_id": student_id}, json={"doubt": "Non capisco la forza"})
    assert started.status_code == 200
    conversation_id = started.json()["conversation_id"]
    duplicate = api_client.post(f"/exams/{exam_id}/doubts", params={"student_id": student_id}, json={"doubt": "Non capisco la forza"})
    assert duplicate.status_code == 200
    assert duplicate.json()["conversation_id"] == conversation_id
    assert starts == ["Non capisco la forza"]
    different = api_client.post(f"/exams/{exam_id}/doubts", params={"student_id": student_id}, json={"doubt": "Un altro dubbio"})
    assert different.status_code == 409
    with SessionLocal() as db:
        assert db.get(DoubtConversation, conversation_id).student_id == student_id

    other_id = api_client.post("/auth/register", json={"username": "altro", "email": "altro@example.com", "password": "secret123", "password_confirm": "secret123"}).json()["student_id"]
    assert api_client.post(f"/exams/{exam_id}/doubts/{conversation_id}/answer", params={"student_id": other_id}, json={"answer": "La massa"}).status_code == 404

    api_client.post("/auth/login", json={"username": "studente", "email": "studente@example.com", "password": "secret123", "password_confirm": "secret123"})
    answered = api_client.post(f"/exams/{exam_id}/doubts/{conversation_id}/answer", params={"student_id": student_id}, json={"answer": "La massa"})
    assert answered.status_code == 200
    assert answered.json()["correct_answer"] == "F = ma"
    assert api_client.post(f"/exams/{exam_id}/doubts/{conversation_id}/answer", params={"student_id": student_id}, json={"answer": "Ancora"}).status_code == 404

    another = api_client.post(f"/exams/{exam_id}/doubts", params={"student_id": student_id}, json={"doubt": "Non capisco la forza"}).json()["conversation_id"]
    assert api_client.delete(f"/exams/{exam_id}/doubts/{another}", params={"student_id": student_id}).status_code == 204
    with SessionLocal() as db:
        assert db.get(DoubtConversation, another) is None

    assert api_client.delete(f"/exams/{exam_id}", params={"student_id": student_id}).status_code == 204
    with SessionLocal() as db:
        assert db.get(DoubtConversation, conversation_id) is None


def test_session_detail_interrupt_resume_notes_and_delete(api_client, monkeypatch):
    student_id, exam_id = _account_and_exam(api_client)
    quiz = {"questions": [{"id": 1, "text": "Che cos'è la forza?", "difficulty": "base", "topic_hint": "forza"}], "material_ids": [7]}
    attempt = {"question_id": 1, "question_text": "Che cos'è la forza?", "student_answer_1": "Una spinta", "final_score": 7, "reliable": True}
    with SessionLocal() as db:
        session = DbSession(exam_id=exam_id, student_id=student_id, quiz_json=json.dumps(quiz), qa_log=json.dumps([attempt]), status="in_corso")
        db.add(session)
        db.commit()
        session_id = session.id

    detail = api_client.get(f"/sessions/{session_id}/detail")
    assert detail.status_code == 200
    assert detail.json()["attempts"][0]["student_answer_1"] == "Una spinta"
    assert detail.json()["material_ids"] == [7]

    def generate_session_notes(attempts, **kwargs):
        assert kwargs["material_ids"] == [7]
        return "# Appunti"

    monkeypatch.setattr(session_extras, "generate_notes_from_log", generate_session_notes)
    assert api_client.post(f"/sessions/{session_id}/notes").json()["notes"] == "# Appunti"
    assert api_client.post(f"/sessions/{session_id}/interrupt").status_code == 200
    assert api_client.get(f"/sessions/{session_id}/detail").json()["status"] == "interrotta"
    assert api_client.post(f"/sessions/{session_id}/resume-active").status_code == 200
    assert api_client.get(f"/sessions/{session_id}/detail").json()["ended_at"] is None
    assert api_client.delete(f"/sessions/{session_id}").status_code == 204
    with SessionLocal() as db:
        assert db.get(DbSession, session_id) is None


def test_interrupted_session_can_be_deleted(api_client):
    student_id, exam_id = _account_and_exam(api_client)
    with SessionLocal() as db:
        session = DbSession(exam_id=exam_id, student_id=student_id, status="in_corso")
        db.add(session)
        db.commit()
        session_id = session.id
    assert api_client.post(f"/sessions/{session_id}/interrupt").status_code == 200
    assert api_client.delete(f"/sessions/{session_id}").status_code == 204
    with SessionLocal() as db:
        assert db.get(DbSession, session_id) is None


def test_new_session_verifies_only_selected_sources(api_client, monkeypatch):
    student_id, exam_id = _account_and_exam(api_client)
    seen = []

    def verify(student_id_arg, exam_id_arg, material_ids=None):
        seen.append(material_ids)
        return {"ok": True, "errors": []}

    monkeypatch.setattr(sessions_api, "verify_exam_material", verify)
    response = api_client.post("/sessions", params={"student_id": student_id},
                               json={"exam_id": exam_id, "material_ids": [7]})
    assert response.status_code == 201
    assert seen == [[7]]


def test_selected_exam_notes_pdf_and_reindex_use_existing_services(api_client, monkeypatch, tmp_path):
    student_id, exam_id = _account_and_exam(api_client)
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"%PDF-test")
    with SessionLocal() as db:
        material = Material(exam_id=exam_id, filename="source.pdf", stored_path=str(pdf_path),
                            file_hash="oldhash", pages_count=1, chunks_count=1)
        db.add(material)
        db.commit()
        material_id = material.id

    seen = []
    monkeypatch.setattr(frontend_extras, "verify_exam_material", lambda *args, **kwargs: {"ok": True, "errors": []})
    monkeypatch.setattr(frontend_extras, "generate_notes_pdf_from_materials",
                        lambda student_id, exam_id, materials, **kwargs: seen.extend(m.id for m in materials) or b"%PDF-notes")
    notes = api_client.post(f"/exams/{exam_id}/notes/pdf", params={"student_id": student_id},
                            json={"material_ids": [material_id]})
    assert notes.status_code == 200
    assert notes.content == b"%PDF-notes"
    assert seen == [material_id]

    from backend.agents.llm_errors import RetryableLLMError
    limited = RetryableLLMError("Limite temporaneo Groq; riprova tra 368.928s")
    limited.status_code = 429
    limited.headers = {"Retry-After": "368.928"}

    def rate_limited(*_args, **_kwargs):
        raise limited

    monkeypatch.setattr(frontend_extras, "generate_notes_pdf_from_materials", rate_limited)
    response = api_client.post(f"/exams/{exam_id}/notes/pdf", params={"student_id": student_id},
                               json={"material_ids": [material_id]})
    assert response.status_code == 429
    assert response.headers["retry-after"] == "368.928"
    assert "368.928s" in response.json()["detail"]

    class FakeCollection:
        def get(self, **kwargs):
            return {"ids": ["old-chunk"], "documents": ["Testo precedente"],
                    "metadatas": [{"file_hash": "oldhash"}]}

        def delete(self, **kwargs):
            seen.append(kwargs["ids"][0])

    monkeypatch.setattr(frontend_extras, "get_collection", lambda *args: FakeCollection())
    monkeypatch.setattr(frontend_extras, "ingest_pdf", lambda *args, **kwargs: {
        "file_hash": "newhash", "chunks": 3, "pages": 2, "pages_with_text": 2, "pages_skipped": 0,
    })
    reindexed = api_client.post(f"/exams/{exam_id}/reindex", params={"student_id": student_id})
    assert reindexed.status_code == 200
    assert "old-chunk" in seen
    with SessionLocal() as db:
        assert db.get(Material, material_id).chunks_count == 3


def test_reindex_restores_previous_chunks_when_ingestion_fails(api_client, monkeypatch, tmp_path):
    student_id, exam_id = _account_and_exam(api_client)
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"%PDF-test")
    with SessionLocal() as db:
        material = Material(exam_id=exam_id, filename="source.pdf", stored_path=str(pdf_path),
                            file_hash="oldhash", pages_count=1, chunks_count=1)
        db.add(material)
        db.commit()
        material_id = material.id

    class FakeCollection:
        def __init__(self):
            self.rows = {"old-chunk": ("Testo precedente", {"file_hash": "oldhash"})}

        def get(self, **kwargs):
            ids = [key for key, (_, meta) in self.rows.items()
                   if all(meta.get(field) == value for field, value in kwargs["where"].items())]
            return {"ids": ids, "documents": [self.rows[key][0] for key in ids],
                    "metadatas": [self.rows[key][1] for key in ids]}

        def delete(self, *, ids):
            for key in ids:
                self.rows.pop(key)

        def add(self, *, ids, documents, metadatas):
            for key, document, metadata in zip(ids, documents, metadatas):
                self.rows[key] = (document, metadata)

    collection = FakeCollection()
    monkeypatch.setattr(frontend_extras, "get_collection", lambda *args: collection)
    monkeypatch.setattr(frontend_extras, "ingest_pdf", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("ingestion failed")))

    response = api_client.post(f"/exams/{exam_id}/reindex", params={"student_id": student_id})
    assert response.status_code == 500
    assert collection.rows == {"old-chunk": ("Testo precedente", {"file_hash": "oldhash"})}
    with SessionLocal() as db:
        stored = db.get(Material, material_id)
        assert stored.file_hash == "oldhash"
        assert stored.chunks_count == 1


def test_reindex_restores_previous_chunks_when_db_commit_fails(api_client, monkeypatch, tmp_path):
    student_id, exam_id = _account_and_exam(api_client)
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"%PDF-test")
    with SessionLocal() as db:
        material = Material(exam_id=exam_id, filename="source.pdf", stored_path=str(pdf_path),
                            file_hash="oldhash", pages_count=1, chunks_count=1)
        db.add(material)
        db.commit()
        material_id = material.id

    rows = {"old-chunk": ("Testo precedente", {"file_hash": "oldhash"})}

    class FakeCollection:
        def get(self, *, where, **kwargs):
            ids = [key for key, (_, meta) in rows.items()
                   if all(meta.get(field) == value for field, value in where.items())]
            return {"ids": ids, "documents": [rows[key][0] for key in ids],
                    "metadatas": [rows[key][1] for key in ids]}

        def delete(self, *, ids):
            for key in ids:
                rows.pop(key)

        def add(self, *, ids, documents, metadatas):
            for key, document, metadata in zip(ids, documents, metadatas):
                assert key not in rows
                rows[key] = (document, metadata)

    collection = FakeCollection()
    monkeypatch.setattr(frontend_extras, "get_collection", lambda *args: collection)

    def ingest(*args, **kwargs):
        collection.add(ids=["new-chunk"], documents=["Testo nuovo"],
                       metadatas=[{"file_hash": "newhash"}])
        return {"file_hash": "newhash", "chunks": 1, "pages": 1,
                "pages_with_text": 1, "pages_skipped": 0}

    monkeypatch.setattr(frontend_extras, "ingest_pdf", ingest)
    original_factory = frontend_extras.SessionLocal

    def failing_factory():
        db = original_factory()
        db.commit = lambda: (_ for _ in ()).throw(OSError("database unavailable"))
        return db

    monkeypatch.setattr(frontend_extras, "SessionLocal", failing_factory)
    response = api_client.post(f"/exams/{exam_id}/reindex", params={"student_id": student_id})
    assert response.status_code == 500
    assert rows == {"old-chunk": ("Testo precedente", {"file_hash": "oldhash"})}
    with SessionLocal() as db:
        stored = db.get(Material, material_id)
        assert stored.file_hash == "oldhash"
        assert stored.chunks_count == 1


def test_report_download_and_regeneration(api_client, monkeypatch, tmp_path):
    student_id, exam_id = _account_and_exam(api_client)
    with SessionLocal() as db:
        session = DbSession(exam_id=exam_id, student_id=student_id, score_total=7,
                            qa_log=json.dumps([{"question_id": 1, "question_text": "Q", "topic_hint": "tema",
                                                "final_score": 7, "reliable": True}]), status="completata")
        db.add(session)
        db.commit()
        session_id = session.id
    pdf_path = tmp_path / "report.pdf"
    pdf_path.write_bytes(b"%PDF-report")
    captured = {}
    monkeypatch.setattr(session_extras, "report_snapshot", lambda *args: {})
    def save_report(report, *_args):
        captured.update(report)
        return {"pdf": str(pdf_path)}
    monkeypatch.setattr(session_extras, "save_report", save_report)
    regenerated = api_client.post(f"/sessions/{session_id}/report/regenerate")
    assert regenerated.status_code == 200
    assert captured["strong_topics"] == []  # A score of 7 is sufficient, not mastered.
    assert api_client.get(f"/sessions/{session_id}/report").content == b"%PDF-report"
    assert api_client.get(f"/sessions/{session_id}").json()["report_available"] is True
