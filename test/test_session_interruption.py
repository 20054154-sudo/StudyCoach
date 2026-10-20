"""Leaving the study screen preserves the partial session for resume."""

import json

from backend.db.models import Exam, Session as DbSession, SessionLocal


def test_interruption_preserves_attempts(api_client):
    student_id = api_client.post("/auth/register", json={"username": "student_one", "email": "student_one@example.com", "password": "secret123", "password_confirm": "secret123"}).json()["student_id"]
    with SessionLocal() as db:
        exam = Exam(student_id=student_id, name="Analisi")
        db.add(exam)
        db.flush()
        session = DbSession(exam_id=exam.id, student_id=student_id, status="in_corso",
                            qa_log=json.dumps([{"question_id": 1, "student_answer_1": "parziale"}]))
        db.add(session)
        db.commit()
        session_id = session.id

    response = api_client.post(f"/sessions/{session_id}/interrupt")
    assert response.status_code == 200
    detail = api_client.get(f"/sessions/{session_id}/detail").json()
    assert detail["status"] == "interrotta"
    assert detail["ended_at"] is not None
    assert detail["attempts"][0]["student_answer_1"] == "parziale"
