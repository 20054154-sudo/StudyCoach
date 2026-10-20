"""Authenticate API requests and check resource ownership before handlers run."""

import json

from fastapi import HTTPException, Request, status

from backend.db.models import Exam, Session as DbSession, SessionLocal
from backend.session_cookie import COOKIE_NAME, verify_token


async def require_api_owner(request: Request) -> None:
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    else:
        token = request.cookies.get(COOKIE_NAME, "")
    student = verify_token(token)
    if student is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Accesso richiesto.")

    owner_id = student.id
    query_student = request.query_params.get("student_id")
    if query_student is not None and query_student != str(owner_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Risorsa non trovata.")

    path = request.path_params
    exam_id = path.get("exam_id") or request.query_params.get("exam_id")
    if request.url.path.rstrip("/") == "/sessions" and request.method == "POST":
        try:
            body = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail="JSON non valido.") from exc
        exam_id = body.get("exam_id") if isinstance(body, dict) else None
    session_id = path.get("session_id")

    with SessionLocal() as db:
        if exam_id is not None:
            try:
                exam_key = int(exam_id)
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=422, detail="ID esame non valido.") from exc
            exam = db.get(Exam, exam_key)
            if exam is None or exam.student_id != owner_id:
                raise HTTPException(status_code=404, detail="Esame non trovato.")
        if session_id is not None:
            try:
                session_key = int(session_id)
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=422, detail="ID sessione non valido.") from exc
            sess = db.get(DbSession, session_key)
            if sess is None:
                raise HTTPException(status_code=404, detail="Sessione non trovata.")
            parent_exam = db.get(Exam, sess.exam_id)
            if parent_exam is None or parent_exam.student_id != owner_id or (
                sess.student_id is not None and sess.student_id != owner_id
            ):
                raise HTTPException(status_code=404, detail="Sessione non trovata.")
