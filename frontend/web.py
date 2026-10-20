"""Server rendered Study Coach interface. Business operations stay in the API modules."""

import asyncio
import base64
import json
import logging
import uuid
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from markdown_it import MarkdownIt
from markupsafe import Markup
from pydantic import ValidationError
from sqlalchemy import func
from starlette.datastructures import UploadFile
from starlette.concurrency import run_in_threadpool

from backend.api import auth, exams, frontend_extras, session_extras, sessions
from backend.agents.llm_errors import is_rate_limit_error
from backend.db.models import DoubtConversation, DoubtThread, Exam, Material, SessionLocal
from backend.models.schemas import (
    Answer1Request, Answer2Request, CloseQuestionRequest, ContestRequest,
    ExamCreate, LoginRequest, NextQuestionRequest, QuestionOut, QuizGenerateRequest,
    RegisterRequest, SessionCloseRequest, SessionCreate, TutorStartRequest,
)
from backend.session_cookie import COOKIE_NAME, verify_token

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/web", include_in_schema=False)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
markdown = MarkdownIt("commonmark", {"html": False})


def _user(request: Request):
    student = verify_token(request.cookies.get(COOKIE_NAME, ""))
    if student is None:
        raise HTTPException(status_code=303, headers={"Location": "/web/login"})
    return student


def _exam(exam_id: int, student_id: int) -> Exam:
    return frontend_extras._owned_exam(exam_id, student_id)


def _session(session_id: int, student_id: int):
    with SessionLocal() as db:
        session = db.get(sessions.DbSession, session_id)
        exam = db.get(Exam, session.exam_id) if session else None
        if not session or not exam or exam.student_id != student_id or (
            session.student_id is not None and session.student_id != student_id
        ):
            raise HTTPException(status_code=404, detail="Sessione non trovata.")
        return session, exam


def _page(request: Request, name: str, **context):
    operation_script_version = (Path(__file__).parent / "static" / "operation-progress.js").stat().st_mtime_ns
    notes_script_version = (Path(__file__).parent / "static" / "notes-progress.js").stat().st_mtime_ns
    return templates.TemplateResponse(request, name, {"user": context.pop("user", None),
                                                      "operation_script_version": operation_script_version,
                                                      "notes_script_version": notes_script_version,
                                                      **context})


def _redirect(url: str):
    return RedirectResponse(url, status_code=303)


def _error(url: str, exc: Exception):
    if isinstance(exc, HTTPException):
        message = _http_error_message(exc)
    elif isinstance(exc, ValidationError):
        message = "Controlla i dati inseriti e riprova."
    else:
        raise exc
    return _redirect(f"{url}{'&' if '?' in url else '?'}error={quote(message)}")


def _http_error_message(exc: HTTPException) -> str:
    return exc.detail if isinstance(exc.detail, str) else "Operazione non riuscita. Controlla i materiali e riprova."


def _operation_stream(work):
    """Stream phases from an async operation while keeping the normal form fallback."""
    async def events():
        queue: asyncio.Queue[dict] = asyncio.Queue()

        def send(message: str, **details):
            queue.put_nowait({"type": "progress", "message": message, **details})

        async def run():
            try:
                result = await work(send)
                await queue.put({"type": "complete", "url": result.headers.get("location", "/web/exams")})
            except Exception as exc:
                if not isinstance(exc, (HTTPException, ValidationError, ValueError)):
                    logger.exception("Web operation failed")
                message = _http_error_message(exc) if isinstance(exc, HTTPException) else (
                    "Controlla i dati inseriti e riprova." if isinstance(exc, (ValidationError, ValueError))
                    else "Operazione non riuscita. Riprova."
                )
                await queue.put({"type": "error", "message": message,
                                 "code": "api_limit" if isinstance(exc, HTTPException) and exc.status_code == 429 else None})

        task = asyncio.create_task(run())
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=15)
            except asyncio.TimeoutError:
                yield '{"type":"heartbeat"}\n'
                continue
            yield json.dumps(event, ensure_ascii=False) + "\n"
            if event["type"] in {"complete", "error"}:
                break
        await task

    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


def _notes_stream(exam_id: int, student_id: int, exam_name: str, materials: list[Material]) -> StreamingResponse:
    """Send real generation stages and the completed PDF to the browser."""
    async def events():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict] = asyncio.Queue()

        def send(event: dict) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        def generate() -> None:
            try:
                pdf, notes_md = frontend_extras.generate_notes_pdf_from_materials(
                    student_id, exam_id, materials, exam_name_fallback=exam_name,
                    progress=lambda event: send({"type": "progress", **event}),
                    include_markdown=True,
                )
            except Exception as exc:
                if is_rate_limit_error(exc):
                    logger.warning("Notes PDF generation rate limited for exam %s", exam_id)
                else:
                    logger.exception("Notes PDF generation failed for exam %s", exam_id)
                send({"type": "error", "message": frontend_extras.public_error(
                    exc, "Impossibile generare gli appunti. Riprova."
                ), "code": "api_limit" if is_rate_limit_error(exc) else None})
            else:
                send({"type": "complete", "filename": "appunti.pdf",
                      "pdf_base64": base64.b64encode(pdf).decode("ascii"),
                      "markdown_filename": "appunti.md", "markdown": notes_md})

        task = asyncio.create_task(asyncio.to_thread(generate))
        yield json.dumps({"type": "progress", "phase": "starting"}, ensure_ascii=False) + "\n"
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=15)
            except asyncio.TimeoutError:
                yield '{"type":"heartbeat"}\n'
                continue
            yield json.dumps(event, ensure_ascii=False) + "\n"
            if event["type"] in {"complete", "error"}:
                break
        await task

    return StreamingResponse(
        events(), media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


def _notice(request: Request):
    return request.query_params.get("error"), request.query_params.get("ok")


def _selected(form):
    try:
        return list(dict.fromkeys(int(value) for value in form.getlist("material_ids")))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Selezione fonti non valida.") from exc


def _verify_selection(exam_id: int, ids: list[int], student_id: int) -> dict:
    try:
        return frontend_extras.verify_materials(
            exam_id, frontend_extras.MaterialSelection(material_ids=ids), student_id
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Material verification failed for exam %s", exam_id)
        raise HTTPException(503, "Verifica delle fonti non disponibile. Riprova.") from exc


@router.get("/login")
def login_page(request: Request):
    if verify_token(request.cookies.get(COOKIE_NAME, "")):
        return _redirect("/web/exams")
    error, _ = _notice(request)
    return _page(request, "auth.html", error=error, auth_mode="login")


@router.get("/register")
def register_page(request: Request):
    if verify_token(request.cookies.get(COOKIE_NAME, "")):
        return _redirect("/web/exams")
    error, _ = _notice(request)
    return _page(request, "auth.html", error=error, auth_mode="register")


@router.post("/login")
async def login(request: Request):
    form = await request.form()
    response = _redirect("/web/exams")
    try:
        auth.login(LoginRequest(username=str(form.get("username", "")), password=str(form.get("password", ""))), response, request)
        return response
    except (HTTPException, ValidationError) as exc:
        return _error("/web/login", exc)


@router.post("/register")
async def register(request: Request):
    form = await request.form()
    response = _redirect("/web/exams")
    try:
        if form.get("password") != form.get("password_confirm"):
            raise HTTPException(422, "Le password non coincidono.")
        auth.register(RegisterRequest(username=str(form.get("username", "")),
                                      email=str(form.get("email", "")),
                                      password=str(form.get("password", "")),
                                      password_confirm=str(form.get("password_confirm", ""))), response)
        return response
    except (HTTPException, ValidationError) as exc:
        return _error("/web/register", exc)


@router.post("/logout")
def logout(request: Request):
    response = _redirect("/web/login")
    auth.logout(request, response)
    return response


@router.get("/exams")
def library(request: Request, user=Depends(_user)):
    items = exams.list_exams(user.id).exams
    with SessionLocal() as db:
        counts = {item.id: db.query(Material).filter(Material.exam_id == item.id).count() for item in items}
    error, ok = _notice(request)
    return _page(request, "exams.html", user=user, exams=items, counts=counts, error=error, ok=ok)


@router.post("/exams")
async def create_exam(request: Request, user=Depends(_user)):
    form = await request.form()
    try:
        created = exams.create_exam(ExamCreate(name=str(form.get("name", ""))), user.id)
        return _redirect(f"/web/exams/{created.id}")
    except (HTTPException, ValidationError) as exc:
        return _error("/web/exams", exc)


@router.post("/exams/{exam_id}/rename")
async def rename_exam(request: Request, exam_id: int, user=Depends(_user)):
    _exam(exam_id, user.id)
    form = await request.form()
    try:
        frontend_extras.rename_exam(exam_id, frontend_extras.RenameExam(name=str(form.get("name", ""))), user.id)
        return _redirect(f"/web/exams/{exam_id}?ok=Nome+aggiornato")
    except (HTTPException, ValidationError) as exc:
        return _error(f"/web/exams/{exam_id}", exc)


@router.post("/exams/{exam_id}/delete")
def delete_exam(exam_id: int, user=Depends(_user)):
    _exam(exam_id, user.id)
    try:
        exams.delete_exam(exam_id, user.id)
        return _redirect("/web/exams?ok=Esame+eliminato")
    except HTTPException as exc:
        return _error("/web/exams", exc)


@router.get("/exams/{exam_id}")
def exam_page(request: Request, exam_id: int, user=Depends(_user)):
    exam = _exam(exam_id, user.id)
    materials = frontend_extras.list_materials(exam_id, user.id)["materials"]
    if materials:
        try:
            verification = _verify_selection(exam_id, [item["id"] for item in materials], user.id)
        except HTTPException as exc:
            verification = {"ok": False, "errors": [exc.detail]}
    else:
        verification = None
    error, ok = _notice(request)
    return _page(request, "exam.html", user=user, exam=exam, materials=materials,
                 verification=verification, error=error, ok=ok)


@router.post("/exams/{exam_id}/upload")
async def upload(request: Request, exam_id: int, user=Depends(_user)):
    _exam(exam_id, user.id)
    form = await request.form()
    files = [value for value in form.getlist("files") if isinstance(value, UploadFile) and value.filename]
    if not files:
        return _error(f"/web/exams/{exam_id}", HTTPException(422, "Seleziona almeno un PDF."))
    if "application/x-ndjson" in request.headers.get("accept", ""):
        return _operation_stream(lambda send: _upload_files(exam_id, user.id, files, send))
    return await _upload_files(exam_id, user.id, files)


async def _upload_files(exam_id, student_id, files, send=None):
    successful, errors = [], []
    for index, file in enumerate(files, 1):
        try:
            if send:
                send(f"File {index}/{len(files)} · {file.filename}: controllo e indicizzazione…")
            await exams.ingest_pdf_endpoint(exam_id, student_id, file)
            successful.append(file.filename)
            if send:
                send(f"File {index}/{len(files)} · {file.filename}: pronto.")
        except HTTPException as exc:
            reason = exc.detail if isinstance(exc.detail, str) else "caricamento non riuscito"
            errors.append(f"{file.filename}: {reason}")
            if send:
                send(f"File {index}/{len(files)} · {file.filename}: {reason}")
    url = f"/web/exams/{exam_id}"
    if successful:
        message = f"{len(successful)} PDF caricati: {', '.join(successful)}"
        url += f"?ok={quote(message)}"
    if errors:
        url += f"{'&' if successful else '?'}error={quote(' · '.join(errors))}"
    if send and not successful:
        raise HTTPException(422, " · ".join(errors))
    return _redirect(url)


@router.post("/exams/{exam_id}/reindex")
def reindex(exam_id: int, user=Depends(_user)):
    _exam(exam_id, user.id)
    try:
        verification = frontend_extras.reindex_materials(exam_id, user.id)
        if not verification["ok"]:
            raise HTTPException(409, "Alcune fonti non sono ancora utilizzabili.")
        return _redirect(f"/web/exams/{exam_id}?ok=Fonti+reindicizzate")
    except HTTPException as exc:
        return _error(f"/web/exams/{exam_id}", exc)


@router.post("/exams/{exam_id}/sources")
async def source_action(request: Request, exam_id: int, user=Depends(_user)):
    exam = _exam(exam_id, user.id)
    form = await request.form()
    wants_progress = "application/x-ndjson" in request.headers.get("accept", "")
    try:
        ids = _selected(form)
        if not ids:
            raise HTTPException(422, "Seleziona almeno una fonte.")
        frontend_extras._selected_materials(exam_id, ids)
        action = form.get("action")
        if action in {"study", "notes"}:
            check = await run_in_threadpool(_verify_selection, exam_id, ids, user.id)
            if not check["ok"]:
                raise HTTPException(409, "Fonti non pronte. Controlla la verifica dei materiali.")
        if action == "study":
            return _page(request, "setup.html", user=user, exam=exam, ids=ids)
        if action == "notes":
            if wants_progress:
                return _notes_stream(exam_id, user.id, exam.name,
                                     frontend_extras._selected_materials(exam_id, ids))
            return await run_in_threadpool(frontend_extras.exam_notes_pdf, exam_id,
                                           frontend_extras.MaterialSelection(material_ids=ids), user.id)
        if action == "delete":
            if form.get("confirm_delete") != "yes":
                raise HTTPException(422, "Conferma l’eliminazione delle fonti selezionate.")
            for material_id in ids:
                await run_in_threadpool(exams.delete_exam_material, exam_id, material_id, user.id)
            return _redirect(f"/web/exams/{exam_id}?ok=Fonti+eliminate")
        raise HTTPException(422, "Azione non valida.")
    except HTTPException as exc:
        if wants_progress:
            return JSONResponse({"error": _http_error_message(exc)}, status_code=exc.status_code)
        return _error(f"/web/exams/{exam_id}", exc)


@router.post("/exams/{exam_id}/study")
async def start_study(request: Request, exam_id: int, user=Depends(_user)):
    exam = _exam(exam_id, user.id)
    form = await request.form()
    if "application/x-ndjson" in request.headers.get("accept", ""):
        return _operation_stream(lambda send: _start_study_work(exam, form, user, send))
    return await _start_study_work(exam, form, user)


async def _start_study_work(exam, form, user, send=None):
    exam_id = exam.id
    session_id = None
    try:
        if send:
            send("Controllo le fonti selezionate…")
        ids = _selected(form)
        if not ids:
            raise HTTPException(422, "Seleziona almeno una fonte.")
        frontend_extras._selected_materials(exam_id, ids)
        count = int(str(form.get("n_questions", "5")))
        if not 3 <= count <= 10:
            raise HTTPException(422, "Scegli tra 3 e 10 domande.")
        if send:
            send("Creo la sessione…")
        created = await run_in_threadpool(sessions.create_session, SessionCreate(exam_id=exam_id, material_ids=ids), user.id)
        session_id = created.id
        if send:
            send("Genero e verifico le domande dalle fonti…")
        await run_in_threadpool(sessions.generate_session_quiz, session_id,
                                QuizGenerateRequest(n_questions=count, source_name=exam.name, material_ids=ids), user.id)
        if send:
            send("Scelgo la prima domanda del tutor…")
        return await run_in_threadpool(advance, session_id, user)
    except (HTTPException, ValidationError, ValueError) as exc:
        if session_id is not None:
            session_extras.delete_incomplete_session(session_id)
        if isinstance(exc, ValueError):
            exc = HTTPException(422, "Numero di domande non valido.")
        if send:
            raise exc
        return _error(f"/web/exams/{exam_id}", exc)


@router.get("/study/{session_id}")
def study(request: Request, session_id: int, user=Depends(_user)):
    _, exam = _session(session_id, user.id)
    detail = session_extras.session_detail(session_id)
    scored = [item["final_score"] for item in detail["attempts"]
              if isinstance(item.get("final_score"), (int, float))
              and not isinstance(item.get("final_score"), bool)]
    average = round(sum(scored) / len(scored), 1) if scored else None
    progress = round(100 * len(detail["attempts"]) / len(detail["questions"])) if detail["questions"] else 0
    duration = int(detail["duration_seconds"])
    duration_text = f"{duration // 3600:02d}:{duration // 60 % 60:02d}:{duration % 60:02d}"
    error, ok = _notice(request)
    return _page(request, "study.html", user=user, exam=exam, detail=detail,
                 average=average, scored_count=len(scored), progress=progress,
                 duration_text=duration_text, error=error, ok=ok)


@router.post("/study/{session_id}/advance")
def advance_route(request: Request, session_id: int, user=Depends(_user)):
    if "application/x-ndjson" in request.headers.get("accept", ""):
        async def work(send):
            send("Scelgo la prossima domanda o preparo il report…")
            return await run_in_threadpool(advance, session_id, user)
        return _operation_stream(work)
    return advance(session_id, user)


def advance(session_id: int, user):
    _, exam = _session(session_id, user.id)
    url = f"/web/study/{session_id}"
    try:
        detail = session_extras.session_detail(session_id)
        if detail["current_turn"] or detail["status"] != "in_corso":
            raise HTTPException(409, "Turno non disponibile.")
        answered = {int(attempt["question_id"]) for attempt in detail["attempts"]}
        remaining = [QuestionOut(**question) for question in detail["questions"] if int(question["id"]) not in answered]
        if not remaining:
            sessions.close_session(session_id, SessionCloseRequest(session_id=session_id, exam_name=exam.name))
        else:
            choice = sessions.next_question(session_id, NextQuestionRequest(session_id=session_id, remaining_questions=remaining))
            if choice.question:
                sessions.tutor_start(session_id, TutorStartRequest(session_id=session_id, question=choice.question), user.id)
        return _redirect(url)
    except (HTTPException, ValidationError) as exc:
        return _error(url, exc)


@router.post("/study/{session_id}/{action}")
async def study_action(request: Request, session_id: int, action: str, user=Depends(_user)):
    _, exam = _session(session_id, user.id)
    url = f"/web/study/{session_id}"
    form = await request.form()
    if "application/x-ndjson" in request.headers.get("accept", ""):
        return _operation_stream(lambda send: _study_action_work(session_id, action, user, exam, form, send))
    return await _study_action_work(session_id, action, user, exam, form)


async def _study_action_work(session_id, action, user, exam, form, send=None):
    url = f"/web/study/{session_id}"
    try:
        detail = session_extras.session_detail(session_id)
        turn = detail["current_turn"]
        if action == "leave":
            session_extras.interrupt_session(session_id)
            return _redirect(f"/web/history/{exam.id}")
        if not turn:
            raise HTTPException(409, "Nessuna domanda attiva.")
        question_id = int(turn["question_id"])
        if action == "answer1":
            if send:
                send("Leggo la risposta e preparo la domanda guida…")
            await run_in_threadpool(sessions.tutor_answer1, session_id, Answer1Request(session_id=session_id, question_id=question_id, answer=str(form.get("answer", "")).strip()))
        elif action == "answer2":
            if send:
                send("Confronto le risposte con le fonti e preparo la valutazione…")
            await run_in_threadpool(sessions.tutor_answer2, session_id, Answer2Request(session_id=session_id, question_id=question_id, answer=str(form.get("answer", "")).strip()))
        elif action == "contest":
            if send:
                send("Riesamino la valutazione e preparo la risposta…")
            await run_in_threadpool(sessions.tutor_contest, session_id, ContestRequest(session_id=session_id, question_id=question_id, argumentation=str(form.get("argumentation", "")).strip()))
        elif action == "close":
            if send:
                send("Registro la risposta nella sessione…")
            await run_in_threadpool(sessions.tutor_close, session_id, CloseQuestionRequest(session_id=session_id, question_id=question_id))
            completed = session_extras.session_detail(session_id)
            if len(completed["attempts"]) >= len(completed["questions"]):
                if send:
                    send("Chiudo la sessione e preparo il report…")
                await run_in_threadpool(sessions.close_session, session_id, SessionCloseRequest(session_id=session_id, exam_name=exam.name))
            else:
                if send:
                    send("Scelgo e preparo la prossima domanda…")
                return await run_in_threadpool(advance, session_id, user)
        else:
            raise HTTPException(404, "Azione non trovata.")
        return _redirect(url)
    except (HTTPException, ValidationError) as exc:
        if send:
            raise
        return _error(url, exc)


@router.get("/chat/{exam_id}")
def chat(request: Request, exam_id: int, user=Depends(_user)):
    exam = _exam(exam_id, user.id)
    with SessionLocal() as db:
        threads = (db.query(DoubtThread).filter_by(student_id=user.id, exam_id=exam_id)
                   .order_by(DoubtThread.created_at.desc(), DoubtThread.id.desc()).all())
        requested = request.query_params.get("thread")
        active = next((item for item in threads if item.id == requested), None) if requested else (threads[0] if threads else None)
        if requested and active is None:
            raise HTTPException(404, "Chat non trovata.")
        conversations = (db.query(DoubtConversation).filter_by(student_id=user.id, exam_id=exam_id, thread_id=active.id)
                         .order_by(DoubtConversation.created_at, DoubtConversation.id).all()) if active else []
        rows = []
        for item in conversations:
            row = {"id": item.id, "doubt": item.doubt_text, "guide": item.guiding_question,
                   "topic": item.topic_hint, "answered": item.answered,
                   "answer": item.student_answer,
                   "result": json.loads(item.result_json) if item.result_json else None}
            if (rows and not rows[-1]["answered"] and not row["answered"]
                    and rows[-1]["doubt"].strip().casefold() == row["doubt"].strip().casefold()):
                rows[-1] = row
            else:
                rows.append(row)
    error, ok = _notice(request)
    return _page(request, "chat.html", user=user, exam=exam, rows=rows, active=active,
                 threads=threads, error=error, ok=ok)


@router.get("/chat/{exam_id}/history")
def chat_history(request: Request, exam_id: int, user=Depends(_user)):
    exam = _exam(exam_id, user.id)
    with SessionLocal() as db:
        threads = (db.query(DoubtThread).filter_by(student_id=user.id, exam_id=exam_id)
                   .order_by(DoubtThread.created_at.desc(), DoubtThread.id.desc()).all())
        counts = dict(db.query(DoubtConversation.thread_id, func.count(DoubtConversation.id))
                      .filter_by(student_id=user.id, exam_id=exam_id)
                      .group_by(DoubtConversation.thread_id).all())
    return _page(request, "chat_history.html", user=user, exam=exam, threads=threads, counts=counts)


@router.post("/chat/{exam_id}")
async def chat_action(request: Request, exam_id: int, user=Depends(_user)):
    _exam(exam_id, user.id)
    form = await request.form()
    if "application/x-ndjson" in request.headers.get("accept", "") and form.get("action") in {"start", "answer"}:
        return _operation_stream(lambda send: _chat_action_work(exam_id, user, form, send))
    return await _chat_action_work(exam_id, user, form)


async def _chat_action_work(exam_id, user, form, send=None):
    url = f"/web/chat/{exam_id}"
    try:
        action = form.get("action")
        if action == "new":
            with SessionLocal() as db:
                thread = DoubtThread(id=str(uuid.uuid4()), student_id=user.id, exam_id=exam_id, title="Nuova chat")
                db.add(thread)
                db.commit()
                return _redirect(f"{url}?thread={thread.id}")
        elif action == "delete":
            if form.get("confirm_delete") != "yes":
                raise HTTPException(422, "Conferma l’eliminazione della chat.")
            thread_id = str(form.get("thread_id", ""))
            with SessionLocal() as db:
                thread = db.query(DoubtThread).filter_by(id=thread_id, student_id=user.id, exam_id=exam_id).first()
                if thread is None:
                    raise HTTPException(404, "Chat non trovata.")
                db.query(DoubtConversation).filter_by(thread_id=thread.id, student_id=user.id, exam_id=exam_id).delete()
                db.delete(thread)
                db.commit()
            return _redirect(f"{url}/history?ok=Chat+eliminata")
        elif action == "start":
            thread_id = str(form.get("thread_id", ""))
            if send:
                send("Cerco i passaggi utili e preparo una domanda guida…")
            await run_in_threadpool(frontend_extras.start_doubt, exam_id,
                                    frontend_extras.DoubtStart(doubt=str(form.get("text", "")).strip(),
                                                               thread_id=thread_id or None), user.id)
            if thread_id:
                return _redirect(f"{url}?thread={thread_id}")
        elif action == "answer":
            thread_id = str(form.get("thread_id", ""))
            conversation_id = str(form.get("conversation_id", ""))
            if thread_id:
                with SessionLocal() as db:
                    owned = db.query(DoubtConversation).filter_by(id=conversation_id, thread_id=thread_id,
                                                                  student_id=user.id, exam_id=exam_id).first()
                    if owned is None:
                        raise HTTPException(404, "Dubbio non trovato in questa chat.")
            if send:
                send("Confronto la risposta con i materiali e preparo il feedback…")
            await run_in_threadpool(frontend_extras.answer_doubt, exam_id, conversation_id,
                                    frontend_extras.DoubtAnswer(answer=str(form.get("text", "")).strip()), user.id)
            if thread_id:
                return _redirect(f"{url}?thread={thread_id}")
        else:
            raise HTTPException(422, "Azione non valida.")
        return _redirect(url)
    except (HTTPException, ValidationError) as exc:
        if send:
            raise
        return _error(url, exc)


@router.get("/history")
def history_index(request: Request, user=Depends(_user)):
    items = exams.list_exams(user.id).exams
    if items:
        return _redirect(f"/web/history/{items[0].id}")
    return _page(request, "history.html", user=user, exams=[], exam=None, sessions=[], details={})


@router.get("/history/{exam_id}")
def history(request: Request, exam_id: int, user=Depends(_user)):
    return _history_page(request, exam_id, user)


def _history_page(request: Request, exam_id: int, user, notes_id: int | None = None,
                  notes_html: Markup | None = None):
    exam = _exam(exam_id, user.id)
    items = exams.list_exams(user.id).exams
    records = sessions.list_sessions(exam_id).sessions
    details = {record.id: session_extras.session_detail(record.id) for record in records}
    durations = {record.id: f"{int(details[record.id]['duration_seconds']) // 3600:02d}:"
                 f"{int(details[record.id]['duration_seconds']) // 60 % 60:02d}:"
                 f"{int(details[record.id]['duration_seconds']) % 60:02d}" for record in records}
    completed = [record.score_total for record in records if record.score_total is not None]
    error, ok = _notice(request)
    return _page(request, "history.html", user=user, exams=items, exam=exam, sessions=records,
                 details=details, durations=durations, completed=len(completed), mean=round(sum(completed) / len(completed), 1) if completed else None,
                 best=max(completed) if completed else None, error=error, ok=ok,
                 notes_id=notes_id, notes_html=notes_html)


@router.post("/history/{exam_id}/{session_id}/{action}")
async def history_action(request: Request, exam_id: int, session_id: int, action: str, user=Depends(_user)):
    session, _ = _session(session_id, user.id)
    if session.exam_id != exam_id:
        raise HTTPException(404, "Sessione non trovata.")
    url = f"/web/history/{exam_id}"
    try:
        if action == "resume":
            session_extras.reactivate_session(session_id)
            return _redirect(f"/web/study/{session_id}")
        if action == "notes":
            notes = (await run_in_threadpool(session_extras.session_notes, session_id))["notes"]
            return _history_page(request, exam_id, user, notes_id=session_id,
                                 notes_html=Markup(markdown.render(notes)))
        if action == "regenerate":
            await run_in_threadpool(session_extras.regenerate_report, session_id)
        elif action == "mark":
            session_extras.mark_incomplete_closed(session_id)
        elif action == "delete":
            form = await request.form()
            if form.get("confirm_delete") != "yes":
                raise HTTPException(422, "Conferma l’eliminazione della sessione.")
            session_extras.delete_incomplete_session(session_id)
        else:
            raise HTTPException(404, "Azione non trovata.")
        return _redirect(f"{url}?ok=Operazione+completata")
    except HTTPException as exc:
        return _error(url, exc)
