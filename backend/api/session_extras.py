"""Session history actions shared by the API and browser interface."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response

from backend.agents.notes_generator import generate_notes_from_log
from backend.agents.report_generator import delete_session_reports, report_snapshot, restore_report_snapshot, save_report
from backend.api.authorization import require_api_owner
from backend.db.models import Exam, Session as DbSession, SessionLocal
from backend.session_clock import elapsed_seconds, pause_clock, resume_clock
from backend.error_messages import public_error

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sessions", tags=["sessions"], dependencies=[Depends(require_api_owner)])


def _session(session_id: int) -> DbSession:
    with SessionLocal() as db:
        session = db.get(DbSession, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Sessione non trovata.")
        return session


@router.get("/{session_id}/detail")
def session_detail(session_id: int) -> dict:
    session = _session(session_id)
    attempts = json.loads(session.qa_log or "[]")
    quiz = json.loads(session.quiz_json or "{}")
    return {
        "id": session.id, "exam_id": session.exam_id,
        "started_at": session.started_at, "ended_at": session.ended_at,
        "duration_seconds": elapsed_seconds(session),
        "timer_running": session.status == "in_corso" and session.active_started_at is not None,
        "status": session.status, "score_total": session.score_total,
        "questions": quiz.get("questions", []), "attempts": attempts,
        "current_turn": json.loads(session.current_turn_json) if session.current_turn_json else None,
        "material_ids": quiz.get("material_ids"),
        "report_available": bool(session.report_path and Path(session.report_path).is_file()),
    }


@router.post("/{session_id}/interrupt")
def interrupt_session(session_id: int) -> dict:
    with SessionLocal() as db:
        session = db.get(DbSession, session_id)
        if session.score_total is None and session.status == "in_corso":
            pause_clock(session)
            session.status = "interrotta"
            session.ended_at = datetime.now(timezone.utc)
            db.commit()
    return {"message": "Sessione interrotta."}


@router.post("/{session_id}/timer")
def save_session_timer(session_id: int) -> dict:
    """Checkpoint the live timer so elapsed time survives an unexpected exit."""
    with SessionLocal() as db:
        session = db.get(DbSession, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Sessione non trovata.")
        if session.status == "in_corso" and session.active_started_at is not None:
            now = datetime.now(timezone.utc)
            pause_clock(session, now)
            resume_clock(session, now)
            db.commit()
        return {"duration_seconds": elapsed_seconds(session), "status": session.status}


@router.post("/{session_id}/resume-active")
def reactivate_session(session_id: int) -> dict:
    with SessionLocal() as db:
        session = db.get(DbSession, session_id)
        if session.status not in {"in_corso", "interrotta"} or not session.quiz_json:
            raise HTTPException(status_code=409, detail="Sessione non riprendibile.")
        resume_clock(session)
        session.status = "in_corso"
        session.ended_at = None
        db.commit()
    return session_detail(session_id)


@router.post("/{session_id}/mark-complete")
def mark_incomplete_closed(session_id: int) -> dict:
    with SessionLocal() as db:
        session = db.get(DbSession, session_id)
        if session.ended_at is None:
            pause_clock(session)
            session.ended_at = datetime.now(timezone.utc)
            session.status = "interrotta"
            db.commit()
    return {"message": "Sessione segnata come conclusa."}


@router.delete("/{session_id}", status_code=204)
def delete_incomplete_session(session_id: int) -> Response:
    with SessionLocal() as db:
        session = db.get(DbSession, session_id)
        if session.status == "completata" or session.score_total is not None:
            raise HTTPException(status_code=409, detail="Solo le sessioni non completate si possono eliminare.")
        exam_id = session.exam_id
        student_id = session.student_id or db.get(Exam, exam_id).student_id
        db.delete(session)
        db.commit()
    delete_session_reports(student_id, exam_id, session_id)
    return Response(status_code=204)


@router.post("/{session_id}/notes")
def session_notes(session_id: int) -> dict:
    session = _session(session_id)
    attempts = json.loads(session.qa_log or "[]")
    material_ids = json.loads(session.quiz_json or "{}").get("material_ids")
    if not attempts:
        raise HTTPException(status_code=409, detail="Nessuna risposta disponibile per gli appunti.")
    try:
        notes = generate_notes_from_log(attempts, student_id=session.student_id,
                                        exam_id=session.exam_id, enrich_with_rag=True,
                                        material_ids=material_ids)
    except Exception as exc:
        logger.exception("Session notes generation failed")
        raise HTTPException(status_code=500, detail=public_error(exc, "Impossibile generare gli appunti. Riprova.")) from exc
    return {"notes": notes}


@router.get("/{session_id}/report")
def download_report(session_id: int) -> Response:
    session = _session(session_id)
    path = Path(session.report_path) if session.report_path else None
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Report PDF non disponibile.")
    return Response(path.read_bytes(), media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="report_sessione_{session_id}.pdf"'})


@router.post("/{session_id}/report/regenerate")
def regenerate_report(session_id: int) -> dict:
    session = _session(session_id)
    if session.score_total is None:
        raise HTTPException(status_code=409, detail="Sessione senza voto finale.")
    attempts = json.loads(session.qa_log or "[]")
    scored = [item for item in attempts if isinstance(item, dict)
              and isinstance(item.get("final_score"), (int, float))
              and not isinstance(item.get("final_score"), bool)]
    if not scored:
        raise HTTPException(status_code=409, detail="Nessuna risposta valutata per il report.")
    topic_scores: dict[str, list[float]] = {}
    for item in scored:
        topic_scores.setdefault(str(item.get("topic_hint") or "generale"), []).append(float(item["final_score"]))
    breakdown = [{"topic": topic, "avg_score": round(sum(scores) / len(scores), 1),
                  "n_attempts": len(scores), "is_weak": sum(scores) / len(scores) < 6}
                 for topic, scores in topic_scores.items()]
    report = {
        "student_id": session.student_id, "exam_id": session.exam_id,
        "overall_avg": round(sum(float(item["final_score"]) for item in scored) / len(scored), 1),
        "n_questions_answered": len(scored),
        "weak_topics": [item["topic"] for item in breakdown if item["is_weak"]],
        "strong_topics": [item["topic"] for item in breakdown if item["avg_score"] >= 8],
        "topic_breakdown": breakdown,
        "summary": "Report rigenerato automaticamente dallo storico sessione.",
    }
    with SessionLocal() as db:
        exam = db.get(Exam, session.exam_id)
        exam_name = exam.name if exam else "Esame"
    snapshot = report_snapshot(report["student_id"], session.exam_id, session_id)
    try:
        paths = save_report(report, session_id, exam_name)
        with SessionLocal() as db:
            stored = db.get(DbSession, session_id)
            stored.report_path = paths["pdf"]
            db.commit()
    except Exception as exc:
        logger.exception("Report regeneration failed")
        restore_report_snapshot(snapshot)
        raise HTTPException(status_code=500, detail="Impossibile rigenerare il report. Riprova.") from exc
    return {"message": "Report PDF rigenerato."}
