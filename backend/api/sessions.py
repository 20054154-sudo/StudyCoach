
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.agents.adaptive_engine import AdaptiveEngine
from backend.api.authorization import require_api_owner
from backend.error_messages import public_error
from backend.agents.quiz_generator import UnverifiableQuizError, generate_quiz
from backend.agents.llm_errors import is_rate_limit_error
from backend.agents.ingestor import verify_exam_material
from backend.agents.ingestor import search_passages
from backend.agents.report_generator import save_report, report_snapshot, restore_report_snapshot
from backend.agents.socratic_tutor import SocraticTutor, QuestionAttempt, TurnState, EvaluationResult, ContestationResult
from backend.db.models import Session as DbSession
from backend.db.models import SessionLocal
from backend.session_clock import elapsed_seconds, pause_clock
from backend.models.schemas import (
    Answer1Request,
    Answer1Response,
    Answer2Request,
    Answer2Response,
    CloseQuestionRequest,
    CloseQuestionResponse,
    ContestRequest,
    ContestResponse,
    NextQuestionRequest,
    NextQuestionResponse,
    QuestionOut,
    QuizGenerateRequest,
    QuizGenerateResponse,
    SessionCloseRequest,
    SessionCreate,
    SessionListResponse,
    SessionReportResponse,
    SessionResponse,
    TopicBreakdown,
    TutorStartRequest,
    TutorStartResponse,
)

router = APIRouter(prefix="/sessions", tags=["sessions"], dependencies=[Depends(require_api_owner)])
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory store per tutor e adaptive engine attivi
# ---------------------------------------------------------------------------

def _engine_from_session(sess: DbSession, *, include_unreliable: bool = False) -> AdaptiveEngine:
    engine = AdaptiveEngine(student_id=sess.student_id or 0, exam_id=sess.exam_id)
    for attempt in json.loads(sess.qa_log or "[]"):
        evaluation = attempt.get("evaluation") or {}
        contestation = attempt.get("contestation_result") or {}
        if not include_unreliable and (attempt.get("reliable") is False or evaluation.get("reliable") is False):
            continue
        score = attempt.get("final_score")
        if score is None:
            score = (contestation.get("final_score", evaluation.get("score"))
                     if contestation.get("reliable", True) else evaluation.get("score"))
        if score is not None:
            engine.record_answer(attempt["question_id"], attempt.get("topic_hint", ""), score)
    return engine


def _tutor_from_session(sess: DbSession, question_id: int, student_id: int) -> SocraticTutor | None:
    if not sess.current_turn_json:
        return None
    data = json.loads(sess.current_turn_json)
    if int(data.get("question_id", -1)) != question_id:
        return None
    material_ids = (json.loads(sess.quiz_json or "{}")).get("material_ids")
    tutor = SocraticTutor(student_id=student_id, exam_id=sess.exam_id, material_ids=material_ids)
    evaluation = data.get("evaluation")
    tutor.attempt = QuestionAttempt(
        question_id=data["question_id"], question_text=data["question_text"],
        difficulty=data.get("difficulty", "intermedia"), topic_hint=data.get("topic_hint", ""),
        student_answer_1=data.get("student_answer_1", ""), guiding_question=data.get("guiding_question", ""),
        positive_feedback=data.get("positive_feedback", ""), focus_area=data.get("focus_area", ""),
        student_answer_2=data.get("student_answer_2", ""), state=TurnState[data.get("state", "WAITING_ANSWER_1")],
    )
    if evaluation:
        tutor.attempt.evaluation = EvaluationResult(
            score=evaluation.get("score", 0), motivation=evaluation.get("motivation", ""),
            correct_points=evaluation.get("correct_points", []), missing_points=evaluation.get("missing_points", []),
            improvement_after_guidance=evaluation.get("improvement_after_guidance", ""),
            reliable=evaluation.get("reliable", True),
        )
    contestation = data.get("contestation_result")
    if contestation:
        tutor.attempt.contestation_result = ContestationResult(**contestation)
    tutor._source_cache = search_passages(
        data["question_text"], student_id, sess.exam_id, n_results=5,
        material_ids=material_ids,
    )
    tutor._context_cache = [passage["text"] for passage in tutor._source_cache]
    return tutor


def _persist_turn(session_id: int, tutor: SocraticTutor) -> None:
    db = SessionLocal()
    try:
        sess = db.query(DbSession).filter(DbSession.id == session_id).first()
        if sess:
            sess.current_turn_json = tutor.to_json()
            db.commit()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Helpers DB
# ---------------------------------------------------------------------------

def _get_session_or_404(session_id: int) -> DbSession:
    db = SessionLocal()
    try:
        sess = db.query(DbSession).filter(DbSession.id == session_id).first()
        if not sess:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sessione {session_id} non trovata.",
            )
        return sess
    finally:
        db.close()


def _session_to_response(sess: DbSession) -> SessionResponse:
    return SessionResponse(
        id=sess.id,
        exam_id=sess.exam_id,
        started_at=sess.started_at,
        ended_at=sess.ended_at,
        duration_seconds=elapsed_seconds(sess),
        score_total=sess.score_total,
        report_path=sess.report_path,
        report_available=bool(sess.report_path and Path(sess.report_path).is_file()),
        has_attempts=bool(sess.qa_log and sess.qa_log.strip() not in {"", "[]"}),
        has_quiz=bool(sess.quiz_json),
        status=sess.status,
    )


# ---------------------------------------------------------------------------
# 1 — Apri sessione
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Apre una nuova sessione di studio",
)
def create_session(
    payload: SessionCreate,
    student_id: int = Query(..., description="ID dello studente"),
) -> SessionResponse:
    """Crea una DbSession e inizializza l'AdaptiveEngine in memoria."""
    material_check = verify_exam_material(student_id, payload.exam_id, material_ids=payload.material_ids)
    if not material_check["ok"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"message": "Materiali dell'esame incoerenti", "errors": material_check["errors"]},
        )
    db = SessionLocal()
    try:
        started = datetime.now(timezone.utc)
        sess = DbSession(exam_id=payload.exam_id, student_id=student_id, status="in_corso",
                         started_at=started, active_started_at=started, duration_seconds=0.0)
        db.add(sess)
        db.commit()
        db.refresh(sess)

        # Inizializza AdaptiveEngine per questa sessione
        return _session_to_response(sess)
    except Exception as exc: # pylint: disable=broad-exception-caught
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Impossibile creare la sessione. Riprova.",
        ) from exc
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 2 — Genera quiz
# ---------------------------------------------------------------------------

@router.post(
    "/{session_id}/quiz",
    response_model=QuizGenerateResponse,
    summary="Genera il quiz per la sessione",
)
def generate_session_quiz(
    session_id: int,
    payload: QuizGenerateRequest,
    student_id: int = Query(..., description="ID dello studente"),
) -> QuizGenerateResponse:
    """
    Usa QuizGenerator (RAG + Groq) per produrre N domande
    calibrate sul materiale indicizzato.
    """
    sess = _require_session_owner(session_id, student_id)
    material_check = verify_exam_material(
        student_id, sess.exam_id, material_ids=payload.material_ids,
    )
    if not material_check["ok"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"message": "Materiali dell'esame incoerenti", "errors": material_check["errors"]},
        )

    try:
        result = generate_quiz(
            student_id=student_id,
            exam_id=sess.exam_id,
            topic_query=payload.topic_query,
            n_questions=payload.n_questions,
            source_name=payload.source_name,
            material_ids=payload.material_ids,
        )
    except Exception as exc:
        if is_rate_limit_error(exc):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=public_error(exc, "Limite di richieste Groq raggiunto. Riprova più tardi."),
            ) from exc
        if isinstance(exc, UnverifiableQuizError):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Il modello non ha prodotto domande con citazioni verificabili. Riprova con queste fonti o selezionane altre.",
            ) from exc
        if isinstance(exc, ValueError):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Impossibile generare le domande con le fonti selezionate.",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=public_error(exc, "Impossibile generare le domande. Riprova."),
        ) from exc

    if result.n_questions != payload.n_questions or len(result.questions) != payload.n_questions:
        produced = len(result.questions)
        label = "domanda verificabile" if produced == 1 else "domande verificabili"
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(f"Sono state generate solo {produced} {label} "
                    f"su {payload.n_questions} richieste. Riprova con queste fonti o selezionane altre."),
        )


    db = SessionLocal()
    try:
        stored = db.query(DbSession).filter(DbSession.id == session_id).first()
        if stored:
            quiz_data = asdict(result)
            if payload.material_ids is not None:
                quiz_data["material_ids"] = payload.material_ids
            stored.quiz_json = json.dumps(quiz_data, ensure_ascii=False)
            db.commit()
    finally:
        db.close()


    return QuizGenerateResponse(
        exam_id=result.exam_id,
        student_id=result.student_id,
        source_name=result.source_name,
        n_questions=result.n_questions,
        prompt_version=result.prompt_version,
        questions=[
            QuestionOut(
                id=q.id,
                text=q.text,
                difficulty=q.difficulty,
                topic_hint=q.topic_hint,
            )
            for q in result.questions
        ],
    )


def _require_session_owner(session_id: int, student_id: int) -> DbSession:
    sess = _get_session_or_404(session_id)
    if sess.student_id != student_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sessione non trovata.")
    return sess


# ---------------------------------------------------------------------------
# 3 — Avvia loop socratico per una domanda
# ---------------------------------------------------------------------------

@router.post(
    "/{session_id}/tutor/start",
    response_model=TutorStartResponse,
    summary="Avvia il tutor socratico per una domanda",
)
def tutor_start(
    session_id: int,
    payload: TutorStartRequest,
    student_id: int = Query(..., description="ID dello studente"),
) -> TutorStartResponse:
    """Inizializza il SocraticTutor e fa il pre-fetch dei chunk RAG."""
    sess = _get_session_or_404(session_id)
    if sess.status != "in_corso":
        raise HTTPException(status_code=409, detail="Riprendi la sessione prima di avviare una domanda.")
    quiz = json.loads(sess.quiz_json or "{}")
    canonical = next((question for question in quiz.get("questions", [])
                      if question.get("id") == payload.question.id), None)
    if canonical is None or QuestionOut(**canonical) != payload.question:
        raise HTTPException(status_code=422, detail="La domanda non appartiene al quiz salvato.")
    answered = {item.get("question_id") for item in json.loads(sess.qa_log or "[]")}
    if payload.question.id in answered or sess.current_turn_json:
        raise HTTPException(status_code=409, detail="Domanda già avviata o completata.")

    tutor = SocraticTutor(
        student_id=student_id,
        exam_id=sess.exam_id,
        material_ids=quiz.get("material_ids"),
    )
    q_dict = canonical
    question_text = tutor.start(q_dict)

    _persist_turn(session_id, tutor)

    return TutorStartResponse(
        question_text=question_text,
        difficulty=payload.question.difficulty,
        topic_hint=payload.question.topic_hint,
    )


# ---------------------------------------------------------------------------
# 4 — Prima risposta → domanda guida
# ---------------------------------------------------------------------------

@router.post(
    "/{session_id}/tutor/answer1",
    response_model=Answer1Response,
    summary="Invia la prima risposta, ricevi la domanda guida socratica",
)
def tutor_answer1(session_id: int, payload: Answer1Request) -> Answer1Response:
    sess = _get_session_or_404(session_id)
    tutor = _tutor_from_session(sess, payload.question_id, student_id=sess.student_id or 0)
    if not tutor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Nessun tutor attivo per sessione {session_id}, domanda {payload.question_id}.",
        )
    try:
        tutor._assert_state(TurnState.WAITING_ANSWER_1)
        tutor.attempt.student_answer_1 = payload.answer.strip()
        _persist_turn(session_id, tutor)
        data = tutor.submit_answer_1(payload.answer)
        _persist_turn(session_id, tutor)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Turno non disponibile. Ricarica la sessione.") from exc
    except Exception as exc: # pylint: disable=broad-exception-caught
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Impossibile generare la domanda guida. Riprova.",
        ) from exc

    return Answer1Response(
        guiding_question=data["guiding_question"],
        positive_feedback=data["positive_feedback"],
        focus_area=data["focus_area"],
    )


# ---------------------------------------------------------------------------
# 5 — Seconda risposta → valutazione
# ---------------------------------------------------------------------------

@router.post(
    "/{session_id}/tutor/answer2",
    response_model=Answer2Response,
    summary="Invia la seconda risposta, ricevi la valutazione",
)
def tutor_answer2(session_id: int, payload: Answer2Request) -> Answer2Response:
    sess = _get_session_or_404(session_id)
    tutor = _tutor_from_session(sess, payload.question_id, student_id=sess.student_id or 0)
    if not tutor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Nessun tutor attivo per sessione {session_id}, domanda {payload.question_id}.",
        )
    try:
        tutor._assert_state(TurnState.WAITING_ANSWER_2)
        tutor.attempt.student_answer_2 = payload.answer.strip()
        _persist_turn(session_id, tutor)
        data = tutor.submit_answer_2(payload.answer)
        _persist_turn(session_id, tutor)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Turno non disponibile. Ricarica la sessione.") from exc
    except Exception as exc: # pylint: disable=broad-exception-caught
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Impossibile valutare la risposta. Riprova.",
        ) from exc

    return Answer2Response(
        score=data["score"],
        motivation=data["motivation"],
        correct_points=data["correct_points"],
        missing_points=data["missing_points"],
        improvement_after_guidance=data["improvement_after_guidance"],
        reliable=data["reliable"],
        can_contest=data["can_contest"],
    )


# ---------------------------------------------------------------------------
# 6 — Contestazione (opzionale)
# ---------------------------------------------------------------------------

@router.post(
    "/{session_id}/tutor/contest",
    response_model=ContestResponse,
    summary="Contesta la valutazione ricevuta",
)
def tutor_contest(session_id: int, payload: ContestRequest) -> ContestResponse:
    sess = _get_session_or_404(session_id)
    tutor = _tutor_from_session(sess, payload.question_id, student_id=sess.student_id or 0)
    if not tutor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Nessun tutor attivo per sessione {session_id}, domanda {payload.question_id}.",
        )
    try:
        data = tutor.submit_contestation(payload.argumentation)
        _persist_turn(session_id, tutor)
    except RuntimeError as exc:
        logger.exception("Contestazione fallita per la sessione %s", session_id)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Turno non disponibile. Ricarica la sessione.") from exc
    except Exception as exc: # pylint: disable=broad-exception-caught
        logger.exception("Riesame fallito per la sessione %s", session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Impossibile rivalutare la risposta. Riprova.",
        ) from exc

    return ContestResponse(
        final_score=data["final_score"],
        score_changed=data["score_changed"],
        score_delta=data["score_delta"],
        contestation_assessment=data["contestation_assessment"],
        final_motivation=data["final_motivation"],
        response_to_student=data["response_to_student"],
    )


# ---------------------------------------------------------------------------
# 7 — Chiudi domanda + aggiorna AdaptiveEngine + persisti qa_log
# ---------------------------------------------------------------------------

@router.post(
    "/{session_id}/tutor/close",
    response_model=CloseQuestionResponse,
    summary="Chiude la domanda corrente e aggiorna l'engine adattivo",
)
def tutor_close(session_id: int, payload: CloseQuestionRequest) -> CloseQuestionResponse:
    sess = _get_session_or_404(session_id)
    tutor = _tutor_from_session(sess, payload.question_id, student_id=sess.student_id or 0)
    if not tutor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Nessun tutor attivo per sessione {session_id}, domanda {payload.question_id}.",
        )
    try:
        attempt = tutor.close()
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Turno non disponibile. Ricarica la sessione.") from exc

    final_score = attempt.final_score
    if final_score is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Impossibile chiudere: nessun voto disponibile.",
        )

    # L'engine viene sempre ricostruito dal log persistito.
    engine = _engine_from_session(sess)

    # Persisti il tentativo nel qa_log della DbSession
    try:
        entry = tutor.to_dict()
        entry["final_score"] = final_score
        entry["reliable"] = bool(attempt.evaluation is None or attempt.evaluation.reliable)
        _append_attempt_to_log(session_id, entry)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Impossibile salvare la risposta. Riprova.",
        ) from exc

    return CloseQuestionResponse(
        final_score=final_score,
        topic_hint=attempt.topic_hint,
    )


def _append_attempt_to_log(session_id: int, attempt_dict: dict) -> None:
    """Record the answer and clear the active turn in one commit."""
    db = SessionLocal()
    try:
        sess = db.query(DbSession).filter(DbSession.id == session_id).first()
        if not sess or not sess.current_turn_json:
            raise RuntimeError("Turno non disponibile.")
        existing: list = json.loads(sess.qa_log) if sess.qa_log else []
        if any(item.get("question_id") == attempt_dict.get("question_id") for item in existing):
            raise RuntimeError("Domanda già registrata.")
        existing.append(attempt_dict)
        sess.qa_log = json.dumps(existing, ensure_ascii=False)
        sess.current_turn_json = None
        db.commit()
    except Exception as exc: # pylint: disable=broad-exception-caught
        db.rollback()
        raise RuntimeError(
            f"Impossibile salvare il tentativo della sessione {session_id}: {exc}"
        ) from exc
    finally:
        db.close()


def _mark_session_interrupted(session_id: int) -> None:
    """Persist an interruption while preserving any partial ``qa_log``."""
    db = SessionLocal()
    try:
        sess = db.query(DbSession).filter(DbSession.id == session_id).first()
        if sess and sess.status == "in_corso":
            pause_clock(sess)
            sess.status = "interrotta"
            sess.ended_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 8 — Prossima domanda (AdaptiveEngine)
# ---------------------------------------------------------------------------

@router.post(
    "/{session_id}/next-question",
    response_model=NextQuestionResponse,
    summary="Ottieni la prossima domanda suggerita dall'AdaptiveEngine",
)
def next_question(session_id: int, payload: NextQuestionRequest) -> NextQuestionResponse:
    sess = _get_session_or_404(session_id)
    engine = _engine_from_session(sess)
    if sess.current_turn_json:
        raise HTTPException(status_code=409, detail="Concludi la domanda corrente prima di continuare.")
    answered = {item.get("question_id") for item in json.loads(sess.qa_log or "[]")}
    remaining = [q for q in json.loads(sess.quiz_json or "{}").get("questions", [])
                 if q.get("id") not in answered]
    chosen = engine.pick_next_question(remaining)

    if chosen is None:
        return NextQuestionResponse(
            question=None,
            is_last=True,
            message="Tutte le domande sono state completate.",
        )

    return NextQuestionResponse(
        question=QuestionOut(**chosen),
        is_last=len(remaining) == 1,
        message=f"Prossima domanda su: {chosen.get('topic_hint', '')}",
    )


# ---------------------------------------------------------------------------
# 9 — Chiudi sessione + genera report
# ---------------------------------------------------------------------------

@router.post(
    "/{session_id}/close",
    response_model=SessionReportResponse,
    summary="Chiude la sessione e genera il report finale",
)
def close_session(session_id: int, payload: SessionCloseRequest) -> SessionReportResponse:
    sess = _get_session_or_404(session_id)
    engine = _engine_from_session(sess, include_unreliable=True)
    report = engine.generate_session_report()
    if not report["n_questions_answered"]:
        raise HTTPException(status_code=409, detail="Completa almeno una domanda prima di chiudere la sessione.")
    snapshot = report_snapshot(report["student_id"], report["exam_id"], session_id)
    paths = save_report(report, session_id, payload.exam_name)

    # Aggiorna DB
    db = SessionLocal()
    try:
        sess = db.query(DbSession).filter(DbSession.id == session_id).first()
        if sess:
            pause_clock(sess)
            sess.ended_at    = datetime.now(timezone.utc)
            sess.score_total = report["overall_avg"]
            sess.report_path = paths.get("pdf")
            sess.status      = "completata"
            sess.current_turn_json = None
            db.commit()
    except Exception as exc: # pylint: disable=broad-exception-caught
        db.rollback()
        try:
            restore_report_snapshot(snapshot)
        except OSError as cleanup_exc:
            raise HTTPException(
                status_code=500,
                detail="Sessione non completata e pulizia del report incompleta. Contatta l'amministratore.",
            ) from cleanup_exc
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Sessione non completata: impossibile salvare il report. Riprova.",
        ) from exc
    finally:
        db.close()

    return SessionReportResponse(
        session_id=session_id,
        student_id=report["student_id"],
        exam_id=report["exam_id"],
        overall_avg=report["overall_avg"],
        n_questions_answered=report["n_questions_answered"],
        weak_topics=report["weak_topics"],
        strong_topics=report["strong_topics"],
        topic_breakdown=[TopicBreakdown(**t) for t in report["topic_breakdown"]],
        summary=report["summary"],
        report_md_path=paths.get("md"),
        report_pdf_path=paths.get("pdf"),
    )


# ---------------------------------------------------------------------------
# 10 — Dettaglio sessione
# ---------------------------------------------------------------------------

@router.get(
    "/{session_id}",
    response_model=SessionResponse,
    summary="Dettaglio sessione",
)
def get_session(session_id: int) -> SessionResponse:
    sess = _get_session_or_404(session_id)
    return _session_to_response(sess)


@router.get("/{session_id}/resume", summary="Ricostruisce lo stato persistito della sessione")
def resume_session(session_id: int) -> dict:
    """Restituisce quiz, turno parziale e statistiche ricostruite dal DB."""
    sess = _get_session_or_404(session_id)
    return {
        "session_id": session_id,
        "status": sess.status,
        "quiz": json.loads(sess.quiz_json) if sess.quiz_json else None,
        "current_turn": json.loads(sess.current_turn_json) if sess.current_turn_json else None,
        "adaptive_engine": _engine_from_session(sess).to_json(),
    }


# ---------------------------------------------------------------------------
# 11 — Lista sessioni per esame
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=SessionListResponse,
    summary="Lista sessioni per un esame",
)
def list_sessions(
    exam_id: int = Query(..., description="ID dell'esame"),
) -> SessionListResponse:
    """Restituisce le sessioni senza modificarle, incluse quelle interrotte."""
    db = SessionLocal()
    try:
        sessions = (
            db.query(DbSession)
            .filter(DbSession.exam_id == exam_id)
            .order_by(DbSession.started_at.desc())
            .all()
        )
        items = [_session_to_response(s) for s in sessions]
        return SessionListResponse(sessions=items, total=len(items))
    finally:
        db.close()
