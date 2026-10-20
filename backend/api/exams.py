
import hashlib
import logging
import os
import uuid
from pathlib import Path
from sqlalchemy.exc import SQLAlchemyError
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from starlette.concurrency import run_in_threadpool

from backend.agents.ingestor import ingest_pdf
from backend.api.authorization import require_api_owner
from backend.materials import delete_material, rollback_uncommitted_ingest
from backend.error_messages import public_error
from backend.exam_deletion import delete_exam as remove_exam, ExamDeletionCleanupError, ExamDeletionRecoveryError
from backend.security import safe_name
from backend.db.models import Exam, Material, Session as DbSession, SessionLocal
from backend.models.schemas import (
    ExamCreate,
    ExamListResponse,
    ExamResponse,
    IngestResponse,
)

router = APIRouter(prefix="/exams", tags=["exams"], dependencies=[Depends(require_api_owner)])
logger = logging.getLogger(__name__)

DEFAULT_UPLOAD_DIR = Path(__file__).resolve().parents[2] / "data" / "uploads"
UPLOAD_DIR = DEFAULT_UPLOAD_DIR
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers DB
# ---------------------------------------------------------------------------

def _get_exam_or_404(exam_id: int, student_id: int) -> Exam:
    db = SessionLocal()
    try:
        exam = (
            db.query(Exam)
            .filter(Exam.id == exam_id, Exam.student_id == student_id)
            .first()
        )
        if not exam:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Esame {exam_id} non trovato per lo studente {student_id}.",
            )
        return exam
    finally:
        db.close()


def _enrich_exam(exam: Exam, db) -> ExamResponse:
    """Aggiunge n_sessions e avg_score all'ExamResponse."""
    sessions = (
        db.query(DbSession)
        .filter(DbSession.exam_id == exam.id)
        .all()
    )
    n_sessions = len(sessions)
    scored = [s.score_total for s in sessions if s.score_total is not None]
    avg = round(sum(scored) / len(scored), 1) if scored else None

    return ExamResponse(
        id=exam.id,
        student_id=exam.student_id,
        name=exam.name,
        created_at=exam.created_at,
        n_sessions=n_sessions,
        avg_score=avg,
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=ExamListResponse,
    summary="Lista esami dello studente",
)
def list_exams(student_id: int = Query(..., description="ID dello studente")) -> ExamListResponse:
    """Restituisce tutti gli esami creati dallo studente, con statistiche aggregate."""
    db = SessionLocal()
    try:
        exams = (
            db.query(Exam)
            .filter(Exam.student_id == student_id)
            .order_by(Exam.created_at.desc())
            .all()
        )
        items = [_enrich_exam(e, db) for e in exams]
        return ExamListResponse(exams=items, total=len(items))
    finally:
        db.close()


@router.post(
    "",
    response_model=ExamResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Crea un nuovo esame",
)
def create_exam(
    payload: ExamCreate,
    student_id: int = Query(..., description="ID dello studente"),
) -> ExamResponse:
    """Crea un nuovo esame vuoto. Il PDF viene caricato separatamente via /ingest."""
    db = SessionLocal()
    try:
        exam = Exam(student_id=student_id, name=payload.name)
        db.add(exam)
        db.commit()
        db.refresh(exam)
        return ExamResponse(
            id=exam.id,
            student_id=exam.student_id,
            name=exam.name,
            created_at=exam.created_at,
            n_sessions=0,
            avg_score=None,
        )
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Failed to create exam")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Impossibile creare l'esame. Riprova.",
        ) from exc
    finally:
        db.close()


@router.delete(
    "/{exam_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Elimina un esame",
)
def delete_exam(
    exam_id: int,
    student_id: int = Query(..., description="ID dello studente"),
) -> None:
    try:
        remove_exam(student_id, exam_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ExamDeletionCleanupError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ExamDeletionRecoveryError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to delete exam %s for student %s", exam_id, student_id)
        raise HTTPException(status_code=500, detail="Impossibile eliminare l'esame. Riprova.") from exc


@router.post(
    "/{exam_id}/ingest",
    response_model=IngestResponse,
    summary="Carica e indicizza un PDF per l'esame",
)
async def ingest_pdf_endpoint(
    exam_id: int,
    student_id: int = Query(..., description="ID dello studente"),
    file: UploadFile = File(..., description="File PDF del materiale di studio"),
) -> IngestResponse:
    """
    Carica un file PDF, lo salva temporaneamente e avvia l'indicizzazione
    tramite ChromaDB + embedding locale.

    Se la collection esiste già, i nuovi chunk vengono aggiunti a quelli esistenti.
    Per sostituire completamente il materiale, elimina l'esame e ricrealo.
    """
    # Verifica che l'esame appartenga allo studente
    _get_exam_or_404(exam_id, student_id)

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Il file deve essere un PDF (.pdf).",
        )

    pdf_path = None
    try:
    # Bound the read before allocating or writing the uploaded content.
        max_bytes = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
        content = await file.read(max_bytes + 1)
        if len(content) > max_bytes:
            raise HTTPException(status_code=413, detail="PDF troppo grande.")
        if not content.startswith(b"%PDF-"):
            raise HTTPException(status_code=400, detail="Il contenuto non è un PDF valido.")
        file_hash = hashlib.sha256(content).hexdigest()
        stored_name = f"{uuid.uuid4().hex}.pdf"
        pdf_path = UPLOAD_DIR / stored_name
        with pdf_path.open("wb") as dest:
            dest.write(content)
    except OSError as exc:
        logger.exception("Failed to save uploaded PDF")
        if pdf_path is not None:
            try:
                pdf_path.unlink(missing_ok=True)
            except OSError:
                logger.exception("Failed to remove partial uploaded PDF")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Impossibile salvare il PDF. Controlla lo spazio disponibile e riprova.",
        ) from exc

    # Indicizzazione
    try:
        stats = await run_in_threadpool(ingest_pdf, str(pdf_path), student_id, exam_id, source_file=safe_name(file.filename))
    except ValueError as exc:
        try:
            pdf_path.unlink(missing_ok=True)
        except OSError:
            logger.exception("Failed to remove PDF after validation error")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=public_error(exc, "PDF non utilizzabile. Controlla il file e riprova."),
        ) from exc
    except Exception as exc:
        logger.exception("Failed to ingest PDF for exam %s", exam_id)
        try:
            pdf_path.unlink(missing_ok=True)
        except OSError:
            logger.exception("Failed to remove PDF after ingest error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=public_error(exc, "Impossibile indicizzare il PDF. Riprova."),
        ) from exc

    db = SessionLocal()
    try:
        existing = db.query(Material).filter(
            Material.exam_id == exam_id,
            Material.file_hash == stats["file_hash"],
        ).first()
        if not existing:
            db.add(Material(
                exam_id=exam_id,
                filename=safe_name(file.filename or "documento.pdf"),
                stored_path=str(pdf_path),
                pages_count=stats["pages"],
                pages_with_text=stats.get("pages_with_text", stats["pages"]),
                pages_skipped=stats.get("pages_skipped", 0),
                chunks_count=stats["chunks"],
                file_hash=stats["file_hash"],
            ))
            db.commit()
        else:
            pdf_path.unlink(missing_ok=True)
    except Exception as exc:
        db.rollback()
        try:
            rollback_uncommitted_ingest(student_id, exam_id, stats["file_hash"])
        except Exception as cleanup_exc:
            logger.exception("Failed to roll back indexed chunks for exam %s", exam_id)
            raise HTTPException(
                status_code=500,
                detail="PDF non salvato e pulizia dell'indice incompleta. Contatta l'amministratore.",
            ) from cleanup_exc
        try:
            pdf_path.unlink(missing_ok=True)
        except OSError:
            logger.exception("Failed to remove uploaded PDF after database error")
        logger.exception("Failed to save material record for exam %s", exam_id)
        raise HTTPException(status_code=500, detail="PDF non salvato. Riprova.") from exc
    finally:
        db.close()
    return IngestResponse(
        file=stats["file"],
        pages=stats["pages"],
        chunks=stats["chunks"],
        collection=stats["collection"],
    )


@router.delete(
    "/{exam_id}/materials/{material_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Elimina una fonte dall'esame",
)
def delete_exam_material(
    exam_id: int,
    material_id: int,
    student_id: int = Query(..., description="ID dello studente"),
) -> None:
    try:
        delete_material(student_id, exam_id, material_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to delete material %s", material_id)
        raise HTTPException(
            status_code=500, detail="Impossibile eliminare la fonte. Riprova."
        ) from exc
