"""HTTP adapters for actions previously performed inside Streamlit pages."""

import logging
import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field

from backend.agents.ingestor import get_collection, ingest_pdf, search_passages, verify_exam_material
from backend.agents.llm_errors import is_rate_limit_error
from backend.agents.notes_generator import generate_notes, generate_notes_pdf_from_materials
from backend.agents.socratic_tutor import DoubtAttempt, DoubtTutor, NoRelevantMaterialError
from backend.api.authorization import require_api_owner
from backend.db.models import DoubtConversation, DoubtThread, Exam, Material, SessionLocal
from backend.error_messages import public_error

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/exams", tags=["exams"], dependencies=[Depends(require_api_owner)])


class RenameExam(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class MaterialSelection(BaseModel):
    material_ids: list[int] | None = None


class DoubtStart(BaseModel):
    doubt: str = Field(min_length=1, max_length=10000)
    thread_id: str | None = None


class DoubtAnswer(BaseModel):
    answer: str = Field(min_length=1, max_length=10000)


def _owned_exam(exam_id: int, student_id: int) -> Exam:
    with SessionLocal() as db:
        exam = db.query(Exam).filter(Exam.id == exam_id, Exam.student_id == student_id).first()
        if exam is None:
            raise HTTPException(status_code=404, detail="Esame non trovato.")
        return exam


def _selected_materials(exam_id: int, ids: list[int] | None) -> list[Material]:
    with SessionLocal() as db:
        materials = db.query(Material).filter(Material.exam_id == exam_id).all()
        if ids is None:
            return materials
        selected = [material for material in materials if material.id in set(ids)]
        if len(selected) != len(set(ids)):
            raise HTTPException(status_code=422, detail="Una o più fonti non appartengono all'esame.")
        return selected


@router.get("/{exam_id}/materials")
def list_materials(exam_id: int, student_id: int) -> dict:
    _owned_exam(exam_id, student_id)
    return {"materials": [
        {"id": m.id, "filename": m.filename, "pages_count": m.pages_count,
         "pages_with_text": m.pages_with_text, "pages_skipped": m.pages_skipped,
         "chunks_count": m.chunks_count, "uploaded_at": m.uploaded_at}
        for m in _selected_materials(exam_id, None)
    ]}


@router.patch("/{exam_id}")
def rename_exam(exam_id: int, payload: RenameExam, student_id: int) -> dict:
    _owned_exam(exam_id, student_id)
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Inserisci un nome per l'esame.")
    with SessionLocal() as db:
        exam = db.get(Exam, exam_id)
        exam.name = name
        db.commit()
    return {"id": exam_id, "name": name}


@router.post("/{exam_id}/verify")
def verify_materials(exam_id: int, payload: MaterialSelection, student_id: int) -> dict:
    _owned_exam(exam_id, student_id)
    return verify_exam_material(student_id, exam_id, material_ids=payload.material_ids)


@router.post("/{exam_id}/reindex")
def reindex_materials(exam_id: int, student_id: int) -> dict:
    _owned_exam(exam_id, student_id)
    materials = _selected_materials(exam_id, None)
    with SessionLocal() as db:
        for material in materials:
            path = Path(material.stored_path)
            if not path.exists():
                raise HTTPException(status_code=409, detail=f"File mancante: {material.filename}")
            collection = get_collection(student_id, exam_id)
            old_where = {"file_hash": material.file_hash} if material.file_hash else {"source_file": material.filename}
            old = collection.get(where=old_where, include=["documents", "metadatas"])
            stats = None
            try:
                if old.get("ids"):
                    collection.delete(ids=old["ids"])
                stats = ingest_pdf(str(path), student_id, exam_id, source_file=material.filename)
                stored = db.get(Material, material.id)
                stored.file_hash = stats.get("file_hash")
                stored.chunks_count = stats.get("chunks", 0)
                stored.pages_count = stats.get("pages", 0)
                stored.pages_with_text = stats.get("pages_with_text", 0)
                stored.pages_skipped = stats.get("pages_skipped", 0)
                db.commit()
            except Exception as exc:
                logger.exception("Reindex failed for material %s", material.id)
                db.rollback()
                try:
                    if stats is not None:
                        new_ids = collection.get(where={"file_hash": stats["file_hash"]}).get("ids") or []
                        if new_ids:
                            collection.delete(ids=new_ids)
                    if old.get("ids"):
                        collection.add(ids=old["ids"], documents=old["documents"],
                                       metadatas=old["metadatas"])
                except Exception as restore_exc:
                    logger.exception("Failed to restore index for material %s", material.id)
                    raise HTTPException(
                        status_code=500,
                        detail="Reindicizzazione fallita e indice precedente non ripristinato. Verifica i materiali.",
                    ) from restore_exc
                raise HTTPException(status_code=500, detail=public_error(exc, "Reindicizzazione fallita. Riprova.")) from exc
    return verify_exam_material(student_id, exam_id)


@router.post("/{exam_id}/notes/pdf")
def exam_notes_pdf(exam_id: int, payload: MaterialSelection, student_id: int) -> Response:
    exam = _owned_exam(exam_id, student_id)
    materials = _selected_materials(exam_id, payload.material_ids)
    if not materials:
        raise HTTPException(status_code=422, detail="Seleziona almeno una fonte.")
    check = verify_exam_material(student_id, exam_id, [m.id for m in materials])
    if not check["ok"]:
        raise HTTPException(status_code=409, detail=check["errors"])
    try:
        pdf = generate_notes_pdf_from_materials(student_id, exam_id, materials, exam_name_fallback=exam.name)
    except Exception as exc:
        if is_rate_limit_error(exc):
            logger.warning("Notes PDF generation rate limited for exam %s", exam_id)
            raise HTTPException(status_code=429,
                                detail=public_error(exc, "Limite temporaneo Groq. Riprova più tardi."),
                                headers=getattr(exc, "headers", None)) from exc
        logger.exception("Notes PDF generation failed")
        raise HTTPException(status_code=500, detail=public_error(exc, "Impossibile generare gli appunti. Riprova.")) from exc
    return Response(pdf, media_type="application/pdf", headers={"Content-Disposition": 'attachment; filename="appunti.pdf"'})


@router.post("/{exam_id}/notes")
def exam_notes(exam_id: int, student_id: int) -> dict:
    exam = _owned_exam(exam_id, student_id)
    check = verify_exam_material(student_id, exam_id)
    if not check["ok"]:
        raise HTTPException(status_code=409, detail=check["errors"])
    try:
        return {"notes": generate_notes(student_id, exam_id, source_name=exam.name)}
    except Exception as exc:
        logger.exception("Notes generation failed")
        raise HTTPException(status_code=500, detail=public_error(exc, "Impossibile generare gli appunti. Riprova.")) from exc


@router.post("/{exam_id}/doubts")
def start_doubt(exam_id: int, payload: DoubtStart, student_id: int) -> dict:
    _owned_exam(exam_id, student_id)
    doubt = payload.doubt.strip()
    with SessionLocal() as db:
        if payload.thread_id:
            thread = db.query(DoubtThread).filter_by(id=payload.thread_id, student_id=student_id, exam_id=exam_id).first()
            if thread is None:
                raise HTTPException(status_code=404, detail="Chat non trovata.")
        else:
            thread = (db.query(DoubtThread).filter_by(student_id=student_id, exam_id=exam_id)
                      .order_by(DoubtThread.created_at.desc(), DoubtThread.id.desc()).first())
            if thread is None:
                thread = DoubtThread(id=str(uuid.uuid4()), student_id=student_id, exam_id=exam_id,
                                     title=doubt[:160])
                db.add(thread)
                db.commit()
        thread_id = thread.id
        pending = (db.query(DoubtConversation)
                   .filter_by(student_id=student_id, exam_id=exam_id, thread_id=thread_id)
                   .order_by(DoubtConversation.created_at.desc(), DoubtConversation.id.desc())
                   .first())
        if pending is not None and not pending.answered:
            if pending.doubt_text.strip().casefold() == doubt.casefold():
                return {"conversation_id": pending.id, "guiding_question": pending.guiding_question,
                        "topic_hint": pending.topic_hint}
            raise HTTPException(status_code=409, detail="Rispondi al dubbio aperto prima di iniziarne un altro.")
    tutor = DoubtTutor(student_id=student_id, exam_id=exam_id)
    try:
        data = tutor.start(doubt)
    except NoRelevantMaterialError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Doubt question generation failed")
        raise HTTPException(status_code=500, detail=public_error(exc, "Impossibile generare la domanda guida. Riprova.")) from exc
    conversation_id = str(uuid.uuid4())
    with SessionLocal() as db:
        thread = db.query(DoubtThread).filter_by(id=thread_id, student_id=student_id, exam_id=exam_id).first()
        if thread is None:
            raise HTTPException(status_code=404, detail="Chat non trovata.")
        if thread.title == "Nuova chat":
            thread.title = doubt[:160]
        db.add(DoubtConversation(id=conversation_id, student_id=student_id, exam_id=exam_id, thread_id=thread_id,
                                 doubt_text=doubt, guiding_question=data["guiding_question"],
                                 topic_hint=data.get("topic_hint", "")))
        db.commit()
    return {"conversation_id": conversation_id, **data}


@router.post("/{exam_id}/doubts/{conversation_id}/answer")
def answer_doubt(exam_id: int, conversation_id: str, payload: DoubtAnswer, student_id: int) -> dict:
    _owned_exam(exam_id, student_id)
    with SessionLocal() as db:
        pending = db.query(DoubtConversation).filter(
            DoubtConversation.id == conversation_id, DoubtConversation.exam_id == exam_id,
            DoubtConversation.student_id == student_id, DoubtConversation.answered == 0,
        ).first()
        if pending is None:
            raise HTTPException(status_code=404, detail="Dubbio non trovato o già concluso.")
        doubt, guide, topic = pending.doubt_text, pending.guiding_question, pending.topic_hint
    tutor = DoubtTutor(student_id=student_id, exam_id=exam_id)
    tutor.attempt = DoubtAttempt(doubt_text=doubt, guiding_question=guide, topic_hint=topic)
    try:
        tutor._source_cache = search_passages(doubt, student_id, exam_id, n_results=tutor.n_chunks)
        tutor._context_cache = [passage["text"] for passage in tutor._source_cache]
        result = tutor.submit_answer(payload.answer)
    except Exception as exc:
        logger.exception("Doubt answer evaluation failed")
        raise HTTPException(status_code=500, detail=public_error(exc, "Impossibile valutare la risposta. Riprova.")) from exc
    with SessionLocal() as db:
        pending = db.get(DoubtConversation, conversation_id)
        pending.answered = 1
        pending.student_answer = payload.answer.strip()
        pending.result_json = json.dumps(result, ensure_ascii=False)
        db.commit()
    return result


@router.delete("/{exam_id}/doubts/{conversation_id}", status_code=204)
def discard_doubt(exam_id: int, conversation_id: str, student_id: int) -> Response:
    _owned_exam(exam_id, student_id)
    with SessionLocal() as db:
        pending = db.query(DoubtConversation).filter(
            DoubtConversation.id == conversation_id,
            DoubtConversation.exam_id == exam_id,
            DoubtConversation.student_id == student_id,
        ).first()
        if pending:
            db.delete(pending)
            db.commit()
    return Response(status_code=204)
