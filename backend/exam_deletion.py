"""Coordinated deletion of an exam and its stored artifacts."""

import logging
import os
import tempfile
from pathlib import Path

from chromadb.errors import NotFoundError

from backend.agents import ingestor, report_generator
from backend.db.models import Exam, Material, SessionLocal

logger = logging.getLogger(__name__)


class ExamDeletionCleanupError(RuntimeError):
    """The exam is gone, but a staged file could not be removed."""


class ExamDeletionRecoveryError(RuntimeError):
    """Deletion failed and at least one rollback step also failed."""


def delete_exam(student_id: int, exam_id: int) -> None:
    """Remove an owned exam, restoring staged files and vectors on failure."""
    with SessionLocal() as db:
        exam = db.query(Exam).filter(Exam.id == exam_id, Exam.student_id == student_id).first()
        if exam is None:
            raise LookupError("Esame non trovato.")

        paths = set()
        for material in db.query(Material).filter(Material.exam_id == exam_id):
            shared = db.query(Material.id).filter(
                Material.id != material.id, Material.stored_path == material.stored_path
            ).first()
            if not shared:
                paths.add(Path(material.stored_path))
        prefix = f"{student_id}_{exam_id}_"
        for path in report_generator.REPORTS_DIR.glob(f"{prefix}*"):
            if path.suffix in (".pdf", ".md") and path.stem.removeprefix(prefix).isdigit():
                paths.add(path)

        name = f"student_{student_id}_exam_{exam_id}"
        chroma = ingestor._get_client()
        try:
            collection = chroma.get_collection(name, embedding_function=ingestor.local_ef)
        except (NotFoundError, ValueError):
            collection = None
        snapshot = collection.get(include=["documents", "metadatas", "embeddings"]) if collection else None
        metadata = dict(collection.metadata or {}) if collection else None

        staged = []
        try:
            for path in paths:
                if path.is_file():
                    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.delete-", delete=False) as temp:
                        backup = Path(temp.name)
                    try:
                        os.replace(path, backup)
                    except Exception:
                        backup.unlink(missing_ok=True)
                        raise
                    staged.append((path, backup))

            if collection is not None:
                ingestor.delete_exam_collection(student_id, exam_id)
            db.delete(exam)
            db.commit()
        except Exception as original_error:
            db.rollback()
            recovery_failed = False
            if collection is not None:
                try:
                    try:
                        restored = chroma.get_collection(name, embedding_function=ingestor.local_ef)
                    except (NotFoundError, ValueError):
                        restored = chroma.get_or_create_collection(
                            name, embedding_function=ingestor.local_ef,
                            **({"metadata": metadata} if metadata else {}),
                        )
                    if snapshot and snapshot["ids"]:
                        present = set(restored.get(ids=snapshot["ids"])["ids"])
                        missing = [i for i, chunk_id in enumerate(snapshot["ids"]) if chunk_id not in present]
                        if missing:
                            restored.add(
                                ids=[snapshot["ids"][i] for i in missing],
                                documents=[snapshot["documents"][i] for i in missing],
                                metadatas=[snapshot["metadatas"][i] for i in missing],
                                embeddings=[snapshot["embeddings"][i].tolist() for i in missing],
                            )
                except Exception:
                    logger.exception("Failed to restore exam collection %s after deletion error", name)
                    recovery_failed = True
            for path, backup in reversed(staged):
                try:
                    os.replace(backup, path)
                except Exception:
                    logger.exception("Failed to restore staged exam file %s", path)
                    recovery_failed = True
            if recovery_failed:
                raise ExamDeletionRecoveryError(
                    "Eliminazione fallita e ripristino incompleto. Contatta l'amministratore."
                ) from original_error
            raise

        for _, backup in staged:
            try:
                backup.unlink()
            except OSError:
                logger.exception("Exam removed but staged file cleanup failed: %s", backup)
                raise ExamDeletionCleanupError(
                    "Esame eliminato, ma la pulizia dei file richiede attenzione."
                )
