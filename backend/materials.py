"""Operations on individual uploaded exam sources."""

import hashlib
import logging
import os
import tempfile
from pathlib import Path

from backend.agents.ingestor import get_collection
from backend.db.models import Exam, Material, SessionLocal

logger = logging.getLogger(__name__)


def rollback_uncommitted_ingest(student_id: int, exam_id: int, file_hash: str) -> None:
    """Remove newly indexed chunks unless another material still owns them."""
    with SessionLocal() as db:
        existing = db.query(Material.id).filter(
            Material.exam_id == exam_id, Material.file_hash == file_hash
        ).first()
    if existing is None:
        collection = get_collection(student_id, exam_id)
        ids = collection.get(where={"file_hash": file_hash}).get("ids") or []
        if ids:
            collection.delete(ids=ids)


def delete_material(student_id: int, exam_id: int, material_id: int) -> None:
    """Remove one owned source from the index, database and stored PDFs."""
    with SessionLocal() as db:
        material = (
            db.query(Material)
            .join(Exam, Material.exam_id == Exam.id)
            .filter(
                Material.id == material_id,
                Material.exam_id == exam_id,
                Exam.student_id == student_id,
            )
            .first()
        )
        if material is None:
            raise LookupError("Fonte non trovata.")

        path = Path(material.stored_path)
        file_hash = material.file_hash
        if not file_hash and path.is_file():
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()

        others = db.query(Material).filter(
            Material.exam_id == exam_id, Material.id != material_id
        ).all()
        shared_file = db.query(Material.id).filter(
            Material.stored_path == material.stored_path,
            Material.id != material_id,
        ).first() is not None
        shared_index = any(
            (file_hash and (
                other.file_hash == file_hash
                or (not other.file_hash and Path(other.stored_path).is_file()
                    and hashlib.sha256(Path(other.stored_path).read_bytes()).hexdigest() == file_hash)
            ))
            or (not file_hash and other.filename == material.filename)
            for other in others
        )

        collection = None
        indexed = None
        chunk_ids = []
        if not shared_index:
            collection = get_collection(student_id, exam_id)
            indexed = collection.get(where={"file_hash": file_hash}) if file_hash else {"ids": []}
            chunk_ids = indexed.get("ids") or []
            if not chunk_ids and not any(other.filename == material.filename for other in others):
                legacy = collection.get(
                    where={"source_file": material.filename}, include=["metadatas"]
                )
                chunk_ids = [
                    chunk_id for chunk_id, metadata in zip(
                        legacy.get("ids") or [], legacy.get("metadatas") or []
                    )
                    if not file_hash or not (metadata or {}).get("file_hash")
                    or (metadata or {}).get("file_hash") == file_hash
                ]
            if chunk_ids:
                indexed = collection.get(ids=chunk_ids, include=["documents", "metadatas", "embeddings"])

        backup = None
        try:
            if not shared_file and path.is_file():
                with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.delete-", delete=False) as temp:
                    backup = Path(temp.name)
                os.replace(path, backup)
            if chunk_ids:
                collection.delete(ids=chunk_ids)
            db.delete(material)
            db.commit()
        except Exception:
            db.rollback()
            if indexed and indexed["ids"]:
                try:
                    present = set(collection.get(ids=indexed["ids"])["ids"])
                    missing = [i for i, chunk_id in enumerate(indexed["ids"]) if chunk_id not in present]
                    if missing:
                        collection.add(
                            ids=[indexed["ids"][i] for i in missing],
                            documents=[indexed["documents"][i] for i in missing],
                            metadatas=[indexed["metadatas"][i] for i in missing],
                            embeddings=[indexed["embeddings"][i].tolist() for i in missing],
                        )
                except Exception:
                    logger.exception("Failed to restore material chunks for id=%s", material_id)
            if backup is not None:
                if backup.exists():
                    if not path.exists():
                        os.replace(backup, path)
                    else:
                        backup.unlink()
            raise

        if backup is not None:
            backup.unlink()
