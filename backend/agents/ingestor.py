import json
import hashlib
import logging
import math
import os
import re
import threading
from pathlib import Path
from pypdf import PdfReader
import chromadb
from chromadb.errors import NotFoundError
from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from backend.db.models import Material, SessionLocal
from backend.agents.upload_errors import upload_error
from backend.agents.storage_errors import EmbeddingModelUnavailable, MaterialIndexUnavailable
from backend.agents.runtime_config import positive_int, nonnegative_int
from backend.observability import trace_event

load_dotenv()

CHROMA_PATH = str(Path(__file__).resolve().parents[2] / "data" / "chroma")
client = None

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_model: SentenceTransformer | None = None
_ocr_engine = None
logger = logging.getLogger(__name__)
_pdf_reader_lock = threading.Lock()


def _retrieval_max_distance() -> float:
    """An empty setting means no distance filter, as in .env.example."""
    raw = os.getenv("RETRIEVAL_MAX_DISTANCE", "").strip()
    if not raw:
        return math.inf
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("RETRIEVAL_MAX_DISTANCE deve essere un numero non negativo.") from exc
    if math.isnan(value) or value < 0:
        raise ValueError("RETRIEVAL_MAX_DISTANCE deve essere un numero non negativo.")
    return value


class _InvalidPdfReferenceFilter(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0
        self.thread_id = threading.get_ident()

    def filter(self, record: logging.LogRecord) -> bool:
        if record.thread == self.thread_id and record.getMessage().startswith("Ignoring wrong pointing object "):
            self.count += 1
            return False
        return True


def _open_pdf(path: str, source_label: str | None = None) -> PdfReader:
    """Condense pypdf's per-object recovery warnings without hiding other warnings."""
    pdf_logger = logging.getLogger("pypdf._reader")
    warning_filter = _InvalidPdfReferenceFilter()
    with _pdf_reader_lock:
        pdf_logger.addFilter(warning_filter)
        try:
            reader = PdfReader(path)
        finally:
            pdf_logger.removeFilter(warning_filter)
    if warning_filter.count:
        logger.warning(
            "PDF %s: %d riferimenti interni non validi ignorati durante la lettura.",
            source_label or Path(path).name, warning_filter.count,
        )
    return reader


def _get_client():
    global client
    if client is None:
        try:
            client = chromadb.PersistentClient(path=CHROMA_PATH)
        except Exception as exc:
            raise MaterialIndexUnavailable(
                "Indice dei materiali non disponibile. Controlla spazio, permessi e percorso data/chroma."
            ) from exc
    return client

"""
Carica (una sola volta) il modello di embedding locale.
Nessun fallback: se sentence-transformers non è installato o il modello
non può essere caricato, l'eccezione si propaga esplicitamente.
"""
def _get_sentence_transformer() -> SentenceTransformer:

    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(EMBEDDING_MODEL)
        except Exception as exc:
            raise EmbeddingModelUnavailable(
                "Modello di embedding non disponibile. Controlla la connessione per il primo download "
                "oppure la cache locale del modello."
            ) from exc
    return _model

    """
    Embedding locale con sentence-transformers.
    Nessuna API key richiesta, funziona offline.
    Modello multilingue con vettori da 384 dimensioni.
    """
class LocalEmbeddingFunction(EmbeddingFunction):

    def __init__(self) -> None:
        pass

    def __call__(self, texts: Documents) -> Embeddings:
        model = _get_sentence_transformer()
        embeddings = model.encode(list(texts), convert_to_numpy=True)
        return embeddings.tolist()

    @staticmethod
    def name() -> str:
        # Chroma persists this identifier; keep it stable so old collections
        # can be opened and re-embedded using the model metadata below.
        # Chroma persists this legacy function ID. Renaming it would make
        # existing collections unreadable; the actual model is recorded in
        # collection metadata and checked by get_collection().
        return "local-sentence-transformers-all-MiniLM-L6-v2"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "LocalEmbeddingFunction":
        return LocalEmbeddingFunction()


local_ef = LocalEmbeddingFunction()


def get_collection(student_id: int, exam_id: int) -> chromadb.Collection:
    """Ogni esame ha la sua collection isolata."""
    name = f"student_{student_id}_exam_{exam_id}"
    try:
        collection = _get_client().get_or_create_collection(
            name=name,
            embedding_function=local_ef,
            metadata={"student_id": student_id, "exam_id": exam_id, "embedding_model": EMBEDDING_MODEL}
        )
    except Exception as exc:
        raise MaterialIndexUnavailable(
            "Impossibile aprire l'indice dei materiali. Controlla data/chroma e riprova."
        ) from exc
    if collection.metadata.get("embedding_model") != EMBEDDING_MODEL:
        # Existing vectors belong to the previous model. Re-embed the stored
        # passages before allowing a query to mix incompatible vector spaces.
        offset = 0
        while True:
            batch = collection.get(include=["documents"], limit=100, offset=offset)
            ids, documents = batch["ids"], batch.get("documents") or []
            if not ids:
                break
            collection.update(ids=ids, embeddings=local_ef(documents))
            offset += len(ids)
        collection.modify(metadata={**collection.metadata, "embedding_model": EMBEDDING_MODEL})
    return collection


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """Prefer paragraph and sentence ends; keep short passages below model limit."""
    if chunk_size is None:
        chunk_size = positive_int("RAG_CHUNK_WORDS", 90)
    if overlap is None:
        overlap = nonnegative_int("RAG_CHUNK_OVERLAP_WORDS", 15)
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size deve superare overlap e essere positivo")
    words = list(re.finditer(r"\S+", text))
    if not words:
        return []
    boundaries = {len(words)}
    for index in range(1, len(words)):
        gap = text[words[index - 1].end():words[index].start()]
        if "\n" in gap or re.search(r"[.!?][\"'”’)]?$", words[index - 1].group()):
            boundaries.add(index)
    chunks = []
    start = 0
    while start < len(words):
        maximum = min(start + chunk_size, len(words))
        candidates = [b for b in boundaries if start < b <= maximum]
        end = max(candidates) if candidates else maximum
        if end - start < chunk_size // 3 and maximum < len(words):
            end = maximum
        chunks.append(text[words[start].start():words[end - 1].end()])
        if end == len(words):
            break
        start = max(start + 1, end - overlap)
    return chunks


def _model_safe_chunks(text: str) -> list[str]:
    """Split again when the embedding tokenizer would truncate a passage."""
    chunks = chunk_text(text)
    model = _get_sentence_transformer()
    tokenizer = getattr(model, "tokenizer", None)
    max_tokens = getattr(model, "max_seq_length", None)
    if tokenizer is None or not isinstance(max_tokens, int):
        return chunks
    pending = chunks[:]
    safe = []
    while pending:
        chunk = pending.pop(0)
        if len(tokenizer.encode(chunk, verbose=False)) <= max_tokens:
            safe.append(chunk)
            continue
        count = len(chunk.split())
        if count <= 1:
            raise ValueError("Una parola del PDF supera il limite di token del modello di embedding")
        size = max(1, count // 2)
        pending[:0] = chunk_text(chunk, chunk_size=size, overlap=min(5, size - 1))
    return safe


def _ocr_page(pdf_path: Path, page_index: int) -> str:
    """Render only pages without extractable text and run local OCR."""
    global _ocr_engine
    import numpy as np
    import pypdfium2 as pdfium
    from rapidocr import RapidOCR

    if _ocr_engine is None:
        _ocr_engine = RapidOCR()
    document = pdfium.PdfDocument(str(pdf_path))
    try:
        image = document[page_index].render(scale=2).to_pil()
        result = _ocr_engine(np.asarray(image.convert("RGB")))
        return "\n".join(result.txts or ())
    finally:
        document.close()


def ingest_pdf(pdf_path: str, student_id: int, exam_id: int, source_file: str | None = None) -> dict:
    path = Path(pdf_path)
    source_label = source_file or path.name
    if not path.exists():
        raise FileNotFoundError(f"PDF non trovato: {pdf_path}")
    if path.suffix.lower() != ".pdf":
        raise upload_error("non_pdf")
    max_bytes = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
    if path.stat().st_size > max_bytes:
        raise upload_error("troppo_grande")

    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        reader = _open_pdf(str(path), source_label)
    except Exception as exc:
        message = str(exc).lower()
        if "password" in message or "decrypt" in message:
            raise upload_error("password") from exc
        raise upload_error("corrotto") from exc
    pages_text: list[tuple[int, str]] = []
    pages_ocr = 0
    ocr_errors = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except UnicodeError as exc:
            raise upload_error("testo_illeggibile") from exc
        if not text.strip():
            try:
                text = _ocr_page(path, i)
            except Exception as exc:
                ocr_errors.append(exc)
                continue
            if text.strip():
                pages_ocr += 1
        if text.strip():
            pages_text.append((i + 1, text))
    pages_total = len(reader.pages)
    pages_with_text = len(pages_text)
    pages_skipped = pages_total - pages_with_text

    collection = get_collection(student_id, exam_id)
    existing = collection.get(where={"file_hash": file_hash})
    existing_ids = existing.get("ids") or []
    if existing_ids:
        expected_ids = []
        chunk_counter = 0
        for page_num, text in pages_text:
            for chunk in _model_safe_chunks(text):
                if len(chunk.strip()) >= 20:
                    expected_ids.append(f"{file_hash}_p{page_num}_c{chunk_counter}")
                    chunk_counter += 1
        if set(existing_ids) != set(expected_ids):
            raise MaterialIndexUnavailable(
                "Indice incompleto per questo PDF. Verifica il materiale e ripristina l'indice prima di riprovare."
            )
        stats = {
            "file": path.name,
            "file_hash": file_hash,
            "pages": pages_total,
            "pages_total": pages_total,
            "pages_with_text": pages_with_text,
            "pages_skipped": pages_skipped,
            "pages_ocr": pages_ocr,
            "chunks": len(existing_ids),
            "collection": f"student_{student_id}_exam_{exam_id}",
        }
        return stats

    if not pages_text:
        if ocr_errors:
            raise upload_error("ocr_fallito") from ocr_errors[0]
        raise upload_error("scansionato")

    all_chunks: list[str] = []
    all_metadata: list[dict] = []
    all_ids: list[str] = []

    chunk_counter = 0
    for page_num, text in pages_text:
        for chunk in _model_safe_chunks(text):
            if len(chunk.strip()) < 20:
                continue
            all_chunks.append(chunk)
            all_metadata.append({
                "student_id": student_id,
                "exam_id": exam_id,
                "page": page_num,
                "source": source_label,
                "source_file": source_label,
                "file_hash": file_hash,
            })
            # Document ID: content hash + source page + chunk index. The hash
            # prevents collisions across PDFs and also enables deduplication
            # when the same bytes are uploaded under another filename.
            all_ids.append(f"{file_hash}_p{page_num}_c{chunk_counter}")
            chunk_counter += 1

    batch_size = 100
    if not all_chunks:
        raise upload_error("zero_chunk")
    try:
        for i in range(0, len(all_chunks), batch_size):
            collection.add(
                documents=all_chunks[i:i + batch_size],
                metadatas=all_metadata[i:i + batch_size],
                ids=all_ids[i:i + batch_size]
            )
        actual_ids = collection.get(where={"file_hash": file_hash}).get("ids") or []
        if set(actual_ids) != set(all_ids):
            collection.delete(ids=actual_ids)
            raise RuntimeError(
                f"Verifica ingestione fallita per {path.name}: attesi {len(all_ids)} chunk, "
                f"trovati {len(actual_ids)}. Chunk parziali rimossi."
            )
    except Exception:
        try:
            collection.delete(ids=all_ids)
        except Exception as cleanup_error:
            logger.exception("Failed to remove partial chunks after ingest failure")
            raise RuntimeError(
                "Indicizzazione fallita e pulizia dei chunk incompleta; controlla l'indice prima di riprovare."
            ) from cleanup_error
        raise

    stats = {
        "file": path.name,
        "file_hash": file_hash,
        "pages": pages_total,
        "pages_total": pages_total,
        "pages_with_text": pages_with_text,
        "pages_skipped": pages_skipped,
        "pages_ocr": pages_ocr,
        "chunks": len(actual_ids),
        "collection": f"student_{student_id}_exam_{exam_id}"
    }
    logger.info("Indexed PDF: %s", stats)
    return stats


def verify_exam_material(
    student_id: int, exam_id: int, material_ids: list[int] | None = None,
) -> dict:
    """Verifica la coerenza tra file, Material DB e chunk Chroma dell'esame."""
    collection = get_collection(student_id, exam_id)
    indexed = collection.get(include=["metadatas"])
    ids = indexed.get("ids") or []
    metadatas = indexed.get("metadatas") or []
    indexed_by_hash: dict[str, list[str]] = {}
    indexed_by_source: dict[str, list[str]] = {}
    for chunk_id, metadata in zip(ids, metadatas):
        metadata = metadata or {}
        if metadata.get("file_hash"):
            indexed_by_hash.setdefault(metadata["file_hash"], []).append(chunk_id)
        if metadata.get("source_file"):
            indexed_by_source.setdefault(metadata["source_file"], []).append(chunk_id)

    db = SessionLocal()
    try:
        materials = db.query(Material).filter(Material.exam_id == exam_id).all()
        errors = []
        details = []
        known_ids: set[str] = set()
        if material_ids is not None:
            requested = set(material_ids)
            materials = [material for material in materials if material.id in requested]
            if len(materials) != len(requested):
                errors.append("Una o più fonti selezionate non appartengono all'esame.")
        for material in materials:
            path = Path(material.stored_path)
            expected_hash = material.file_hash
            if not expected_hash and path.exists():
                expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            chunk_ids = indexed_by_hash.get(expected_hash or "", [])
            if not chunk_ids:
                chunk_ids = indexed_by_source.get(material.filename, [])
            known_ids.update(chunk_ids)
            file_exists = path.exists()
            actual_count = len(chunk_ids)
            detail = {
                "material_id": material.id,
                "filename": material.filename,
                "file_exists": file_exists,
                "expected_chunks": material.chunks_count or 0,
                "actual_chunks": actual_count,
            }
            details.append(detail)
            if not file_exists:
                errors.append(f"File mancante: {material.filename}")
            elif expected_hash and hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                errors.append(f"Contenuto del file modificato: {material.filename}")
            if actual_count != (material.chunks_count or 0):
                errors.append(
                    f"Chunk incoerenti per {material.filename}: attesi {material.chunks_count or 0}, trovati {actual_count}"
                )
            if actual_count == 0:
                errors.append(f"Materiale senza chunk: {material.filename}")

        orphan_ids = (
            [chunk_id for chunk_id in ids if chunk_id not in known_ids]
            if material_ids is None else []
        )
        if orphan_ids:
            errors.append(f"Chunk orfani rilevati: {len(orphan_ids)}")
        return {"ok": not errors, "errors": errors, "materials": details, "orphan_chunks": len(orphan_ids)}
    finally:
        db.close()


def _material_where_clause(collection, exam_id: int, material_ids: list[int]) -> dict:
    """Build an index filter from material IDs belonging to this exam."""
    unique_ids = set(material_ids)
    with SessionLocal() as db:
        materials = db.query(Material).filter(
            Material.exam_id == exam_id, Material.id.in_(unique_ids)
        ).all()
        same_exam = db.query(Material).filter(Material.exam_id == exam_id).all()
    if len(materials) != len(unique_ids):
        raise ValueError("Una o più fonti selezionate non appartengono all'esame.")

    conditions = []
    for material in materials:
        file_hash = material.file_hash
        path = Path(material.stored_path)
        if not file_hash and path.is_file():
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if file_hash and collection.get(where={"file_hash": file_hash}).get("ids"):
            condition = {"file_hash": file_hash}
        else:
            if sum(other.filename == material.filename for other in same_exam) > 1:
                raise ValueError(f"Fonti omonime da reindicizzare: {material.filename}")
            condition = {"source_file": material.filename}
        if condition not in conditions:
            conditions.append(condition)
    return conditions[0] if len(conditions) == 1 else {"$or": conditions}


def retrieve_chunks(
    query: str, student_id: int, exam_id: int, n_results: int = 5,
    material_ids: list[int] | None = None,
) -> list[str]:
    if material_ids is not None and not material_ids:
        return []
    collection = get_collection(student_id, exam_id)
    options = {"query_texts": [query], "n_results": n_results}
    if material_ids is not None:
        options["where"] = _material_where_clause(collection, exam_id, material_ids)
    results = collection.query(**options)
    trace_event("retrieval", exam_id=exam_id, selected_sources=len(material_ids) if material_ids is not None else "all",
                requested=n_results, returned=len((results.get("documents") or [[]])[0]))
    documents = results.get("documents")
    if not documents or not documents[0]:
        return []
    threshold = _retrieval_max_distance()
    distances = (results.get("distances") or [[]])[0]
    if distances and threshold != float("inf"):
        return [doc for doc, distance in zip(documents[0], distances) if distance <= threshold]
    return documents[0]


def retrieve_chunks_by_source(student_id: int, exam_id: int, source_file: str, limit: int = 12) -> list[str]:
    collection = get_collection(student_id, exam_id)
    try:
        results = collection.get(
            where={"source_file": source_file},
            limit=limit,
            include=["documents"],
        )
    except Exception as exc:
        raise RuntimeError(
            f"Errore lettura Chroma per la fonte '{source_file}' dell'esame {exam_id}"
        ) from exc

    documents = results.get("documents") or []
    return [doc for doc in documents if isinstance(doc, str) and doc.strip()]


def _material_name_maps(exam_id: int) -> tuple[dict[str, str], dict[str, str]]:
    db = SessionLocal()
    try:
        materials = db.query(Material).filter(Material.exam_id == exam_id).all()
        return (
            {material.file_hash: material.filename for material in materials if material.file_hash},
            {Path(material.stored_path).name: material.filename for material in materials},
        )
    finally:
        db.close()


def retrieve_passages(
    student_id: int,
    exam_id: int,
    source_file: str | None = None,
    *,
    file_hash: str | None = None,
    display_source: str | None = None,
    material_ids: list[int] | None = None,
) -> list[dict]:
    """Return every indexed passage with its source and page, in document order."""
    if material_ids is not None and not material_ids:
        return []
    collection = get_collection(student_id, exam_id)
    options = {"include": ["documents", "metadatas"]}
    if material_ids is not None:
        options["where"] = _material_where_clause(collection, exam_id, material_ids)
    elif file_hash is not None:
        options["where"] = {"file_hash": file_hash}
    elif source_file is not None:
        options["where"] = {"source_file": source_file}
    rows = collection.get(**options)
    names_by_hash, names_by_path = _material_name_maps(exam_id)
    passages = []
    for chunk_id, document, metadata in zip(
        rows.get("ids") or [], rows.get("documents") or [], rows.get("metadatas") or []
    ):
        if not isinstance(document, str) or not document.strip():
            continue
        metadata = metadata or {}
        indexed_source = metadata.get("source_file") or metadata.get("source") or "materiale di studio"
        passages.append({
            "text": document,
            "source": display_source or names_by_hash.get(metadata.get("file_hash"))
            or names_by_path.get(indexed_source) or indexed_source,
            "page": metadata.get("page"),
            "chunk_id": chunk_id,
        })
    passages.sort(key=lambda passage: (
        passage["source"],
        passage["page"] if isinstance(passage["page"], int) else 0,
        int(passage["chunk_id"].rsplit("_c", 1)[-1])
        if "_c" in passage["chunk_id"] and passage["chunk_id"].rsplit("_c", 1)[-1].isdigit()
        else 0,
    ))
    return passages


def search_passages(
    query: str, student_id: int, exam_id: int, n_results: int = 5,
    material_ids: list[int] | None = None,
) -> list[dict]:
    """Semantic search retaining the source metadata needed for citations."""
    if material_ids is not None and not material_ids:
        return []
    collection = get_collection(student_id, exam_id)
    options = {
        "query_texts": [query], "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if material_ids is not None:
        options["where"] = _material_where_clause(collection, exam_id, material_ids)
    rows = collection.query(**options)
    trace_event("retrieval", exam_id=exam_id, selected_sources=len(material_ids) if material_ids is not None else "all",
                requested=n_results, returned=len((rows.get("documents") or [[]])[0]))
    documents = (rows.get("documents") or [[]])[0]
    metadatas = (rows.get("metadatas") or [[]])[0]
    distances = (rows.get("distances") or [[]])[0]
    threshold = _retrieval_max_distance()
    names_by_hash, names_by_path = _material_name_maps(exam_id)
    passages = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        if isinstance(document, str) and document.strip() and distance <= threshold:
            metadata = metadata or {}
            indexed_source = metadata.get("source_file") or metadata.get("source") or "materiale di studio"
            passages.append({
                "text": document,
                "source": names_by_hash.get(metadata.get("file_hash"))
                or names_by_path.get(indexed_source) or indexed_source,
                "page": metadata.get("page"),
            })
    return passages


def delete_exam_collection(student_id: int, exam_id: int) -> None:
    name = f"student_{student_id}_exam_{exam_id}"
    try:
        _get_client().delete_collection(name)
    except (ValueError, NotFoundError):
        # Collection assente: operazione idempotente; altri errori propagano.
        return
