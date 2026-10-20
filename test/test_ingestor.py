"""Unit/integration test — backend/agents/ingestor.py."""

import pytest

from backend.agents import ingestor
from backend.db.models import Material, SessionLocal
from backend.agents.upload_errors import UploadError


# ---------------------------------------------------------------------------
# chunk_text — logica pura
# ---------------------------------------------------------------------------

def test_chunk_text_single_chunk_for_short_text():
    chunks = ingestor.chunk_text("parola " * 10, chunk_size=500, overlap=50)
    assert len(chunks) == 1


def test_chunk_text_splits_long_text_with_overlap():
    text = " ".join(f"parola{i}" for i in range(1000))
    chunks = ingestor.chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) > 1
    # L'overlap fa si' che le ultime parole di un chunk ricompaiano nel successivo.
    last_words_first_chunk = chunks[0].split()[-10:]
    assert any(w in chunks[1] for w in last_words_first_chunk)


def test_chunk_text_empty_string_returns_empty_list():
    assert ingestor.chunk_text("") == []


def test_chunk_text_preserves_poetry_line_breaks():
    verse = "Sempre caro mi fu quest'ermo colle,\ne questa siepe, che da tanta parte"
    assert ingestor.chunk_text(verse, chunk_size=30, overlap=2) == [verse]


def test_chunk_text_prefers_sentence_and_section_boundaries():
    text = "Prima frase con diverse parole utili. Seconda frase con altre parole.\n\nNuova sezione con altri concetti."
    chunks = ingestor.chunk_text(text, chunk_size=12, overlap=2)
    assert chunks[0].endswith("parole.")
    assert any("Nuova sezione" in chunk for chunk in chunks)
    assert max(len(chunk.split()) for chunk in chunks) <= 12


def test_embedding_uses_multilingual_model():
    assert ingestor.EMBEDDING_MODEL.endswith("paraphrase-multilingual-MiniLM-L12-v2")
    assert ingestor.LocalEmbeddingFunction.name() == "local-sentence-transformers-all-MiniLM-L6-v2"


def test_model_safe_chunks_fit_tokenizer_limit(monkeypatch):
    class Tokenizer:
        def encode(self, text, **kwargs):
            assert kwargs.get("verbose") is False
            return [0] * (3 * len(text.split()) + 2)

    class Model:
        tokenizer = Tokenizer()
        max_seq_length = 20

    monkeypatch.setattr(ingestor, "_get_sentence_transformer", lambda: Model())
    chunks = ingestor._model_safe_chunks(" ".join(f"parola{i}" for i in range(35)))
    assert len(chunks) > 1
    assert all(len(Model.tokenizer.encode(chunk, verbose=False)) <= Model.max_seq_length for chunk in chunks)


def test_malformed_pdf_xref_warnings_are_summarized(monkeypatch, caplog):
    import logging

    class Reader:
        def __init__(self, _path):
            for index in range(10):
                logging.getLogger("pypdf._reader").warning(
                    "Ignoring wrong pointing object %s 0 (offset 0)", index
                )
            logging.getLogger("pypdf._reader").warning("Other PDF warning")

    monkeypatch.setattr(ingestor, "PdfReader", Reader)
    with caplog.at_level(logging.WARNING):
        ingestor._open_pdf("example.pdf")
    assert "10 riferimenti interni" in caplog.text
    assert "Other PDF warning" in caplog.text
    assert "Ignoring wrong pointing object" not in caplog.text


# ---------------------------------------------------------------------------
# get_collection — naming/isolamento per studente+esame
# ---------------------------------------------------------------------------

def test_get_collection_naming_is_isolated_per_student_and_exam(isolated_chroma):
    coll_a = ingestor.get_collection(student_id=1, exam_id=1)
    coll_b = ingestor.get_collection(student_id=1, exam_id=2)
    coll_c = ingestor.get_collection(student_id=2, exam_id=1)

    assert coll_a.name == "student_1_exam_1"
    assert coll_b.name == "student_1_exam_2"
    assert coll_c.name == "student_2_exam_1"
    assert coll_a.name != coll_b.name != coll_c.name


def test_selected_material_queries_exclude_other_pdfs(tmp_path):
    from fpdf import FPDF
    from backend.db.models import Exam, Student

    with SessionLocal() as db:
        student = Student(username="selected_sources", password_hash="hash")
        db.add(student)
        db.flush()
        exam = Exam(student_id=student.id, name="Esame")
        db.add(exam)
        db.commit()
        student_id, exam_id = student.id, exam.id

    material_ids = []
    for filename, sentence in (
        ("bayes.pdf", "Il teorema di Bayes descrive la probabilita condizionata."),
        ("matrici.pdf", "Le matrici rappresentano trasformazioni lineari."),
    ):
        path = tmp_path / filename
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 8, (sentence + " ") * 15)
        pdf.output(str(path))
        stats = ingestor.ingest_pdf(str(path), student_id, exam_id)
        with SessionLocal() as db:
            material = Material(
                exam_id=exam_id, filename=filename, stored_path=str(path),
                file_hash=stats["file_hash"], chunks_count=stats["chunks"],
            )
            db.add(material)
            db.commit()
            material_ids.append(material.id)

    passages = ingestor.retrieve_passages(student_id, exam_id, material_ids=[material_ids[0]])
    chunks = ingestor.retrieve_chunks("probabilita", student_id, exam_id, material_ids=[material_ids[0]])
    searched = ingestor.search_passages("probabilita", student_id, exam_id, material_ids=[material_ids[0]])
    assert passages and chunks and searched
    assert all("Bayes" in passage["text"] for passage in passages)
    assert all("Bayes" in chunk for chunk in chunks)
    assert all("Bayes" in passage["text"] for passage in searched)
    assert not ingestor.retrieve_passages(student_id, exam_id, material_ids=[])
    (tmp_path / "matrici.pdf").unlink()
    assert not ingestor.verify_exam_material(student_id, exam_id)["ok"]
    assert ingestor.verify_exam_material(
        student_id, exam_id, material_ids=[material_ids[0]],
    )["ok"]
    with pytest.raises(ValueError, match="non appartengono"):
        ingestor.retrieve_passages(student_id, exam_id, material_ids=[999999])


def test_two_real_pdfs_match_index_and_tutor_retrieval(tmp_path):
    """Check PDF bytes, indexed text, and the tutor's selected-source path together."""
    from fpdf import FPDF

    from backend.agents.socratic_tutor import SocraticTutor
    from backend.db.models import Exam, Student

    with SessionLocal() as db:
        student = Student(username="two_pdf_audit", password_hash="hash")
        db.add(student)
        db.flush()
        exam = Exam(student_id=student.id, name="Due dispense")
        db.add(exam)
        db.commit()
        student_id, exam_id = student.id, exam.id

    sources = (
        ("bayes.pdf", "Il teorema di Bayes aggiorna la probabilita usando una evidenza osservata."),
        ("matrici.pdf", "Il prodotto tra matrici combina trasformazioni lineari compatibili."),
    )
    material_ids = []
    hashes = []
    for filename, sentence in sources:
        path = tmp_path / filename
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 8, sentence * 4)
        pdf.output(str(path))
        stats = ingestor.ingest_pdf(str(path), student_id, exam_id)
        assert stats["chunks"] > 0
        hashes.append(stats["file_hash"])
        with SessionLocal() as db:
            material = Material(
                exam_id=exam_id, filename=filename, stored_path=str(path),
                file_hash=stats["file_hash"], chunks_count=stats["chunks"],
            )
            db.add(material)
            db.commit()
            material_ids.append(material.id)

    collection = ingestor.get_collection(student_id, exam_id)
    assert len(set(hashes)) == 2
    for (filename, sentence), file_hash in zip(sources, hashes):
        rows = collection.get(where={"file_hash": file_hash}, include=["documents", "metadatas"])
        assert rows["ids"]
        assert len(rows["ids"]) == len(set(rows["ids"]))
        assert sentence in " ".join(rows["documents"])
        assert all(metadata["file_hash"] == file_hash for metadata in rows["metadatas"])
    assert collection.count() == sum(
        len(collection.get(where={"file_hash": file_hash})["ids"]) for file_hash in hashes
    )
    assert ingestor.verify_exam_material(student_id, exam_id)["ok"]

    for material_id, (filename, sentence) in zip(material_ids, sources):
        tutor = SocraticTutor(student_id, exam_id, material_ids=[material_id])
        tutor.start({"id": filename, "text": sentence})
        assert tutor._source_cache
        assert all(passage["source"] == filename for passage in tutor._source_cache)
        assert sentence in " ".join(passage["text"] for passage in tutor._source_cache)


def test_get_collection_reembeds_legacy_vectors(isolated_chroma, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(
            ingestor.LocalEmbeddingFunction, "name",
            staticmethod(lambda: "local-sentence-transformers-all-MiniLM-L6-v2"),
        )
        old = isolated_chroma.get_or_create_collection(
            name="student_1_exam_1", embedding_function=ingestor.local_ef,
            metadata={"student_id": 1, "exam_id": 1},
        )
        old.add(ids=["old"], documents=["testo italiano sull'apprendimento"])
    migrated = ingestor.get_collection(1, 1)
    assert migrated.metadata["embedding_model"] == ingestor.EMBEDDING_MODEL
    assert migrated.count() == 1


# ---------------------------------------------------------------------------
# ingest_pdf / retrieve_chunks / retrieve_chunks_by_source (con embedding finti)
# ---------------------------------------------------------------------------

def test_ingest_pdf_missing_file_raises(isolated_chroma):
    with pytest.raises(FileNotFoundError):
        ingestor.ingest_pdf("/percorso/inesistente.pdf", student_id=1, exam_id=1)


@pytest.mark.parametrize(("code", "reader_error"), [
    ("corrotto", RuntimeError("EOF marker not found")),
    ("password", RuntimeError("File has not been decrypted")),
])
def test_ingest_pdf_reports_structured_pdf_errors(tmp_path, monkeypatch, code, reader_error):
    class Reader:
        def __init__(self, _path):
            raise reader_error
    monkeypatch.setattr(ingestor, "PdfReader", Reader)
    path = tmp_path / "documento.pdf"
    path.write_bytes(b"not-a-pdf")
    with pytest.raises(UploadError, match=code if code == "corrotto" else "password"):
        ingestor.ingest_pdf(str(path), 1, 1)


def test_ingest_pdf_rejects_non_pdf_and_zero_chunks(tmp_path, monkeypatch):
    non_pdf = tmp_path / "documento.txt"
    non_pdf.write_text("testo")
    with pytest.raises(UploadError, match="non è un PDF"):
        ingestor.ingest_pdf(str(non_pdf), 1, 1)

    class Page:
        def extract_text(self):
            return "testo sufficiente per essere estratto"
    class Reader:
        pages = [Page()]
        def __init__(self, _path):
            pass
    monkeypatch.setattr(ingestor, "PdfReader", Reader)
    monkeypatch.setattr(ingestor, "chunk_text", lambda _text: ["breve"])
    zero = tmp_path / "zero.pdf"
    zero.write_bytes(b"pdf")
    with pytest.raises(UploadError, match="chunk utili"):
        ingestor.ingest_pdf(str(zero), 1, 1)


def test_ingest_pdf_reports_total_text_and_skipped_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(ingestor, "_ocr_page", lambda *_: "")
    class Page:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class Reader:
        def __init__(self, _path):
            self.pages = [Page("contenuto disciplinare " * 20), Page(""), Page("altro contenuto " * 20)]

    monkeypatch.setattr(ingestor, "PdfReader", Reader)
    path = tmp_path / "misto.pdf"
    path.write_bytes(b"mixed")
    stats = ingestor.ingest_pdf(str(path), 1, 1)

    assert stats["pages"] == 3
    assert stats["pages_total"] == 3
    assert stats["pages_with_text"] == 2
    assert stats["pages_skipped"] == 1


def test_ingest_pdf_ocr_indexes_scanned_page(tmp_path, monkeypatch):
    class Page:
        def extract_text(self):
            return ""

    class Reader:
        def __init__(self, _path):
            self.pages = [Page()]

    monkeypatch.setattr(ingestor, "PdfReader", Reader)
    monkeypatch.setattr(ingestor, "_ocr_page", lambda *_: "Leopardi parla della memoria e del desiderio.")
    path = tmp_path / "scansione.pdf"
    path.write_bytes(b"scanned")
    stats = ingestor.ingest_pdf(str(path), 1, 1)
    assert stats["pages_ocr"] == 1
    assert stats["pages_skipped"] == 0
    assert "Leopardi parla" in ingestor.retrieve_passages(1, 1)[0]["text"]


def test_ingest_pdf_keeps_readable_pages_when_one_ocr_page_fails(tmp_path, monkeypatch):
    class Page:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class Reader:
        def __init__(self, _path):
            self.pages = [Page("Pagina leggibile con abbastanza contenuto per l'indice."), Page("")]

    monkeypatch.setattr(ingestor, "PdfReader", Reader)
    monkeypatch.setattr(ingestor, "_ocr_page", lambda *_: (_ for _ in ()).throw(RuntimeError("OCR failed")))
    path = tmp_path / "misto.pdf"
    path.write_bytes(b"mixed")
    stats = ingestor.ingest_pdf(str(path), 1, 1)
    assert stats["pages_with_text"] == 1
    assert stats["pages_skipped"] == 1


def test_real_ocr_recognizes_image_only_pdf(tmp_path):
    from PIL import Image, ImageDraw, ImageFont
    from fpdf import FPDF

    image = Image.new("RGB", (1400, 250), "white")
    font_path = "C:/Windows/Fonts/arial.ttf"
    font = ImageFont.truetype(font_path if __import__("os").path.exists(font_path) else "DejaVuSans.ttf", 72)
    ImageDraw.Draw(image).text((40, 60), "Materiale di studio", font=font, fill="black")
    png_path = tmp_path / "pagina.png"
    image.save(png_path)
    pdf = FPDF()
    pdf.add_page()
    pdf.image(str(png_path), x=10, y=10, w=190)
    pdf_path = tmp_path / "pagina.pdf"
    pdf.output(str(pdf_path))
    assert "Materiale di studio" in ingestor._ocr_page(pdf_path, 0)


def test_verify_exam_material_coherent_and_orphan(isolated_chroma, sample_pdf_path):
    stats = ingestor.ingest_pdf(str(sample_pdf_path), 1, 1)
    db = SessionLocal()
    db.add(Material(
        exam_id=1,
        filename=sample_pdf_path.name,
        stored_path=str(sample_pdf_path),
        chunks_count=stats["chunks"],
        pages_count=stats["pages"],
        file_hash=stats["file_hash"],
    ))
    db.commit()
    db.close()

    assert ingestor.verify_exam_material(1, 1)["ok"] is True

    collection = ingestor.get_collection(1, 1)
    collection.add(
        documents=["chunk orfano di test"],
        metadatas=[{"file_hash": "hash-orfano", "source_file": "orfano.pdf"}],
        ids=["hash-orfano_p1_c0"],
    )
    verification = ingestor.verify_exam_material(1, 1)
    assert verification["ok"] is False
    assert verification["orphan_chunks"] == 1


def test_multi_pdf_no_collision(tmp_path, monkeypatch):
    class FakeCollection:
        def __init__(self):
            self.rows = {}

        def add(self, documents, metadatas, ids):
            self.rows.update(dict(zip(ids, zip(documents, metadatas))))

        def get(self, where=None, ids=None, include=None):
            selected = self.rows
            if ids is not None:
                selected = {key: self.rows[key] for key in ids if key in self.rows}
            if where:
                selected = {key: value for key, value in selected.items() if all(value[1].get(k) == v for k, v in where.items())}
            result = {"ids": list(selected)}
            if include and "metadatas" in include:
                result["metadatas"] = [value[1] for value in selected.values()]
            return result

        def delete(self, ids):
            for key in ids:
                self.rows.pop(key, None)

    collection = FakeCollection()
    monkeypatch.setattr(ingestor, "get_collection", lambda *args, **kwargs: collection)
    monkeypatch.setattr(ingestor, "chunk_text", lambda text: [(text + " contenuto" * 10) + str(i) for i in range(2 if "A" in text else 3)])

    class Page:
        def __init__(self, text):
            self.text = text
        def extract_text(self):
            return self.text

    class Reader:
        def __init__(self, path):
            self.pages = [Page("A" if "a.pdf" in path else "B")]

    monkeypatch.setattr(ingestor, "PdfReader", Reader)
    first = tmp_path / "a.pdf"
    second = tmp_path / "b.pdf"
    same_name_different_content = tmp_path / "copy.pdf"
    first.write_bytes(b"pdf-A")
    second.write_bytes(b"pdf-B")
    same_name_different_content.write_bytes(b"pdf-C")

    assert ingestor.ingest_pdf(str(first), 1, 1)["chunks"] == 2
    assert ingestor.ingest_pdf(str(second), 1, 1)["chunks"] == 3
    assert len(collection.rows) == 5
    assert ingestor.ingest_pdf(str(first), 1, 1)["chunks"] == 2
    assert len(collection.rows) == 5
    assert ingestor.ingest_pdf(str(same_name_different_content), 1, 1)["chunks"] == 3
    assert len(collection.rows) == 8
    assert len({key.split("_p")[0] for key in collection.rows}) == 3


def test_reingest_rejects_partial_existing_index(isolated_chroma, sample_pdf_path):
    stats = ingestor.ingest_pdf(str(sample_pdf_path), 1, 1)
    collection = ingestor.get_collection(1, 1)
    indexed = collection.get(where={"file_hash": stats["file_hash"]})["ids"]
    assert indexed
    collection.delete(ids=[indexed[0]])

    with pytest.raises(ingestor.MaterialIndexUnavailable, match="Indice incompleto"):
        ingestor.ingest_pdf(str(sample_pdf_path), 1, 1)
    assert collection.get(where={"file_hash": stats["file_hash"]})["ids"] == indexed[1:]


def test_material_verification_detects_changed_pdf_bytes(isolated_chroma, sample_pdf_path):
    stats = ingestor.ingest_pdf(str(sample_pdf_path), 1, 1)
    with SessionLocal() as db:
        db.add(Material(exam_id=1, filename=sample_pdf_path.name,
                        stored_path=str(sample_pdf_path), chunks_count=stats["chunks"],
                        file_hash=stats["file_hash"]))
        db.commit()
    assert ingestor.verify_exam_material(1, 1)["ok"]
    sample_pdf_path.write_bytes(sample_pdf_path.read_bytes() + b"\nchanged")
    verification = ingestor.verify_exam_material(1, 1)
    assert not verification["ok"]
    assert any("modificato" in error for error in verification["errors"])


def test_ingest_pdf_indexes_chunks_with_correct_stats(isolated_chroma, sample_pdf_path):
    stats = ingestor.ingest_pdf(str(sample_pdf_path), student_id=1, exam_id=1)
    assert stats["file"] == sample_pdf_path.name
    assert stats["pages"] >= 1
    assert stats["chunks"] > 0
    assert stats["collection"] == "student_1_exam_1"


def test_blank_retrieval_distance_from_example_means_no_filter(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MAX_DISTANCE", "")
    assert ingestor._retrieval_max_distance() == float("inf")
    monkeypatch.setenv("RETRIEVAL_MAX_DISTANCE", "0.8")
    assert ingestor._retrieval_max_distance() == 0.8


def test_invalid_retrieval_distance_is_rejected(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MAX_DISTANCE", "not-a-number")
    with pytest.raises(ValueError, match="RETRIEVAL_MAX_DISTANCE"):
        ingestor._retrieval_max_distance()


def test_retrieve_chunks_returns_indexed_content(isolated_chroma, sample_pdf_path):
    ingestor.ingest_pdf(str(sample_pdf_path), student_id=1, exam_id=1)
    chunks = ingestor.retrieve_chunks("machine learning", student_id=1, exam_id=1, n_results=3)
    assert len(chunks) > 0
    assert any("machine learning" in c.lower() for c in chunks)


def test_retrieve_chunks_isolated_per_exam(isolated_chroma, sample_pdf_path):
    ingestor.ingest_pdf(str(sample_pdf_path), student_id=1, exam_id=1)
    # Nessun materiale indicizzato per l'esame 2: nessun risultato.
    chunks = ingestor.retrieve_chunks("machine learning", student_id=1, exam_id=2, n_results=3)
    assert chunks == []


def test_retrieve_chunks_by_source_filters_by_filename(isolated_chroma, sample_pdf_path):
    ingestor.ingest_pdf(str(sample_pdf_path), student_id=1, exam_id=1)
    chunks = ingestor.retrieve_chunks_by_source(student_id=1, exam_id=1, source_file=sample_pdf_path.name)
    assert len(chunks) > 0

    empty = ingestor.retrieve_chunks_by_source(student_id=1, exam_id=1, source_file="altro_file.pdf")
    assert empty == []


def test_retrieve_passages_preserves_pages_and_source(isolated_chroma, sample_pdf_path):
    ingestor.ingest_pdf(str(sample_pdf_path), student_id=1, exam_id=1)
    passages = ingestor.retrieve_passages(1, 1, source_file=sample_pdf_path.name)
    assert passages
    assert all(passage["source"] == sample_pdf_path.name for passage in passages)
    assert all(isinstance(passage["page"], int) for passage in passages)
    assert ingestor.retrieve_passages(1, 1, source_file="assente.pdf") == []


def test_ingest_pdf_uses_original_filename_for_stored_uuid(isolated_chroma, sample_pdf_path):
    ingestor.ingest_pdf(str(sample_pdf_path), 3, 4, source_file="Letteratura.pdf")
    passages = ingestor.retrieve_passages(3, 4, source_file="Letteratura.pdf")
    assert passages
    assert {passage["source"] for passage in passages} == {"Letteratura.pdf"}


def test_delete_exam_collection_removes_indexed_data(isolated_chroma, sample_pdf_path):
    ingestor.ingest_pdf(str(sample_pdf_path), student_id=1, exam_id=1)
    ingestor.delete_exam_collection(student_id=1, exam_id=1)

    chunks = ingestor.retrieve_chunks("machine learning", student_id=1, exam_id=1, n_results=3)
    assert chunks == []


def test_delete_exam_collection_missing_collection_does_not_raise(isolated_chroma):
    ingestor.delete_exam_collection(student_id=999, exam_id=999)


def test_chroma_initialization_failure_has_actionable_error(monkeypatch):
    import chromadb
    import pytest

    monkeypatch.setattr(ingestor, "client", None)
    def unavailable(*args, **kwargs):
        raise OSError("internal SECRET path")
    monkeypatch.setattr(chromadb, "PersistentClient", unavailable)
    with pytest.raises(ingestor.MaterialIndexUnavailable, match="data/chroma") as error:
        ingestor.get_collection(1, 2)
    assert "SECRET" not in str(error.value)


def test_embedding_download_failure_has_actionable_error(monkeypatch):
    import pytest

    monkeypatch.setattr(ingestor, "_model", None)
    def unavailable(*args, **kwargs):
        raise OSError("internal SECRET token")
    monkeypatch.setattr(ingestor, "SentenceTransformer", unavailable)
    with pytest.raises(ingestor.EmbeddingModelUnavailable, match="download") as error:
        ingestor._get_sentence_transformer()
    assert "SECRET" not in str(error.value)
