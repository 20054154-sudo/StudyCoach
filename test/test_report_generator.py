"""Unit/integration test — backend/agents/report_generator.py."""

import io
import importlib
from pathlib import Path

from backend.agents import report_generator as rg


def test_import_report_generator_does_not_write_files(monkeypatch):
    original_mkdir = Path.mkdir
    original_write_text = Path.write_text
    reports_dir = rg.REPORTS_DIR

    def forbid_reports_mkdir(path, *args, **kwargs):
        if path.name == "reports":
            raise AssertionError("mkdir during import")
        return original_mkdir(path, *args, **kwargs)

    def forbid_reports_write(path, *args, **kwargs):
        if path.parent.name == "reports":
            raise AssertionError("write during import")
        return original_write_text(path, *args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(Path, "mkdir", forbid_reports_mkdir)
            patch.setattr(Path, "write_text", forbid_reports_write)
            importlib.reload(rg)
    finally:
        rg.REPORTS_DIR = reports_dir


def _sample_report(avg=7.5):
    return {
        "student_id": 1,
        "exam_id": 2,
        "overall_avg": avg,
        "n_questions_answered": 3,
        "weak_topics": ["derivate"],
        "strong_topics": ["limiti"],
        "topic_breakdown": [
            {"topic": "limiti", "avg_score": 9.0, "n_attempts": 2, "is_weak": False},
            {"topic": "derivate", "avg_score": 4.0, "n_attempts": 1, "is_weak": True},
        ],
        "summary": "Prestazione sufficiente con margini di miglioramento.",
    }


# ---------------------------------------------------------------------------
# generate_markdown
# ---------------------------------------------------------------------------

def test_generate_markdown_contains_key_sections():
    md = rg.generate_markdown(_sample_report(), session_id=5, exam_name="Analisi 1")
    assert "Analisi 1" in md
    assert "Sessione ID:** 5" in md
    assert "7.5/10" in md
    assert "derivate" in md
    assert "limiti" in md


def test_generate_markdown_verdict_thresholds():
    assert "🟢 Ottimo" in rg.generate_markdown(_sample_report(avg=9), session_id=1)
    assert "🟡 Sufficiente" in rg.generate_markdown(_sample_report(avg=6.5), session_id=1)
    assert "🔴 Insufficiente" in rg.generate_markdown(_sample_report(avg=3), session_id=1)


def test_generate_markdown_no_weak_topics_message():
    report = _sample_report()
    report["weak_topics"] = []
    md = rg.generate_markdown(report, session_id=1)
    assert "Nessuna area critica" in md


def test_generate_markdown_no_strong_topics_message():
    report = _sample_report()
    report["strong_topics"] = []
    md = rg.generate_markdown(report, session_id=1)
    assert "Continua a studiare" in md


# ---------------------------------------------------------------------------
# generate_pdf
# ---------------------------------------------------------------------------

def test_generate_pdf_returns_nonempty_bytes():
    pdf_bytes = rg.generate_pdf(_sample_report(), session_id=1, exam_name="Analisi 1")
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes[:4] == b"%PDF"
    assert len(pdf_bytes) > 500


def test_pdf_text_preserves_accents_and_uses_readable_symbol_fallback():
    result = rg.pdf_text("Perché è già così: σ, μ, ∑ e ∫")
    assert "Perché è già così" in result
    assert "sigma" in result and "mu" in result


def test_generate_pdf_preserves_accents_and_symbols():
    report = _sample_report()
    report["summary"] = "Perché è già così: sigma, mu, somma e l'integrale"
    pdf_bytes = rg.generate_pdf(report, session_id=1, exam_name="Analisi 1")
    from pypdf import PdfReader
    text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(pdf_bytes)).pages)
    assert "Perché è già così" in text


# ---------------------------------------------------------------------------
# save_report (I/O isolato in tmp_path via fixture isolated_reports_dir)
# ---------------------------------------------------------------------------

def test_save_report_writes_md_and_pdf(isolated_reports_dir):
    paths = rg.save_report(_sample_report(), session_id=9, exam_name="Analisi 1")

    md_path = isolated_reports_dir / paths["md"].split("\\")[-1].split("/")[-1]
    pdf_path = isolated_reports_dir / paths["pdf"].split("\\")[-1].split("/")[-1]

    assert md_path.exists()
    assert pdf_path.exists()
    assert md_path.read_text(encoding="utf-8").startswith("# Report Sessione")
    assert pdf_path.read_bytes()[:4] == b"%PDF"


def test_save_report_filename_includes_ids(isolated_reports_dir):
    paths = rg.save_report(_sample_report(), session_id=9, exam_name="Analisi 1")
    assert "1_2_9" in paths["md"]
    assert "1_2_9" in paths["pdf"]


def test_save_report_restores_previous_files_if_pdf_replace_fails(isolated_reports_dir, monkeypatch):
    from backend.agents import report_generator

    md = isolated_reports_dir / "1_2_9.md"
    pdf = isolated_reports_dir / "1_2_9.pdf"
    md.write_bytes(b"old markdown")
    pdf.write_bytes(b"old pdf")
    real_replace = report_generator.os.replace if hasattr(report_generator, "os") else __import__("os").replace

    def fail_pdf(source, target):
        if Path(target) == pdf:
            raise OSError("disk full SECRET")
        return real_replace(source, target)

    monkeypatch.setattr("os.replace", fail_pdf)
    import pytest
    with pytest.raises(OSError, match="disk full"):
        rg.save_report(_sample_report(), session_id=9)
    assert md.read_bytes() == b"old markdown"
    assert pdf.read_bytes() == b"old pdf"
    assert not list(isolated_reports_dir.glob(".1_2_9.*"))


def test_report_cleanup_is_scoped_to_session_and_exam(isolated_reports_dir):
    for stem in ("1_2_9", "1_2_10", "1_3_9", "2_2_9"):
        for extension in ("md", "pdf"):
            (isolated_reports_dir / f"{stem}.{extension}").write_text("report")
    rg.delete_session_reports(1, 2, 9)
    assert not (isolated_reports_dir / "1_2_9.md").exists()
    assert (isolated_reports_dir / "1_2_10.pdf").exists()
    rg.delete_exam_reports(1, 2)
    assert not (isolated_reports_dir / "1_2_10.pdf").exists()
    assert (isolated_reports_dir / "1_3_9.md").exists()
    assert (isolated_reports_dir / "2_2_9.pdf").exists()


def test_weak_topic_markdown_has_consistent_status_icon():
    md = rg.generate_markdown(_sample_report(), session_id=1)
    assert "| derivate | 4.0/10 | 1 | ❌ Da rivedere |" in md


def test_sufficient_topic_is_not_marked_mastered_in_report():
    report = _sample_report()
    report["topic_breakdown"] = [{"topic": "integrali", "avg_score": 7.0,
                                  "n_attempts": 1, "is_weak": False}]
    report["strong_topics"] = []
    md = rg.generate_markdown(report, session_id=1)
    assert "| integrali | 7.0/10 | 1 | 🟡 Sufficiente |" in md
    assert "Padroneggiato" not in md


def test_history_session_deletion_removes_report(isolated_reports_dir):
    from backend.api.session_extras import delete_incomplete_session
    from backend.db.models import Exam, Session, SessionLocal, Student

    with SessionLocal() as db:
        student = Student(username="report_owner", password_hash="test")
        db.add(student)
        db.flush()
        exam = Exam(student_id=student.id, name="Esame")
        db.add(exam)
        db.flush()
        session = Session(exam_id=exam.id, student_id=student.id)
        db.add(session)
        db.commit()
        student_id, exam_id, session_id = student.id, exam.id, session.id
    path = isolated_reports_dir / f"{student_id}_{exam_id}_{session_id}.pdf"
    path.write_bytes(b"report")
    delete_incomplete_session(session_id)
    assert not path.exists()
    with SessionLocal() as db:
        assert db.get(Session, session_id) is None
