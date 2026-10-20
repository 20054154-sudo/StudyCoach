
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from fpdf import FPDF
from backend.agents.pdf_utils import pdf_text, register_unicode_fonts

REPORTS_DIR = Path(__file__).parent.parent.parent / "reports"


def _topic_status(topic: dict) -> str:
    if topic["is_weak"]:
        return "Da rivedere"
    if topic["avg_score"] >= 8:
        return "Padroneggiato"
    return "Sufficiente"


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def generate_markdown(report: dict,session_id: int,exam_name: str = "Esame",) -> str:
    now = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
    avg = report.get("overall_avg", 0)
    n_q = report.get("n_questions_answered", 0)
    weak = report.get("weak_topics", [])
    strong = report.get("strong_topics", [])
    breakdown = report.get("topic_breakdown", [])
    summary = report.get("summary", "")

    if avg >= 8:
        verdict = "🟢 Ottimo"
    elif avg >= 6:
        verdict = "🟡 Sufficiente"
    else:
        verdict = "🔴 Insufficiente"

    lines = [ f"# Report Sessione — {exam_name}", f"", f"**Data:** {now}  ", f"**Sessione ID:** {session_id}  ",
        f"**Studente ID:** {report.get('student_id')}  ", f"**Esame ID:** {report.get('exam_id')}  ", f"",f"---",
        f"", f"## Risultato Complessivo",f"", f"| Metrica | Valore |", f"|---|---|",f"| Media sessione | **{avg}/10** |",
        f"| Domande completate | {n_q} |",f"| Verdetto | {verdict} |",f"",f"---",f"",f"## Performance per Argomento",
        f"",f"| Argomento | Media | Tentativi | Stato |",f"|---|---|---|---|",]

    for t in breakdown:
        stato = f"{'❌' if t['is_weak'] else '✅' if t['avg_score'] >= 8 else '🟡'} {_topic_status(t)}"
        lines.append(f"| {t['topic']} | {t['avg_score']}/10 | {t['n_attempts']} | {stato} |")

    lines += [ f"", f"---", f"", f"## Aree da Rivedere", f"",]
    if weak:
        for t in weak:
            lines.append(f"- ❌ {t}")
    else:
        lines.append("Nessuna area critica. Ottimo lavoro!")

    lines += [ f"",  f"## Punti di Forza",  f"",]
    if strong:
        for t in strong:
            lines.append(f"- ✅ {t}")
    else:
        lines.append("Continua a studiare per consolidare gli argomenti.")

    lines += [f"",f"---",f"",f"## Sintesi",f"",f"{summary}",f"",f"---",f"",f"*Report generato automaticamente da Study Coach*",]

    return "\n".join(lines)


# PDF

class _ReportPDF(FPDF):
    """Sottoclasse FPDF con header e footer personalizzati."""

    def __init__(self):
        super().__init__()
        register_unicode_fonts(self)

    def header(self) -> None:
        self.set_font("Arial", "B", 10)
        self.set_text_color(80, 80, 80)
        self.cell(0, 8, "Study Coach - Report Sessione")
        self.ln(4)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Pagina {self.page_no()}", align="C")


def _add_section_title(pdf: FPDF, title: str) -> None:
    pdf.set_font("Arial", "B", 13)
    pdf.set_text_color(40, 40, 40)
    pdf.ln(4)
    pdf.cell(0, 8, pdf_text(title))
    pdf.ln(8)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)


def _add_body(pdf: FPDF, text: str) -> None:
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(60, 60, 60)
    pdf.multi_cell(0, 6, pdf_text(text))
    pdf.ln(2)


def generate_pdf( report: dict, session_id: int, exam_name: str = "Esame",) -> bytes:
    avg = report.get("overall_avg", 0)
    n_q = report.get("n_questions_answered", 0)
    weak = report.get("weak_topics", [])
    strong = report.get("strong_topics", [])
    breakdown = report.get("topic_breakdown", [])
    summary = report.get("summary", "")
    now = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")

    if avg >= 8:
        verdict = "Ottimo"
    elif avg >= 6:
        verdict = "Sufficiente"
    else:
        verdict = "Insufficiente"

    pdf = _ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Titolo
    pdf.set_font("Arial", "B", 18)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 12, pdf_text(f"Report Sessione - {exam_name}"))
    pdf.ln(12)
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, pdf_text(f"Generato il {now} | Sessione #{session_id}"))
    pdf.ln(10)

    # Risultato complessivo
    _add_section_title(pdf, "Risultato Complessivo")
    if avg >= 8:
        pdf.set_text_color(0, 140, 0)
    elif avg >= 6:
        pdf.set_text_color(200, 140, 0)
    else:
        pdf.set_text_color(200, 0, 0)
    pdf.set_font("Arial", "B", 28)
    pdf.cell(0, 14, f"{avg}/10")
    pdf.ln(14)
    pdf.set_text_color(60, 60, 60)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 6, pdf_text(f"Verdetto: {verdict}"))
    pdf.ln(6)
    pdf.cell(0, 6, pdf_text(f"Domande completate: {n_q}"))
    pdf.ln(10)

    # Tabella topic
    _add_section_title(pdf, "Performance per Argomento")
    pdf.set_font("Arial", "B", 10)
    pdf.set_fill_color(240, 240, 240)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(90, 7, "Argomento", border=1, fill=True)
    pdf.cell(25, 7, "Media", border=1, fill=True, align="C")
    pdf.cell(25, 7, "Tentativi", border=1, fill=True, align="C")
    pdf.cell(40, 7, "Stato", border=1, fill=True, align="C")
    pdf.ln()

    pdf.set_font("Arial", "", 10)
    for t in breakdown:
        stato = _topic_status(t)
        if t["is_weak"]:
            pdf.set_text_color(180, 0, 0)
        elif t["avg_score"] < 8:
            pdf.set_text_color(150, 110, 0)
        else:
            pdf.set_text_color(0, 130, 0)
        pdf.cell(90, 6, pdf_text(str(t["topic"])[:45]), border=1)
        pdf.cell(25, 6, f"{t['avg_score']}/10", border=1, align="C")
        pdf.cell(25, 6, str(t["n_attempts"]), border=1, align="C")
        pdf.cell(40, 6, pdf_text(stato), border=1, align="C")
        pdf.ln()

    pdf.set_text_color(60, 60, 60)
    pdf.ln(4)

    # Aree da rivedere
    _add_section_title(pdf, "Aree da Rivedere")
    if weak:
        for t in weak:
            _add_body(pdf, f"  - {t}")
    else:
        _add_body(pdf, "Nessuna area critica. Ottimo lavoro!")

    # Punti di forza
    _add_section_title(pdf, "Punti di Forza")
    if strong:
        for t in strong:
            _add_body(pdf, f"  - {t}")
    else:
        _add_body(pdf, "Continua a studiare per consolidare gli argomenti.")

    # Sintesi
    _add_section_title(pdf, "Sintesi")
    _add_body(pdf, summary)

    out = pdf.output()
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    if isinstance(out, str):
        return out.encode("latin-1", "replace")
    raise TypeError(f"Tipo output PDF non supportato: {type(out).__name__}")


# Salvataggio su disco

def save_report(report: dict,session_id: int,exam_name: str = "Esame",) -> dict[str, str]:
    """
    Genera e salva Markdown e PDF su disco.
    Returns: {"md": path_md, "pdf": path_pdf}

    Nessun fallback: qualunque errore nella generazione o nel salvataggio
    dei file si propaga esplicitamente al chiamante.
    """
    student_id = report.get("student_id", 0)
    exam_id = report.get("exam_id", 0)
    base_name = f"{student_id}_{exam_id}_{session_id}"

    md_path = REPORTS_DIR / f"{base_name}.md"
    pdf_path = REPORTS_DIR / f"{base_name}.pdf"

    md_content = generate_markdown(report, session_id, exam_name)
    pdf_bytes = generate_pdf(report, session_id, exam_name)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Prepare both artifacts before replacing either published file. Restore
    # existing reports if the second replacement fails.
    import os
    import tempfile

    staged = []
    previous = {md_path: md_path.read_bytes() if md_path.exists() else None,
                pdf_path: pdf_path.read_bytes() if pdf_path.exists() else None}
    try:
        for target, content in ((md_path, md_content.encode("utf-8")), (pdf_path, pdf_bytes)):
            with tempfile.NamedTemporaryFile(dir=REPORTS_DIR, prefix=f".{base_name}.", delete=False) as temp:
                staged.append(Path(temp.name))
                temp.write(content)
        replaced = []
        try:
            for source, target in zip(staged, (md_path, pdf_path)):
                os.replace(source, target)
                replaced.append(target)
        except Exception:
            for target in reversed(replaced):
                old_content = previous[target]
                if old_content is None:
                    target.unlink(missing_ok=True)
                else:
                    target.write_bytes(old_content)
            raise
    finally:
        for source in staged:
            source.unlink(missing_ok=True)

    return {"md": str(md_path), "pdf": str(pdf_path)}


def report_snapshot(student_id: int, exam_id: int, session_id: int) -> dict[Path, bytes | None]:
    """Capture existing artifacts so a failed database commit can undo a save."""
    stem = f"{int(student_id)}_{int(exam_id)}_{int(session_id)}"
    paths = (REPORTS_DIR / f"{stem}.md", REPORTS_DIR / f"{stem}.pdf")
    return {path: path.read_bytes() if path.exists() else None for path in paths}


def restore_report_snapshot(snapshot: dict[Path, bytes | None]) -> None:
    for path, content in snapshot.items():
        if content is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(content)


def delete_session_reports(student_id: int, exam_id: int, session_id: int) -> None:
    """Remove only the two files belonging to this session."""
    stem = f"{int(student_id)}_{int(exam_id)}_{int(session_id)}"
    for extension in ("md", "pdf"):
        (REPORTS_DIR / f"{stem}.{extension}").unlink(missing_ok=True)


def delete_exam_reports(student_id: int, exam_id: int) -> None:
    """Remove report artifacts for sessions cascaded with an exam."""
    prefix = f"{int(student_id)}_{int(exam_id)}_"
    for path in REPORTS_DIR.glob(f"{prefix}*.md"):
        if path.stem.removeprefix(prefix).isdigit():
            path.unlink()
    for path in REPORTS_DIR.glob(f"{prefix}*.pdf"):
        if path.stem.removeprefix(prefix).isdigit():
            path.unlink()
