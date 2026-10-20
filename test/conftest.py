"""
test/conftest.py
"""

import hashlib
import sys
import types
from pathlib import Path

import pytest
from sqlalchemy import create_engine

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "app"))

# ---------------------------------------------------------------------------
# Stub di agno.models.groq — evita import pesanti/rete per tutta la suite.
# Deve avvenire PRIMA che qualunque modulo backend.agents.* venga importato.
# ---------------------------------------------------------------------------

if "agno.models.groq" not in sys.modules:
    _groq_stub = types.ModuleType("agno.models.groq")

    class Groq:  # pragma: no cover - test stub
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    _groq_stub.Groq = Groq
    sys.modules["agno.models.groq"] = _groq_stub


@pytest.fixture(autouse=True)
def default_test_model_routing(monkeypatch):
    """Keep tests independent of a developer's private .env provider choices."""
    for feature in ("TUTOR", "QUIZ", "NOTES"):
        monkeypatch.setenv(f"{feature}_PROVIDER", "groq")
        monkeypatch.delenv(f"{feature}_MODEL", raising=False)
        monkeypatch.delenv(f"{feature}_FALLBACK_PROVIDER", raising=False)
        monkeypatch.delenv(f"{feature}_FALLBACK_MODEL", raising=False)


# ---------------------------------------------------------------------------
# Isolamento database (autouse: si applica a OGNI test della suite)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Ogni test parte da un SQLite temporaneo e vuoto, mai dal DB reale."""
    from backend.db import models as db_models

    test_engine = create_engine(f"sqlite:///{tmp_path / 'test_study_coach.db'}")
    monkeypatch.setattr(db_models, "engine", test_engine)
    db_models.SessionLocal.configure(bind=test_engine)
    db_models.init_db()

    yield test_engine

    db_models.SessionLocal.configure(bind=db_models.engine)


# ---------------------------------------------------------------------------
# Isolamento ChromaDB + embeddings finti (autouse)
# ---------------------------------------------------------------------------

def _fake_embedding(text: str, dim: int = 16) -> list[float]:
    """Vettore deterministico basato sull'hash del testo: stabile fra le chiamate,
    a dimensione fissa, senza scaricare/eseguire il modello reale."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [b / 255.0 for b in digest[:dim]]


class _FakeSentenceTransformer:
    def encode(self, texts, convert_to_numpy=True):
        import numpy as np
        vectors = [_fake_embedding(t) for t in texts]
        return np.array(vectors) if convert_to_numpy else vectors


@pytest.fixture(autouse=True)
def isolated_chroma(tmp_path, monkeypatch):
    """Ogni test usa una collection ChromaDB temporanea con embedding finti."""
    import chromadb
    from backend.agents import ingestor

    fake_client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    monkeypatch.setattr(ingestor, "client", fake_client)
    monkeypatch.setattr(ingestor, "SentenceTransformer", lambda *args, **kwargs: _FakeSentenceTransformer())
    monkeypatch.setattr(ingestor, "_model", None)

    yield fake_client


# ---------------------------------------------------------------------------
# Isolamento cartella reports/ (usata da report_generator e notes_generator)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_upload_dir(tmp_path, monkeypatch):
    """I PDF caricati via /exams/{id}/ingest finiscono in una cartella temporanea,
    mai in data/uploads/ del progetto reale."""
    from backend.api import exams as exams_api

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(exams_api, "UPLOAD_DIR", upload_dir)
    yield upload_dir


@pytest.fixture(autouse=True)
def isolated_reports_dir(tmp_path, monkeypatch):
    from backend.agents import report_generator

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(report_generator, "REPORTS_DIR", reports_dir)
    yield reports_dir


# ---------------------------------------------------------------------------
# Doppio di test per gli agenti Agno (nessuna chiamata LLM reale)
# ---------------------------------------------------------------------------

class FakeAgentOutput:
    def __init__(self, content: str):
        self._content = content

    def get_content_as_string(self) -> str:
        return self._content


class FakeAgent:
    """Sostituto di agno.agent.Agent: restituisce risposte pre-cotte in sequenza.

    `responses` può essere una stringa singola (riusata per ogni chiamata) o
    una lista (una risposta per ogni .run() successiva, l'ultima si ripete
    se le chiamate superano gli elementi disponibili).
    """

    def __init__(self, responses):
        self._responses = [responses] if isinstance(responses, str) else list(responses)
        self.calls: list[str] = []

    def run(self, prompt, stream: bool = False):
        self.calls.append(prompt)
        content = self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]
        return FakeAgentOutput(content)


@pytest.fixture
def fake_agent_factory():
    """Restituisce una funzione factory(responses) -> callable(model_name) -> FakeAgent,
    pronta per essere passata a monkeypatch.setattr al posto di _guide_agent/_eval_agent/ecc."""

    def _factory(responses):
        agent = FakeAgent(responses)
        return lambda model_name=None: agent

    return _factory


# ---------------------------------------------------------------------------
# Client FastAPI (integrazione API)
# ---------------------------------------------------------------------------

@pytest.fixture
def api_client(isolated_db, isolated_chroma):
    """TestClient FastAPI su DB e ChromaDB isolati (dipendenza esplicita per
    garantire che l'isolamento sia attivo PRIMA che il lifespan dell'app
    chiami init_db())."""
    from fastapi.testclient import TestClient
    from backend.api.main import app

    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------------------
# PDF di esempio per i test di ingestion
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_pdf_bytes() -> bytes:
    from pypdf import PdfWriter
    import io

    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


@pytest.fixture
def sample_pdf_path(tmp_path) -> Path:
    """PDF reale con testo estraibile, per testare ingest_pdf() end-to-end."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    text = (
        "Il machine learning e' una branca dell'intelligenza artificiale "
        "che si occupa di algoritmi in grado di apprendere dai dati. "
    ) * 30
    pdf.multi_cell(0, 8, text)
    path = tmp_path / "materiale.pdf"
    pdf.output(str(path))
    return path
