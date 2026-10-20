from backend.agents.upload_errors import upload_error
from backend.agents.llm_errors import ModelServiceUnavailable
from backend.error_messages import public_error


def test_public_error_hides_internal_details_but_preserves_upload_guidance():
    assert public_error(RuntimeError("SECRET database path"), "Riprova.") == "Riprova."
    assert "Azione suggerita" in public_error(upload_error("non_pdf"), "Riprova.")


def test_public_error_preserves_safe_provider_outage_message():
    error = ModelServiceUnavailable("Gemini è temporaneamente non disponibile. Riprova tra qualche minuto.")
    error.status_code = 503
    assert public_error(error, "Impossibile generare gli appunti.") == str(error)


def test_error_messages_import_does_not_require_reloaded_ingestor(monkeypatch):
    """A running process may retain the old ingestor module during a reload."""
    import importlib
    from backend.agents import ingestor
    from backend import error_messages

    with monkeypatch.context() as patch:
        patch.delattr(ingestor, "EmbeddingModelUnavailable")
        patch.delattr(ingestor, "MaterialIndexUnavailable")
        reloaded = importlib.reload(error_messages)
        assert reloaded.public_error(RuntimeError("private"), "Riprova.") == "Riprova."
