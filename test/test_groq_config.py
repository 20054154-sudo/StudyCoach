import importlib

from backend.agents import groq_config
from backend.agents import model_router, notes_generator, quiz_generator, socratic_tutor


def test_agents_share_groq_model_default():
    expected = "openai/gpt-oss-120b"
    assert groq_config.DEFAULT_GROQ_MODEL == expected
    assert notes_generator.DEFAULT_MODEL == expected
    assert quiz_generator.DEFAULT_MODEL == expected
    assert socratic_tutor.DEFAULT_MODEL == expected


def test_groq_model_can_be_overridden_by_environment(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    config = importlib.reload(groq_config)
    assert config.GROQ_MODEL == "llama-3.3-70b-versatile"
    monkeypatch.delenv("GROQ_MODEL")
    importlib.reload(groq_config)


def test_quiz_agent_initializes_groq_with_configured_model(monkeypatch):
    captured = {}

    class FakeGroq:
        def __init__(self, **kwargs):
            captured["groq_kwargs"] = kwargs

    class FakeAgent:
        def __init__(self, **kwargs):
            captured["agent_kwargs"] = kwargs

    monkeypatch.setattr(model_router, "Groq", FakeGroq)
    monkeypatch.setattr(quiz_generator, "Agent", FakeAgent)

    quiz_generator._make_quiz_agent(quiz_generator.DEFAULT_MODEL)

    assert captured["groq_kwargs"] == {"id": "openai/gpt-oss-120b", "timeout": 30.0}
    assert captured["agent_kwargs"]["markdown"] is False


def test_groq_error_redacts_api_key_and_bearer_token(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-secret-value")
    message = groq_config.safe_groq_error(
        RuntimeError("request failed with gsk-secret-value and Bearer abc.def-token")
    )
    assert "gsk-secret-value" not in message
    assert "abc.def-token" not in message
    assert "[REDACTED]" in message
