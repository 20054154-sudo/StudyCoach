"""Routing of the three AI workflows without contacting provider APIs."""

import pytest
from types import SimpleNamespace

from backend.agents import model_router, notes_generator, quiz_generator, socratic_tutor
from backend.agents.llm_errors import is_rate_limit_error, rate_limit_from_response


def test_each_feature_uses_its_configured_provider(monkeypatch):
    from agno.models.google import Gemini
    from agno.models.mistral import MistralChat

    monkeypatch.setenv("TUTOR_PROVIDER", "groq")
    monkeypatch.setenv("QUIZ_PROVIDER", "mistral")
    monkeypatch.setenv("QUIZ_MODEL", "mistral-small-latest")
    monkeypatch.setenv("NOTES_PROVIDER", "gemini")
    monkeypatch.setenv("NOTES_MODEL", "gemini-3.5-flash")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    for module in (socratic_tutor, quiz_generator, notes_generator):
        monkeypatch.setattr(module, "Agent", lambda **kwargs: SimpleNamespace(**kwargs))

    tutor = socratic_tutor._guide_agent(None)
    quiz = quiz_generator._make_quiz_agent(None)
    notes = notes_generator._make_notes_agent(None)
    review = notes_generator._make_notes_review_agent(None)

    assert tutor.model.kwargs["id"] == model_router.GROQ_MODEL
    assert isinstance(quiz.model, MistralChat)
    assert quiz.model.id == "mistral-small-latest"
    assert isinstance(notes.model, Gemini)
    assert notes.model.id == "gemini-3.5-flash"
    assert notes.model.api_key == "test-gemini-key"
    assert review.model.request_params["config"]["response_mime_type"] == "application/json"
    assert review.model.request_params["config"]["max_output_tokens"] == notes_generator.MAX_REVIEW_TOKENS


def test_model_name_can_be_overridden_for_one_call(monkeypatch):
    monkeypatch.setenv("QUIZ_PROVIDER", "mistral")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    assert model_router.make_model("quiz", "mistral-large-latest").id == "mistral-large-latest"
    assert model_router.model_choice("quiz").model_id == "mistral-small-latest"


def test_missing_provider_key_has_safe_configuration_error(monkeypatch):
    monkeypatch.setenv("QUIZ_PROVIDER", "mistral")
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(model_router.ModelConfigurationError, match="MISTRAL_API_KEY"):
        model_router.make_model("quiz")

    monkeypatch.setenv("NOTES_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(model_router.ModelConfigurationError, match="GOOGLE_API_KEY"):
        model_router.make_model("notes")


def test_unsupported_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("TUTOR_PROVIDER", "unknown")
    with pytest.raises(model_router.ModelConfigurationError, match="Provider non valido"):
        model_router.make_model("tutor")


def test_mistral_error_output_is_a_rate_limit_not_a_bad_quiz(monkeypatch):
    monkeypatch.setenv("QUIZ_PROVIDER", "mistral")
    monkeypatch.setattr(quiz_generator, "retrieve_passages", lambda **_: [
        {"text": "La fotosintesi trasforma la luce in energia chimica.", "source": "bio.pdf", "page": 1}
    ])
    calls = []

    class ErrorOutput:
        status = "ERROR"

        def get_content_as_string(self):
            return ('API error occurred: Status 429. Body: '
                    '{"object":"error","message":"Rate limit exceeded",'
                    '"type":"rate_limited","code":"1300","raw_status_code":429}')

    class FakeAgent:
        def run(self, prompt, stream=False):
            calls.append(prompt)
            return ErrorOutput()

    monkeypatch.setattr(quiz_generator, "_make_quiz_agent", lambda *_: FakeAgent())
    with pytest.raises(Exception) as caught:
        quiz_generator.generate_quiz(1, 1, n_questions=1)
    assert is_rate_limit_error(caught.value)
    assert "Mistral" in str(caught.value)
    assert "citazioni" not in str(caught.value)
    assert len(calls) == 1


def test_prefixed_error_is_ignored_without_failed_run_status():
    text = 'Il testo cita API error occurred: Status 429. Body: {"raw_status_code":429}'
    assert rate_limit_from_response(text, "Mistral") is None


def test_gemini_daily_request_quota_has_specific_message():
    payload = ('{"error":{"code":429,"status":"RESOURCE_EXHAUSTED",'
               '"message":"Quota exceeded",'
               '"details":[{"quotaId":"GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}}')
    error = rate_limit_from_response(payload, "Gemini", run_failed=True)
    assert error is not None and is_rate_limit_error(error)
    assert error.quota_period == "daily"
    assert "Quota giornaliera di richieste Gemini" in str(error)


def test_gemini_daily_quota_reports_provider_wait_without_raw_payload():
    payload = ('{"error":{"code":429,"status":"RESOURCE_EXHAUSTED",'
               '"message":"Quota exceeded",'
               '"details":[{"quotaId":"GenerateRequestsPerDayPerProjectPerModel-FreeTier"},'
               '{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"39303s"}]}}')
    error = rate_limit_from_response(payload, "Gemini", run_failed=True)
    assert error is not None and error.retry_after_seconds == 39303
    assert "10 ore e 56 minuti" in str(error)
    assert "Quota giornaliera di richieste Gemini" in str(error)
    assert "googleapis.com" not in str(error)


def test_quiz_reports_both_provider_limits_after_fallback(monkeypatch):
    monkeypatch.setenv("QUIZ_PROVIDER", "mistral")
    monkeypatch.setenv("QUIZ_FALLBACK_PROVIDER", "gemini")
    monkeypatch.setattr(quiz_generator, "retrieve_passages", lambda **_: [
        {"text": "La fotosintesi trasforma la luce in energia chimica.", "source": "bio.pdf", "page": 1}
    ])

    class FakeAgent:
        def __init__(self, provider):
            self.provider = provider

        def run(self, prompt, stream=False):
            if self.provider == "mistral":
                content = ('API error occurred: Status 429. Body: '
                           '{"raw_status_code":429,"message":"Rate limit exceeded"}')
            else:
                content = ('{"error":{"code":429,"status":"RESOURCE_EXHAUSTED",'
                           '"message":"Quota exceeded",'
                           '"details":[{"quotaId":"GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}}')
            return SimpleNamespace(status="ERROR", get_content_as_string=lambda: content)

    monkeypatch.setattr(quiz_generator, "_make_quiz_agent",
                        lambda _model_name, *, choice=None: FakeAgent(choice.provider if choice else "mistral"))
    with pytest.raises(Exception) as caught:
        quiz_generator.generate_quiz(1, 1, n_questions=1)
    assert is_rate_limit_error(caught.value)
    assert "Mistral" in str(caught.value) and "Gemini" in str(caught.value)
    assert "Quota giornaliera" in str(caught.value)


def test_quiz_uses_gemini_after_mistral_429_on_same_selected_source(monkeypatch):
    import json

    monkeypatch.setenv("QUIZ_PROVIDER", "mistral")
    monkeypatch.setenv("QUIZ_FALLBACK_PROVIDER", "gemini")
    monkeypatch.setenv("QUIZ_FALLBACK_MODEL", "gemini-3.5-flash")
    seen_material_ids = []

    def passages(**kwargs):
        seen_material_ids.append(kwargs["material_ids"])
        return [{"text": "La fotosintesi trasforma la luce in energia chimica.",
                 "source": "bio.pdf", "page": 1}]

    monkeypatch.setattr(quiz_generator, "retrieve_passages", passages)
    calls = []

    class FakeAgent:
        def __init__(self, provider):
            self.provider = provider

        def run(self, prompt, stream=False):
            calls.append((self.provider, prompt))
            if self.provider == "mistral":
                content = ('API error occurred: Status 429. Body: '
                           '{"object":"error","raw_status_code":429,"message":"Rate limit exceeded"}')
                status = "ERROR"
            else:
                content = json.dumps({"questions": [{"id": 1, "text": "Che cosa produce la fotosintesi?",
                                                   "difficulty": "base", "topic_hint": "biologia",
                                                   "source_quote": "La fotosintesi trasforma la luce"}]})
                status = "COMPLETED"
            return SimpleNamespace(status=status, get_content_as_string=lambda: content)

    monkeypatch.setattr(quiz_generator, "_make_quiz_agent",
                        lambda _model_name, *, choice=None: FakeAgent(choice.provider if choice else "mistral"))
    result = quiz_generator.generate_quiz(1, 1, n_questions=1, material_ids=[7])
    assert result.n_questions == 1
    assert [provider for provider, _ in calls] == ["mistral", "gemini"]
    assert seen_material_ids == [[7]]
    assert "bio.pdf" in calls[1][1]
