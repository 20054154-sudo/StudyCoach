import logging

import pytest

from backend.agents.ingestor import chunk_text
from backend.agents.runtime_config import positive_int
from backend.observability import end_trace, run_agent, start_trace


def test_chunk_settings_are_configurable_and_validated(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_WORDS", "4")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP_WORDS", "1")
    assert chunk_text("uno due tre quattro cinque sei sette") == [
        "uno due tre quattro", "quattro cinque sei sette"
    ]
    monkeypatch.setenv("RAG_CHUNK_OVERLAP_WORDS", "4")
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("uno due tre quattro cinque")
    monkeypatch.setenv("RAG_CHUNK_WORDS", "oops")
    with pytest.raises(ValueError, match="RAG_CHUNK_WORDS"):
        chunk_text("uno due tre")


def test_positive_int_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("TEST_LIMIT", "0")
    with pytest.raises(ValueError, match="TEST_LIMIT"):
        positive_int("TEST_LIMIT", 4)


def test_trace_records_stage_without_prompt(caplog):
    class FakeModel:
        id = "test-model"

    class FakeAgent:
        model = FakeModel()

        def run(self, prompt, stream=False):
            assert prompt == "secret student response"
            return "ok"

    trace_id, token, _ = start_trace()
    try:
        with caplog.at_level(logging.INFO, logger="uvicorn.error"):
            assert run_agent(FakeAgent(), "secret student response", "tutor_test") == "ok"
    finally:
        end_trace(token)
    assert trace_id in caplog.text
    assert "tutor_test" in caplog.text
    assert "secret student response" not in caplog.text
