"""Unit/integration test — backend/agents/quiz_generator.py."""

import json

import pytest

from backend.agents import quiz_generator as qg
from backend.agents.llm_errors import is_rate_limit_error

VALID_QUIZ_JSON = json.dumps({
    "source_name": "Analisi 1",
    "questions": [
        {"id": 1, "text": "Cos'e' un limite?", "difficulty": "base", "topic_hint": "limiti", "source_quote": "materiale sui limiti"},
        {"id": 2, "text": "Spiega la derivata.", "difficulty": "intermedia", "topic_hint": "derivate", "source_quote": "materiale sui limiti"},
    ],
})


# ---------------------------------------------------------------------------
# _parse_llm_response
# ---------------------------------------------------------------------------

def test_parse_llm_response_valid_json():
    data = qg._parse_llm_response(VALID_QUIZ_JSON)
    assert data["source_name"] == "Analisi 1"
    assert len(data["questions"]) == 2


def test_parse_llm_response_strips_markdown_fences():
    wrapped = f"```json\n{VALID_QUIZ_JSON}\n```"
    data = qg._parse_llm_response(wrapped)
    assert len(data["questions"]) == 2


def test_parse_llm_response_strips_surrounding_text():
    wrapped = f"Ecco il JSON richiesto:\n{VALID_QUIZ_JSON}\nSpero vada bene!"
    data = qg._parse_llm_response(wrapped)
    assert len(data["questions"]) == 2


def test_parse_llm_response_rejects_trailing_commas():
    broken = '{"questions": [{"id": 1, "text": "domanda?", "difficulty": "base", "topic_hint": "x", "source_quote": "materiale sui limiti"},]}'
    with pytest.raises(ValueError, match="JSON quiz"):
        qg._parse_llm_response(broken)


def test_parse_llm_response_preserves_punctuation_inside_question():
    question = "Quando compare la sequenza ,} oppure ,] nel testo?"
    raw = json.dumps({"questions": [{"id": 1, "text": question, "difficulty": "base", "topic_hint": "sintassi", "source_quote": "materiale sui limiti"}]})
    assert qg._parse_llm_response(f"Risposta:\n```json\n{raw}\n```")["questions"][0]["text"] == question


def test_parse_llm_response_empty_raises():
    with pytest.raises(ValueError, match="vuota"):
        qg._parse_llm_response("")


def test_parse_llm_response_missing_questions_field_raises():
    with pytest.raises(ValueError, match="questions"):
        qg._parse_llm_response('{"source_name": "x"}')


def test_parse_llm_response_questions_not_a_list_raises():
    with pytest.raises(ValueError):
        qg._parse_llm_response('{"questions": "not a list"}')


def test_parse_llm_response_accepts_empty_quiz_as_insufficient_source():
    assert qg._parse_llm_response('{"questions": []}')["questions"] == []


def test_parse_llm_response_rejects_invalid_or_duplicate_questions():
    payload = json.loads(VALID_QUIZ_JSON)
    payload["questions"][0]["difficulty"] = "molto difficile"
    with pytest.raises(ValueError, match="Quiz non conforme"):
        qg._parse_llm_response(json.dumps(payload))
    payload = json.loads(VALID_QUIZ_JSON)
    payload["questions"][1]["id"] = 1
    with pytest.raises(ValueError, match="Quiz non conforme"):
        qg._parse_llm_response(json.dumps(payload))
    payload = json.loads(VALID_QUIZ_JSON)
    del payload["questions"][0]["source_quote"]
    with pytest.raises(ValueError, match="Quiz non conforme"):
        qg._parse_llm_response(json.dumps(payload))


def test_quiz_contexts_cover_all_sources_and_pages(monkeypatch):
    monkeypatch.setattr(qg, "MAX_QUIZ_CONTEXT_CHARS", 180)
    passages = [
        {"text": f"Concetto distintivo pagina {page} " * 5, "source": source, "page": page}
        for source in ("a.pdf", "b.pdf") for page in range(1, 8)
    ]
    contexts = qg._quiz_contexts(passages)
    assert len(contexts) > 1
    joined = "\n".join(contexts)
    assert all(f"[Fonte: {source}, p. {page}]" in joined for source in ("a.pdf", "b.pdf") for page in range(1, 8))
    assert max(map(len, contexts)) <= 180


def test_quiz_grounding_requires_quote_in_source():
    with pytest.raises(ValueError, match="Citazione"):
        qg._validate_quiz_grounding(json.loads(VALID_QUIZ_JSON), "Materiale completamente diverso")
    metadata_quote = json.loads(VALID_QUIZ_JSON)
    metadata_quote["questions"][0]["source_quote"] = "Fonte: poesia.pdf"
    with pytest.raises(ValueError, match="Citazione"):
        qg._validate_quiz_grounding(metadata_quote, "[Fonte: poesia.pdf]\nmateriale sui limiti")


def test_generate_quiz_selects_from_first_and_last_batches(monkeypatch):
    monkeypatch.setattr(qg, "MAX_QUIZ_CONTEXT_CHARS", 110)
    passages = [
        {"text": f"Materiale{index} descrive un concetto specifico della fonte. ",
         "source": f"fonte{index}.pdf", "page": 1}
        for index in range(4)
    ]
    monkeypatch.setattr(qg, "retrieve_passages", lambda **_: passages)

    class Output:
        def __init__(self, raw):
            self.raw = raw

        def get_content_as_string(self):
            return self.raw

    class Agent:
        def run(self, prompt, stream=False):
            index = next(index for index in range(4) if f"Materiale{index}" in prompt)
            return Output(json.dumps({"questions": [{
                "id": 1, "text": f"Che cosa descrive il Materiale{index} della fonte?",
                "difficulty": "base", "topic_hint": f"tema{index}",
                "source_quote": f"Materiale{index} descrive un concetto specifico",
            }]}))

    monkeypatch.setattr(qg, "_make_quiz_agent", lambda *_: Agent())
    result = qg.generate_quiz(1, 1, n_questions=2)
    assert [question.topic_hint for question in result.questions] == ["tema0", "tema3"]


def test_generate_quiz_bounds_model_calls_across_large_material(monkeypatch):
    monkeypatch.setattr(qg, "MAX_QUIZ_CONTEXT_CHARS", 110)
    passages = [
        {"text": f"Materiale{index} descrive un concetto specifico della fonte.",
         "source": "dispensa.pdf", "page": index + 1}
        for index in range(40)
    ]
    monkeypatch.setattr(qg, "retrieve_passages", lambda **_: passages)
    seen = []

    class Agent:
        def run(self, prompt, stream=False):
            index = int(__import__("re").search(r"Materiale(\d+)", prompt).group(1))
            seen.append(index)
            payload = {"questions": [{"id": 1, "text": f"Che cosa descrive Materiale{index}?",
                        "difficulty": "base", "topic_hint": f"tema{index}",
                        "source_quote": f"Materiale{index} descrive un concetto specifico"}]}
            return type("Output", (), {"get_content_as_string": lambda self: json.dumps(payload)})()

    monkeypatch.setattr(qg, "_make_quiz_agent", lambda *_: Agent())
    result = qg.generate_quiz(1, 1, n_questions=3)
    assert len(seen) <= 3
    assert seen[0] == 0 and seen[-1] == 39
    assert result.n_questions == 3


def test_context_sampling_keeps_small_selected_source(monkeypatch):
    monkeypatch.setattr(qg, "MAX_QUIZ_CONTEXT_CHARS", 110)
    passages = [
        {"text": f"Fisica{index} descrive un argomento molto ampio e distinto.",
         "source": "fisica.pdf", "page": index + 1}
        for index in range(40)
    ] + [{"text": "Letteratura descrive una poesia e il suo autore.",
          "source": "letteratura.pdf", "page": 1}]
    selected = qg._select_quiz_contexts(qg._quiz_contexts(passages), 3)
    assert len(selected) <= 3
    assert any("Letteratura descrive" in context for context in selected)
    assert any("Fisica39 descrive" in context for context in selected)


def test_context_sampling_keeps_every_source_above_normal_batch_limit(monkeypatch):
    monkeypatch.setattr(qg, "MAX_QUIZ_CONTEXT_CHARS", 90)
    passages = [{"text": f"Fonte {index} presenta un concetto distinto e verificabile.",
                 "source": f"fonte{index}.pdf", "page": 1} for index in range(7)]
    selected = qg._select_quiz_contexts(qg._quiz_contexts(passages), 7)
    assert len(selected) >= 7
    for index in range(7):
        assert any(f"Fonte {index} presenta" in context for context in selected)


def test_generate_quiz_corrects_invalid_quote_on_retry(monkeypatch):
    monkeypatch.setattr(qg, "retrieve_passages", lambda **_: [
        {"text": "La fotosintesi trasforma la luce in energia chimica.", "source": "bio.pdf", "page": 1}
    ])
    monkeypatch.setattr(qg.time, "sleep", lambda *_: None)
    prompts = []

    class Agent:
        def run(self, prompt, stream=False):
            prompts.append(prompt)
            quote = "La fotosintesi produce oro" if len(prompts) == 1 else "La fotosintesi trasforma la luce"
            payload = {"questions": [{"id": 1, "text": "Che cosa fa la fotosintesi?",
                        "difficulty": "base", "topic_hint": "biologia", "source_quote": quote}]}
            return type("Output", (), {"get_content_as_string": lambda self: json.dumps(payload)})()

    monkeypatch.setattr(qg, "_make_quiz_agent", lambda *_: Agent())
    assert qg.generate_quiz(1, 1, n_questions=1).n_questions == 1
    assert len(prompts) == 2
    assert "citazione" in prompts[1].lower() and prompts[1] != prompts[0]


def test_invalid_batch_does_not_discard_grounded_batch(monkeypatch):
    monkeypatch.setattr(qg, "MAX_QUIZ_CONTEXT_CHARS", 105)
    monkeypatch.setattr(qg, "MAX_LLM_ATTEMPTS", 2)
    monkeypatch.setattr(qg.time, "sleep", lambda *_: None)
    monkeypatch.setattr(qg, "retrieve_passages", lambda **_: [
        {"text": "Materiale0 spiega un tema valido.", "source": "a.pdf", "page": 1},
        {"text": "Materiale1 spiega un secondo tema valido.", "source": "b.pdf", "page": 1},
    ])
    calls = []

    class Agent:
        def run(self, prompt, stream=False):
            calls.append(prompt)
            first = "Materiale0" in prompt
            payload = {"questions": [{"id": 1, "text": "Quale tema spiega la fonte?",
                        "difficulty": "base", "topic_hint": "tema",
                        "source_quote": "citazione inventata" if first else "Materiale1 spiega un secondo tema"}]}
            return type("Output", (), {"get_content_as_string": lambda self: json.dumps(payload)})()

    monkeypatch.setattr(qg, "_make_quiz_agent", lambda *_: Agent())
    assert qg.generate_quiz(1, 1, n_questions=2).n_questions == 1
    assert len(calls) == 4


def test_invalid_batch_recovers_missing_question_from_other_selected_context(monkeypatch):
    monkeypatch.setattr(qg, "MAX_QUIZ_CONTEXT_CHARS", 110)
    monkeypatch.setattr(qg.time, "sleep", lambda *_: None)
    monkeypatch.setattr(qg, "retrieve_passages", lambda **_: [
        {"text": "Materiale0 descrive il primo concetto e anche un secondo concetto distinto.",
         "source": "a.pdf", "page": 1},
        {"text": "Materiale1 descrive il terzo concetto.", "source": "b.pdf", "page": 1},
    ])
    calls = []

    class Agent:
        def run(self, prompt, stream=False):
            calls.append(prompt)
            if "Genera UNA domanda diversa" in prompt:
                text, quote = "Qual è il secondo concetto?", "un secondo concetto distinto"
            elif "Materiale1" in prompt:
                text, quote = "Qual è il terzo concetto?", "citazione inventata"
            else:
                text, quote = "Qual è il primo concetto?", "Materiale0 descrive il primo concetto"
            payload = {"questions": [{"id": 1, "text": text, "difficulty": "base",
                        "topic_hint": "concetti", "source_quote": quote}]}
            return type("Output", (), {"get_content_as_string": lambda self: json.dumps(payload)})()

    monkeypatch.setattr(qg, "_make_quiz_agent", lambda *_: Agent())
    result = qg.generate_quiz(1, 1, n_questions=2)
    assert len(result.questions) == 2
    assert [question.text for question in result.questions] == ["Qual è il primo concetto?", "Qual è il secondo concetto?"]
    assert len(calls) == 4


def test_unverifiable_quiz_stops_after_one_corrective_retry(monkeypatch):
    monkeypatch.setattr(qg, "retrieve_passages", lambda **_: [
        {"text": "La fotosintesi trasforma la luce in energia chimica.", "source": "bio.pdf", "page": 1}
    ])
    monkeypatch.setattr(qg.time, "sleep", lambda *_: None)
    prompts = []

    class Agent:
        def run(self, prompt, stream=False):
            prompts.append(prompt)
            payload = {"questions": [{"id": 1, "text": "Che cosa fa la fotosintesi?",
                        "difficulty": "base", "topic_hint": "biologia",
                        "source_quote": "citazione inventata"}]}
            return type("Output", (), {"get_content_as_string": lambda self: json.dumps(payload)})()

    monkeypatch.setattr(qg, "_make_quiz_agent", lambda *_: Agent())
    with pytest.raises(qg.UnverifiableQuizError):
        qg.generate_quiz(1, 1, n_questions=1)
    assert len(prompts) == 2


# ---------------------------------------------------------------------------
# _is_rate_limit_error
# ---------------------------------------------------------------------------

def test_rate_limit_text_is_not_an_error():
    assert not is_rate_limit_error(RuntimeError("la media e' 429 e la varianza 500"))


def test_rate_limit_http_429_exception_is_detected():
    error = RuntimeError("Too Many Requests")
    error.status_code = 429
    assert is_rate_limit_error(error)


def test_generate_quiz_propagates_groq_daily_quota_response(monkeypatch):
    monkeypatch.setattr(qg, "retrieve_passages", lambda **_: [
        {"text": "La fotosintesi trasforma la luce in energia chimica.", "source": "bio.pdf", "page": 1}
    ])
    calls = []

    class Agent:
        def run(self, prompt, stream=False):
            calls.append(prompt)
            payload = {"error": {"code": "rate_limit_exceeded", "message":
                       "Rate limit reached on tokens per day (TPD): Limit 200000, Used 194147, Requested 6707."}}
            return type("Output", (), {"get_content_as_string": lambda self: json.dumps(payload)})()

    monkeypatch.setattr(qg, "_make_quiz_agent", lambda *_: Agent())
    with pytest.raises(Exception) as caught:
        qg.generate_quiz(1, 1, n_questions=1)
    assert is_rate_limit_error(caught.value)
    assert "giornalier" in str(caught.value).lower()
    assert "citazion" not in str(caught.value).lower()
    assert len(calls) == 1


def test_generate_quiz_propagates_http_429_without_retry(monkeypatch):
    monkeypatch.setattr(qg, "retrieve_passages", lambda **_: [
        {"text": "La fotosintesi trasforma la luce in energia chimica.", "source": "bio.pdf", "page": 1}
    ])
    calls = []

    class Agent:
        def run(self, prompt, stream=False):
            calls.append(prompt)
            error = RuntimeError("Rate limit reached on tokens per day (TPD)")
            error.status_code = 429
            raise error

    monkeypatch.setattr(qg, "_make_quiz_agent", lambda *_: Agent())
    with pytest.raises(Exception) as caught:
        qg.generate_quiz(1, 1, n_questions=1)
    assert is_rate_limit_error(caught.value)
    assert "giornalier" in str(caught.value).lower()
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# generate_quiz (integrazione con FakeAgent + RAG finto)
# ---------------------------------------------------------------------------

def test_generate_quiz_raises_without_indexed_material(monkeypatch):
    monkeypatch.setattr(qg, "retrieve_passages", lambda **kwargs: [])
    with pytest.raises(ValueError, match="Nessun chunk trovato"):
        qg.generate_quiz(student_id=1, exam_id=1)


def test_generate_quiz_returns_parsed_questions(monkeypatch, fake_agent_factory):
    monkeypatch.setattr(qg, "retrieve_passages", lambda **kwargs: [{"text": "chunk di materiale sui limiti", "source": "Analisi", "page": 1}])
    monkeypatch.setattr(qg, "_make_quiz_agent", fake_agent_factory(VALID_QUIZ_JSON))

    result = qg.generate_quiz(student_id=1, exam_id=2, n_questions=2, source_name="Analisi 1")

    assert result.exam_id == 2
    assert result.student_id == 1
    assert result.n_questions == 2
    assert result.questions[0].text == "Cos'e' un limite?"
    assert result.questions[0].difficulty == "base"


def test_generate_quiz_uses_only_requested_material_ids(monkeypatch, fake_agent_factory):
    received = {}

    def selected_passages(**kwargs):
        received.update(kwargs)
        return [{"text": "chunk di materiale sui limiti", "source": "limiti.pdf", "page": 1}]

    monkeypatch.setattr(qg, "retrieve_passages", selected_passages)
    monkeypatch.setattr(qg, "_make_quiz_agent", fake_agent_factory(VALID_QUIZ_JSON))
    qg.generate_quiz(student_id=1, exam_id=2, n_questions=2, material_ids=[7])
    assert received["material_ids"] == [7]


def test_generate_quiz_accepts_fewer_supported_questions(monkeypatch, fake_agent_factory):
    monkeypatch.setattr(qg, "retrieve_passages", lambda **kwargs: [{"text": "chunk di materiale sui limiti", "source": "Analisi", "page": 1}])
    monkeypatch.setattr(qg.time, "sleep", lambda *_: None)

    too_few = json.dumps({"source_name": "x", "questions": [
        {"id": 1, "text": "una sola domanda", "difficulty": "base", "topic_hint": "limiti", "source_quote": "materiale sui limiti"},
    ]})
    monkeypatch.setattr(qg, "_make_quiz_agent", fake_agent_factory([too_few, VALID_QUIZ_JSON]))

    result = qg.generate_quiz(student_id=1, exam_id=1, n_questions=2)
    assert result.n_questions == 1


def test_quiz_prompt_adapts_to_literature_and_source_limits():
    template = qg._load_prompt_template()
    assert "Per letteratura" in template
    assert "restituiscine meno" in template
    assert "istruzioni" in template


def test_prompt_loader_preserves_horizontal_rule_in_body(tmp_path, monkeypatch):
    from backend.agents import notes_generator as ng, socratic_tutor as tutor

    path = tmp_path / "prompt.txt"
    path.write_text("Prima parte\n---\n\nSeconda parte", encoding="utf-8")
    monkeypatch.setattr(qg, "PROMPT_PATH", path)
    monkeypatch.setattr(ng, "PROMPT_PATH", path)
    monkeypatch.setattr(tutor, "PROMPTS_DIR", tmp_path)
    (tmp_path / "custom.txt").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    assert qg._load_prompt_template() == "Prima parte\n---\n\nSeconda parte"
    assert ng._load_prompt_template() == "Prima parte\n---\n\nSeconda parte"
    assert tutor._load_prompt("custom") == "Prima parte\n---\n\nSeconda parte"


def test_generate_quiz_rejects_zero_supported_questions(monkeypatch, fake_agent_factory):
    monkeypatch.setattr(qg, "retrieve_passages", lambda **kwargs: [{"text": "materiale leggibile", "source": "x", "page": 1}])
    monkeypatch.setattr(qg, "_make_quiz_agent", fake_agent_factory('{"questions": []}'))
    with pytest.raises(ValueError, match="Materiale insufficiente"):
        qg.generate_quiz(student_id=1, exam_id=1)


def test_quiz_to_dict_and_json_roundtrip(monkeypatch, fake_agent_factory):
    monkeypatch.setattr(qg, "retrieve_passages", lambda **kwargs: [{"text": "chunk di materiale sui limiti", "source": "Analisi", "page": 1}])
    monkeypatch.setattr(qg, "_make_quiz_agent", fake_agent_factory(VALID_QUIZ_JSON))

    result = qg.generate_quiz(student_id=1, exam_id=1, n_questions=2)
    d = qg.quiz_to_dict(result)
    assert d["exam_id"] == 1

    payload = qg.quiz_to_json(result)
    reparsed = json.loads(payload)
    assert reparsed["student_id"] == 1
