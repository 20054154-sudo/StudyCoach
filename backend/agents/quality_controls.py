"""Controlli deterministici per rubriche, grounding e sicurezza del prompt."""

import re
import json
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from typing import Literal


class LLMJsonError(ValueError):
    """Risposta JSON non conforme allo schema atteso."""


class GuidingQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")
    guiding_question: str = Field(min_length=1)
    positive_feedback: str = ""
    focus_area: str = ""


class QuizQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: int = Field(gt=0)
    text: str = Field(min_length=12)
    difficulty: Literal["base", "intermedia", "avanzata"]
    topic_hint: str = Field(min_length=2)
    source_quote: str = Field(min_length=8)


class QuizPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")
    questions: list[QuizQuestion]

    @model_validator(mode="after")
    def unique_questions(self):
        ids = [question.id for question in self.questions]
        texts = [re.sub(r"\s+", " ", question.text).casefold() for question in self.questions]
        if len(ids) != len(set(ids)) or len(texts) != len(set(texts)):
            raise ValueError("domande duplicate")
        return self


def parse_json_object(raw: object) -> dict:
    """Extract the first complete JSON object without rewriting its contents."""
    if raw is None or not str(raw).strip():
        raise LLMJsonError("contenuto modello vuoto")
    content = str(raw)
    start = content.find("{")
    if start < 0:
        raise LLMJsonError("JSON del modello assente")
    try:
        value, _ = json.JSONDecoder().raw_decode(content, start)
    except json.JSONDecodeError as exc:
        raise LLMJsonError("JSON del modello troncato o non valido") from exc
    if not isinstance(value, dict):
        raise LLMJsonError("atteso un oggetto JSON")
    return value


def parse_model_json(model, raw: object):
    try:
        return model.model_validate(parse_json_object(raw))
    except ValidationError as exc:
        raise LLMJsonError(f"JSON non conforme allo schema: {exc}") from exc


def validate_notes_structure(notes: str) -> str | None:
    """Return the first structural problem in generated Markdown notes."""
    headings = re.findall(r"(?m)^\s*#{1,2}\s+(.+)$", notes)
    if not any(not re.match(r"(?i)\s*riepilogo\b", heading) for heading in headings):
        return "Manca: heading di capitolo"
    summary_heading = re.search(r"(?im)^\s*#{1,6}\s+[*_`]*\s*riepilogo\b[^\n]*$", notes)
    if not summary_heading:
        return "Manca: sezione ## Riepilogo"
    if not notes[summary_heading.end():].strip():
        return "Riepilogo vuoto"
    if not re.sub(r"(?m)^\s*#{1,6}[^\n]*$", "", notes[:summary_heading.start()]).strip():
        return "Manca: contenuto degli appunti"
    if re.search(r'\\[a-zA-Z]+\^?\*', notes):
        return "LaTeX corrotto"
    return None


def rubric_score(correct_points: list[str], missing_points: list[str], *, weights=(2, 1)) -> int:
    correct = len([p for p in correct_points if isinstance(p, str) and p.strip()])
    missing = len([p for p in missing_points if isinstance(p, str) and p.strip()])
    if correct == 0 and missing:
        return 0
    return max(0, min(10, round(10 * correct * weights[0] / max(1, correct * weights[0] + missing * weights[1]))))


def add_citations(feedback: str, metadata: list[dict] | None) -> str:
    sources = sorted({
        (str(item.get("source", "")).strip(), item.get("page"))
        for item in (metadata or [])
        if item.get("source") and isinstance(item.get("page"), int)
    })
    if not sources:
        return feedback
    citations = "; ".join(f"{source}, p. {page}" for source, page in sources)
    return f"{feedback.rstrip()} (fonti consultate: {citations})"


def is_relevant_distance(distance: float | None, threshold: float) -> bool:
    return distance is not None and distance <= threshold


def is_spoiler(question: str, reference_answer: str, known_text: str = "") -> bool:
    """Detect a copied answer phrase that was not already in the student's input."""
    tokens = lambda text: re.findall(r"\w+", text.casefold())
    guide, source, known = tokens(question), tokens(reference_answer), tokens(known_text)
    if len(guide) < 5 or len(source) < 5:
        return False
    known_phrases = {tuple(known[i:i + 5]) for i in range(len(known) - 4)}
    source_phrases = {tuple(source[i:i + 5]) for i in range(len(source) - 4)}
    return any(
        tuple(guide[i:i + 5]) in source_phrases
        and tuple(guide[i:i + 5]) not in known_phrases
        for i in range(len(guide) - 4)
    )


def delimit_contestation(text: str) -> str:
    return f"<CONTESTAZIONE_STUDENTE>\n{text.strip()}\n</CONTESTAZIONE_STUDENTE>"
