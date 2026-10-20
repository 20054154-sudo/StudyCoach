"""Deterministic validation rules for judge and contestation payloads."""

from dataclasses import dataclass, field
from typing import Any


VALID_CONTESTATION_ASSESSMENTS = {"valido", "parzialmente_valido", "non_valido"}
DEFAULT_VALIDATION_THRESHOLDS = {
    "high_score": 8,
    "high_score_max_missing": 1,
    "low_score": 3,
    "low_score_min_missing": 1,
    "adequate_score": 6,
    "adequate_min_correct": 1,
    "very_low_score_max": 1,
    "very_low_max_correct": 1,
    "perfect_score": 10,
}


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)


def _string_list(data: dict[str, Any], name: str, errors: list[str]) -> None:
    if name not in data:
        errors.append(f"campo assente: {name}")
        return
    value = data[name]
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        errors.append(f"{name} deve essere una lista di stringhe non vuote")


def validate_evaluation(data: dict[str, Any], thresholds: dict[str, int] | None = None) -> ValidationResult:
    """Validate the structural and score/content coherence rules."""
    cfg = {**DEFAULT_VALIDATION_THRESHOLDS, **(thresholds or {})}
    errors: list[str] = []
    required = ("score", "motivation", "correct_points", "missing_points", "improvement_after_guidance")
    for name in required:
        if name not in data:
            errors.append(f"campo assente: {name}")
    score = data.get("score")
    if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 10:
        errors.append("score deve essere un intero tra 0 e 10")
    if not isinstance(data.get("motivation"), str) or not data.get("motivation", "").strip():
        errors.append("motivation non può essere vuota")
    _string_list(data, "correct_points", errors)
    _string_list(data, "missing_points", errors)
    if not isinstance(data.get("improvement_after_guidance"), str):
        errors.append("improvement_after_guidance deve essere presente")
    if errors or not isinstance(score, int):
        return ValidationResult(False, errors)

    correct = data["correct_points"]
    missing = data["missing_points"]
    if score >= cfg["high_score"] and len(missing) > cfg["high_score_max_missing"]:
        errors.append("voto alto incoerente con molte lacune")
    if score <= cfg["low_score"] and len(missing) < cfg["low_score_min_missing"]:
        errors.append("voto basso senza lacune dichiarate")
    if score >= cfg["adequate_score"] and len(correct) < cfg["adequate_min_correct"]:
        errors.append("voto sufficiente senza punti corretti")
    if score <= cfg["very_low_score_max"] and len(correct) > cfg["very_low_max_correct"]:
        errors.append("voto molto basso con molti punti corretti")
    if score == cfg["perfect_score"] and missing:
        errors.append("voto 10 con punti mancanti")
    return ValidationResult(not errors, errors)


def validate_contestation(data: dict[str, Any], *, original_score: int) -> ValidationResult:
    errors: list[str] = []
    required = ("final_score", "score_changed", "score_delta", "contestation_assessment")
    for name in required:
        if name not in data:
            errors.append(f"campo assente: {name}")
    final_score = data.get("final_score")
    delta = data.get("score_delta")
    changed = data.get("score_changed")
    assessment = data.get("contestation_assessment")
    if not isinstance(final_score, int) or isinstance(final_score, bool) or not 0 <= final_score <= 10:
        errors.append("final_score deve essere un intero tra 0 e 10")
    if assessment not in VALID_CONTESTATION_ASSESSMENTS:
        errors.append("contestation_assessment non ammesso")
    if not isinstance(delta, int) or isinstance(delta, bool) or not isinstance(changed, bool):
        errors.append("score_delta/score_changed hanno tipo non valido")
    if not errors:
        if assessment == "non_valido" and (delta != 0 or changed is not False):
            errors.append("non_valido richiede delta 0 e score_changed False")
        if assessment == "valido" and (delta <= 0 or not str(data.get("response_to_student", "")).strip()):
            errors.append("valido richiede delta positivo ed evidenza citata")
        if changed != (delta != 0):
            errors.append("score_changed non coerente con score_delta")
        if final_score != original_score + delta:
            errors.append("final_score non coerente con il delta")
    return ValidationResult(not errors, errors)
