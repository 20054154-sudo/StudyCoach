import pytest

from backend.agents.evaluation_validator import validate_contestation, validate_evaluation


def evaluation(**overrides):
    value = {
        "score": 7,
        "motivation": "Valutazione motivata.",
        "correct_points": ["Definizione corretta"],
        "missing_points": ["Manca un esempio"],
        "improvement_after_guidance": "Migliorata.",
    }
    value.update(overrides)
    return value


@pytest.mark.parametrize("field", ["score", "motivation", "correct_points", "missing_points", "improvement_after_guidance"])
def test_evaluation_missing_required_field_is_invalid(field):
    data = evaluation()
    del data[field]
    assert not validate_evaluation(data).valid


@pytest.mark.parametrize("data", [
    evaluation(score=8, missing_points=["a", "b"]),
    evaluation(score=3, missing_points=[]),
    evaluation(score=6, correct_points=[]),
    evaluation(score=1, correct_points=["a", "b"]),
    evaluation(score=10, missing_points=["a"]),
])
def test_evaluation_score_point_coherence_rules(data):
    assert not validate_evaluation(data).valid


def test_evaluation_positive_and_a1_regression_cases():
    assert validate_evaluation(evaluation(score=8, missing_points=[])).valid
    assert not validate_evaluation(evaluation(score=2, missing_points=[])).valid


def test_contestation_rules():
    valid = {
        "final_score": 8, "score_changed": True, "score_delta": 1,
        "contestation_assessment": "valido", "response_to_student": "Evidenza: la risposta cita il teorema."
    }
    assert validate_contestation(valid, original_score=7).valid
    assert not validate_contestation({**valid, "contestation_assessment": "altro"}, original_score=7).valid
    assert not validate_contestation({**valid, "contestation_assessment": "non_valido", "score_delta": 1}, original_score=7).valid
    assert not validate_contestation({**valid, "final_score": 11}, original_score=7).valid


def test_a2_substantial_non_so_is_not_a_format_error():
    data = evaluation(motivation="Non so la formula esatta, ma il teorema afferma che la media campionaria converge alla media vera")
    assert validate_evaluation(data).valid
