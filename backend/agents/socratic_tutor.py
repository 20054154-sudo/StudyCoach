"""
Socratic Tutor Agent — Study Coach
=====================================
Gestisce il loop di interrogazione socratica per una singola domanda.
"""

import json
import re
import time
from dataclasses import asdict, dataclass
from enum import Enum, auto
from pathlib import Path
from typing import cast

from agno.agent import Agent
from dotenv import load_dotenv

from backend.agents.ingestor import search_passages
from backend.agents.groq_config import GROQ_MODEL, safe_groq_error
from backend.agents.model_router import make_model
from backend.agents.llm_errors import (MAX_LLM_ATTEMPTS, is_retryable_error, retry_delay)
from backend.agents.evaluation_validator import validate_contestation, validate_evaluation
from backend.agents.quality_controls import add_citations, delimit_contestation, is_spoiler, rubric_score, parse_json_object
from backend.observability import run_agent

load_dotenv()

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

PROMPTS_DIR = Path(__file__).parent / "prompts"
PROMPT_GUIDE = "socratic_tutor_guide_v1"
PROMPT_EVAL = "socratic_tutor_eval_v1"


class NoRelevantMaterialError(ValueError):
    """The requested topic has no supporting passage in the selected files."""
PROMPT_CONTEST = "socratic_tutor_contest_v1"
PROMPT_DOUBT_OPEN = "socratic_tutor_doubt_open_v1"
PROMPT_DOUBT_EVAL = "socratic_tutor_doubt_eval_v1"

DEFAULT_MODEL = GROQ_MODEL
DEFAULT_N_CHUNKS = 5
RETRY_DELAY = 3
# Risposte piu' lunghe possono contenere contenuto disciplinare anche se
# includono una formula di incertezza.  La soglia e' intenzionalmente
# configurabile per poterla tarare sul materiale e sul modello usato.
NON_ANSWER_MAX_WORDS = 5
NON_ANSWER_MIN_EXTRA_WORDS = 3

NON_ANSWER_PATTERNS = (
    r"\bnon\s+so\b",
    r"\bnon\s+lo\s+so\b",
    r"\bnon\s+saprei\b",
    r"\bboh\b",
    r"\bnon\s+ricordo\b",
    r"\bnon\s+ne\s+ho\s+idea\b",
    r"\bnessuna\s+idea\b",
    r"\bidk\b",
    r"\bi\s+don'?t\s+know\b",
)


# ---------------------------------------------------------------------------
# Helper per rate limit
# ---------------------------------------------------------------------------

def _parse_retry_after(raw: str) -> int | None:
    """Estrae il tempo di attesa dal messaggio di errore (in secondi)."""
    # Pattern: 1h18m31.392s
    match = re.search(r'(?:try again in\s*)?(\d+)h(\d+)m(\d+\.?\d*)s', raw, re.IGNORECASE)
    if match:
        h, m, s = int(match.group(1)), int(match.group(2)), float(match.group(3))
        return int(h * 3600 + m * 60 + s)
    # Solo minuti
    match = re.search(r'(\d+)\s*(?:minute|min|m)', raw, re.IGNORECASE)
    if match:
        return int(match.group(1)) * 60
    # Solo secondi
    match = re.search(r'(\d+)\s*(?:second|sec|s)', raw, re.IGNORECASE)
    if match:
        return int(match.group(1))
    # Fallback
    return None


# ---------------------------------------------------------------------------
# Stato del loop
# ---------------------------------------------------------------------------

class TurnState(Enum):
    WAITING_ANSWER_1 = auto()
    WAITING_ANSWER_2 = auto()
    EVALUATED = auto()
    CONTESTED = auto()
    CLOSED = auto()


@dataclass
class EvaluationResult:
    score: int
    motivation: str
    correct_points: list[str]
    missing_points: list[str]
    improvement_after_guidance: str
    reliable: bool = True


class InsufficientEvaluationMaterialError(RuntimeError):
    """The retrieved source cannot support an academic assessment."""


@dataclass
class ContestationResult:
    final_score: int
    score_changed: bool
    score_delta: int
    contestation_assessment: str
    final_motivation: str
    response_to_student: str
    reliable: bool = True


class DoubtState(Enum):
    WAITING_ANSWER = auto()
    ANSWERED = auto()


@dataclass
class DoubtEvaluationResult:
    correct_points: list[str]
    missing_points: list[str]
    correct_answer: str


@dataclass
class DoubtAttempt:
    doubt_text: str
    guiding_question: str = ""
    topic_hint: str = ""
    student_answer: str = ""
    evaluation: DoubtEvaluationResult | None = None
    state: DoubtState = DoubtState.WAITING_ANSWER


@dataclass
class QuestionAttempt:
    question_id: int
    question_text: str
    difficulty: str
    topic_hint: str

    student_answer_1: str = ""
    guiding_question: str = ""
    positive_feedback: str = ""
    focus_area: str = ""
    student_answer_2: str = ""

    evaluation: EvaluationResult | None = None
    contestation_text: str = ""
    contestation_result: ContestationResult | None = None
    state: TurnState = TurnState.WAITING_ANSWER_1

    @property
    def final_score(self) -> int | None:
        if self.contestation_result is not None:
            return self.contestation_result.final_score
        if self.evaluation is not None:
            return self.evaluation.score
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_prompt(version_name: str) -> str:
    path = PROMPTS_DIR / f"{version_name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt non trovato: {path}")
    return path.read_text(encoding="utf-8")


def _parse_json(raw: str) -> dict:
    return parse_json_object(raw)


def _is_non_answer(text: str) -> bool:
    normalized = " ".join(text.lower().strip().split())
    if not normalized:
        return True

    matches = [re.search(pattern, normalized) for pattern in NON_ANSWER_PATTERNS]
    matches = [match for match in matches if match is not None]
    if not matches:
        return False

    # Un pattern da solo non basta: rimuoviamo le espressioni di incertezza
    # e consideriamo il resto. Una risposta breve e quasi interamente fatta
    # di "non so/boh" resta non-risposta; testo extra sufficientemente lungo
    # viene invece lasciato al valutatore disciplinare.
    remainder = normalized
    for match in sorted(matches, key=lambda item: item.start(), reverse=True):
        remainder = remainder[:match.start()] + remainder[match.end():]
    extra_words = re.findall(r"\b[\wÀ-ÿ']+\b", remainder)
    total_words = re.findall(r"\b[\wÀ-ÿ']+\b", normalized)
    return (
        len(total_words) <= NON_ANSWER_MAX_WORDS
        and len(extra_words) < NON_ANSWER_MIN_EXTRA_WORDS
    )


def _is_role_spoofing_only(text: str) -> bool:
    """Reject a reply that consists entirely of a forged instruction role."""
    return bool(re.match(r"(?is)^\s*(?:\[system\]|<\|system\|>).*", text))


def _apply_severity_guardrails(attempt: QuestionAttempt, evaluation: EvaluationResult) -> EvaluationResult:
    if not (_is_non_answer(attempt.student_answer_2)
            or _is_role_spoofing_only(attempt.student_answer_2)):
        return evaluation

    missing_points = list(evaluation.missing_points)
    strict_missing = "Risposta finale non valutabile: assenza di contenuto disciplinare verificabile."
    if strict_missing not in missing_points:
        missing_points.append(strict_missing)

    strict_motivation = (
        "Valutazione severa applicata: la seconda risposta non contiene "
        "una risposta disciplinare valutabile."
    )
    if evaluation.motivation:
        strict_motivation = f"{strict_motivation} {evaluation.motivation}"

    return EvaluationResult(
        score=0,
        motivation=strict_motivation,
        correct_points=evaluation.correct_points,
        missing_points=missing_points,
        improvement_after_guidance=evaluation.improvement_after_guidance,
    )


_CONTENT_STOPWORDS = {
    "anche", "ancora", "come", "con", "dalla", "delle", "dello", "dentro",
    "dopo", "essere", "fatto", "forse", "importante", "inoltre", "molto",
    "nella", "nelle", "nello", "perche", "perché", "questo", "questa",
    "quindi", "risposta", "secondo", "sopra", "studiare", "studiato",
    "tutto", "vero",
}


def _lacks_verifiable_content(attempt: QuestionAttempt) -> bool:
    """Flag short replies consisting only of generic or study-related words."""
    answer = attempt.student_answer_2.casefold()
    words = re.findall(r"[\wÀ-ÿ]+", answer)
    if len(words) > 18 or re.search(r"[=+*/^<>∑∫√]", answer):
        return False
    question_terms = set(re.findall(r"[\wÀ-ÿ]+", attempt.question_text.casefold()))
    answer_terms = {
        word for word in words
        if len(word) >= 4 and word not in _CONTENT_STOPWORDS and word not in question_terms
    }
    return not answer_terms


def _apply_content_guardrail(
    attempt: QuestionAttempt, evaluation: EvaluationResult
) -> EvaluationResult:
    if evaluation.score <= 1 or not _lacks_verifiable_content(attempt):
        return evaluation
    return EvaluationResult(
        score=1,
        motivation=(evaluation.motivation + " Punteggio limitato a 1/10: la risposta finale "
                    "contiene solo frasi generiche sullo studio.").strip(),
        correct_points=evaluation.correct_points,
        missing_points=evaluation.missing_points + ["Nessuna affermazione disciplinare nella risposta finale."],
        improvement_after_guidance=evaluation.improvement_after_guidance,
        reliable=evaluation.reliable,
    )


# Frasi con cui il valutatore dichiara esplicitamente l'assenza di lacune,
# pur non lasciando "missing_points" tecnicamente vuoto (es. una singola
# voce come "nessun punto significativo mancante").
NO_GAPS_PATTERNS = (
    r"nessun\s+punto\s+(significativo\s+)?mancante",
    r"nessuna\s+lacuna",
    r"non\s+ci\s+sono\s+lacune",
    r"tutti\s+i\s+punti\s+(chiave\s+)?(sono\s+)?(corretti|presenti|coperti)",
    r"nessun\s+elemento\s+mancante",
)


def _has_no_real_gaps(missing_points: list[str]) -> bool:
    """
    True se la lista "missing_points" non descrive alcuna lacuna reale:
    o è vuota, o contiene solo frasi che dichiarano esplicitamente
    l'assenza di lacune.
    """
    if not missing_points:
        return True
    for point in missing_points:
        normalized = " ".join(str(point).lower().strip().split())
        if not normalized:
            continue
        if not any(re.search(pattern, normalized) for pattern in NO_GAPS_PATTERNS):
            return False
    return True


def _apply_consistency_guardrails(
    evaluation: EvaluationResult,
    *,
    missing_points_present: bool = True,
) -> EvaluationResult:
    """
    Garantisce coerenza tra punteggio e lacune segnalate: se il valutatore
    non ha indicato alcuna lacuna disciplinare reale, il punteggio non può
    restare in una fascia "insufficiente/sufficiente" (es. 6-7). Questo
    corregge deterministicamente il caso in cui l'LLM elenca tutti i punti
    come corretti e nessun punto mancante, ma assegna comunque un voto
    parziale in modo incoerente.
    """
   
    if not missing_points_present or evaluation.score >= 9:
        return evaluation
    if not _has_no_real_gaps(evaluation.missing_points):
        return evaluation

    note = (
        "Punteggio corretto automaticamente a 9/10 per coerenza: la valutazione "
        "non ha segnalato alcuna lacuna disciplinare significativa."
    )
    motivation = f"{evaluation.motivation} {note}" if evaluation.motivation else note

    return EvaluationResult(
        score=9,
        motivation=motivation,
        correct_points=evaluation.correct_points,
        missing_points=evaluation.missing_points,
        improvement_after_guidance=evaluation.improvement_after_guidance,
    )


# ---------------------------------------------------------------------------
# Agno agent factories
# ---------------------------------------------------------------------------

def _guide_agent(model_name: str) -> Agent:
    return Agent(
        name="SocraticGuideAgent",
        model=make_model("tutor", model_name),
        description=(
            "Sei un tutor socratico universitario. "
            "NON dare mai la risposta corretta. "
            "Guida lo studente con domande mirate. "
            "Rispondi SOLO con JSON valido, senza markdown."
        ),
        markdown=False,
    )


def _eval_agent(model_name: str) -> Agent:
    return Agent(
        name="EvaluatorAgent",
        model=make_model("tutor", model_name),
        description=(
            "Sei un valutatore universitario esperto e obiettivo. "
            "Valuta la preparazione dello studente con un voto 0-10. "
            "Rispondi SOLO con JSON valido, senza markdown."
        ),
        markdown=False,
    )


def _contest_agent(model_name: str) -> Agent:
    return Agent(
        name="ContestationAgent",
        model=make_model("tutor", model_name),
        description=(
            "Sei un valutatore universitario imparziale. "
            "Riesamina la valutazione con obiettività dopo la contestazione. "
            "Rispondi SOLO con JSON valido, senza markdown."
        ),
        markdown=False,
    )


def _doubt_agent(model_name: str) -> Agent:
    return Agent(
        name="DoubtTutorAgent",
        model=make_model("tutor", model_name),
        description=(
            "Sei un tutor socratico universitario. Aiuti lo studente a chiarire "
            "un proprio dubbio guidandolo con domande mirate, senza mai rivelare "
            "subito la risposta. Rispondi SOLO con JSON valido, senza markdown."
        ),
        markdown=False,
    )


# ---------------------------------------------------------------------------
# Step LLM con retry e gestione rate limit
# ---------------------------------------------------------------------------

def _generate_guiding_question(
    attempt: QuestionAttempt,
    context_chunks: list[str],
    model_name: str,
) -> dict:
    template = _load_prompt(PROMPT_GUIDE)
    prompt = template.format(
        question=attempt.question_text,
        student_answer_1=attempt.student_answer_1,
        context_chunks="\n\n---\n\n".join(context_chunks),
    )
    agent = _guide_agent(model_name)

    for attempt_num in range(1, MAX_LLM_ATTEMPTS + 1):
        try:
            output = run_agent(agent, prompt, "tutor_guide")
            raw = output.get_content_as_string()

            data = _parse_json(raw)
            question = data.get("guiding_question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("Domanda guida assente o vuota")
            if any(is_spoiler(question, chunk, attempt.question_text + " " + attempt.student_answer_1) for chunk in context_chunks):
                if attempt_num == MAX_LLM_ATTEMPTS:
                    return {
                        "guiding_question": "Quale passaggio del materiale può aiutarti a precisare la tua risposta?",
                        "positive_feedback": "",
                        "focus_area": "Rileggi il passaggio pertinente del materiale.",
                    }
                raise ValueError("Domanda guida anticipa la risposta")
            return data

        except Exception as e:
            if not is_retryable_error(e):
                raise
            print(f"[SocraticGuideAgent] Tentativo {attempt_num} fallito ({safe_groq_error(e)}). Ritento...")
            if attempt_num < MAX_LLM_ATTEMPTS:
                time.sleep(retry_delay(attempt_num))
    raise RuntimeError(f"SocraticGuideAgent: esauriti {MAX_LLM_ATTEMPTS} tentativi")


def _evaluate_one(
    attempt: QuestionAttempt,
    context_chunks: list[str],
    model_name: str,
) -> EvaluationResult:
    if not context_chunks:
        raise InsufficientEvaluationMaterialError(
            "Materiale insufficiente per valutare la risposta dello studente."
        )
    template = _load_prompt(PROMPT_EVAL)
    prompt = template.format(
        question=attempt.question_text,
        student_answer_1=attempt.student_answer_1,
        guiding_question=attempt.guiding_question,
        student_answer_2=attempt.student_answer_2,
        context_chunks="\n\n---\n\n".join(context_chunks),
    )
    agent = _eval_agent(model_name)

    attempt_num = 0
    validation_failures = 0
    for attempt_num in range(1, MAX_LLM_ATTEMPTS + 1):
        try:
            output = run_agent(agent, prompt, "tutor_evaluate")
            raw = output.get_content_as_string()

            data = _parse_json(raw)
            if data.get("source_sufficient") is False:
                raise InsufficientEvaluationMaterialError(
                    "Materiale insufficiente per valutare la risposta dello studente."
                )
            validation = validate_evaluation(data)
            if not validation.valid:
                validation_failures += 1
                if validation_failures < 2:
                    raise ValueError("Giudizio non valido: " + "; ".join(validation.errors))
                score_value = data.get("score", 0)
                score_value = score_value if isinstance(score_value, int) and not isinstance(score_value, bool) else 0
                return EvaluationResult(
                    score=max(0, min(10, score_value)),
                    motivation="Valutazione non affidabile: " + "; ".join(validation.errors),
                    correct_points=data.get("correct_points", []) if isinstance(data.get("correct_points", []), list) else [],
                    missing_points=data.get("missing_points", []) if isinstance(data.get("missing_points", []), list) else [],
                    improvement_after_guidance=data.get("improvement_after_guidance", ""),
                    reliable=False,
                )
            if "score" not in data or "missing_points" not in data:
                raise ValueError("JSON di valutazione mancante di 'score' o 'missing_points'")
            if not isinstance(data["missing_points"], list):
                raise ValueError("'missing_points' deve essere una lista")
            score = rubric_score(data.get("correct_points", []), data["missing_points"])
            evaluation = EvaluationResult(
                score=score,
                motivation=data.get("motivation", "") or "Valutazione non affidabile.",
                correct_points=data.get("correct_points", []),
                missing_points=data["missing_points"],
                improvement_after_guidance=data.get("improvement_after_guidance", ""),
                reliable=validation.valid,
            )
            evaluation = _apply_severity_guardrails(attempt, evaluation)
            evaluation = _apply_consistency_guardrails(
                evaluation,
                missing_points_present=True,
            )
            return _apply_content_guardrail(attempt, evaluation)

        except Exception as e:
            if not is_retryable_error(e):
                raise
            print(f"[EvaluatorAgent] Tentativo {attempt_num} fallito ({safe_groq_error(e)}). Ritento...")
            if attempt_num < MAX_LLM_ATTEMPTS:
                time.sleep(retry_delay(attempt_num))
    raise RuntimeError(f"EvaluatorAgent: esauriti {MAX_LLM_ATTEMPTS} tentativi")


def _evaluate(
    attempt: QuestionAttempt,
    context_chunks: list[str],
    model_name: str,
) -> EvaluationResult:
    """Obtain two independent judgments and expose material disagreement."""
    first = _evaluate_one(attempt, context_chunks, model_name)
    if not first.reliable:
        return first
    second = _evaluate_one(attempt, context_chunks, model_name)
    if not second.reliable:
        first.reliable = False
        first.motivation += " Secondo giudizio non affidabile; voto escluso dalle statistiche."
        return first
    chosen = first if first.score <= second.score else second
    if abs(first.score - second.score) > 1:
        chosen.reliable = False
        chosen.motivation += (
            f" Giudizi discordanti ({first.score}/10 e {second.score}/10); "
            "voto escluso dalle statistiche."
        )
    elif first.score != second.score:
        chosen.motivation += f" Verifica indipendente: {first.score}/10 e {second.score}/10; mantenuto il voto prudente."
    return chosen


def _re_evaluate_after_contestation(
    attempt: QuestionAttempt,
    context_chunks: list[str],
    model_name: str,
) -> ContestationResult:
    if attempt.evaluation is None:
        raise ValueError("Nessuna valutazione iniziale: impossibile rivalutare.")

    template = _load_prompt(PROMPT_CONTEST)
    prompt = template.format(
        question=attempt.question_text,
        student_answer_1=attempt.student_answer_1,
        guiding_question=attempt.guiding_question,
        student_answer_2=attempt.student_answer_2,
        original_score=attempt.evaluation.score,
        original_motivation=attempt.evaluation.motivation,
        contestation=delimit_contestation(attempt.contestation_text),
        context_chunks="\n\n---\n\n".join(context_chunks),
    )
    agent = _contest_agent(model_name)

    for attempt_num in range(1, MAX_LLM_ATTEMPTS + 1):
        try:
            output = run_agent(agent, prompt, "tutor_contest")
            raw = output.get_content_as_string()

            data = _parse_json(raw)
            final_score = max(0, min(10, int(data.get("final_score", attempt.evaluation.score))))
            delta = final_score - attempt.evaluation.score

            return ContestationResult(
                final_score=final_score,
                score_changed=delta != 0,
                score_delta=delta,
                contestation_assessment=data.get("contestation_assessment", "non_valido"),
                final_motivation=data.get("final_motivation", ""),
                response_to_student=data.get("response_to_student", ""),
                reliable=validate_contestation(
                    data, original_score=attempt.evaluation.score
                ).valid,
            )

        except Exception as e:
            if not is_retryable_error(e):
                raise
            print(f"[ContestationAgent] Tentativo {attempt_num} fallito ({safe_groq_error(e)}). Ritento...")
            if attempt_num < MAX_LLM_ATTEMPTS:
                time.sleep(retry_delay(attempt_num))
    raise RuntimeError(f"ContestationAgent: esauriti {MAX_LLM_ATTEMPTS} tentativi")


# ---------------------------------------------------------------------------
# Step LLM — flusso "dubbio libero" (chatbot avviato dallo studente)
# ---------------------------------------------------------------------------

def _generate_opening_question(
    doubt_text: str,
    context_chunks: list[str],
    model_name: str,
) -> dict:
    template = _load_prompt(PROMPT_DOUBT_OPEN)
    prompt = template.format(
        doubt_text=doubt_text,
        context_chunks="\n\n---\n\n".join(context_chunks),
    )
    agent = _doubt_agent(model_name)

    for attempt_num in range(1, MAX_LLM_ATTEMPTS + 1):
        try:
            output = run_agent(agent, prompt, "doubt_open")
            raw = output.get_content_as_string()

            data = _parse_json(raw)
            if "guiding_question" not in data:
                raise ValueError(f"Risposta mancante di 'guiding_question': {data}")
            return data

        except Exception as e:
            if not is_retryable_error(e):
                raise
            print(f"[DoubtTutorAgent] Tentativo {attempt_num} fallito ({safe_groq_error(e)}). Ritento...")
            if attempt_num < MAX_LLM_ATTEMPTS:
                time.sleep(retry_delay(attempt_num))
    raise RuntimeError(f"DoubtTutorAgent: esauriti {MAX_LLM_ATTEMPTS} tentativi")


def _evaluate_doubt(
    attempt: "DoubtAttempt",
    context_chunks: list[str],
    model_name: str,
) -> DoubtEvaluationResult:
    template = _load_prompt(PROMPT_DOUBT_EVAL)
    prompt = template.format(
        doubt_text=attempt.doubt_text,
        guiding_question=attempt.guiding_question,
        student_answer=attempt.student_answer,
        context_chunks="\n\n---\n\n".join(context_chunks),
    )
    agent = _doubt_agent(model_name)

    for attempt_num in range(1, MAX_LLM_ATTEMPTS + 1):
        try:
            output = run_agent(agent, prompt, "doubt_evaluate")
            raw = output.get_content_as_string()

            data = _parse_json(raw)
            if "correct_answer" not in data:
                raise ValueError(f"Risposta mancante di 'correct_answer': {data}")
            return DoubtEvaluationResult(
                correct_points=data.get("correct_points", []),
                missing_points=data.get("missing_points", []),
                correct_answer=data["correct_answer"],
            )

        except Exception as e:
            if not is_retryable_error(e):
                raise
            print(f"[DoubtTutorAgent] Tentativo {attempt_num} fallito ({safe_groq_error(e)}). Ritento...")
            if attempt_num < MAX_LLM_ATTEMPTS:
                time.sleep(retry_delay(attempt_num))
    raise RuntimeError(f"DoubtTutorAgent: esauriti {MAX_LLM_ATTEMPTS} tentativi")


# ---------------------------------------------------------------------------
# SocraticTutor — classe principale
# ---------------------------------------------------------------------------

class SocraticTutor:
    """
    Gestisce il loop socratico per una singola domanda usando Agno agents.
    """

    def __init__(
        self,
        student_id: int,
        exam_id: int,
        model_name: str | None = None,
        n_chunks: int = DEFAULT_N_CHUNKS,
        material_ids: list[int] | None = None,
    ):
        self.student_id = student_id
        self.exam_id = exam_id
        self.model_name = model_name
        self.n_chunks = n_chunks
        self.material_ids = material_ids
        self.attempt: QuestionAttempt | None = None
        self._context_cache: list[str] = []
        self._source_cache: list[dict] = []

    def start(self, question_data: dict) -> str:
        self.attempt = QuestionAttempt(
            question_id=question_data["id"],
            question_text=question_data["text"],
            difficulty=question_data.get("difficulty", "intermedia"),
            topic_hint=question_data.get("topic_hint", ""),
            state=TurnState.WAITING_ANSWER_1,
        )
        self._source_cache = search_passages(
            query=question_data["text"],
            student_id=self.student_id,
            exam_id=self.exam_id,
            n_results=self.n_chunks,
            material_ids=self.material_ids,
        )
        self._context_cache = [passage["text"] for passage in self._source_cache]
        return self.attempt.question_text

    def submit_answer_1(self, answer: str) -> dict:
        self._assert_state(TurnState.WAITING_ANSWER_1)
        attempt = cast(QuestionAttempt, self.attempt)
        attempt.student_answer_1 = answer.strip()

        data = _generate_guiding_question(attempt, self._context_cache, self.model_name)
        attempt.guiding_question = data["guiding_question"]
        attempt.positive_feedback = data.get("positive_feedback", "")
        attempt.focus_area = data.get("focus_area", "")
        attempt.state = TurnState.WAITING_ANSWER_2

        print(f"[SocraticGuideAgent] focus={attempt.focus_area} | q_id={attempt.question_id}")
        return {
            "guiding_question": attempt.guiding_question,
            "positive_feedback": attempt.positive_feedback,
            "focus_area": attempt.focus_area,
        }

    def submit_answer_2(self, answer: str) -> dict:
        self._assert_state(TurnState.WAITING_ANSWER_2)
        attempt = cast(QuestionAttempt, self.attempt)
        attempt.student_answer_2 = answer.strip()

        eval_result = _evaluate(attempt, self._context_cache, self.model_name)
        eval_result.motivation = add_citations(eval_result.motivation, self._source_cache)
        attempt.evaluation = eval_result
        attempt.state = TurnState.EVALUATED

        print(f"[EvaluatorAgent] score={eval_result.score}/10 | q_id={attempt.question_id}")
        return {
            "score": eval_result.score,
            "motivation": eval_result.motivation,
            "correct_points": eval_result.correct_points,
            "missing_points": eval_result.missing_points,
            "improvement_after_guidance": eval_result.improvement_after_guidance,
            "reliable": eval_result.reliable,
            "can_contest": True,
        }

    def submit_contestation(self, argumentation: str) -> dict:
        self._assert_state(TurnState.EVALUATED)
        attempt = cast(QuestionAttempt, self.attempt)
        attempt.contestation_text = argumentation.strip()

        contest_result = _re_evaluate_after_contestation(attempt, self._context_cache, self.model_name)
        contest_result.final_motivation = add_citations(contest_result.final_motivation, self._source_cache)
        contest_result.response_to_student = add_citations(contest_result.response_to_student, self._source_cache)
        attempt.contestation_result = contest_result
        attempt.state = TurnState.CONTESTED

        print(
            f"[ContestationAgent] {contest_result.contestation_assessment} | "
            f"{attempt.evaluation.score} -> {contest_result.final_score} | q_id={attempt.question_id}"
        )
        return {
            "final_score": contest_result.final_score,
            "score_changed": contest_result.score_changed,
            "score_delta": contest_result.score_delta,
            "contestation_assessment": contest_result.contestation_assessment,
            "final_motivation": contest_result.final_motivation,
            "response_to_student": contest_result.response_to_student,
        }

    def close(self) -> QuestionAttempt:
        if self.attempt is None:
            raise RuntimeError("Tutor non inizializzato.")
        attempt = self.attempt
        if attempt.state not in (TurnState.EVALUATED, TurnState.CONTESTED):
            raise RuntimeError(
                f"Non si può chiudere in stato {attempt.state.name}. "
                "Lo studente deve aver completato almeno due risposte."
            )
        attempt.state = TurnState.CLOSED
        return attempt

    def to_dict(self) -> dict:
        if self.attempt is None:
            return {}
        d = asdict(self.attempt)
        d["state"] = self.attempt.state.name
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def _assert_state(self, expected: TurnState) -> None:
        if self.attempt is None:
            raise RuntimeError("Tutor non inizializzato. Chiama start() prima.")
        if self.attempt.state != expected:
            raise RuntimeError(
                f"Stato '{self.attempt.state.name}' non valido. "
                f"Atteso: '{expected.name}'."
            )


# ---------------------------------------------------------------------------
# DoubtTutor — chatbot per dubbi liberi dello studente
# ---------------------------------------------------------------------------

class DoubtTutor:
    """
    Gestisce il loop socratico avviato da un dubbio libero dello studente:
    lo studente scrive un dubbio -> il tutor risponde con una domanda guida
    socratica (senza svelare la risposta) -> lo studente risponde -> il tutor
    mostra dove ha sbagliato e fornisce la risposta corretta e migliorata,
    basandosi sui chunk RAG del materiale caricato per l'esame.
    """

    def __init__(
        self,
        student_id: int,
        exam_id: int,
        model_name: str | None = None,
        n_chunks: int = DEFAULT_N_CHUNKS,
    ):
        self.student_id = student_id
        self.exam_id = exam_id
        self.model_name = model_name
        self.n_chunks = n_chunks
        self.attempt: DoubtAttempt | None = None
        self._context_cache: list[str] = []
        self._source_cache: list[dict] = []

    def start(self, doubt_text: str) -> dict:
        doubt_text = doubt_text.strip()
        self._source_cache = search_passages(
            query=doubt_text,
            student_id=self.student_id,
            exam_id=self.exam_id,
            n_results=self.n_chunks,
        )
        self._context_cache = [passage["text"] for passage in self._source_cache]
        if not self._context_cache:
            raise NoRelevantMaterialError(
                "Non ho trovato materiale caricato pertinente a questo dubbio. "
                "Carica un PDF sull'argomento o riformula la domanda."
            )

        data = _generate_opening_question(doubt_text, self._context_cache, self.model_name)
        self.attempt = DoubtAttempt(
            doubt_text=doubt_text,
            guiding_question=data["guiding_question"],
            topic_hint=data.get("topic_hint", ""),
        )
        return {
            "guiding_question": self.attempt.guiding_question,
            "topic_hint": self.attempt.topic_hint,
        }

    def submit_answer(self, answer: str) -> dict:
        self._assert_state(DoubtState.WAITING_ANSWER)
        attempt = cast(DoubtAttempt, self.attempt)
        attempt.student_answer = answer.strip()

        eval_result = _evaluate_doubt(attempt, self._context_cache, self.model_name)
        eval_result.correct_answer = add_citations(eval_result.correct_answer, self._source_cache)
        attempt.evaluation = eval_result
        attempt.state = DoubtState.ANSWERED

        print(f"[DoubtTutorAgent] valutato dubbio '{attempt.topic_hint}'")
        return {
            "correct_points": eval_result.correct_points,
            "missing_points": eval_result.missing_points,
            "correct_answer": eval_result.correct_answer,
        }

    def to_dict(self) -> dict:
        if self.attempt is None:
            return {}
        d = asdict(self.attempt)
        d["state"] = self.attempt.state.name
        return d

    def _assert_state(self, expected: DoubtState) -> None:
        if self.attempt is None:
            raise RuntimeError("Tutor non inizializzato. Chiama start() prima.")
        if self.attempt.state != expected:
            raise RuntimeError(
                f"Stato '{self.attempt.state.name}' non valido. "
                f"Atteso: '{expected.name}'."
            )
