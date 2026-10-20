
import json
import logging
import re
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from agno.agent import Agent
from dotenv import load_dotenv

from backend.agents.ingestor import retrieve_passages
from backend.agents.groq_config import GROQ_MODEL, safe_groq_error
from backend.agents.runtime_config import positive_int
from backend.observability import run_agent
from backend.agents.model_router import ModelChoice, fallback_choice, make_model, provider_name
from backend.agents.llm_errors import LLM_TIMEOUT_SECONDS, MAX_LLM_ATTEMPTS, RetryableLLMError, is_rate_limit_error, is_retryable_error, provider_rate_limit_error, rate_limit_from_response, retry_delay
from backend.agents.quality_controls import LLMJsonError, QuizPayload, parse_model_json, parse_json_object

load_dotenv()

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

PROMPT_VERSION = "quiz_generator_v1"
PROMPT_PATH = Path(__file__).parent / "prompts" / f"{PROMPT_VERSION}.txt"
DEFAULT_N_QUESTIONS = 5
DEFAULT_N_CHUNKS = None
MAX_QUIZ_CONTEXT_CHARS = positive_int("QUIZ_MAX_CONTEXT_CHARS", 8000)
MAX_QUIZ_BATCHES = positive_int("QUIZ_MAX_BATCHES", 6)
MAX_INVALID_QUIZ_ATTEMPTS = positive_int("QUIZ_MAX_INVALID_ATTEMPTS", 2)
MAX_RECOVERY_QUIZ_CALLS = positive_int("QUIZ_MAX_RECOVERY_CALLS", 2)
DEFAULT_MODEL = GROQ_MODEL
RETRY_DELAY = 3
logger = logging.getLogger(__name__)


class InsufficientMaterialError(ValueError):
    """The indexed source does not support any verifiable question."""


class UnverifiableQuizError(ValueError):
    """The model returned no question backed by a literal source quote."""


def _quiz_limit_error(error: Exception, provider: str, *, previous_provider: str | None = None) -> RetryableLLMError:
    limit = error if isinstance(error, RetryableLLMError) else provider_rate_limit_error(str(error), provider)
    if previous_provider is None:
        return limit
    combined = RetryableLLMError(
        f"Limite chiamate API per il quiz: {previous_provider} ha raggiunto il limite di richieste; "
        f"anche {provider} non è disponibile. {limit}"
    )
    combined.status_code = 429
    combined.quota_period = getattr(limit, "quota_period", None)
    combined.retry_after_seconds = getattr(limit, "retry_after_seconds", None)
    return combined


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Question:
    id: int
    text: str
    difficulty: str
    topic_hint: str


@dataclass
class QuizResult:
    questions: list[Question]
    source_name: str
    n_questions: int
    prompt_version: str
    exam_id: int
    student_id: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_prompt_template() -> str:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Prompt template non trovato: {PROMPT_PATH}\n"
            "Assicurati che il file esista in backend/agents/prompts/"
        )
    return PROMPT_PATH.read_text(encoding="utf-8")


def _parse_llm_response(raw: str) -> dict:
    """Extract a complete JSON object without rewriting its string values."""
    if raw is None or not str(raw).strip():
        raise ValueError("Risposta vuota/None dal modello. Riprova la generazione.")
    try:
        data = parse_json_object(raw)
    except LLMJsonError as exc:
        raise ValueError(f"JSON quiz troncato o non valido: {exc}") from exc

    if "questions" not in data or not isinstance(data["questions"], list):
        raise ValueError(
            f"Campo 'questions' mancante o non è una lista.\n"
            f"Campi ricevuti: {list(data.keys())}\n"
            f"JSON: {json.dumps(data, indent=2)[:500]}"
        )

    try:
        parse_model_json(QuizPayload, json.dumps(data, ensure_ascii=False))
    except LLMJsonError as exc:
        raise ValueError(f"Quiz non conforme: {exc}") from exc
    return data


def _make_quiz_agent(model_name: str | None, *, choice: ModelChoice | None = None) -> Agent:
    """Crea un Agent Agno con il provider configurato per il quiz."""
    return Agent(
        name="QuizGeneratorAgent",
        model=make_model("quiz", model_name, choice=choice),
        description=(
            "Sei un docente universitario esperto. "
            "Generi domande di esame a risposta aperta basandoti "
            "ESCLUSIVAMENTE sul materiale fornito. "
            "Rispondi SOLO con JSON valido, senza markdown, senza testo extra."
        ),
        markdown=False,
    )


# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------

def _quiz_contexts(passages: list[dict]) -> list[str]:
    """Visit all sources in rotation and include every passage in bounded batches."""
    by_source: dict[str, deque] = defaultdict(deque)
    for passage in passages:
        if passage.get("text", "").strip():
            by_source[str(passage.get("source") or "materiale di studio")].append(passage)
    contexts, current = [], ""
    while any(by_source.values()):
        for source in sorted(by_source):
            if not by_source[source]:
                continue
            passage = by_source[source].popleft()
            page = passage.get("page")
            label = f"[Fonte: {source}" + (f", p. {page}" if page is not None else "") + "]\n"
            remaining = str(passage["text"]).strip()
            while remaining:
                capacity = MAX_QUIZ_CONTEXT_CHARS - len(label) - 2
                part = remaining[:capacity]
                if len(remaining) > capacity:
                    cut = part.rfind(" ")
                    if cut > capacity // 2:
                        part = part[:cut]
                remaining = remaining[len(part):].lstrip()
                block = label + part
                if current and len(current) + len(block) + 2 > MAX_QUIZ_CONTEXT_CHARS:
                    contexts.append(current)
                    current = ""
                current += ("\n\n" if current else "") + block
    if current:
        contexts.append(current)
    return contexts


def _validate_quiz_grounding(data: dict, context: str) -> None:
    passages = [
        re.sub(r"\s+", " ", part).casefold()
        for part in re.split(r"(?m)^\[Fonte: [^\n]*\]\n", context)[1:]
    ]
    for question in data["questions"]:
        quote = re.sub(r"\s+", " ", question["source_quote"]).casefold()
        if not any(quote in passage for passage in passages):
            raise ValueError(f"Citazione di prova assente dal materiale per domanda {question['id']}")


def _select_quiz_contexts(contexts: list[str], n_questions: int) -> list[str]:
    """Bound model calls while covering selected sources and distant sections."""
    source_names = {
        source for context in contexts
        for source in re.findall(r"(?m)^\[Fonte: (.*?)(?:, p\. [^\]]+)?\]$", context)
    }
    # Every selected source needs at least one batch, even when there are
    # more sources than the usual model-call budget.
    budget = min(len(contexts), max(len(source_names), min(MAX_QUIZ_BATCHES, max(2, n_questions))))
    if budget == len(contexts):
        return contexts
    selected: set[int] = set()
    seen_sources: set[str] = set()
    for index, context in enumerate(contexts):
        for source in re.findall(r"(?m)^\[Fonte: (.*?)(?:, p\. [^\]]+)?\]$", context):
            if source not in seen_sources and len(selected) < budget:
                seen_sources.add(source)
                selected.add(index)
    for index in (0, len(contexts) - 1):
        if len(selected) < budget:
            selected.add(index)
    for i in range(budget):
        if len(selected) >= budget:
            break
        selected.add(round(i * (len(contexts) - 1) / (budget - 1)))
    # Rounding can collide with mandatory source batches; fill any remaining slots.
    for index in range(len(contexts)):
        if len(selected) >= budget:
            break
        selected.add(index)
    return [contexts[index] for index in sorted(selected)]


def generate_quiz(
        student_id: int,
        exam_id: int,
        topic_query: str = "concetti principali del corso",
        n_questions: int = DEFAULT_N_QUESTIONS,
        n_chunks: int | None = DEFAULT_N_CHUNKS,
        source_name: str = "materiale di studio",
        model_name: str | None = None,
        material_ids: list[int] | None = None,
) -> QuizResult:
    """Generate questions from all indexed passages, validating each against its batch."""
    if n_questions < 1:
        raise ValueError("n_questions deve essere positivo")
    passages = retrieve_passages(
        student_id=student_id, exam_id=exam_id, material_ids=material_ids,
    )
    contexts = _select_quiz_contexts(_quiz_contexts(passages), n_questions)
    if not contexts:
        raise ValueError(
            f"Nessun chunk trovato per student_id={student_id}, exam_id={exam_id}.\n"
            "Indicizza il materiale prima di generare il quiz."
        )
    template = _load_prompt_template()
    collected: list[Question] = []
    seen: set[str] = set()
    per_batch = max(1, (n_questions + len(contexts) - 1) // len(contexts))
    invalid_batches = 0
    successful_contexts: list[str] = []
    backup = fallback_choice("quiz")
    using_backup = False
    primary_rate_limited = False
    for batch_number, context in enumerate(contexts, 1):
        filled_prompt = template.format(
            n_questions=per_batch, source_name=source_name,
            topic_query=topic_query, context_chunks=context,
        )
        last_error = None
        retry_prompt = filled_prompt
        last_invalid_response = False
        invalid_attempts = 0
        for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
            last_invalid_response = False
            try:
                agent = (_make_quiz_agent(None, choice=backup) if using_backup
                         else _make_quiz_agent(model_name))
                output = run_agent(agent, retry_prompt, "quiz_generate")
                raw = output.get_content_as_string()
                failed = getattr(output, "status", None) == "ERROR"
                active_provider = backup.provider.capitalize() if using_backup else provider_name("quiz")
                rate_limit = rate_limit_from_response(raw, active_provider, run_failed=failed)
                if rate_limit is not None:
                    raise rate_limit
                if failed:
                    raise RuntimeError(f"{active_provider} ha interrotto la generazione delle domande.")
                try:
                    data = _parse_llm_response(raw)
                    _validate_quiz_grounding(data, context)
                except ValueError as exc:
                    last_invalid_response = True
                    retry_prompt = (
                        filled_prompt + "\n\nLa risposta precedente è stata rifiutata: "
                        + str(exc) + ". Copia ogni source_quote come sequenza letterale "
                        "dal testo di un singolo passaggio [Fonte], senza parafrasi, "
                        "correzioni o parole dell'etichetta. Ometti le domande senza prova."
                    )
                    raise RetryableLLMError(f"Quiz non valido: {exc}") from exc
                for q in data["questions"][:per_batch]:
                    normalized = re.sub(r"\s+", " ", q["text"]).casefold()
                    if normalized not in seen:
                        seen.add(normalized)
                        collected.append(Question(
                            id=len(collected) + 1, text=q["text"],
                            difficulty=q["difficulty"], topic_hint=q["topic_hint"],
                        ))
                if data["questions"]:
                    successful_contexts.append(context)
                break
            except Exception as exc:
                if is_rate_limit_error(exc):
                    active_provider = backup.provider.capitalize() if using_backup else provider_name("quiz")
                    logger.warning("Quiz: quota %s raggiunta nel gruppo %d/%d", active_provider, batch_number, len(contexts))
                    if backup is not None and not using_backup and attempt < MAX_LLM_ATTEMPTS:
                        using_backup = True
                        primary_rate_limited = True
                        logger.warning("Quiz: continuo con %s per le fonti selezionate", backup.provider.capitalize())
                        continue
                    limit_error = _quiz_limit_error(
                        exc, active_provider,
                        previous_provider=provider_name("quiz") if primary_rate_limited else None,
                    )
                    if limit_error is exc:
                        raise
                    raise limit_error from exc
                if not is_retryable_error(exc):
                    raise
                last_error = exc
                logger.warning(
                    "Quiz: gruppo %d/%d, tentativo %d/%d fallito (%s)",
                    batch_number, len(contexts), attempt, MAX_LLM_ATTEMPTS,
                    safe_groq_error(exc),
                )
                if last_invalid_response:
                    invalid_attempts += 1
                    if invalid_attempts >= MAX_INVALID_QUIZ_ATTEMPTS:
                        invalid_batches += 1
                        break
                if attempt < MAX_LLM_ATTEMPTS:
                    time.sleep(retry_delay(attempt))
        else:
            if last_invalid_response:
                invalid_batches += 1
                continue
            raise RuntimeError(
                f"QuizGenerator: esauriti {MAX_LLM_ATTEMPTS} tentativi; ultimo errore: {last_error}"
            ) from last_error

    # A rejected quote in one batch does not prove the other selected passages
    # cannot support another distinct question. Try a bounded recovery using
    # only contexts that have already yielded grounded output.
    if invalid_batches and 0 < len(collected) < n_questions:
        for context in successful_contexts[:MAX_RECOVERY_QUIZ_CALLS]:
            if len(collected) >= n_questions:
                break
            existing = "\n".join(f"- {question.text}" for question in collected)
            recovery_prompt = template.format(
                n_questions=1, source_name=source_name,
                topic_query=topic_query, context_chunks=context,
            ) + (
                "\n\nGenera UNA domanda diversa dalle seguenti, se il passaggio "
                "contiene un altro fatto verificabile. Copia source_quote "
                "letteralmente da un singolo passaggio [Fonte]. "
                "Se non puoi, restituisci {\"questions\": []}.\n" + existing
            )
            try:
                active_provider = backup.provider.capitalize() if using_backup else provider_name("quiz")
                agent = (_make_quiz_agent(None, choice=backup) if using_backup
                         else _make_quiz_agent(model_name))
                output = run_agent(agent, recovery_prompt, "quiz_recovery")
                raw = output.get_content_as_string()
                failed = getattr(output, "status", None) == "ERROR"
                rate_limit = rate_limit_from_response(raw, active_provider, run_failed=failed)
                if rate_limit is not None:
                    raise rate_limit
                if failed:
                    raise RetryableLLMError("Generazione integrativa interrotta dal modello.")
                data = _parse_llm_response(raw)
                _validate_quiz_grounding(data, context)
                for q in data["questions"][:1]:
                    normalized = re.sub(r"\s+", " ", q["text"]).casefold()
                    if normalized not in seen:
                        seen.add(normalized)
                        collected.append(Question(
                            id=len(collected) + 1, text=q["text"],
                            difficulty=q["difficulty"], topic_hint=q["topic_hint"],
                        ))
            except Exception as exc:
                if is_rate_limit_error(exc):
                    limit_error = _quiz_limit_error(
                        exc, active_provider,
                        previous_provider=provider_name("quiz") if primary_rate_limited else None,
                    )
                    if limit_error is exc:
                        raise
                    raise limit_error from exc
                if not is_retryable_error(exc):
                    raise
                logger.warning("Quiz: recupero domanda non riuscito (%s)", safe_groq_error(exc))
    if not collected:
        if invalid_batches:
            raise UnverifiableQuizError(
                "Il modello non ha prodotto domande con citazioni verificabili. Riprova la generazione."
            )
        raise InsufficientMaterialError("Materiale insufficiente per formulare domande verificabili.")
    if len(collected) > n_questions > 1:
        positions = [round(i * (len(collected) - 1) / (n_questions - 1)) for i in range(n_questions)]
        questions = [collected[position] for position in positions]
    elif len(collected) > n_questions == 1:
        questions = [collected[len(collected) // 2]]
    else:
        questions = collected
    for index, question in enumerate(questions, 1):
        question.id = index
    return QuizResult(
        questions=questions, source_name=source_name, n_questions=len(questions),
        prompt_version=PROMPT_VERSION, exam_id=exam_id, student_id=student_id,
    )


# ---------------------------------------------------------------------------
# Utilità
# ---------------------------------------------------------------------------

def quiz_to_dict(result: QuizResult) -> dict:
    return asdict(result)


def quiz_to_json(result: QuizResult, indent: int = 2) -> str:
    return json.dumps(quiz_to_dict(result), ensure_ascii=False, indent=indent)
