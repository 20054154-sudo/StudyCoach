"""Shared classification and bounded retry policy for LLM calls."""

import os
import json
import random
import re
import socket
import time

MAX_LLM_ATTEMPTS = int(os.getenv("LLM_MAX_ATTEMPTS", "4"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
LLM_RETRY_BASE_SECONDS = float(os.getenv("LLM_RETRY_BASE_SECONDS", "1"))


class RetryableLLMError(RuntimeError):
    """Errore di risposta del modello che può essere corretto ritentando."""


class NotesReviewUnavailable(RuntimeError):
    """La verifica degli appunti non ha restituito dati utilizzabili."""


class ModelServiceUnavailable(RetryableLLMError):
    """Safe error for an upstream outage or connection interruption."""


def provider_rate_limit_error(message: str, provider: str = "Groq", *, retry_after_seconds: int | None = None) -> RetryableLLMError:
    """Return a public, status-bearing error without exposing provider diagnostics."""
    daily_tokens = bool(re.search(r"tokens per day|tokensperday|\bTPD\b", message, re.IGNORECASE))
    daily_requests = bool(re.search(r"requests per day|requestsperday|requests_per_day", message, re.IGNORECASE))
    text = (
        f"Quota giornaliera di token {provider} insufficiente per questa richiesta. "
        "Riprova quando la quota si rinnova."
        if daily_tokens else
        f"Quota giornaliera di richieste {provider} esaurita. Riprova quando la quota si rinnova."
        if daily_requests else f"Limite di richieste {provider} raggiunto. Riprova più tardi."
    )
    if retry_after_seconds is not None and retry_after_seconds > 0:
        hours, minutes = divmod((retry_after_seconds + 59) // 60, 60)
        wait = f"{hours} ore e {minutes} minuti" if hours else f"{minutes} minuti"
        text += f" Il servizio indica un'attesa di circa {wait}."
    error = RetryableLLMError(text)
    error.status_code = 429
    error.retry_after_seconds = retry_after_seconds
    if daily_tokens or daily_requests:
        error.quota_period = "daily"
    return error


def rate_limit_from_response(text: str, provider: str = "Groq", *, run_failed: bool = False) -> RetryableLLMError | None:
    """Recognize only the structured provider error envelope, not model prose."""
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        if not run_failed or not isinstance(text, str) or "{" not in text:
            return None
        try:
            payload = json.loads(text[text.find("{"):])
        except ValueError:
            return None
    if not isinstance(payload, dict):
        return None
    details = payload.get("error")
    if not isinstance(details, dict):
        if not run_failed or payload.get("raw_status_code") != 429:
            return None
        details = payload
    if (details.get("code") not in ("rate_limit_exceeded", 429)
            and details.get("status") != "RESOURCE_EXHAUSTED"
            and details.get("status_code") != 429
            and details.get("raw_status_code") != 429):
        return None
    retry_after_seconds = None
    for item in details.get("details") or []:
        if not isinstance(item, dict) or not str(item.get("@type", "")).endswith("RetryInfo"):
            continue
        match = re.fullmatch(r"(\d+)s", str(item.get("retryDelay", "")))
        if match:
            retry_after_seconds = int(match.group(1))
            break
    return provider_rate_limit_error(json.dumps(details, ensure_ascii=False), provider,
                                     retry_after_seconds=retry_after_seconds)


def is_rate_limit_error(error: BaseException | None) -> bool:
    if error is None:
        return False
    response = getattr(error, "response", None)
    return (
        getattr(error, "status_code", None) == 429
        or getattr(response, "status_code", None) == 429
    )


def http_status(error: BaseException | None) -> int | None:
    if error is None:
        return None
    response = getattr(error, "response", None)
    return getattr(error, "status_code", None) or getattr(response, "status_code", None)


def is_retryable_error(error: BaseException | None) -> bool:
    """Retry only transport failures, 429 and HTTP 5xx responses."""
    if error is None:
        return False
    status = http_status(error)
    if status is not None:
        return status == 429 or 500 <= status <= 599
    # ValueError copre il JSON malformato prodotto dal modello; KeyError e'
    # intenzionalmente escluso per non ritentare template.format/dati mancanti.
    return isinstance(error, (RetryableLLMError, ValueError, TimeoutError, socket.timeout, ConnectionError))


def retry_delay(attempt: int) -> float:
    base = LLM_RETRY_BASE_SECONDS * (2 ** max(0, attempt - 1))
    return base + random.uniform(0, min(1.0, base * 0.1))


def sleep_before_retry(attempt: int) -> None:
    time.sleep(retry_delay(attempt))
