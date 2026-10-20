"""Privacy-conscious request tracing: IDs and timings, never prompts or answers."""

import contextvars
import logging
import time
import uuid


trace_id_var = contextvars.ContextVar("studycoach_trace_id", default="-")

logger = logging.getLogger("uvicorn.error")


def start_trace() -> tuple[str, object, float]:
    trace_id = uuid.uuid4().hex
    token = trace_id_var.set(trace_id)
    return trace_id, token, time.perf_counter()


def end_trace(token: object) -> None:
    trace_id_var.reset(token)


def trace_event(stage: str, **fields: object) -> None:
    safe = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("trace_id=%s stage=%s %s", trace_id_var.get(), stage, safe)


def run_agent(agent, prompt: str, stage: str):
    started = time.perf_counter()
    model = getattr(getattr(agent, "model", None), "id", "unknown")
    try:
        result = agent.run(prompt, stream=False)
    except Exception:
        trace_event("llm_error", operation=stage, model=model,
                    elapsed_ms=round((time.perf_counter() - started) * 1000))
        raise
    trace_event("llm", operation=stage, model=model,
                elapsed_ms=round((time.perf_counter() - started) * 1000))
    return result
