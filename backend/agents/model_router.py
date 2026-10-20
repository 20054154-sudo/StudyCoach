"""Choose an LLM provider independently for tutor, quiz, and notes."""

import os
from dataclasses import dataclass

from agno.models.groq import Groq
from dotenv import load_dotenv

from backend.agents.groq_config import GROQ_MODEL
from backend.agents.llm_errors import LLM_TIMEOUT_SECONDS

load_dotenv()

FEATURES = {"tutor", "quiz", "notes"}
DEFAULT_MODELS = {
    "groq": GROQ_MODEL,
    "mistral": "mistral-small-latest",
    "gemini": "gemini-3.8-flash",
}


class ModelConfigurationError(RuntimeError):
    """A selected provider lacks a usable local configuration."""


@dataclass(frozen=True)
class ModelChoice:
    provider: str
    model_id: str


def model_choice(feature: str, model_name: str | None = None) -> ModelChoice:
    if feature not in FEATURES:
        raise ValueError(f"Funzione LLM sconosciuta: {feature}")
    provider = os.getenv(f"{feature.upper()}_PROVIDER", "groq").strip().lower() or "groq"
    if provider not in DEFAULT_MODELS:
        raise ModelConfigurationError(f"Provider non valido per {feature}: {provider}")
    model_id = (model_name or os.getenv(f"{feature.upper()}_MODEL") or DEFAULT_MODELS[provider]).strip()
    if not model_id:
        raise ModelConfigurationError(f"Modello non configurato per {feature}.")
    return ModelChoice(provider, model_id)


def provider_name(feature: str) -> str:
    return model_choice(feature).provider.capitalize()


def fallback_choice(feature: str) -> ModelChoice | None:
    if feature not in FEATURES:
        raise ValueError(f"Funzione LLM sconosciuta: {feature}")
    provider = os.getenv(f"{feature.upper()}_FALLBACK_PROVIDER", "").strip().lower()
    if not provider:
        return None
    if provider not in DEFAULT_MODELS:
        raise ModelConfigurationError(f"Provider di riserva non valido per {feature}: {provider}")
    if provider == model_choice(feature).provider:
        return None
    model_id = (os.getenv(f"{feature.upper()}_FALLBACK_MODEL") or DEFAULT_MODELS[provider]).strip()
    if not model_id:
        raise ModelConfigurationError(f"Modello di riserva non configurato per {feature}.")
    return ModelChoice(provider, model_id)


def make_model(feature: str, model_name: str | None = None, *, max_tokens: int | None = None,
               json_mode: bool = False, choice: ModelChoice | None = None):
    choice = choice or model_choice(feature, model_name)
    if choice.provider == "groq":
        params = {"id": choice.model_id, "timeout": LLM_TIMEOUT_SECONDS}
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if json_mode:
            request_params = {"response_format": {"type": "json_object"}}
            if choice.model_id.startswith("openai/gpt-oss-"):
                request_params["reasoning_effort"] = "low"
            params["request_params"] = request_params
        return Groq(**params)

    if choice.provider == "mistral":
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ModelConfigurationError("Configura MISTRAL_API_KEY nel file .env per usare Mistral.")
        from agno.models.mistral import MistralChat
        from mistralai import Mistral

        client = Mistral(api_key=api_key, timeout_ms=int(LLM_TIMEOUT_SECONDS * 1000))
        params = {"id": choice.model_id, "mistral_client": client}
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if json_mode:
            params["request_params"] = {"response_format": {"type": "json_object"}}
        return MistralChat(**params)

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ModelConfigurationError("Configura GOOGLE_API_KEY o GEMINI_API_KEY nel file .env per usare Gemini.")
    from agno.models.google import Gemini

    params = {"id": choice.model_id, "api_key": api_key, "timeout": LLM_TIMEOUT_SECONDS}
    if max_tokens is not None:
        params["max_output_tokens"] = max_tokens
    if json_mode:
        config = {"response_mime_type": "application/json"}
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens
        params["request_params"] = {"config": config}
    return Gemini(**params)
