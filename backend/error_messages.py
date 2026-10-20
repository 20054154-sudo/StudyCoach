"""Messages that may safely be shown to a student."""

from backend.agents.storage_errors import EmbeddingModelUnavailable, MaterialIndexUnavailable
from backend.agents.llm_errors import ModelServiceUnavailable, NotesReviewUnavailable, RetryableLLMError, is_rate_limit_error
from backend.agents.upload_errors import UploadError
from backend.agents.model_router import ModelConfigurationError


def public_error(error: Exception, fallback: str) -> str:
    if isinstance(error, RetryableLLMError) and is_rate_limit_error(error):
        return str(error)
    if isinstance(error, (UploadError, EmbeddingModelUnavailable, MaterialIndexUnavailable, NotesReviewUnavailable, ModelConfigurationError, ModelServiceUnavailable)):
        return str(error)
    return fallback
