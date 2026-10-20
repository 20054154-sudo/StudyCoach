"""Public, stable error types for the material index and embedding model."""


class MaterialIndexUnavailable(RuntimeError):
    """The local vector store cannot be opened or used."""


class EmbeddingModelUnavailable(RuntimeError):
    """The local embedding model could not be loaded."""
