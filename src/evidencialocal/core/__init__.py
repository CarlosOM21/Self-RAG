"""Modelos de dominio y utilidades core."""

from .exceptions import (
    ConfigurationError,
    EvaluationError,
    EvidenciaLocalError,
    GenerationError,
    IndexingError,
    ModelNotFoundError,
    RetrievalError,
)
from .models import (
    AskRequest,
    AskResponse,
    Chunk,
    ChunkMetadata,
    CritiqueResult,
    Document,
    EvaluationResult,
    EvaluationSample,
    GenerationResult,
    IndexManifest,
    IndexStatus,
    Mode,
    RetrievalResult,
)

__all__ = [
    "Document",
    "Chunk",
    "ChunkMetadata",
    "RetrievalResult",
    "GenerationResult",
    "CritiqueResult",
    "AskRequest",
    "AskResponse",
    "IndexManifest",
    "IndexStatus",
    "Mode",
    "EvaluationSample",
    "EvaluationResult",
    "EvidenciaLocalError",
    "ConfigurationError",
    "ModelNotFoundError",
    "IndexingError",
    "RetrievalError",
    "GenerationError",
    "EvaluationError",
]
