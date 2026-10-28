"""Modelos de dominio (Pydantic) para EvidenciaLocal."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class IndexStatus(str, Enum):
    """Estado de indexación de un documento."""
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class Mode(str, Enum):
    """Modos de operación del RAG."""
    BASELINE = "baseline"
    CORRECTIVE = "corrective"


class Document(BaseModel):
    """Documento original antes de segmentar."""
    id: UUID = Field(default_factory=uuid4)
    source_path: Path
    filename: str
    content_hash: str  # SHA256 del contenido
    page_count: int
    created_at: datetime = Field(default_factory=datetime.now)
    status: IndexStatus = IndexStatus.PENDING
    error_message: str | None = None

    @field_validator("source_path", mode="before")
    @classmethod
    def _validate_path(cls, v: str | Path) -> Path:
        return Path(v) if isinstance(v, str) else v


class ChunkMetadata(BaseModel):
    """Metadatos de un chunk para SQLite."""
    chunk_id: UUID
    document_id: UUID
    document_filename: str
    page_number: int | None = None
    chunk_index: int  # índice dentro del documento
    token_count: int
    char_start: int
    char_end: int
    content_hash: str  # SHA256 del chunk


class Chunk(BaseModel):
    """Chunk de texto con embedding y metadatos."""
    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    text: str
    embedding: list[float]  # vector de 384 dimensiones
    metadata: ChunkMetadata

    def to_chroma_format(self) -> dict[str, Any]:
        """Convierte a formato compatible con ChromaDB."""
        return {
            "id": str(self.id),
            "document": self.text,
            "embedding": self.embedding,
            "metadata": self.metadata.model_dump(mode="json"),
        }


class RetrievalResult(BaseModel):
    """Resultado de una recuperación individual."""
    chunk_id: UUID
    document_id: UUID
    document_filename: str
    page_number: int | None = None
    text: str
    score: float  # similitud coseno (0-1, mayor = más similar)
    rank: int  # posición en ranking (1-based)


class GenerationResult(BaseModel):
    """Resultado de generación del LLM."""
    answer: str
    citations: list[UUID]  # IDs de chunks citados
    abstained: bool = False
    abstention_reason: str | None = None
    tokens_used: int | None = None
    latency_ms: float
    model: str
    prompt_hash: str  # hash del prompt usado


class CritiqueResult(BaseModel):
    """Resultado de la evaluación crítica (modo correctivo)."""
    sufficient: bool  # True = contexto suficiente, False = necesita más recuperación
    reason: str  # razón breve (no cadena de pensamiento)
    missing_aspects: list[str] = Field(default_factory=list)
    retry_query: str | None = None  # consulta sugerida para siguiente recuperación
    tokens_used: int | None = None
    latency_ms: float
    model: str
    prompt_hash: str


class AskRequest(BaseModel):
    """Petición de pregunta al sistema."""
    question: str
    mode: Mode
    top_k: int | None = None  # override opcional
    session_id: str | None = None


class AskResponse(BaseModel):
    """Respuesta completa del sistema."""
    question: str
    mode: Mode
    answer: str
    citations: list[UUID]
    abstained: bool
    abstention_reason: str | None = None
    retrievals: list[list[RetrievalResult]]  # lista por cada ronda de recuperación
    critiques: list[CritiqueResult] = Field(default_factory=list)  # solo modo correctivo
    total_latency_ms: float
    total_tokens: int | None = None
    model: str
    timestamp: datetime = Field(default_factory=datetime.now)
    seed: int | None = None


class IndexManifest(BaseModel):
    """Manifiesto de índice para versionado y idempotencia."""
    version: int = 1
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    embedding_model: str
    embedding_dimension: int
    chunk_size: int
    chunk_overlap: int
    tokenizer_model: str
    documents: dict[str, Document] = Field(default_factory=dict)  # key = content_hash
    total_chunks: int = 0
    total_documents: int = 0

    def get_document_by_hash(self, content_hash: str) -> Document | None:
        return self.documents.get(content_hash)

    def add_document(self, doc: Document) -> None:
        self.documents[doc.content_hash] = doc
        self.total_documents = len(self.documents)
        self.updated_at = datetime.now()


class EvaluationSample(BaseModel):
    """Muestra de evaluación (pregunta + gold opcional)."""
    id: str
    question: str
    gold_answer: str | None = None
    gold_citations: list[UUID] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Resultado de evaluación de una muestra."""
    sample_id: str
    mode: Mode
    question: str
    predicted_answer: str
    predicted_citations: list[UUID]
    abstained: bool
    latency_ms: float
    tokens_used: int | None = None
    retrieval_count: int
    critique_count: int = 0
    # Métricas (se llenan tras anotación humana)
    correctness: float | None = None  # 0-1
    faithfulness: float | None = None  # 0-1
    citation_precision: float | None = None
    citation_recall: float | None = None
    annotated: bool = False
    annotator_notes: str | None = None


class ExperimentManifest(BaseModel):
    """Manifiesto de experimento para reproducibilidad."""
    id: UUID = Field(default_factory=uuid4)
    name: str
    created_at: datetime = Field(default_factory=datetime.now)
    config_hash: str  # hash de la configuración congelada
    settings_snapshot: dict[str, Any]
    git_commit: str | None = None
    random_seed: int
    modes_evaluated: list[Mode]
    dataset_path: str
    total_samples: int
    completed_samples: int = 0
