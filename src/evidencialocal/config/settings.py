"""Configuración centralizada con Pydantic Settings."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingSettings(BaseSettings):
    """Configuración de embeddings."""
    model_name: str = "intfloat/multilingual-e5-small"
    dimension: int = 384
    batch_size: int = 32
    normalize: bool = True
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    device: str = "auto"  # auto, cpu, cuda, mps


class ChromaSettings(BaseSettings):
    """Configuración de ChromaDB."""
    persist_directory: str = "data/chroma"
    collection_name: str = "evidencialocal"
    distance_metric: Literal["cosine", "l2", "ip"] = "cosine"


class SQLiteSettings(BaseSettings):
    """Configuración de SQLite para metadatos y trazas."""
    database_path: str = "data/sqlite/metadata.db"
    enable_wal: bool = True


class LlamaCppSettings(BaseSettings):
    """Configuración de llama.cpp."""
    model_path: str = "qwen2.5-3b-instruct-q4_k_m.gguf"
    n_ctx: int = 4096
    n_threads: int = 4
    n_batch: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 2048


class IndexingSettings(BaseSettings):
    """Configuración de indexación."""
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    tokenizer_model: str = "intfloat/multilingual-e5-small"


class RetrievalSettings(BaseSettings):
    """Configuración de recuperación."""
    top_k: int = 5
    max_retrievals: int = 3  # máx 3 recuperaciones (1 baseline, hasta 3 correctivo)
    score_threshold: float = 0.0  # umbral coseno, 0 = sin filtro


class CorrectiveSettings(BaseSettings):
    """Configuración específica del modo correctivo."""
    max_retries: int = 2
    critique_temperature: float = 0.0  # determinista para evaluación


class EvaluationSettings(BaseSettings):
    """Configuración de evaluación."""
    random_seed: int = 42
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95


class Settings(BaseSettings):
    """Configuración principal de la aplicación."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chroma: ChromaSettings = Field(default_factory=ChromaSettings)
    sqlite: SQLiteSettings = Field(default_factory=SQLiteSettings)
    llama_cpp: LlamaCppSettings = Field(default_factory=LlamaCppSettings)
    indexing: IndexingSettings = Field(default_factory=IndexingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    corrective: CorrectiveSettings = Field(default_factory=CorrectiveSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)

    # Directorios base
    data_dir: Path = Path("data")
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    config_dir: Path = Path("configs")
    experiments_dir: Path = Path("experiments/results")

    # Logging
    log_level: str = "INFO"
    log_file: Path = Path("logs/evidencialocal.log")


_settings: Settings | None = None


def get_settings() -> Settings:
    """Obtiene la instancia singleton de configuración."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Resetea la configuración (útil para tests)."""
    global _settings
    _settings = None
