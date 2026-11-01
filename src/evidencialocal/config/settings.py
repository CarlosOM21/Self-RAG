"""Configuración centralizada con valores por defecto, YAML y entorno."""

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import dotenv_values
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..core.exceptions import ConfigurationError


class EmbeddingSettings(BaseSettings):
    """Configuración de embeddings."""

    model_name: str = "intfloat/multilingual-e5-small"
    dimension: int = Field(default=384, gt=0)
    batch_size: int = Field(default=32, gt=0)
    normalize: bool = True
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    device: str = "auto"  # auto, cpu, cuda, mps


class ChromaSettings(BaseSettings):
    """Configuración de ChromaDB."""

    persist_directory: str = "data/chroma"
    collection_name: str = "evidencialocal"
    distance_metric: Literal["cosine"] = "cosine"


class SQLiteSettings(BaseSettings):
    """Configuración de SQLite para metadatos y trazas."""

    database_path: str = "data/sqlite/metadata.db"
    enable_wal: bool = True


class LlamaCppSettings(BaseSettings):
    """Configuración de inferencia local con llama.cpp."""

    model_path: str = "qwen2.5-3b-instruct-q4_k_m.gguf"
    n_ctx: int = Field(default=8192, gt=0)
    n_threads: int = Field(default=4, gt=0)
    n_batch: int = Field(default=512, gt=0)
    temperature: float = Field(default=0.1, ge=0)
    top_p: float = Field(default=0.9, gt=0, le=1)
    max_tokens: int = Field(default=512, gt=0)


class IndexingSettings(BaseSettings):
    """Configuración de segmentación de documentos."""

    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    tokenizer_model: str = "intfloat/multilingual-e5-small"


class RetrievalSettings(BaseSettings):
    """Configuración de recuperación."""

    top_k: int = Field(default=5, ge=1, le=20)
    score_threshold: float = Field(default=0.0, ge=0, le=1)


class CorrectiveSettings(BaseSettings):
    """Configuración específica del modo correctivo."""

    max_retries: int = Field(default=2, ge=0)
    critique_temperature: float = Field(default=0.0, ge=0)
    max_context_chunks: int = Field(default=8, ge=1)


class EvaluationSettings(BaseSettings):
    """Configuración de evaluación."""

    random_seed: int = 42
    bootstrap_samples: int = Field(default=1000, ge=1)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)


class Settings(BaseSettings):
    """Configuración principal de la aplicación."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_nested_delimiter="__", extra="ignore"
    )

    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chroma: ChromaSettings = Field(default_factory=ChromaSettings)
    sqlite: SQLiteSettings = Field(default_factory=SQLiteSettings)
    llama_cpp: LlamaCppSettings = Field(default_factory=LlamaCppSettings)
    indexing: IndexingSettings = Field(default_factory=IndexingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    corrective: CorrectiveSettings = Field(default_factory=CorrectiveSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)

    data_dir: Path = Path("data")
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    config_dir: Path = Path("configs")
    experiments_dir: Path = Path("experiments/results")

    log_level: str = "INFO"


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    """Combina diccionarios anidados sin descartar claves no especificadas."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _apply_explicit_environment(merged: dict[str, Any]) -> None:
    """Superpone variables realmente presentes en .env y después en el proceso."""
    environment: dict[str, str] = {}
    env_path = Path(".env")
    if env_path.is_file():
        environment.update(
            {key: value for key, value in dotenv_values(env_path).items() if value is not None}
        )
    environment.update(os.environ)

    for name, value in environment.items():
        parts = name.lower().split("__")
        if len(parts) == 2:
            section, field = parts
            if section in merged and isinstance(merged[section], dict) and field in merged[section]:
                merged[section][field] = value
        elif len(parts) == 1 and parts[0] in Settings.model_fields:
            merged[parts[0]] = value


def load_settings(config_path: Path | None = None) -> Settings:
    """Carga configuración con precedencia: defaults < YAML < .env < entorno."""
    if config_path is None:
        return Settings()
    if not config_path.exists():
        raise ConfigurationError(f"Archivo de configuración no existe: {config_path}")

    try:
        yaml_data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"No se pudo leer {config_path}: {exc}") from exc
    if not isinstance(yaml_data, dict):
        raise ConfigurationError("El archivo YAML debe contener un objeto en la raíz")

    # model_construct rellena defaults sin consultar fuentes externas.
    merged = Settings.model_construct().model_dump(mode="python")
    _deep_update(merged, yaml_data)
    _apply_explicit_environment(merged)
    return Settings(**merged)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Obtiene la instancia singleton de configuración sin YAML explícito."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Resetea la configuración (útil para tests)."""
    global _settings
    _settings = None
