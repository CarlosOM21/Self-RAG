"""Tests unitarios para configuración."""


import pytest

from src.evidencialocal.config.settings import (
    ChromaSettings,
    CorrectiveSettings,
    EmbeddingSettings,
    EvaluationSettings,
    IndexingSettings,
    LlamaCppSettings,
    RetrievalSettings,
    Settings,
    SQLiteSettings,
    get_settings,
    reset_settings,
)


class TestSettings:
    """Tests para configuración."""

    def setup_method(self):
        reset_settings()

    def teardown_method(self):
        reset_settings()

    def test_default_settings(self):
        settings = Settings()
        assert settings.embedding.model_name == "intfloat/multilingual-e5-small"
        assert settings.embedding.dimension == 384
        assert settings.chroma.collection_name == "evidencialocal"
        assert settings.chroma.distance_metric == "cosine"
        assert settings.llama_cpp.model_path == "qwen2.5-3b-instruct-q4_k_m.gguf"
        assert settings.llama_cpp.n_ctx == 4096
        assert settings.llama_cpp.n_threads == 4
        assert settings.llama_cpp.n_batch == 512
        assert settings.indexing.chunk_size == 512
        assert settings.indexing.chunk_overlap == 50
        assert settings.retrieval.top_k == 5
        assert settings.retrieval.max_retrievals == 3
        assert settings.corrective.max_retries == 2
        assert settings.evaluation.random_seed == 42

    def test_settings_from_env(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING__MODEL_NAME", "test-model")
        monkeypatch.setenv("EMBEDDING__DIMENSION", "768")
        monkeypatch.setenv("LLAMA_CPP__MODEL_PATH", "test-model.gguf")
        monkeypatch.setenv("LLAMA_CPP__N_CTX", "2048")
        monkeypatch.setenv("LLAMA_CPP__N_THREADS", "8")
        monkeypatch.setenv("LLAMA_CPP__N_BATCH", "256")
        monkeypatch.setenv("CHROMA__PERSIST_DIRECTORY", "/tmp/test_chroma")
        monkeypatch.setenv("SQLITE__DATABASE_PATH", "/tmp/test.db")
        monkeypatch.setenv("INDEXING__CHUNK_SIZE", "256")
        monkeypatch.setenv("RETRIEVAL__TOP_K", "10")
        monkeypatch.setenv("CORRECTIVE__MAX_RETRIES", "3")
        monkeypatch.setenv("EVALUATION__RANDOM_SEED", "123")

        reset_settings()
        settings = get_settings()

        assert settings.embedding.model_name == "test-model"
        assert settings.embedding.dimension == 768
        assert settings.llama_cpp.model_path == "test-model.gguf"
        assert settings.llama_cpp.n_ctx == 2048
        assert settings.llama_cpp.n_threads == 8
        assert settings.llama_cpp.n_batch == 256
        assert settings.chroma.persist_directory == "/tmp/test_chroma"
        assert settings.sqlite.database_path == "/tmp/test.db"
        assert settings.indexing.chunk_size == 256
        assert settings.retrieval.top_k == 10
        assert settings.corrective.max_retries == 3
        assert settings.evaluation.random_seed == 123

    def test_nested_settings_objects(self):
        settings = Settings()
        assert isinstance(settings.embedding, EmbeddingSettings)
        assert isinstance(settings.chroma, ChromaSettings)
        assert isinstance(settings.sqlite, SQLiteSettings)
        assert isinstance(settings.llama_cpp, LlamaCppSettings)
        assert isinstance(settings.indexing, IndexingSettings)
        assert isinstance(settings.retrieval, RetrievalSettings)
        assert isinstance(settings.corrective, CorrectiveSettings)
        assert isinstance(settings.evaluation, EvaluationSettings)

    def test_singleton_pattern(self):
        reset_settings()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_reset_settings(self):
        reset_settings()
        s1 = get_settings()
        reset_settings()
        s2 = get_settings()
        assert s1 is not s2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
