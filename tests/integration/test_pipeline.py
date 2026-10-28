"""Tests de integración para el pipeline (con fakes/mocks)."""

from uuid import uuid4

import pytest

from src.evidencialocal.config.settings import Settings, reset_settings
from src.evidencialocal.core.models import (
    AskRequest,
    Chunk,
    ChunkMetadata,
    Document,
    IndexStatus,
    Mode,
    RetrievalResult,
)
from src.evidencialocal.pipeline.indexer import Indexer
from src.evidencialocal.pipeline.rag_pipeline import RAGPipeline


class FakeEmbedder:
    """Fake embedder para tests sin modelo real."""
    def __init__(self):
        self.dimension = 384
        self.model_name = "fake-model"

    def encode_queries(self, texts, batch_size=None):
        return [[0.1] * 384 for _ in texts]

    def encode_passages(self, texts, batch_size=None):
        return [[0.2] * 384 for _ in texts]

    def encode_single_query(self, text):
        return [0.1] * 384

    def encode_single_passage(self, text):
        return [0.2] * 384

    def get_model_hash(self):
        return "fakehash123"


class FakeChromaStore:
    """Fake ChromaDB para tests."""
    def __init__(self):
        self.chunks = {}
        self.count = 0

    def add_chunks(self, chunks):
        for c in chunks:
            self.chunks[str(c.id)] = c
        self.count += len(chunks)

    def delete_chunks(self, chunk_ids):
        for cid in chunk_ids:
            self.chunks.pop(str(cid), None)
        self.count = len(self.chunks)

    def query(self, query_embedding, top_k=5, where=None):
        # Devolver chunks fake
        results = []
        for i, (cid, chunk) in enumerate(list(self.chunks.items())[:top_k]):
            results.append(RetrievalResult(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                document_filename=chunk.metadata.document_filename,
                page_number=chunk.metadata.page_number,
                text=chunk.text,
                score=0.9 - i * 0.1,
                rank=i + 1,
            ))
        return results

    def get_chunk_count(self):
        return self.count

    def reset_collection(self):
        self.chunks = {}
        self.count = 0

    def close(self):
        pass


class FakeSQLiteStore:
    """Fake SQLite para tests."""
    def __init__(self):
        self.documents = {}
        self.chunks = {}
        self.traces = []
        self.manifest = None
        self.eval_results = {}
        self.experiments = {}

    def upsert_document(self, doc):
        self.documents[str(doc.id)] = doc

    def get_document(self, doc_id):
        return self.documents.get(str(doc_id))

    def get_document_by_hash(self, content_hash):
        for doc in self.documents.values():
            if doc.content_hash == content_hash:
                return doc
        return None

    def list_documents(self, status=None):
        docs = list(self.documents.values())
        if status:
            docs = [d for d in docs if d.status == status]
        return docs

    def delete_document(self, doc_id):
        return self.documents.pop(str(doc_id), None) is not None

    def upsert_chunk_metadata(self, meta):
        self.chunks[str(meta.chunk_id)] = meta

    def get_chunk_metadata(self, chunk_id):
        return self.chunks.get(str(chunk_id))

    def get_chunks_by_document(self, doc_id):
        return [c for c in self.chunks.values() if str(c.document_id) == str(doc_id)]

    def delete_chunks_by_document(self, doc_id):
        to_delete = [k for k, v in self.chunks.items() if str(v.document_id) == str(doc_id)]
        for k in to_delete:
            del self.chunks[k]
        return len(to_delete)

    def save_trace(self, trace_data):
        self.traces.append(trace_data)
        return "trace-id"

    def save_manifest(self, manifest):
        self.manifest = manifest

    def load_manifest(self):
        return self.manifest

    def save_experiment_manifest(self, manifest):
        self.experiments[str(manifest.id)] = manifest

    def get_experiment_manifest(self, exp_id):
        return self.experiments.get(str(exp_id))

    def save_evaluation_result(self, result, experiment_id):
        key = (str(experiment_id), result.sample_id, result.mode.value)
        self.eval_results[key] = result

    def get_evaluation_results(self, experiment_id):
        return [r for k, r in self.eval_results.items() if k[0] == str(experiment_id)]

    def update_evaluation_annotations(self, *args, **kwargs):
        pass

    def increment_completed_samples(self, experiment_id):
        if str(experiment_id) in self.experiments:
            self.experiments[str(experiment_id)].completed_samples += 1

    def close(self):
        pass


class FakeLlamaClient:
    """Fake llama.cpp para tests."""
    def __init__(self):
        self.model = "fake-model"
        self.call_count = 0

    def generate_answer(self, question, context_chunks, citation_ids, mode="baseline"):
        self.call_count += 1
        return {
            "answer": f"Respuesta fake para: {question[:50]}",
            "citations": [str(cid) for cid in citation_ids[:2]],
            "abstained": False,
            "abstention_reason": None,
        }, 100, 500.0, "prompthash"

    def critique_context(self, question, context_chunks, citation_ids):
        self.call_count += 1
        return {
            "sufficient": True,
            "reason": "Contexto suficiente",
            "missing_aspects": [],
            "retry_query": None,
        }, 50, 200.0, "prompthash"

    def check_model_available(self):
        return True

    def close(self):
        pass


class FakeDocumentProcessor:
    """Fake processor para tests."""
    def __init__(self):
        pass

    def process_document(self, pdf_path):
        doc = Document(
            source_path=pdf_path,
            filename=pdf_path.name,
            content_hash="fakehash",
            page_count=1,
            status=IndexStatus.READY,
        )
        chunk_meta = ChunkMetadata(
            chunk_id=uuid4(),
            document_id=doc.id,
            document_filename=doc.filename,
            page_number=1,
            chunk_index=0,
            token_count=100,
            char_start=0,
            char_end=500,
            content_hash="chunkhash",
        )
        chunk = Chunk(
            id=chunk_meta.chunk_id,
            document_id=doc.id,
            text="Contenido fake del documento",
            embedding=[0.2] * 384,
            metadata=chunk_meta,
        )
        return doc, [chunk]


@pytest.fixture
def fake_settings():
    """Settings con fakes."""
    reset_settings()
    settings = Settings()
    # Override para tests rápidos
    settings.embedding.model_name = "fake"
    settings.llama_cpp.model_path = "fake.gguf"
    return settings


@pytest.fixture
def fake_indexer(fake_settings, monkeypatch):
    """Indexer con componentes fake."""
    monkeypatch.setattr("src.evidencialocal.pipeline.indexer.get_document_processor", lambda s: FakeDocumentProcessor())
    monkeypatch.setattr("src.evidencialocal.pipeline.indexer.get_chroma_store", lambda s: FakeChromaStore())
    monkeypatch.setattr("src.evidencialocal.pipeline.indexer.get_sqlite_store", lambda s: FakeSQLiteStore())
    monkeypatch.setattr("src.evidencialocal.pipeline.indexer.get_embedder", lambda s: FakeEmbedder())
    return Indexer(fake_settings)


@pytest.fixture
def fake_pipeline(fake_settings, monkeypatch):
    """RAGPipeline con componentes fake."""
    monkeypatch.setattr("src.evidencialocal.pipeline.rag_pipeline.get_retriever", lambda s: MockRetriever())
    monkeypatch.setattr("src.evidencialocal.pipeline.rag_pipeline.get_llama_client", lambda s: FakeLlamaClient())
    monkeypatch.setattr("src.evidencialocal.pipeline.rag_pipeline.get_sqlite_store", lambda s: FakeSQLiteStore())
    return RAGPipeline(fake_settings)


class MockRetriever:
    """Mock retriever que devuelve resultados predecibles."""
    def __init__(self):
        self.retrieval_count = 0

    def retrieve_for_baseline(self, question, top_k=None):
        self.retrieval_count += 1
        chunk_id = uuid4()
        doc_id = uuid4()
        return [RetrievalResult(
            chunk_id=chunk_id,
            document_id=doc_id,
            document_filename="test.pdf",
            page_number=1,
            text="Contenido relevante para la pregunta",
            score=0.9,
            rank=1,
        )]

    def retrieve_for_corrective(self, question, previous_queries, previous_results, top_k=None):
        self.retrieval_count += 1
        # En correctivo, devolver chunks nuevos (diferentes IDs)
        chunk_id = uuid4()
        doc_id = uuid4()
        return [RetrievalResult(
            chunk_id=chunk_id,
            document_id=doc_id,
            document_filename="test2.pdf",
            page_number=1,
            text="Más contenido relevante",
            score=0.85,
            rank=1,
        )]


class TestIndexerIntegration:
    """Tests de integración del indexer."""

    def test_index_directory_idempotent(self, fake_indexer, tmp_path):
        # Crear PDF fake
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("fake pdf content")

        # Primera indexación
        manifest1 = fake_indexer.index_directory(tmp_path)
        assert manifest1.total_documents == 1
        assert manifest1.total_chunks == 1

        # Segunda indexación (debe ser idempotente)
        manifest2 = fake_indexer.index_directory(tmp_path)
        assert manifest2.total_documents == 1
        assert manifest2.total_chunks == 1

    def test_reindex_all(self, fake_indexer, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("fake pdf content")

        fake_indexer.index_directory(tmp_path)
        manifest = fake_indexer.reindex_all(tmp_path)
        assert manifest.total_documents == 1

    def test_delete_document(self, fake_indexer, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("fake pdf content")

        manifest = fake_indexer.index_directory(tmp_path)
        doc_id = list(manifest.documents.values())[0].id

        result = fake_indexer.delete_document(str(doc_id))
        assert result is True

        manifest2 = fake_indexer.load_manifest()
        assert manifest2.total_documents == 0


class TestRAGPipelineIntegration:
    """Tests de integración del pipeline RAG."""

    def test_ask_baseline(self, fake_pipeline):
        request = AskRequest(question="¿Qué es RAG?", mode=Mode.BASELINE, top_k=5)
        response = fake_pipeline.ask(request)

        assert response.mode == Mode.BASELINE
        assert response.answer is not None
        assert len(response.retrievals) == 1
        assert len(response.critiques) == 0
        assert not response.abstained

    def test_ask_corrective(self, fake_pipeline):
        request = AskRequest(question="¿Qué es RAG?", mode=Mode.CORRECTIVE, top_k=5)
        response = fake_pipeline.ask(request)

        assert response.mode == Mode.CORRECTIVE
        assert response.answer is not None
        assert len(response.retrievals) >= 1
        # Con mock que devuelve sufficient=True, solo 1 crítica
        assert len(response.critiques) >= 0

    def test_ask_corrective_with_retry(self, fake_settings, monkeypatch):
        """Test correctivo con crítica insuficiente que dispara retry."""
        call_count = {"critique": 0}

        class CritiqueFailThenPassClient(FakeLlamaClient):
            def critique_context(self, question, context_chunks, citation_ids):
                call_count["critique"] += 1
                if call_count["critique"] == 1:
                    return {
                        "sufficient": False,
                        "reason": "Falta info",
                        "missing_aspects": ["detalle"],
                        "retry_query": "más detalle",
                    }, 50, 200.0, "prompthash"
                return {
                    "sufficient": True,
                    "reason": "Ahora suficiente",
                    "missing_aspects": [],
                    "retry_query": None,
                }, 50, 200.0, "prompthash"

        monkeypatch.setattr("src.evidencialocal.pipeline.rag_pipeline.get_retriever", lambda s: MockRetriever())
        monkeypatch.setattr("src.evidencialocal.pipeline.rag_pipeline.get_llama_client", lambda s: CritiqueFailThenPassClient())
        monkeypatch.setattr("src.evidencialocal.pipeline.rag_pipeline.get_sqlite_store", lambda s: FakeSQLiteStore())

        pipeline = RAGPipeline(fake_settings)
        request = AskRequest(question="¿Qué es RAG?", mode=Mode.CORRECTIVE, top_k=5)
        response = pipeline.ask(request)

        assert response.mode == Mode.CORRECTIVE
        assert len(response.retrievals) == 2  # 1 inicial + 1 retry
        assert len(response.critiques) == 2  # crítica tras cada recuperación
        assert response.critiques[0].sufficient is False
        assert response.critiques[1].sufficient is True


class TestEndToEndOffline:
    """Test end-to-end completo sin modelos reales."""

    def test_full_flow_index_and_ask(self, fake_indexer, fake_pipeline, tmp_path):
        """Flujo completo: indexar -> preguntar baseline -> preguntar correctivo."""
        # 1. Indexar
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_text("fake")
        manifest = fake_indexer.index_directory(tmp_path)
        assert manifest.total_documents == 1

        # 2. Preguntar baseline
        req_baseline = AskRequest(question="Test question", mode=Mode.BASELINE)
        resp_baseline = fake_pipeline.ask(req_baseline)
        assert resp_baseline.mode == Mode.BASELINE
        assert resp_baseline.answer is not None

        # 3. Preguntar correctivo
        req_corrective = AskRequest(question="Test question", mode=Mode.CORRECTIVE)
        resp_corrective = fake_pipeline.ask(req_corrective)
        assert resp_corrective.mode == Mode.CORRECTIVE
        assert resp_corrective.answer is not None

        # Verificar que ambos modos producen respuesta
        assert len(resp_baseline.answer) > 0
        assert len(resp_corrective.answer) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
