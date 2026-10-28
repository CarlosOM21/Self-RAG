"""Tests unitarios para modelos Pydantic."""

from uuid import UUID, uuid4

import pytest

from src.evidencialocal.core.models import (
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


class TestDocument:
    """Tests para modelo Document."""

    def test_document_creation(self):
        doc = Document(
            source_path="/tmp/test.pdf",
            filename="test.pdf",
            content_hash="abc123",
            page_count=5,
        )
        assert doc.filename == "test.pdf"
        assert doc.page_count == 5
        assert doc.status == IndexStatus.PENDING
        assert isinstance(doc.id, UUID)

    def test_document_status_transitions(self):
        doc = Document(
            source_path="/tmp/test.pdf",
            filename="test.pdf",
            content_hash="abc123",
            page_count=5,
        )
        assert doc.status == IndexStatus.PENDING
        doc.status = IndexStatus.READY
        assert doc.status == IndexStatus.READY
        doc.status = IndexStatus.FAILED
        assert doc.status == IndexStatus.FAILED


class TestChunkMetadata:
    """Tests para ChunkMetadata."""

    def test_chunk_metadata_creation(self):
        doc_id = uuid4()
        chunk_id = uuid4()
        meta = ChunkMetadata(
            chunk_id=chunk_id,
            document_id=doc_id,
            document_filename="test.pdf",
            page_number=1,
            chunk_index=0,
            token_count=100,
            char_start=0,
            char_end=500,
            content_hash="hash123",
        )
        assert meta.chunk_id == chunk_id
        assert meta.document_id == doc_id
        assert meta.page_number == 1
        assert meta.token_count == 100


class TestChunk:
    """Tests para Chunk."""

    def test_chunk_to_chroma_format(self):
        doc_id = uuid4()
        chunk_id = uuid4()
        meta = ChunkMetadata(
            chunk_id=chunk_id,
            document_id=doc_id,
            document_filename="test.pdf",
            page_number=1,
            chunk_index=0,
            token_count=100,
            char_start=0,
            char_end=500,
            content_hash="hash123",
        )
        chunk = Chunk(
            id=chunk_id,
            document_id=doc_id,
            text="Texto de prueba",
            embedding=[0.1] * 384,
            metadata=meta,
        )
        chroma_format = chunk.to_chroma_format()
        assert chroma_format["id"] == str(chunk_id)
        assert chroma_format["document"] == "Texto de prueba"
        assert chroma_format["embedding"] == [0.1] * 384
        assert "metadata" in chroma_format


class TestRetrievalResult:
    """Tests para RetrievalResult."""

    def test_retrieval_result_creation(self):
        result = RetrievalResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            document_filename="test.pdf",
            page_number=1,
            text="Contenido del chunk",
            score=0.85,
            rank=1,
        )
        assert result.score == 0.85
        assert result.rank == 1
        assert result.page_number == 1


class TestGenerationResult:
    """Tests para GenerationResult."""

    def test_generation_result_with_citations(self):
        citations = [uuid4(), uuid4()]
        result = GenerationResult(
            answer="Respuesta de prueba",
            citations=citations,
            abstained=False,
            tokens_used=150,
            latency_ms=1200.5,
            model="qwen2.5:3b",
            prompt_hash="abc123",
        )
        assert len(result.citations) == 2
        assert result.abstained is False
        assert result.tokens_used == 150


class TestCritiqueResult:
    """Tests para CritiqueResult."""

    def test_critique_sufficient(self):
        critique = CritiqueResult(
            sufficient=True,
            reason="Contexto completo",
            missing_aspects=[],
            retry_query=None,
            tokens_used=50,
            latency_ms=300.0,
            model="qwen2.5:3b",
            prompt_hash="def456",
        )
        assert critique.sufficient is True
        assert critique.retry_query is None

    def test_critique_insufficient(self):
        critique = CritiqueResult(
            sufficient=False,
            reason="Falta información sobre X",
            missing_aspects=["aspecto1", "aspecto2"],
            retry_query="consulta más específica",
            tokens_used=50,
            latency_ms=300.0,
            model="qwen2.5:3b",
            prompt_hash="def456",
        )
        assert critique.sufficient is False
        assert len(critique.missing_aspects) == 2
        assert critique.retry_query == "consulta más específica"


class TestAskRequestResponse:
    """Tests para AskRequest y AskResponse."""

    def test_ask_request(self):
        request = AskRequest(
            question="¿Qué es RAG?",
            mode=Mode.BASELINE,
            top_k=5,
        )
        assert request.question == "¿Qué es RAG?"
        assert request.mode == Mode.BASELINE
        assert request.top_k == 5

    def test_ask_response_baseline(self):
        response = AskResponse(
            question="¿Qué es RAG?",
            mode=Mode.BASELINE,
            answer="RAG es...",
            citations=[uuid4()],
            abstained=False,
            retrievals=[[]],
            total_latency_ms=1500.0,
            model="qwen2.5:3b",
        )
        assert response.mode == Mode.BASELINE
        assert response.abstained is False

    def test_ask_response_corrective_with_critiques(self):
        critique = CritiqueResult(
            sufficient=False,
            reason="Falta info",
            missing_aspects=["detalle"],
            retry_query="más detalle",
            tokens_used=30,
            latency_ms=200.0,
            model="qwen2.5:3b",
            prompt_hash="hash",
        )
        response = AskResponse(
            question="¿Qué es RAG?",
            mode=Mode.CORRECTIVE,
            answer="RAG mejorado...",
            citations=[uuid4()],
            abstained=False,
            retrievals=[[], []],
            critiques=[critique],
            total_latency_ms=3000.0,
            model="qwen2.5:3b",
        )
        assert response.mode == Mode.CORRECTIVE
        assert len(response.critiques) == 1
        assert response.critiques[0].sufficient is False


class TestIndexManifest:
    """Tests para IndexManifest."""

    def test_manifest_creation(self):
        manifest = IndexManifest(
            embedding_model="intfloat/multilingual-e5-small",
            embedding_dimension=384,
            chunk_size=512,
            chunk_overlap=50,
            tokenizer_model="intfloat/multilingual-e5-small",
        )
        assert manifest.version == 1
        assert manifest.total_documents == 0
        assert manifest.total_chunks == 0

    def test_manifest_add_document(self):
        manifest = IndexManifest(
            embedding_model="test",
            embedding_dimension=384,
            chunk_size=512,
            chunk_overlap=50,
            tokenizer_model="test",
        )
        doc = Document(
            source_path="/tmp/test.pdf",
            filename="test.pdf",
            content_hash="hash123",
            page_count=5,
        )
        manifest.add_document(doc)
        assert manifest.total_documents == 1
        assert manifest.get_document_by_hash("hash123") == doc


class TestEvaluationModels:
    """Tests para modelos de evaluación."""

    def test_evaluation_sample(self):
        sample = EvaluationSample(
            id="q1",
            question="¿Qué es ML?",
            gold_answer="Machine Learning...",
            gold_citations=[uuid4()],
        )
        assert sample.id == "q1"
        assert sample.gold_answer is not None

    def test_evaluation_result(self):
        result = EvaluationResult(
            sample_id="q1",
            mode=Mode.BASELINE,
            question="¿Qué es ML?",
            predicted_answer="ML es...",
            predicted_citations=[uuid4()],
            abstained=False,
            latency_ms=1000.0,
            retrieval_count=1,
        )
        assert result.sample_id == "q1"
        assert result.annotated is False
        # Marcar como anotado
        result.correctness = 1.0
        result.faithfulness = 0.9
        result.annotated = True
        assert result.annotated is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
