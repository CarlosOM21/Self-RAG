"""Regresiones para el bucle correctivo y el contrato textual del LLM."""

import json
from uuid import uuid4

import pytest

from src.evidencialocal.config.settings import Settings, load_settings
from src.evidencialocal.core.models import AskRequest, Mode, RetrievalResult
from src.evidencialocal.llm.llama_client import LlamaClient
from src.evidencialocal.pipeline.rag_pipeline import RAGPipeline
from src.evidencialocal.retrieval.retriever import Retriever


def make_result(text: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=uuid4(),
        document_id=uuid4(),
        document_filename="turing.pdf",
        page_number=1,
        text=text,
        score=0.9,
        rank=1,
    )


class TraceStore:
    def __init__(self):
        self.traces = []

    def save_trace(self, trace):
        self.traces.append(trace)


class CorrectiveLLM:
    def __init__(self, answer="Turing propuso la máquina de Turing y sentó bases de la IA."):
        self.answer = answer
        self.critique_calls = 0
        self.generation_calls = 0

    def critique_context(self, question, context_chunks, citation_ids):
        self.critique_calls += 1
        if self.critique_calls == 1:
            parsed = {
                "sufficient": False,
                "reason": "Faltan contribuciones concretas",
                "missing_aspects": ["contribuciones de Alan Turing"],
                "retry_query": "contribuciones de Alan Turing a la inteligencia artificial",
            }
        else:
            parsed = {
                "sufficient": True,
                "reason": "La evidencia ya permite responder",
                "missing_aspects": [],
                "retry_query": None,
            }
        return parsed, 10, 1.0, "hash"

    def generate_answer(self, question, context_chunks, citation_ids, mode="corrective"):
        self.generation_calls += 1
        return (
            {
                "answer": self.answer,
                "citations": [str(citation_ids[0])],
                "abstained": False,
                "abstention_reason": None,
            },
            20,
            1.0,
            "hash",
        )


class CorrectiveRetriever:
    def __init__(self, return_new=True):
        self.first = make_result("Alan Turing y las primeras computadoras")
        self.second = make_result("El test de Turing estudia si una máquina puede pensar")
        self.return_new = return_new
        self.corrective_queries = []

    def retrieve_for_baseline(self, question, top_k=None):
        return [self.first]

    def retrieve_for_corrective(self, question, previous_queries, previous_results, top_k=None):
        self.corrective_queries.append(previous_queries[-1])
        return [self.second] if self.return_new else []


def build_pipeline(monkeypatch, retriever, llm):
    settings = Settings(_env_file=None)
    monkeypatch.setattr(
        "src.evidencialocal.pipeline.rag_pipeline.get_retriever", lambda _: retriever
    )
    monkeypatch.setattr("src.evidencialocal.pipeline.rag_pipeline.get_llama_client", lambda _: llm)
    store = TraceStore()
    monkeypatch.setattr(
        "src.evidencialocal.pipeline.rag_pipeline.get_sqlite_store", lambda _: store
    )
    return RAGPipeline(settings)


def test_corrective_uses_retry_query_and_generates_final_answer(monkeypatch):
    retriever = CorrectiveRetriever()
    llm = CorrectiveLLM()
    pipeline = build_pipeline(monkeypatch, retriever, llm)

    response = pipeline.ask(AskRequest(question="¿Qué hizo Turing?", mode=Mode.CORRECTIVE))

    assert retriever.corrective_queries == [
        "contribuciones de Alan Turing a la inteligencia artificial"
    ]
    assert len(response.retrievals) == 2
    assert len(response.critiques) == 2
    assert llm.generation_calls == 1
    assert response.answer.startswith("Turing propuso")
    assert not response.abstained
    assert response.abstention_reason is None


def test_corrective_generates_with_existing_context_when_no_new_chunks(monkeypatch):
    llm = CorrectiveLLM()
    pipeline = build_pipeline(monkeypatch, CorrectiveRetriever(return_new=False), llm)

    response = pipeline.ask(AskRequest(question="¿Qué hizo Turing?", mode=Mode.CORRECTIVE))

    assert llm.generation_calls == 1
    assert len(response.retrievals) == 1
    assert response.answer.startswith("Turing propuso")
    assert not response.abstained


def test_citation_only_never_becomes_visible_answer(monkeypatch):
    llm = CorrectiveLLM(answer="[DOC_1 ID:123e4567-e89b-42d3-a456-426614174000]")
    pipeline = build_pipeline(monkeypatch, CorrectiveRetriever(), llm)

    response = pipeline.ask(AskRequest(question="¿Qué hizo Turing?", mode=Mode.CORRECTIVE))

    assert response.abstained
    assert not response.answer.startswith("[DOC_")
    assert response.abstention_reason == "El modelo no produjo una respuesta textual válida"


def test_llama_retries_citation_only_payload():
    class FakeLlama:
        def __init__(self):
            self.calls = 0

        def create_chat_completion(self, **kwargs):
            self.calls += 1
            payload = {
                "answer": "[DOC_1 ID:123e4567-e89b-42d3-a456-426614174000]",
                "citations": [],
                "abstained": False,
                "abstention_reason": None,
            }
            if self.calls == 2:
                payload["answer"] = "Turing formalizó una máquina de computación abstracta."
            return {
                "choices": [{"message": {"content": json.dumps(payload)}}],
                "usage": {"completion_tokens": 10},
            }

    client = LlamaClient(Settings(_env_file=None))
    fake_llama = FakeLlama()
    client._llm = fake_llama

    parsed, tokens, _ = client.generate(
        "Responde en JSON", LlamaClient.GENERATION_SCHEMA, max_retries=1
    )

    assert fake_llama.calls == 2
    assert parsed["answer"].startswith("Turing formalizó")
    assert tokens == 20


def test_corrective_retriever_overfetches_and_deduplicates():
    old = make_result("viejo")
    new = make_result("nuevo")
    settings = Settings(_env_file=None)
    retriever = Retriever.__new__(Retriever)
    retriever.settings = settings
    retriever.retrieval_settings = settings.retrieval

    class Embedder:
        def encode_single_query(self, query):
            return [0.1]

    class Chroma:
        def __init__(self):
            self.top_k = None

        def query(self, embedding, top_k, where=None):
            self.top_k = top_k
            return [old, new]

    chroma = Chroma()
    retriever.embedder = Embedder()
    retriever.chroma = chroma

    results = retriever.retrieve_for_corrective(
        "pregunta original", ["pregunta original", "consulta corregida"], [[old]], top_k=1
    )

    assert chroma.top_k == 3
    assert results == [new]


def test_yaml_and_environment_precedence(tmp_path, monkeypatch):
    config = tmp_path / "settings.yaml"
    config.write_text("retrieval:\n  top_k: 7\nllama_cpp:\n  n_ctx: 2048\n", encoding="utf-8")
    monkeypatch.setenv("RETRIEVAL__TOP_K", "9")

    settings = load_settings(config)

    assert settings.retrieval.top_k == 9
    assert settings.llama_cpp.n_ctx == 2048

    # También debe prevalecer un valor explícito aunque coincida con el default.
    monkeypatch.setenv("RETRIEVAL__TOP_K", "5")
    assert load_settings(config).retrieval.top_k == 5


def test_empty_question_is_rejected():
    with pytest.raises(ValueError):
        AskRequest(question="   ", mode=Mode.BASELINE)
