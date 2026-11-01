"""Recuperación vectorial para los modos baseline y correctivo."""

import logging

from ..config.settings import get_settings
from ..core.models import RetrievalResult
from ..embeddings.embedder import get_embedder
from ..vectorstore.chroma_store import get_chroma_store

logger = logging.getLogger(__name__)


class Retriever:
    """Recupera chunks relevantes usando embeddings E5 y ChromaDB."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.retrieval_settings = self.settings.retrieval
        self.chroma = get_chroma_store(settings)
        self.embedder = get_embedder(settings)

    def retrieve(
        self, query: str, top_k: int | None = None, where: dict | None = None
    ) -> list[RetrievalResult]:
        """Recupera los chunks más similares a una consulta."""
        top_k = self.retrieval_settings.top_k if top_k is None else top_k
        query_embedding = self.embedder.encode_single_query(query)
        results = self.chroma.query(query_embedding, top_k=top_k, where=where)

        threshold = self.retrieval_settings.score_threshold
        if threshold > 0:
            results = [result for result in results if result.score >= threshold]

        logger.debug("Recuperados %d chunks para: %s", len(results), query[:80])
        return results

    def retrieve_for_baseline(
        self, question: str, top_k: int | None = None
    ) -> list[RetrievalResult]:
        """Ejecuta la única recuperación del modo baseline."""
        return self.retrieve(question, top_k)

    def retrieve_for_corrective(
        self,
        question: str,
        previous_queries: list[str],
        previous_results: list[list[RetrievalResult]],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Recupera evidencia nueva usando la última consulta propuesta por el crítico.

        Se solicitan más candidatos que `top_k` porque los primeros pueden ser los
        mismos chunks de rondas anteriores. Si la reformulación no aporta nada, se
        hace un último intento con la pregunta original y ese mismo ranking amplio.
        """
        top_k = self.retrieval_settings.top_k if top_k is None else top_k
        seen_chunk_ids = {
            result.chunk_id for round_results in previous_results for result in round_results
        }
        query = next(
            (candidate.strip() for candidate in reversed(previous_queries) if candidate.strip()),
            question,
        )
        candidate_k = max(top_k * 3, top_k + len(seen_chunk_ids))

        new_results = self._retrieve_new(query, candidate_k, top_k, seen_chunk_ids)
        if not new_results and query != question:
            logger.info("La reformulación no aportó chunks; se reintenta con la pregunta original")
            new_results = self._retrieve_new(question, candidate_k, top_k, seen_chunk_ids)

        logger.debug("Recuperación correctiva con %r: %d chunks nuevos", query, len(new_results))
        return new_results

    def _retrieve_new(
        self, query: str, candidate_k: int, top_k: int, seen_chunk_ids: set
    ) -> list[RetrievalResult]:
        """Filtra IDs conocidos y reconstruye el ranking de los nuevos resultados."""
        candidates = self.retrieve(query, candidate_k)
        filtered = [result for result in candidates if result.chunk_id not in seen_chunk_ids]
        filtered = filtered[:top_k]
        for rank, result in enumerate(filtered, start=1):
            result.rank = rank
        return filtered


_retriever_instance = None


def get_retriever(settings=None):
    """Obtiene la instancia singleton del recuperador."""
    global _retriever_instance
    if _retriever_instance is None or settings is not None:
        _retriever_instance = Retriever(settings)
    return _retriever_instance


def reset_retriever():
    """Resetea el recuperador (útil para tests)."""
    global _retriever_instance
    _retriever_instance = None
