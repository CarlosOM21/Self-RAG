"""Recuperador para modos baseline y correctivo."""

import logging

from ..config.settings import get_settings
from ..core.models import RetrievalResult
from ..embeddings.embedder import get_embedder
from ..vectorstore.chroma_store import get_chroma_store

logger = logging.getLogger(__name__)


class Retriever:
    """Recupera chunks relevantes usando embeddings y ChromaDB."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.retrieval_settings = self.settings.retrieval
        self.chroma = get_chroma_store(settings)
        self.embedder = get_embedder(settings)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        where: dict | None = None,
    ) -> list[RetrievalResult]:
        """Recupera top-k chunks más similares a la consulta."""
        top_k = top_k or self.retrieval_settings.top_k
        query_embedding = self.embedder.encode_single_query(query)
        results = self.chroma.query(query_embedding, top_k=top_k, where=where)

        # Filtrar por umbral de score si está configurado
        threshold = self.retrieval_settings.score_threshold
        if threshold > 0:
            results = [r for r in results if r.score >= threshold]

        logger.debug(f"Recuperados {len(results)} chunks para: {query[:50]}...")
        return results

    def retrieve_for_baseline(
        self,
        question: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Recuperación única para modo baseline."""
        return self.retrieve(question, top_k)

    def retrieve_for_corrective(
        self,
        question: str,
        previous_queries: list[str],
        previous_results: list[list[RetrievalResult]],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Recuperación para modo correctivo.
        Evita duplicados con recuperaciones previas.
        """
        # Obtener IDs ya recuperados
        seen_chunk_ids = set()
        for round_results in previous_results:
            for r in round_results:
                seen_chunk_ids.add(r.chunk_id)

        # Recuperar nuevos
        new_results = self.retrieve(question, top_k)

        # Filtrar duplicados
        filtered = [r for r in new_results if r.chunk_id not in seen_chunk_ids]

        logger.debug(
            f"Recuperación correctiva: {len(new_results)} totales, "
            f"{len(filtered)} nuevos (después de dedup)"
        )
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
