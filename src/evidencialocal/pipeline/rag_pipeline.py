"""Pipeline RAG con modos baseline y correctivo."""

import logging
import re
import time
from uuid import UUID

from ..config.settings import get_settings
from ..core.exceptions import GenerationError
from ..core.models import (
    AskRequest,
    AskResponse,
    CritiqueResult,
    Mode,
    RetrievalResult,
)
from ..llm.llama_client import get_llama_client
from ..metadata.sqlite_store import get_sqlite_store
from ..retrieval.retriever import get_retriever

logger = logging.getLogger(__name__)

# Patrón para extraer UUID de cadenas como "DOC_1 ID:uuid" o solo "uuid"
UUID_PATTERN = re.compile(
    r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})',
    re.IGNORECASE
)


def extract_uuid_from_citation(citation: str) -> UUID | None:
    """
    Extrae un UUID válido de una cadena de cita.

    Acepta formatos:
    - UUID puro: "ae591a91-11d0-4756-9d73-d50835681a82"
    - Con prefijo DOC: "DOC_1 ID:ae591a91-11d0-4756-9d73-d50835681a82"
    - Cualquier cadena que contenga un UUID válido

    Returns:
        UUID si se encuentra uno válido, None en caso contrario.
    """
    if not citation:
        return None

    match = UUID_PATTERN.search(citation)
    if match:
        try:
            return UUID(match.group(1))
        except ValueError:
            return None
    return None


class RAGPipeline:
    """Pipeline principal para consultas RAG en ambos modos."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.retriever = get_retriever(settings)
        self.llm = get_llama_client(settings)
        self.sqlite = get_sqlite_store(settings)
        self.retrieval_settings = self.settings.retrieval
        self.corrective_settings = self.settings.corrective

    def ask(self, request: AskRequest) -> AskResponse:
        """Procesa una pregunta según el modo solicitado."""
        if request.mode == Mode.BASELINE:
            return self._ask_baseline(request)
        elif request.mode == Mode.CORRECTIVE:
            return self._ask_corrective(request)
        else:
            raise ValueError(f"Modo desconocido: {request.mode}")

    def _ask_baseline(self, request: AskRequest) -> AskResponse:
        """Modo baseline: una recuperación + generación con abstención."""
        start_time = time.perf_counter()

        # 1. Recuperación única
        retrievals = self.retriever.retrieve_for_baseline(request.question, request.top_k)

        # 2. Preparar contexto para LLM
        context_chunks = [r.text for r in retrievals]
        citation_ids = [r.chunk_id for r in retrievals]

        # 3. Generación
        try:
            parsed, tokens, gen_latency, prompt_hash = self.llm.generate_answer(
                request.question,
                context_chunks,
                citation_ids,
                mode="baseline",
            )
        except GenerationError as e:
            logger.error(f"Error en generación baseline: {e}")
            raise

        # 4. Construir respuesta
        answer = parsed.get("answer", "")
        citations = [extract_uuid_from_citation(c) for c in parsed.get("citations", []) if c]
        citations = [c for c in citations if c is not None]
        abstained = parsed.get("abstained", False)
        abstention_reason = parsed.get("abstention_reason")

        # Validar que las citas existan en lo recuperado
        valid_citation_ids = set(citation_ids)
        citations = [c for c in citations if c in valid_citation_ids]

        total_latency = (time.perf_counter() - start_time) * 1000

        response = AskResponse(
            question=request.question,
            mode=Mode.BASELINE,
            answer=answer,
            citations=citations,
            abstained=abstained,
            abstention_reason=abstention_reason,
            retrievals=[retrievals],
            critiques=[],
            total_latency_ms=total_latency,
            total_tokens=tokens,
            model=self.settings.llama_cpp.model_path,
            seed=self.settings.evaluation.random_seed,
        )

        # Guardar traza
        self._save_trace(response)

        return response

    def _ask_corrective(self, request: AskRequest) -> AskResponse:
        """Modo correctivo: recuperación iterativa con evaluación crítica."""
        start_time = time.perf_counter()

        all_retrievals: list[list[RetrievalResult]] = []
        all_critiques: list[CritiqueResult] = []
        accumulated_chunks: list[RetrievalResult] = []
        previous_queries: list[str] = []

        # Pregunta original inmutable
        original_question = request.question

        for attempt in range(self.corrective_settings.max_retries + 1):
            # 1. Recuperación (con deduplicación en intentos > 0)
            if attempt == 0:
                new_retrievals = self.retriever.retrieve_for_baseline(original_question, request.top_k)
            else:
                new_retrievals = self.retriever.retrieve_for_corrective(
                    original_question,
                    previous_queries,
                    all_retrievals,
                    request.top_k,
                )

            if not new_retrievals and attempt > 0:
                logger.info("No hay nuevos chunks para recuperar, terminando")
                final_answer = answer if not abstained else "No puedo responder con la información disponible."
                final_abstained = abstained
                final_reason = abstention_reason or "No hay más información disponible"
                break

            all_retrievals.append(new_retrievals)
            accumulated_chunks.extend(new_retrievals)

            # 2. Preparar contexto acumulado
            context_chunks = [r.text for r in accumulated_chunks]
            citation_ids = [r.chunk_id for r in accumulated_chunks]

            # 3. Generación de respuesta tentativa
            try:
                parsed, tokens, gen_latency, prompt_hash = self.llm.generate_answer(
                    original_question,
                    context_chunks,
                    citation_ids,
                    mode="corrective",
                )
            except GenerationError as e:
                logger.error(f"Error en generación correctivo (intento {attempt}): {e}")
                raise

            answer = parsed.get("answer", "")
            citations = [extract_uuid_from_citation(c) for c in parsed.get("citations", []) if c]
            citations = [c for c in citations if c is not None]
            abstained = parsed.get("abstained", False)
            abstention_reason = parsed.get("abstention_reason")

            # Validar citas
            valid_citation_ids = set(citation_ids)
            citations = [c for c in citations if c in valid_citation_ids]

            # 4. Evaluación crítica (excepto si ya respondió o es último intento)
            if abstained or attempt == self.corrective_settings.max_retries:
                # Si se abstiene o es último intento, terminar
                final_answer = answer if not abstained else "No puedo responder con la información disponible."
                final_abstained = abstained
                final_reason = abstention_reason
                break

            # Criticar contexto
            try:
                critique_parsed, critique_tokens, critique_latency, critique_prompt_hash = self.llm.critique_context(
                    original_question,
                    context_chunks,
                    citation_ids,
                )
            except GenerationError as e:
                logger.error(f"Error en crítica (intento {attempt}): {e}")
                # Si falla la crítica, continuar sin ella
                critique_parsed = {
                    "sufficient": True,
                    "reason": "Error en evaluación crítica",
                    "missing_aspects": [],
                    "retry_query": None,
                }
                critique_tokens = 0
                critique_latency = 0
                critique_prompt_hash = ""

            critique = CritiqueResult(
                sufficient=critique_parsed.get("sufficient", True),
                reason=critique_parsed.get("reason", ""),
                missing_aspects=critique_parsed.get("missing_aspects") or [],
                retry_query=critique_parsed.get("retry_query"),
                tokens_used=critique_tokens,
                latency_ms=critique_latency,
                model=self.settings.llama_cpp.model_path,
                prompt_hash=critique_prompt_hash,
            )
            all_critiques.append(critique)

            # 5. Decidir si continuar
            if critique.sufficient:
                logger.info(f"Contexto suficiente en intento {attempt + 1}")
                final_answer = answer
                final_abstained = False
                final_reason = None
                break

            # Preparar siguiente consulta
            next_query = critique.retry_query or original_question
            previous_queries.append(next_query)
            logger.info(f"Contexto insuficiente, reintentando con: {next_query[:80]}...")

        else:
            # Se agotaron intentos
            final_answer = answer if not abstained else "No puedo responder con la información disponible tras reintentos."
            final_abstained = abstained
            final_reason = abstention_reason or "Contexto insuficiente tras máximos reintentos"

        total_latency = (time.perf_counter() - start_time) * 1000
        total_tokens = sum(
            c.tokens_used or 0 for c in all_critiques
        ) + tokens  # tokens de la última generación

        response = AskResponse(
            question=original_question,
            mode=Mode.CORRECTIVE,
            answer=final_answer,
            citations=citations,
            abstained=final_abstained,
            abstention_reason=final_reason,
            retrievals=all_retrievals,
            critiques=all_critiques,
            total_latency_ms=total_latency,
            total_tokens=total_tokens,
            model=self.settings.llama_cpp.model_path,
            seed=self.settings.evaluation.random_seed,
        )

        # Guardar traza
        self._save_trace(response)

        return response

    def _save_trace(self, response: AskResponse) -> None:
        """Guarda traza de la consulta en SQLite."""
        trace_data = {
            "timestamp": response.timestamp.isoformat(),
            "question": response.question,
            "mode": response.mode.value,
            "answer": response.answer,
            "citations": [str(c) for c in response.citations],
            "abstained": response.abstained,
            "abstention_reason": response.abstention_reason,
            "retrievals": [
                [
                    {
                        "chunk_id": str(r.chunk_id),
                        "document_id": str(r.document_id),
                        "document_filename": r.document_filename,
                        "page_number": r.page_number,
                        "text": r.text,
                        "score": r.score,
                        "rank": r.rank,
                    }
                    for r in round_results
                ]
                for round_results in response.retrievals
            ],
            "critiques": [
                {
                    "sufficient": c.sufficient,
                    "reason": c.reason,
                    "missing_aspects": c.missing_aspects,
                    "retry_query": c.retry_query,
                    "tokens_used": c.tokens_used,
                    "latency_ms": c.latency_ms,
                    "model": c.model,
                    "prompt_hash": c.prompt_hash,
                }
                for c in response.critiques
            ],
            "total_latency_ms": response.total_latency_ms,
            "total_tokens": response.total_tokens,
            "model": response.model,
            "seed": response.seed,
        }
        self.sqlite.save_trace(trace_data)


_rag_pipeline_instance = None


def get_rag_pipeline(settings=None):
    """Obtiene la instancia singleton del pipeline RAG."""
    global _rag_pipeline_instance
    if _rag_pipeline_instance is None or settings is not None:
        _rag_pipeline_instance = RAGPipeline(settings)
    return _rag_pipeline_instance


def reset_rag_pipeline():
    """Resetea el pipeline (útil para tests)."""
    global _rag_pipeline_instance
    _rag_pipeline_instance = None
