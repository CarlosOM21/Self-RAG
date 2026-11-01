"""Pipeline RAG con modos baseline y correctivo."""

import logging
import re
import time
from uuid import UUID

from ..config.settings import get_settings
from ..core.exceptions import GenerationError
from ..core.models import AskRequest, AskResponse, CritiqueResult, Mode, RetrievalResult
from ..llm.llama_client import get_llama_client
from ..metadata.sqlite_store import get_sqlite_store
from ..retrieval.retriever import get_retriever

logger = logging.getLogger(__name__)

UUID_PATTERN = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", re.IGNORECASE
)
CITATION_ONLY_PATTERN = re.compile(
    r"^\s*(?:\[DOC_\d+\s+ID:[0-9a-f-]+\]|[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})\s*[.!?]?\s*$",
    re.IGNORECASE,
)
ABSTENTION_ANSWER = "No puedo responder con la información disponible en los documentos."


def extract_uuid_from_citation(citation: str) -> UUID | None:
    """Extrae un UUID de una cita textual o devuelve None si no contiene uno."""
    if not citation:
        return None
    match = UUID_PATTERN.search(citation)
    if not match:
        return None
    try:
        return UUID(match.group(1))
    except ValueError:
        return None


def is_citation_only(answer: str) -> bool:
    """Detecta salidas degeneradas que contienen únicamente una cita o un UUID."""
    return bool(CITATION_ONLY_PATTERN.fullmatch(answer or ""))


class RAGPipeline:
    """Orquesta recuperación, evaluación crítica, generación y trazabilidad."""

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
        if request.mode == Mode.CORRECTIVE:
            return self._ask_corrective(request)
        raise ValueError(f"Modo desconocido: {request.mode}")

    def _ask_baseline(self, request: AskRequest) -> AskResponse:
        """Ejecuta una recuperación y una generación, sin bucle correctivo."""
        start_time = time.perf_counter()
        retrievals = self.retriever.retrieve_for_baseline(request.question, request.top_k)

        if not retrievals:
            return self._abstain_without_context(
                request,
                Mode.BASELINE,
                [[]],
                start_time,
                "No se recuperó evidencia para la pregunta",
            )

        context_retrievals = retrievals[: max(1, self.corrective_settings.max_context_chunks)]
        parsed, tokens, _, _ = self.llm.generate_answer(
            request.question,
            [result.text for result in context_retrievals],
            [result.chunk_id for result in context_retrievals],
            mode="baseline",
        )
        answer, citations, abstained, reason = self._parse_generation(parsed, context_retrievals)
        response = AskResponse(
            question=request.question,
            mode=Mode.BASELINE,
            answer=answer,
            citations=citations,
            abstained=abstained,
            abstention_reason=reason,
            retrievals=[retrievals],
            critiques=[],
            total_latency_ms=(time.perf_counter() - start_time) * 1000,
            total_tokens=tokens,
            model=self.settings.llama_cpp.model_path,
            seed=self.settings.evaluation.random_seed,
        )
        self._save_trace(response)
        return response

    def _ask_corrective(self, request: AskRequest) -> AskResponse:
        """
        Evalúa el contexto antes de generar y reformula la recuperación si falta evidencia.

        La respuesta se genera una sola vez, cuando el crítico acepta el contexto,
        cuando se agotan los reintentos o cuando ya no aparecen chunks nuevos.
        """
        start_time = time.perf_counter()
        original_question = request.question
        all_retrievals: list[list[RetrievalResult]] = []
        all_critiques: list[CritiqueResult] = []
        previous_queries = [original_question]
        total_tokens = 0

        for attempt in range(self.corrective_settings.max_retries + 1):
            if attempt == 0:
                new_retrievals = self.retriever.retrieve_for_baseline(
                    original_question, request.top_k
                )
            else:
                new_retrievals = self.retriever.retrieve_for_corrective(
                    original_question, previous_queries, all_retrievals, request.top_k
                )

            if new_retrievals:
                all_retrievals.append(new_retrievals)
            elif not all_retrievals:
                return self._abstain_without_context(
                    request,
                    Mode.CORRECTIVE,
                    [[]],
                    start_time,
                    "No se recuperó evidencia para la pregunta",
                )
            else:
                logger.info("No hay chunks nuevos; se genera con la mejor evidencia acumulada")
                break

            context_retrievals = self._select_context(all_retrievals)
            try:
                critique_parsed, critique_tokens, critique_latency, critique_hash = (
                    self.llm.critique_context(
                        original_question,
                        [result.text for result in context_retrievals],
                        [result.chunk_id for result in context_retrievals],
                    )
                )
            except GenerationError as exc:
                logger.warning("La crítica falló; se continúa con el contexto actual: %s", exc)
                critique_parsed = {
                    "sufficient": True,
                    "reason": "No se pudo completar la evaluación crítica",
                    "missing_aspects": [],
                    "retry_query": None,
                }
                critique_tokens = 0
                critique_latency = 0.0
                critique_hash = ""

            total_tokens += critique_tokens
            critique = CritiqueResult(
                sufficient=bool(critique_parsed.get("sufficient", True)),
                reason=str(critique_parsed.get("reason") or ""),
                missing_aspects=critique_parsed.get("missing_aspects") or [],
                retry_query=critique_parsed.get("retry_query"),
                tokens_used=critique_tokens,
                latency_ms=critique_latency,
                model=self.settings.llama_cpp.model_path,
                prompt_hash=critique_hash,
            )
            all_critiques.append(critique)

            if critique.sufficient or attempt == self.corrective_settings.max_retries:
                break

            next_query = self._next_query(original_question, critique, previous_queries)
            previous_queries.append(next_query)
            logger.info("Contexto insuficiente; nueva búsqueda: %s", next_query)

        context_retrievals = self._select_context(all_retrievals)
        try:
            parsed, generation_tokens, _, _ = self.llm.generate_answer(
                original_question,
                [result.text for result in context_retrievals],
                [result.chunk_id for result in context_retrievals],
                mode="corrective",
            )
        except GenerationError:
            logger.exception("Error en la generación correctiva final")
            raise
        total_tokens += generation_tokens
        answer, citations, abstained, reason = self._parse_generation(parsed, context_retrievals)

        response = AskResponse(
            question=original_question,
            mode=Mode.CORRECTIVE,
            answer=answer,
            citations=citations,
            abstained=abstained,
            abstention_reason=reason,
            retrievals=all_retrievals,
            critiques=all_critiques,
            total_latency_ms=(time.perf_counter() - start_time) * 1000,
            total_tokens=total_tokens,
            model=self.settings.llama_cpp.model_path,
            seed=self.settings.evaluation.random_seed,
        )
        self._save_trace(response)
        return response

    def _select_context(self, rounds: list[list[RetrievalResult]]) -> list[RetrievalResult]:
        """Limita el contexto conservando evidencia de cada ronda, no solo de la última."""
        limit = max(1, self.corrective_settings.max_context_chunks)
        selected: list[RetrievalResult] = []
        selected_ids: set[UUID] = set()
        rank = 0
        while len(selected) < limit:
            added = False
            for round_results in rounds:
                if rank >= len(round_results):
                    continue
                added = True
                result = round_results[rank]
                if result.chunk_id not in selected_ids:
                    selected.append(result)
                    selected_ids.add(result.chunk_id)
                    if len(selected) == limit:
                        break
            if not added:
                break
            rank += 1
        return selected

    @staticmethod
    def _next_query(question: str, critique: CritiqueResult, previous_queries: list[str]) -> str:
        """Obtiene una reformulación no vacía y evita repetir consultas exactas."""
        candidate = (critique.retry_query or "").strip()
        if candidate and candidate not in previous_queries:
            return candidate
        missing = ", ".join(critique.missing_aspects).strip()
        fallback = f"{question} {missing}".strip()
        return fallback if fallback not in previous_queries else question

    @staticmethod
    def _parse_generation(
        parsed: dict, context_retrievals: list[RetrievalResult]
    ) -> tuple[str, list[UUID], bool, str | None]:
        """Normaliza la salida y descarta citas que no pertenecen al contexto enviado."""
        valid_ids = {result.chunk_id for result in context_retrievals}
        citations = []
        for citation in parsed.get("citations") or []:
            if not isinstance(citation, str):
                continue
            citation_id = extract_uuid_from_citation(citation)
            if citation_id in valid_ids and citation_id not in citations:
                citations.append(citation_id)

        answer = str(parsed.get("answer") or "").strip()
        abstained = bool(parsed.get("abstained", False))
        reason = parsed.get("abstention_reason")
        if not abstained and (not answer or is_citation_only(answer)):
            abstained = True
            reason = "El modelo no produjo una respuesta textual válida"
        if abstained:
            return ABSTENTION_ANSWER, citations, True, reason or "Evidencia insuficiente"
        return answer, citations, False, None

    def _abstain_without_context(
        self,
        request: AskRequest,
        mode: Mode,
        retrievals: list[list[RetrievalResult]],
        start_time: float,
        reason: str,
    ) -> AskResponse:
        """Evita invocar al LLM cuando no existe ningún chunk que fundamentar."""
        response = AskResponse(
            question=request.question,
            mode=mode,
            answer=ABSTENTION_ANSWER,
            citations=[],
            abstained=True,
            abstention_reason=reason,
            retrievals=retrievals,
            critiques=[],
            total_latency_ms=(time.perf_counter() - start_time) * 1000,
            total_tokens=0,
            model=self.settings.llama_cpp.model_path,
            seed=self.settings.evaluation.random_seed,
        )
        self._save_trace(response)
        return response

    def _save_trace(self, response: AskResponse) -> None:
        """Guarda la traza sin impedir que el usuario reciba una respuesta ya generada."""
        trace_data = {
            "timestamp": response.timestamp.isoformat(),
            "question": response.question,
            "mode": response.mode.value,
            "answer": response.answer,
            "citations": [str(citation) for citation in response.citations],
            "abstained": response.abstained,
            "abstention_reason": response.abstention_reason,
            "retrievals": [
                [
                    {
                        "chunk_id": str(result.chunk_id),
                        "document_id": str(result.document_id),
                        "document_filename": result.document_filename,
                        "page_number": result.page_number,
                        "text": result.text,
                        "score": result.score,
                        "rank": result.rank,
                    }
                    for result in round_results
                ]
                for round_results in response.retrievals
            ],
            "critiques": [
                {
                    "sufficient": critique.sufficient,
                    "reason": critique.reason,
                    "missing_aspects": critique.missing_aspects,
                    "retry_query": critique.retry_query,
                    "tokens_used": critique.tokens_used,
                    "latency_ms": critique.latency_ms,
                    "model": critique.model,
                    "prompt_hash": critique.prompt_hash,
                }
                for critique in response.critiques
            ],
            "total_latency_ms": response.total_latency_ms,
            "total_tokens": response.total_tokens,
            "model": response.model,
            "seed": response.seed,
        }
        try:
            self.sqlite.save_trace(trace_data)
        except Exception as exc:
            logger.warning("No se pudo guardar la traza: %s", exc)


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
