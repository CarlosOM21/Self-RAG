"""Interfaz Streamlit centrada en indexar documentos y consultar los dos modos RAG."""

from pathlib import Path

import streamlit as st

from ..config.settings import Settings, get_settings
from ..core.exceptions import EvidenciaLocalError
from ..core.models import AskRequest, AskResponse, Mode
from ..llm.llama_client import get_llama_client
from ..pipeline.indexer import get_indexer
from ..pipeline.rag_pipeline import get_rag_pipeline

st.set_page_config(
    page_title="EvidenciaLocal - RAG Educativo Local",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_cached_settings() -> Settings:
    return get_settings()


@st.cache_resource
def get_cached_indexer(_settings: Settings):
    return get_indexer(_settings)


@st.cache_resource
def get_cached_pipeline(_settings: Settings):
    return get_rag_pipeline(_settings)


def main() -> None:
    """Punto de entrada de la aplicación."""
    settings = get_cached_settings()
    indexer = get_cached_indexer(settings)
    pipeline = get_cached_pipeline(settings)

    with st.sidebar:
        st.title("📚 EvidenciaLocal")
        st.caption("RAG local · Baseline y Correctivo")
        page = st.radio(
            "Navegación",
            ["Inicio", "Documentos", "Consultar", "Configuración"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(f"LLM: {settings.llama_cpp.model_path}")
        st.caption(f"Embeddings: {settings.embedding.model_name}")

    if page == "Inicio":
        render_home(indexer, settings)
    elif page == "Documentos":
        render_documents(indexer)
    elif page == "Consultar":
        render_ask(pipeline, settings)
    else:
        render_settings(settings)


def render_home(indexer, settings: Settings) -> None:
    """Muestra el estado y el flujo esencial de uso."""
    st.header("Estado del sistema")
    col1, col2, col3 = st.columns(3)
    try:
        manifest = indexer.load_manifest()
        col1.metric("Documentos", manifest.total_documents)
        col1.metric("Chunks", manifest.total_chunks)
    except Exception:
        col1.metric("Índice", "No disponible")

    llama_ok = get_llama_client(settings).check_model_available()
    col2.metric("Modelo GGUF", "Disponible" if llama_ok else "No encontrado")
    col3.metric("Dimensión embedding", settings.embedding.dimension)
    if not llama_ok:
        st.warning(f"No se encontró el modelo: `{settings.llama_cpp.model_path}`")

    st.divider()
    st.subheader("Flujo de trabajo")
    st.markdown(
        "1. Añade tus PDF en **Documentos** e indexa el corpus.\n"
        "2. Escribe una pregunta en **Consultar**.\n"
        "3. Usa **Baseline** para una sola recuperación o **Correctivo** para "
        "evaluar la evidencia y reformular la búsqueda cuando sea necesario."
    )
    with st.expander("Diferencia entre los modos"):
        st.markdown(
            "**Baseline:** pregunta → una recuperación → respuesta.\n\n"
            "**Correctivo:** pregunta → recuperación → crítica de suficiencia → "
            "hasta dos recuperaciones reformuladas con deduplicación → respuesta final. "
            "La pregunta original nunca se sustituye al generar."
        )


def render_documents(indexer) -> None:
    """Gestiona el corpus real del usuario; no genera documentos artificiales."""
    st.header("Gestión de documentos")
    upload_tab, list_tab, delete_tab = st.tabs(["Subir e indexar", "Listado", "Eliminar"])

    with upload_tab:
        uploaded_files = st.file_uploader(
            "Archivos PDF con texto extraíble",
            type=["pdf"],
            accept_multiple_files=True,
            help="Máximo 50 MB por archivo. Los PDF escaneados requieren OCR previo.",
        )
        if uploaded_files and st.button("Indexar archivos", type="primary"):
            raw_dir = Path("data/raw")
            raw_dir.mkdir(parents=True, exist_ok=True)
            saved_paths = []
            for uploaded in uploaded_files:
                if uploaded.size > 50 * 1024 * 1024:
                    st.error(f"{uploaded.name}: excede 50 MB")
                    continue
                save_path = raw_dir / Path(uploaded.name).name
                save_path.write_bytes(uploaded.read())
                saved_paths.append(save_path)
            if saved_paths:
                with st.spinner("Extrayendo texto y generando embeddings..."):
                    try:
                        manifest = indexer.index_directory(raw_dir)
                        st.success(
                            f"Índice actualizado: {manifest.total_documents} documentos, "
                            f"{manifest.total_chunks} chunks"
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Error indexando: {exc}")

        st.divider()
        input_dir = st.text_input("Directorio local con PDFs", value="data/raw")
        col1, col2 = st.columns(2)
        if col1.button("Indexar directorio"):
            with st.spinner("Indexando..."):
                try:
                    manifest = indexer.index_directory(Path(input_dir))
                    st.success(
                        f"{manifest.total_documents} documentos y "
                        f"{manifest.total_chunks} chunks en el índice"
                    )
                except Exception as exc:
                    st.error(f"Error: {exc}")
        confirm_reindex = col2.checkbox("Confirmar reindexación completa")
        if col2.button("Reindexar todo", disabled=not confirm_reindex):
            with st.spinner("Reconstruyendo el índice..."):
                try:
                    manifest = indexer.reindex_all(Path(input_dir))
                    st.success(f"Reindexación completa: {manifest.total_documents} documentos")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Error: {exc}")

    with list_tab:
        try:
            manifest = indexer.load_manifest()
            if not manifest.documents:
                st.info("No hay documentos indexados. Sube tus PDF para comenzar.")
            for doc in manifest.documents.values():
                with st.expander(f"{doc.filename} · {doc.status.value}"):
                    chunks = indexer.sqlite.get_chunks_by_document(doc.id)
                    st.write(f"**Páginas con texto:** {doc.page_count}")
                    st.write(f"**Chunks:** {len(chunks)}")
                    st.write(f"**ID del documento:** `{doc.id}`")
                    if doc.error_message:
                        st.error(doc.error_message)
        except Exception as exc:
            st.error(f"Error cargando índice: {exc}")

    with delete_tab:
        try:
            manifest = indexer.load_manifest()
            if not manifest.documents:
                st.info("No hay documentos para eliminar.")
            else:
                options = {
                    f"{doc.filename} ({str(doc.id)[:8]})": doc.id
                    for doc in manifest.documents.values()
                }
                selected = st.selectbox("Documento", options=list(options))
                if st.button("Eliminar del índice"):
                    if indexer.delete_document(str(options[selected])):
                        st.success("Documento eliminado del índice")
                        st.rerun()
                    st.error("No se pudo eliminar el documento")
        except Exception as exc:
            st.error(f"Error: {exc}")


def render_ask(pipeline, settings: Settings) -> None:
    """Ejecuta consultas y presenta respuesta, fuentes y decisiones del crítico."""
    st.header("Consultar al sistema")
    col1, col2, col3 = st.columns([2, 1, 1])
    mode = col1.radio(
        "Modo",
        [Mode.BASELINE, Mode.CORRECTIVE],
        format_func=lambda value: value.value.capitalize(),
        horizontal=True,
    )
    top_k = col2.number_input("Top-K", min_value=1, max_value=20, value=settings.retrieval.top_k)
    show_context = col3.checkbox("Mostrar evidencia", value=True)
    question = st.text_area("Pregunta", placeholder="Ej.: ¿Qué hizo Turing para la IA?", height=100)

    if st.button("Responder", type="primary", disabled=not question.strip()):
        with st.spinner("Buscando evidencia y generando la respuesta..."):
            try:
                request = AskRequest(question=question, mode=mode, top_k=int(top_k))
                st.session_state["last_response"] = pipeline.ask(request)
            except EvidenciaLocalError as exc:
                st.error(f"Error: {exc}")
            except Exception as exc:
                st.error(f"Error inesperado: {exc}")

    response = st.session_state.get("last_response")
    if response:
        render_response(response, show_context)


def render_response(response: AskResponse, show_context: bool) -> None:
    """Renderiza una respuesta almacenada sin repetir la inferencia en cada rerun."""
    st.caption(f"Consulta: {response.question} · Modo: {response.mode.value}")
    if response.abstained:
        st.warning(response.answer)
        st.write(f"**Motivo:** {response.abstention_reason}")
    else:
        st.success("Respuesta")
        st.markdown(response.answer)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latencia", f"{response.total_latency_ms / 1000:.1f} s")
    col2.metric("Tokens", response.total_tokens if response.total_tokens is not None else "N/A")
    col3.metric("Recuperaciones", len(response.retrievals))
    col4.metric("Citas", len(response.citations))

    citation_map = {
        result.chunk_id: result for round_results in response.retrievals for result in round_results
    }
    if response.citations:
        st.subheader("Fuentes citadas")
        for index, citation_id in enumerate(response.citations, start=1):
            result = citation_map.get(citation_id)
            if result:
                page = f", página {result.page_number}" if result.page_number else ""
                st.markdown(f"{index}. **{result.document_filename}**{page}")
                st.caption(f"Chunk: {citation_id}")
            else:
                st.caption(str(citation_id))

    if show_context:
        for index, round_results in enumerate(response.retrievals, start=1):
            with st.expander(f"Evidencia · ronda {index} ({len(round_results)} chunks)"):
                for result in round_results:
                    page = f", pág. {result.page_number}" if result.page_number else ""
                    st.markdown(
                        f"**[{result.rank}] {result.document_filename}**{page} · "
                        f"similitud: {result.score:.3f}"
                    )
                    st.caption(result.text)

    if response.critiques:
        st.subheader("Evaluación crítica")
        for index, critique in enumerate(response.critiques, start=1):
            status = "suficiente" if critique.sufficient else "insuficiente"
            with st.expander(f"Ronda {index}: evidencia {status}"):
                st.write(critique.reason)
                if critique.missing_aspects:
                    st.write(f"**Aspectos faltantes:** {', '.join(critique.missing_aspects)}")
                if critique.retry_query:
                    st.write(f"**Nueva búsqueda:** {critique.retry_query}")


def render_settings(settings: Settings) -> None:
    """Muestra la configuración efectiva sin exponer el contenido de .env."""
    st.header("Configuración actual")
    st.json(settings.model_dump(mode="json"), expanded=False)
    st.caption(
        "Precedencia: valores por defecto < archivo YAML (CLI) < variables de entorno/.env. "
        "La interfaz web usa valores por defecto y entorno; el YAML se indica en la CLI con --config."
    )
    st.info(
        f"Modelo: {settings.llama_cpp.model_path}\n\n"
        f"Índice vectorial: {settings.chroma.persist_directory}\n\n"
        f"Metadatos y trazas: {settings.sqlite.database_path}"
    )


if __name__ == "__main__":
    main()
