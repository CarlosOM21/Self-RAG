"""Aplicación Streamlit para EvidenciaLocal - Interfaz web en español."""

import time
from pathlib import Path

import streamlit as st

# Configuración de página
st.set_page_config(
    page_title="EvidenciaLocal - RAG Educativo Local",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Importar módulos internos
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import Settings, get_settings
from core.exceptions import EvidenciaLocalError
from core.models import AskRequest, Mode
from pipeline.indexer import get_indexer
from pipeline.rag_pipeline import get_rag_pipeline


# Cache para modelos (reutilizar por sesión)
@st.cache_resource
def get_cached_settings() -> Settings:
    return get_settings()

@st.cache_resource
def get_cached_indexer(_settings: Settings):
    return get_indexer(_settings)

@st.cache_resource
def get_cached_pipeline(_settings: Settings):
    return get_rag_pipeline(_settings)


def main():
    """Punto de entrada principal de la aplicación Streamlit."""
    settings = get_cached_settings()
    indexer = get_cached_indexer(settings)
    pipeline = get_cached_pipeline(settings)

    # Sidebar
    with st.sidebar:
        st.title("📚 EvidenciaLocal")
        st.caption("RAG Educativo Local - Baseline vs Correctivo")

        page = st.radio(
            "Navegación",
            ["🏠 Inicio", "📄 Documentos", "❓ Consultar", "📊 Evaluación", "⚙️ Configuración"],
            label_visibility="collapsed",
        )

        st.divider()
        st.caption(f"Modelo LLM: {settings.llama_cpp.model_path}")
        st.caption(f"Embeddings: {settings.embedding.model_name}")

    # Router de páginas
    if page == "🏠 Inicio":
        render_home(indexer, settings)
    elif page == "📄 Documentos":
        render_documents(indexer, settings)
    elif page == "❓ Consultar":
        render_ask(pipeline, settings)
    elif page == "📊 Evaluación":
        render_evaluation(settings)
    elif page == "⚙️ Configuración":
        render_settings(settings)


def render_home(indexer, settings: Settings):
    """Página de inicio con estado del sistema."""
    st.header("🏠 Inicio")

    # Estado del sistema
    col1, col2, col3 = st.columns(3)

    with col1:
        try:
            manifest = indexer.load_manifest()
            st.metric("Documentos indexados", manifest.total_documents)
            st.metric("Chunks totales", manifest.total_chunks)
        except Exception:
            st.metric("Documentos indexados", "Error")
            st.metric("Chunks totales", "Error")

    with col2:
        # Verificar llama.cpp
        from llm.llama_client import get_llama_client
        llama_client = get_llama_client(settings)
        llama_ok = llama_client.check_model_available()
        st.metric("llama.cpp", "✅ Disponible" if llama_ok else "❌ No disponible")
        if not llama_ok:
            st.warning(f"Modelo `{settings.llama_cpp.model_path}` no encontrado")

    with col3:
        st.metric("Modo embedding", settings.embedding.model_name)
        st.metric("Dimensión", settings.embedding.dimension)

    st.divider()

    # Acciones rápidas
    st.subheader("Acciones rápidas")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📥 Generar datos demo", use_container_width=True):
            with st.spinner("Generando documentos sintéticos..."):
                from cli.demo import generate_demo_data
                generate_demo_data(Path("data/raw"), force=True)
                st.success("Datos demo generados. Vaya a 'Documentos' para indexar.")

    with col2:
        if st.button("🔍 Indexar documentos", use_container_width=True):
            st.switch_page("📄 Documentos")

    with col3:
        if st.button("❓ Hacer una pregunta", use_container_width=True):
            st.switch_page("❓ Consultar")

    # Información del proyecto
    with st.expander("ℹ️ Acerca de EvidenciaLocal"):
        st.markdown("""
        **EvidenciaLocal** es un sistema RAG (Retrieval-Augmented Generation) educativo
        que funciona completamente en local, sin APIs de pago ni servicios en la nube.

        **Dos modos comparables:**
        - **Baseline**: Recuperación única + generación con permiso de abstención
        - **Correctivo**: Evaluación crítica del contexto, hasta 2 reintentos de recuperación,
          deduplicación y abstención explícita

        **Tecnologías:**
        - Embeddings: `intfloat/multilingual-e5-small` (384 dims, coseno)
        - Vector DB: ChromaDB persistente
        - Metadatos: SQLite con WAL
        - LLM: llama.cpp local (`qwen2.5-3b-instruct-q4_k_m.gguf`)
        - Interfaz: Streamlit
        """)


def render_documents(indexer, settings: Settings):
    """Página de gestión de documentos."""
    st.header("📄 Gestión de Documentos")

    tab1, tab2, tab3 = st.tabs(["📤 Subir e Indexar", "📋 Listado", "🗑️ Eliminar"])

    with tab1:
        st.subheader("Subir archivos PDF")
        uploaded_files = st.file_uploader(
            "Seleccione archivos PDF",
            type=["pdf"],
            accept_multiple_files=True,
            help="Máx. 50MB por archivo. Solo PDFs con texto extraíble.",
        )

        if uploaded_files:
            st.info(f"{len(uploaded_files)} archivo(s) seleccionado(s)")

            if st.button("📥 Indexar archivos", type="primary"):
                # Guardar archivos temporalmente
                raw_dir = Path("data/raw")
                raw_dir.mkdir(parents=True, exist_ok=True)

                saved_paths = []
                for uploaded in uploaded_files:
                    if uploaded.size > 50 * 1024 * 1024:
                        st.error(f"{uploaded.name}: excede 50MB")
                        continue
                    save_path = raw_dir / uploaded.name
                    save_path.write_bytes(uploaded.read())
                    saved_paths.append(save_path)

                if saved_paths:
                    with st.spinner("Indexando documentos..."):
                        try:
                            manifest = indexer.index_directory(raw_dir)
                            st.success(f"✅ Indexados {manifest.total_documents} documentos, {manifest.total_chunks} chunks")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error indexando: {e}")

        st.divider()
        st.subheader("Indexar directorio existente")
        input_dir = st.text_input("Ruta del directorio con PDFs", value="data/raw")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📂 Indexar (idempotente)"):
                with st.spinner("Indexando..."):
                    try:
                        manifest = indexer.index_directory(Path(input_dir))
                        st.success(f"✅ {manifest.total_documents} docs, {manifest.total_chunks} chunks")
                    except Exception as e:
                        st.error(f"Error: {e}")
        with col2:
            if st.button("🔄 Reindexar todo (borra anterior)", type="secondary"):
                if st.checkbox("Confirmo que quiero borrar el índice actual"):
                    with st.spinner("Reindexando..."):
                        try:
                            manifest = indexer.reindex_all(Path(input_dir))
                            st.success(f"✅ Reindexación completa: {manifest.total_documents} docs")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

    with tab2:
        st.subheader("Documentos indexados")
        try:
            manifest = indexer.load_manifest()
            if manifest.documents:
                for doc_hash, doc in manifest.documents.items():
                    with st.expander(f"📄 {doc.filename} ({doc.page_count} págs, {doc.status.value})"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**ID:** {doc.id}")
                            st.write(f"**Hash:** {doc.content_hash[:16]}...")
                            st.write(f"**Estado:** {doc.status.value}")
                            if doc.error_message:
                                st.error(doc.error_message)
                        with col2:
                            chunks = indexer.sqlite.get_chunks_by_document(doc.id)
                            st.write(f"**Chunks:** {len(chunks)}")
                            if chunks:
                                st.write(f"**Tokens totales:** {sum(c.token_count for c in chunks)}")
            else:
                st.info("No hay documentos indexados. Suba archivos o use datos demo.")
        except Exception as e:
            st.error(f"Error cargando índice: {e}")

    with tab3:
        st.subheader("Eliminar documentos")
        st.warning("⚠️ Esta acción elimina el documento del índice (Chroma + SQLite + manifiesto)")
        try:
            manifest = indexer.load_manifest()
            if manifest.documents:
                doc_options = {f"{doc.filename} ({str(doc.id)[:8]})": doc.id for doc in manifest.documents.values()}
                selected = st.selectbox("Seleccione documento", options=list(doc_options.keys()))
                if st.button("🗑️ Eliminar seleccionado", type="secondary"):
                    doc_id = doc_options[selected]
                    if indexer.delete_document(str(doc_id)):
                        st.success("Documento eliminado")
                        st.rerun()
                    else:
                        st.error("Error eliminando")
            else:
                st.info("No hay documentos para eliminar")
        except Exception as e:
            st.error(f"Error: {e}")


def render_ask(pipeline, settings: Settings):
    """Página de consulta interactiva."""
    st.header("❓ Consultar al Sistema")

    # Controles
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        mode = st.radio(
            "Modo",
            [Mode.BASELINE, Mode.CORRECTIVE],
            format_func=lambda m: f"{'🔵' if m == Mode.BASELINE else '🟢'} {m.value.capitalize()}",
            horizontal=True,
        )
    with col2:
        top_k = st.number_input("Top-K", min_value=1, max_value=20, value=settings.retrieval.top_k)
    with col3:
        show_context = st.checkbox("Mostrar evidencia", value=True)

    # Input de pregunta
    question = st.text_area(
        "Pregunta",
        placeholder="Ej: ¿Qué es el aprendizaje supervisado?",
        height=100,
    )

    if st.button("🔍 Responder", type="primary", disabled=not question.strip()):
        with st.spinner("Procesando..."):
            start = time.perf_counter()
            try:
                request = AskRequest(question=question.strip(), mode=mode, top_k=top_k)
                response = pipeline.ask(request)
                elapsed = (time.perf_counter() - start) * 1000

                # Mostrar respuesta
                if response.abstained:
                    st.warning("⚠️ **ABSTENCIÓN**")
                    st.write(f"Razón: {response.abstention_reason}")
                else:
                    st.success("✅ **RESPUESTA**")
                    st.write(response.answer)

                # Métricas
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Latencia", f"{response.total_latency_ms:.0f} ms")
                col2.metric("Tokens", response.total_tokens or "N/A")
                col3.metric("Recuperaciones", len(response.retrievals))
                col4.metric("Citas", len(response.citations))

                if response.citations:
                    st.subheader("📎 Citas")
                    for cid in response.citations:
                        st.code(str(cid), language=None)

                # Evidencia expandible
                if show_context and response.retrievals:
                    st.subheader("🔍 Evidencia recuperada")
                    for i, round_results in enumerate(response.retrievals):
                        with st.expander(f"Ronda {i+1} - {len(round_results)} chunks"):
                            for r in round_results:
                                page_str = f" (pág. {r.page_number})" if r.page_number else ""
                                st.markdown(f"**[{r.rank}] {r.document_filename}{page_str}** — score: {r.score:.3f}")
                                st.caption(r.text[:300] + ("..." if len(r.text) > 300 else ""))

                # Críticas (modo correctivo)
                if response.critiques:
                    st.subheader("🧠 Evaluación crítica")
                    for i, critique in enumerate(response.critiques):
                        with st.expander(f"Crítica {i+1}: {'✅ Suficiente' if critique.sufficient else '❌ Insuficiente'}"):
                            st.write(f"**Razón:** {critique.reason}")
                            if critique.missing_aspects:
                                st.write(f"**Aspectos faltantes:** {', '.join(critique.missing_aspects)}")
                            if critique.retry_query:
                                st.write(f"**Consulta sugerida:** {critique.retry_query}")
                            st.caption(f"Latencia: {critique.latency_ms:.1f}ms | Tokens: {critique.tokens_used or 'N/A'}")

            except EvidenciaLocalError as e:
                st.error(f"Error: {e}")
            except Exception as e:
                st.error(f"Error inesperado: {e}")


def render_evaluation(settings: Settings):
    """Página de evaluación comparativa."""
    st.header("📊 Evaluación Comparativa")

    st.info("Esta sección ejecuta evaluaciones por lotes sobre datasets JSONL.")

    col1, col2 = st.columns(2)
    with col1:
        dataset_file = st.file_uploader("Dataset JSONL", type=["jsonl"])
    with col2:
        modes = st.multiselect(
            "Modos",
            [Mode.BASELINE, Mode.CORRECTIVE],
            default=[Mode.BASELINE, Mode.CORRECTIVE],
            format_func=lambda m: m.value.capitalize(),
        )

    limit = st.number_input("Límite de muestras (0 = sin límite)", min_value=0, value=0)
    output_dir = st.text_input("Directorio de salida", value="experiments/results/demo")

    if st.button("▶️ Ejecutar evaluación", type="primary", disabled=not dataset_file or not modes):
        if dataset_file:
            # Guardar dataset temporalmente
            dataset_path = Path("datasets/temp_eval.jsonl")
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_path.write_bytes(dataset_file.read())

            with st.spinner("Ejecutando evaluación..."):
                try:
                    from cli.evaluate import run_evaluation
                    run_evaluation(
                        dataset_path,
                        modes,
                        Path(output_dir),
                        settings,
                        limit if limit > 0 else None,
                    )
                    st.success("✅ Evaluación completada")
                    st.info(f"Resultados en: {output_dir}")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()

    # Exportar/Importar anotaciones
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📤 Exportar plantilla")
        results_dir = st.text_input("Directorio resultados", value="experiments/results/demo", key="export_dir")
        if st.button("Generar CSV para anotación"):
            try:
                from cli.evaluate import export_annotations_template
                out_path = Path(results_dir) / "annotations_template.csv"
                export_annotations_template(Path(results_dir), out_path)
                st.success(f"Plantilla generada: {out_path}")
            except Exception as e:
                st.error(f"Error: {e}")

    with col2:
        st.subheader("📥 Importar anotaciones")
        annotations_file = st.file_uploader("CSV con anotaciones", type=["csv"], key="ann_upload")
        if annotations_file and st.button("Importar"):
            try:
                ann_path = Path("annotations_imported.csv")
                ann_path.write_bytes(annotations_file.read())
                from cli.evaluate import import_annotations
                import_annotations(Path(results_dir), ann_path)
                st.success("Anotaciones importadas")
            except Exception as e:
                st.error(f"Error: {e}")

    # Generar reporte
    st.divider()
    if st.button("📄 Generar reporte Markdown"):
        try:
            from cli.evaluate import generate_report
            out_path = Path(results_dir) / "report.md"
            generate_report(Path(results_dir), out_path)
            st.success(f"Reporte generado: {out_path}")
            with open(out_path, encoding="utf-8") as f:
                st.markdown(f.read())
        except Exception as e:
            st.error(f"Error: {e}")


def render_settings(settings: Settings):
    """Página de configuración."""
    st.header("⚙️ Configuración")

    st.subheader("Configuración actual (solo lectura)")
    st.json(settings.model_dump(mode="json"), expanded=False)

    st.divider()
    st.subheader("Archivos de configuración")
    st.markdown("""
    La configuración se carga desde:
    1. Valores por defecto en código
    2. Archivo `.env` (variables de entorno)
    3. Archivo YAML pasado con `--config` (CLI)

    **Variables de entorno principales:**
    - `LLAMA_CPP__MODEL_PATH` - Ruta del modelo (default: qwen2.5-3b-instruct-q4_k_m.gguf)
    - `LLAMA_CPP__N_CTX` - Contexto (default: 4096)
    - `LLAMA_CPP__N_THREADS` - Hilos (default: 4)
    - `LLAMA_CPP__N_BATCH` - Batch size (default: 512)
    - `LLAMA_CPP__TEMPERATURE` - Temperatura (default: 0.1)
    - `LLAMA_CPP__TOP_P` - Top-p (default: 0.9)
    - `LLAMA_CPP__MAX_TOKENS` - Max tokens (default: 2048)
    - `EMBEDDING__MODEL_NAME` - Modelo embeddings (default: intfloat/multilingual-e5-small)
    - `CHROMA__PERSIST_DIRECTORY` - Directorio ChromaDB
    - `SQLITE__DATABASE_PATH` - Archivo SQLite
    """)

    # Verificar .env
    env_path = Path(".env")
    if env_path.exists():
        st.success("✅ Archivo .env encontrado")
        with st.expander("Ver .env"):
            st.code(env_path.read_text())
    else:
        st.info("ℹ️ No hay archivo .env. Cree uno basado en `.env.example`")


if __name__ == "__main__":
    main()
