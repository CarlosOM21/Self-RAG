# Arquitectura de EvidenciaLocal

## Visión general

EvidenciaLocal es un sistema RAG (Retrieval-Augmented Generation) educativo que implementa **dos modos comparables** para responder preguntas basándose en un corpus local de documentos PDF.

```
┌─────────────────────────────────────────────────────────────────┐
│                        USUARIO / CLI / UI                        │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RAGPipeline (Controlador)                  │
│  ┌─────────────────────┐    ┌─────────────────────────────┐    │
│  │    Modo Baseline    │    │     Modo Correctivo         │    │
│  │  1. Retrieve (1x)   │    │  1. Retrieve (1x)           │    │
│  │  2. Generate        │    │  2. Generate (tentativa)    │    │
│  │  3. Return          │    │  3. Critique                │    │
│  │                     │    │  4. Retrieve (hasta 2x más) │    │
│  │                     │    │  5. Generate final          │    │
│  └─────────────────────┘    └─────────────────────────────┘    │
└────────────────────────────────┬────────────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ChromaDB      │    │    SQLite       │    │   Ollama LLM    │
│  (Vectores +    │    │  (Metadatos +   │    │  (Generación +  │
│   Coseno)       │    │   Trazas)       │    │   Crítica)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                       ▲
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Indexer (Offline)                          │
│  PDF → Extract → Chunk (E5 tokenizer) → Embed → Store           │
└─────────────────────────────────────────────────────────────────┘
```

## Módulos principales

### 1. Config (`src/evidencialocal/config/`)
- **Settings**: Pydantic Settings con validación, variables de entorno, YAML
- Configuración anidada: Embedding, Chroma, SQLite, Ollama, Indexing, Retrieval, Corrective, Evaluation

### 2. Core (`src/evidencialocal/core/`)
- **Models**: Todos los modelos Pydantic del dominio (Document, Chunk, RetrievalResult, GenerationResult, CritiqueResult, AskRequest/Response, IndexManifest, EvaluationSample/Result, ExperimentManifest)
- **Exceptions**: Jerarquía de excepciones tipadas

### 3. Embeddings (`src/evidencialocal/embeddings/`)
- **Embedder**: Wrapper de SentenceTransformer con prefijos E5 (`query:`/`passage:`) y normalización L2 obligatoria
- Singleton pattern para reutilización

### 4. VectorStore (`src/evidencialocal/vectorstore/`)
- **ChromaStore**: Wrapper ChromaDB persistente, colección con `hnsw:space=cosine`
- Métodos: `add_chunks`, `query`, `delete_chunks`, `reset_collection`

### 5. Metadata (`src/evidencialocal/metadata/`)
- **SQLiteStore**: SQLite con WAL, Foreign Keys, esquema versionado
- Tablas: documents, chunks, traces, index_manifest, evaluation_results, experiment_manifest
- Índices para consultas frecuentes

### 6. LLM (`src/evidencialocal/llm/`)
- **OllamaClient**: Cliente HTTP tipado con timeouts, esquemas JSON, reparación única de formato
- Esquemas: `GENERATION_SCHEMA` (answer, citations, abstained, abstention_reason) y `CRITIQUE_SCHEMA` (sufficient, reason, missing_aspects, retry_query)
- Prompts en español, documentos tratados como datos

### 7. Retrieval (`src/evidencialocal/retrieval/`)
- **DocumentProcessor**: Extracción PDF (pypdf), chunking por tokens (tokenizer E5), embeddings
- **Retriever**: Recuperación con deduplicación en modo correctivo

### 8. Pipeline (`src/evidencialocal/pipeline/`)
- **Indexer**: Indexación idempotente con manifiesto, verificación compatibilidad config, reindexación completa
- **RAGPipeline**: Orquesta baseline y correctivo, guarda trazas en SQLite

### 9. CLI (`src/evidencialocal/cli/`)
- **main.py**: Comandos Typer exactos según especificación
- **demo.py**: Generador datos sintéticos
- **evaluate.py**: Evaluación comparativa, export/import anotaciones, reporte

### 10. UI (`src/evidencialocal/ui/`)
- **streamlit_app.py**: Interfaz web en español, cache de modelos por sesión

## Flujo de datos

### Indexación
```
PDFs en data/raw
    │
    ▼
DocumentProcessor.extract_pdf_pages() → [(page_num, text), ...]
    │
    ▼
DocumentProcessor.chunk_text_by_tokens() → [(chunk_text, char_start, char_end), ...]
    │
    ▼
Embedder.encode_passages() → embeddings (384 dims, normalizados)
    │
    ▼
ChromaStore.add_chunks() + SQLiteStore.upsert_document/chunk_metadata()
    │
    ▼
IndexManifest actualizado (JSON en SQLite)
```

### Consulta Baseline
```
Pregunta
    │
    ▼
Embedder.encode_queries() → query_embedding
    │
    ▼
ChromaStore.query() → top-k RetrievalResult (score coseno)
    │
    ▼
OllamaClient.generate_answer() con contexto + instrucciones abstención
    │
    ▼
AskResponse (answer, citations, abstained, retrievals, métricas)
    │
    ▼
SQLiteStore.save_trace()
```

### Consulta Correctiva
```
Pregunta original (inmutable)
    │
    ▼
Para intento en 0..max_retries:
    │
    ├── Retrieve (baseline si 0, corrective con dedup si >0)
    ├── Acumular chunks
    ├── Generate tentativa
    ├── Si abstained o último intento → RETURN
    ├── Critique_context()
    ├── Si sufficient → RETURN
    └── Siguiente intento con retry_query o pregunta original
    │
    ▼
AskResponse final (answer, citations, abstained, retrievals[], critiques[])
```

## Decisiones de diseño clave

1. **Controlador explícito**: No hay agente autónomo; el pipeline orquesta pasos deterministas
2. **Idempotencia**: Indexación basada en hash de contenido (SHA256)
3. **Versionado**: Manifiesto con hash de config, rechazo si incompatibilidad
4. **Abstención obligatoria**: Ambos modos pueden abstenerse si contexto insuficiente
5. **Citas verificables**: UUIDs de chunks reales, página real o null
6. **Trazabilidad completa**: Cada consulta guardada con timestamps, latencias, tokens, prompts hashes
7. **Evaluación ciega**: Plantillas de anotación ocultan el modo
8. **Sin métricas falsas**: Sin gold → métricas NA, no se inventan números

## Dependencias externas

| Dependencia | Uso | Versión mínima |
|-------------|-----|----------------|
| sentence-transformers | Embeddings E5 | 3.0.0 |
| chromadb | Vector store | 0.5.0 |
| pypdf | Extracción PDF | 4.0.0 |
| ollama | Cliente LLM local | 0.3.0 |
| streamlit | UI web | 1.35.0 |
| typer | CLI | 0.12.0 |
| pydantic/pydantic-settings | Config/Models | 2.8.0/2.3.0 |
| transformers | Tokenizer E5 | 4.41.0 |
| torch | Backend embeddings | 2.3.0 |

## Esquema de base de datos (SQLite)

Ver `src/evidencialocal/metadata/sqlite_store.py` - constante `DDL` con todas las tablas e índices.