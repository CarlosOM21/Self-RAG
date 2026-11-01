# EvidenciaLocal

**RAG local con dos modos comparables: Baseline y Correctivo.**

EvidenciaLocal responde preguntas usando únicamente un corpus de PDF indexado en el equipo. No necesita APIs de pago ni servicios de inferencia remotos: los embeddings se calculan con Sentence Transformers y la generación se ejecuta directamente con `llama-cpp-python` y un modelo GGUF.

## Modos de consulta

### Baseline

```text
Pregunta → embedding de consulta → una recuperación top-k → generación → respuesta
```

Realiza una única búsqueda vectorial. El modelo puede abstenerse si la evidencia no permite responder.

### Correctivo

```text
Pregunta original
  → recuperación inicial
  → crítica de suficiencia del contexto
      ├─ suficiente → generación final
      └─ insuficiente → consulta reformulada → nueva recuperación sin duplicados
                         → nueva crítica (máximo dos reintentos)
  → una única generación final con la evidencia seleccionada
```

La pregunta original se conserva siempre para la crítica y la respuesta. La reformulación solo se utiliza para buscar más evidencia. Si no aparecen chunks nuevos o se agotan los reintentos, el sistema genera con la mejor evidencia acumulada en lugar de devolver una respuesta tentativa incompleta.

## Componentes

| Componente | Implementación |
|---|---|
| Extracción | `pypdf`, página por página |
| Segmentación | Ventanas de 512 tokens con 50 de solape; tokenizer E5 |
| Embeddings | `intfloat/multilingual-e5-small`, 384 dimensiones, prefijos `query:`/`passage:`, normalización L2 |
| Base vectorial | ChromaDB persistente, distancia coseno |
| Metadatos y trazas | SQLite con WAL y claves foráneas |
| LLM | `llama-cpp-python`, GGUF local; por defecto `qwen2.5-3b-instruct-q4_k_m.gguf` |
| Interfaces | CLI con Typer y UI con Streamlit |
| Evaluación | JSONL, anotación humana ciega y reporte Markdown con bootstrap pareado |

## Requisitos

- Python 3.11 o superior.
- Un modelo GGUF compatible con `llama-cpp-python`.
- Los modelos de Hugging Face requieren conexión la primera vez que se descargan; después pueden reutilizar la caché local.
- PDFs con texto extraíble. No se incluye OCR.

El repositorio de trabajo utiliza por defecto el archivo `qwen2.5-3b-instruct-q4_k_m.gguf` situado en la raíz. Los archivos `*.gguf` están excluidos de Git.

## Instalación

```bash
cd Self-RAG
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Si el modelo está en otra ruta:

```bash
export LLAMA_CPP__MODEL_PATH=/ruta/absoluta/modelo.gguf
```

Comprobación del sistema:

```bash
evidencialocal doctor
```

`doctor` verifica el GGUF, ChromaDB, SQLite y la carga del modelo de embeddings. No genera documentos ni descarga otro LLM.

## Uso

### 1. Añadir e indexar documentos reales

Copia tus PDF a `data/raw` o indica otro directorio:

```bash
evidencialocal index --input data/raw
evidencialocal list-docs
```

La indexación incremental utiliza SHA-256 del archivo para omitir documentos ya procesados. Para reconstruir todo el índice:

```bash
evidencialocal index --input data/raw --force-reindex
```

Para borrar el índice sin volver a indexar:

```bash
evidencialocal delete-index
# o, sin confirmación interactiva:
evidencialocal delete-index --yes
```

### 2. Consultar desde la CLI

La pregunta es un argumento posicional:

```bash
evidencialocal ask "¿Qué hizo Turing para la IA?" --mode baseline
evidencialocal ask "¿Qué hizo Turing para la IA?" --mode corrective
```

Opciones útiles:

```bash
evidencialocal ask "¿Qué es RAG?" --mode corrective --top-k 5 --show-context
```

La salida incluye la respuesta, las fuentes citadas, latencia, tokens de salida, número de recuperaciones y críticas. `--show-context` añade los chunks recuperados.

### 3. Consultar desde Streamlit

```bash
streamlit run src/evidencialocal/ui/streamlit_app.py
```

La interfaz permite:

- consultar el estado del índice y del GGUF;
- subir e indexar PDFs propios;
- consultar en modo Baseline o Correctivo;
- inspeccionar respuesta, fuentes, evidencia y críticas;
- eliminar documentos del índice;
- revisar la configuración efectiva.

No incluye generación de datos demo, creación de PDFs sintéticos ni utilidades de exportación innecesarias para la consulta habitual.

## Configuración

La configuración vive en `src/evidencialocal/config/settings.py`, `configs/default.yaml` y `.env.example`.

Precedencia en la CLI:

```text
valores por defecto < YAML indicado con --config < .env < variables de entorno del proceso
```

Ejemplo:

```bash
cp configs/default.yaml configs/local.yaml
# editar configs/local.yaml
evidencialocal index --input data/raw --config configs/local.yaml
evidencialocal ask "¿Qué es RAG?" --config configs/local.yaml
```

La UI usa defaults y entorno/`.env`; no dispone de selector YAML.

Parámetros principales:

| Variable | Default | Finalidad |
|---|---:|---|
| `LLAMA_CPP__N_CTX` | 8192 | Ventana total del LLM |
| `LLAMA_CPP__MAX_TOKENS` | 512 | Máximo de tokens de salida por llamada |
| `LLAMA_CPP__N_THREADS` | 4 | Hilos CPU de llama.cpp |
| `EMBEDDING__DEVICE` | auto | `auto`, `cpu`, `cuda` o `mps` |
| `EMBEDDING__BATCH_SIZE` | 32 | Lote de embeddings |
| `INDEXING__CHUNK_SIZE` | 512 | Tokens máximos por chunk |
| `INDEXING__CHUNK_OVERLAP` | 50 | Solape entre ventanas |
| `RETRIEVAL__TOP_K` | 5 | Chunks por recuperación |
| `CORRECTIVE__MAX_RETRIES` | 2 | Recuperaciones adicionales |
| `CORRECTIVE__MAX_CONTEXT_CHUNKS` | 8 | Límite de evidencia enviada al LLM |

Si cambias el modelo de embeddings, su dimensión, el tokenizer o el chunking, reconstruye el índice con `--force-reindex`.

## Evaluación comparativa

Crea un JSONL propio, sin generadores demo:

```jsonl
{"id":"q1","question":"¿Qué hizo Turing para la IA?","gold_answer":"Respuesta de referencia opcional","gold_citations":[]}
{"id":"q2","question":"¿Qué es el test de Turing?","gold_answer":null,"gold_citations":[]}
```

Ejecuta ambos modos:

```bash
evidencialocal evaluate \
  --dataset datasets/mis_preguntas.jsonl \
  --modes baseline,corrective \
  --output experiments/results/experimento_01
```

Anotación humana ciega al modo:

```bash
evidencialocal export-annotations \
  --results experiments/results/experimento_01 \
  --output experiments/results/experimento_01/annotations.csv

# Completar correctness, faithfulness, citation_precision y citation_recall (0–1).

evidencialocal import-annotations \
  --results experiments/results/experimento_01 \
  --annotations experiments/results/experimento_01/annotations.csv

evidencialocal report \
  --results experiments/results/experimento_01 \
  --output experiments/results/experimento_01/report.md
```

El CSV público contiene `annotation_id`, pero no el modo. `annotations_mapping.json` permite reimportar cada fila sin ambigüedad y debe conservarse junto a los resultados. El reporte solo calcula métricas humanas cuando existen anotaciones; no sustituye la calidad factual por similitud de embeddings.

## Estructura

```text
src/evidencialocal/
├── config/settings.py              # defaults, YAML, entorno
├── core/                           # modelos Pydantic y excepciones
├── embeddings/embedder.py          # E5 y normalización
├── retrieval/
│   ├── document_processor.py       # PDF → chunks → embeddings por lote
│   └── retriever.py                # búsqueda y deduplicación correctiva
├── vectorstore/chroma_store.py     # persistencia y similitud coseno
├── metadata/sqlite_store.py        # documentos, chunks, trazas y evaluación
├── llm/llama_client.py             # prompts, JSON, validación y reintentos
├── pipeline/
│   ├── indexer.py                  # indexación idempotente y manifiesto
│   └── rag_pipeline.py             # Baseline y Correctivo
├── cli/                            # comandos y evaluación
└── ui/streamlit_app.py             # interfaz web
```

## Desarrollo y verificación

```bash
pytest -q
ruff check src tests
ruff format --check src tests
python -m compileall -q src tests
```

Los tests usan dobles para los modelos pesados e incluyen regresiones del bucle correctivo: uso real de `retry_query`, deduplicación con sobre-recuperación, generación final cuando no hay chunks nuevos y rechazo de respuestas que contienen solo una cita.

## Limitaciones deliberadas

- No hay OCR para PDFs escaneados.
- No se usa Ollama ni un servidor HTTP para el LLM: la inferencia es directa con llama.cpp.
- La similitud coseno ordena candidatos, pero no demuestra relevancia ni factualidad.
- Las citas validan que el UUID pertenezca al contexto enviado; la valoración de si sustentan bien cada afirmación corresponde a la evaluación humana.
- El crítico y el generador comparten modelo, por lo que la crítica no es una garantía independiente de corrección.

## Documentación

- `docs/architecture.md`: arquitectura resumida.
- `docs/usage.md`: guía operativa y resolución de problemas.

## Licencia

MIT.
