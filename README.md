# EvidenciaLocal

> **Sistema RAG educativo local con evaluación comparativa rigurosa entre dos estrategias de recuperación-generación**

Un sistema *Retrieval-Augmented Generation* (RAG) completamente local, sin dependencias de nube ni APIs de pago, diseñado para experimentación científica controlada. Implementa dos pipelines comparables —**Baseline** y **Correctivo**— bajo un mismo marco de evaluación ciega con métricas humanas y bootstrap pareado.

---

## Propósito y filosofía

EvidenciaLocal nace de la necesidad de **comparar estrategias RAG de forma honesta y reproducible**. La mayoría de benchmarks públicos favorecen arquitecturas complejas sin aislar el efecto de cada componente. Este proyecto:

- **Aísla variables**: misma recuperación, mismos embeddings, mismo LLM, mismo prompt base —solo cambia la estrategia de generación
- **Exige evidencia**: toda métrica de calidad requiere anotación humana ciega al modo
- **Registra todo**: semillas, hashes de configuración, versión de modelo, latencia, tokens, recursos
- **Rechaza métricas proxy**: similitud coseno ≠ relevancia, score LLM ≠ certeza, JSON válido ≠ respuesta fundamentada

---

## Arquitectura técnica

```
┌─────────────────────────────────────────────────────────────────┐
│                        EVIDENCIALOCAL                             │
├─────────────────────────────────────────────────────────────────┤
│  CLI (Typer)          │  Web UI (Streamlit)                     │
├─────────────────────────────────────────────────────────────────┤
│                    RAGPipeline (Controller)                      │
│  ┌─────────────────┐  ┌─────────────────┐                       │
│  │   Baseline      │  │   Corrective    │                       │
│  │   • 1 retrieval │  │   • ≤3 retrieval│                       │
│  │   • 1 gen + opt │  │   • Critic eval │                       │
│  │     abstention  │  │   • Dedup +     │                       │
│  │                 │  │     explicit    │                       │
│  │                 │  │     abstention  │                       │
│  └────────┬────────┘  └────────┬────────┘                       │
│           │                    │                                 │
│           └────────┬───────────┘                                 │
│                    ▼                                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Shared Components                       │    │
│  │  Embedder (E5) → ChromaDB (cosine) → SQLite (traces)      │    │
│  │  Ollama Client → llama.cpp (qwen2.5:3b / configurable)   │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Componentes clave

| Capa | Tecnología | Detalles |
|------|------------|----------|
| **Embeddings** | `sentence-transformers` | `intfloat/multilingual-e5-small` · 384 dims · prefijos `query:`/`passage:` · normalizados L2 |
| **Vector Store** | `chromadb` | Persistente · distancia coseno · colecciones versionadas |
| **Metadatos/Traza** | `sqlite3` (WAL) | Documentos, chunks, índices, runs, evaluaciones, anotaciones |
| **LLM Local** | `ollama` + `llama.cpp` | Modelo por defecto: `qwen2.5:3b` · soporta cualquier modelo GGUF vía Ollama |
| **Chunking** | Tokenizer E5 | Ventana 512 tokens · solape 50 · respeto a fronteras de oración |
| **CLI** | `typer` | 10 comandos tipados con validación Pydantic |
| **Web UI** | `streamlit` | Interfaz en español · comparación lado a lado · exportación CSV |
| **Evaluación** | Propia | Plantillas ciegas · bootstrap pareado (10k resamples) · intervalos IC95 |

---

## Los dos modos RAG

### Baseline (línea base)
```
Pregunta → Recuperación única (top-k) → Generación con permiso de abstención → Respuesta
```
- **Una sola ronda** de recuperación
- **Generación única** con instrucción: *"Si el contexto no basta, abstente: 'No hay evidencia suficiente'."*
- Sin validación posterior del contexto recuperado

### Correctivo (con bucle de corrección)
```
Pregunta → Recuperación → Crítico evalúa contexto → 
    ├─ Suficiente → Generación + abstención explícita → Respuesta
    └─ Insuficiente → Reintento (hasta 2) → Deduplicación → Generación → Respuesta
```
- **Hasta 3 recuperaciones** (1 inicial + 2 reintentos)
- **Crítico LLM** evalúa: *¿El contexto recuperado permite responder con evidencia?*
- **Deduplicación** de chunks por `doc_id` + `chunk_index`
- **Abstención explícita** diferenciada: *"Contexto insuficiente"* vs *"Evidencia contradictoria"*

### Contrato inmutable (ambos modos)
1. **Pregunta original inmutable** — nunca se reformula ni expande
2. **Máximo 3 recuperaciones** — hard limit
3. **Citas obligatorias** — apuntan a `chunk_id` existentes; `page` real o `null`
4. **JSON estructurado** — `{answer, citations, abstained, abstention_reason, confidence}` — pero **JSON válido ≠ evidencia suficiente**

---

## Modelo de lenguaje: Ollama + llama.cpp

EvidenciaLocal **no incluye pesos de modelo**. Usa **Ollama** como servidor de inferencia local, que a su vez ejecuta modelos **GGUF** mediante **llama.cpp**.

### Por qué Ollama + llama.cpp
| Ventaja | Detalle |
|---------|---------|
| **Cero configuración** | `ollama pull qwen2.5:3b` y listo |
| **GPU/CPU automático** | Offload parcial a VRAM si hay GPU (Metal/CUDA/ROCm) |
| **Quantización GGUF** | Modelos 4-bit (Q4_K_M) ≈ 2 GB RAM · calidad near-fp16 |
| **API OpenAI-compatible** | Cliente simple, streaming nativo, `tool_calls` |
| **Modelos intercambiables** | Cambia `LLM_MODEL` en config sin tocar código |

### Modelos probados / recomendados

| Modelo | Tamaño (Q4_K_M) | RAM mínima | Caso de uso |
|--------|-----------------|------------|-------------|
| `qwen2.5:3b` | ~2.0 GB | 4 GB | **Default** · equilibrado español/inglés · instrucciones |
| `qwen2.5:7b` | ~4.7 GB | 8 GB | Mayor razonamiento · contexto largo |
| `llama3.2:3b` | ~2.0 GB | 4 GB | Alternativa fuerte en inglés |
| `gemma2:2b` | ~1.6 GB | 3 GB | Muy ligero · buen seguimiento de formato |
| `phi3.5:3.8b` | ~2.3 GB | 4 GB | Excelente en razonamiento estructurado |

> **Nota**: El modelo `qwen2.5:3b` se usa en toda la documentación y configs por ser el *punto de partida reproducible*. Cualquier modelo GGUF servido por Ollama funciona cambiando `llm.model` en `configs/local.yaml` o la variable `LLM_MODEL`.

### Configuración de Ollama
```bash
# Instalación (Linux/macOS)
curl -fsSL https://ollama.ai/install.sh | sh

# Iniciar servidor (persistente)
ollama serve

# Descargar modelo por defecto
ollama pull qwen2.5:3b

# Ver modelos disponibles
ollama list
```

El cliente Ollama en EvidenciaLocal (`src/evidencialocal/llm/ollama_client.py`):
- Timeout configurable (default 120s)
- Reintentos con backoff exponencial
- Streaming token-a-token para UI
- Validación de respuesta JSON con esquema Pydantic
- Métricas de latencia y tokens (prompt/completion/total)

---

## Instalación

### Requisitos previos
- **Python 3.11+** (probado en 3.11, 3.12, 3.13)
- **Ollama** instalado y `ollama serve` ejecutándose
- **Modelo base**: `ollama pull qwen2.5:3b` (o el que prefieras)
- Opcional: GPU con drivers CUDA/Metal/ROCm para aceleración

### Instalación del paquete
```bash
# 1. Entrar al directorio
cd Self-RAG

# 2. Entorno virtual
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 3. Instalar en modo desarrollo (incluye deps de test/lint)
pip install -e ".[dev]"

# 4. Verificar instalación completa
evidencialocal doctor
# ✓ Python 3.11+
# ✓ Ollama reachable at http://localhost:11434
# ✓ Model qwen2.5:3b available
# ✓ ChromaDB writable
# ✓ SQLite writable
# ✓ Embeddings model loaded
```

### Configuración
```bash
# Variables de entorno (opcional)
cp .env.example .env
# Edita .env si necesitas: OLLAMA_HOST, LLM_MODEL, EMBEDDING_MODEL, etc.

# Configuración YAML (recomendado)
cp configs/default.yaml configs/local.yaml
# Edita configs/local.yaml para personalizar:
#   - chunk_size, chunk_overlap
#   - retrieval.top_k
#   - llm.temperature, llm.num_ctx
#   - paths de datos/índices
```

---

## Uso rápido

### 1. Verificar sistema
```bash
evidencialocal doctor
```

### 2. Generar datos de demostración (sintéticos, no tocan corpus real)
```bash
evidencialocal demo
# Crea: data/raw/*.pdf + datasets/demo/questions.jsonl
```

### 3. Indexar documentos
```bash
# Indexación básica (idempotente: solo añade nuevos)
evidencialocal index --input data/raw

# Con configuración personalizada
evidencialocal index --input data/raw --config configs/local.yaml

# Reindexar todo (borra índice anterior)
evidencialocal index --input data/raw --force-reindex

# Ver documentos indexados
evidencialocal list-docs
```

### 4. Hacer preguntas (CLI)
```bash
# Modo baseline
evidencialocal ask --mode baseline --question "¿Qué es el aprendizaje supervisado?"

# Modo correctivo
evidencialocal ask --mode corrective --question "¿Qué es el aprendizaje supervisado?"

# Ver contexto recuperado (debug)
evidencialocal ask --mode corrective --question "¿Qué es ML?" --show-context

# Salida JSON para piping
evidencialocal ask --mode corrective --question "..." --format json
```

### 5. Evaluación comparativa (núcleo científico)
```bash
# Dataset JSONL con preguntas (+ gold opcional)
evidencialocal evaluate \
  --dataset datasets/demo/questions.jsonl \
  --modes baseline,corrective \
  --output experiments/results/mi_experimento

# Exportar plantilla para anotación HUMANA (ciega al modo)
evidencialocal export-annotations \
  --results experiments/results/mi_experimento \
  --output experiments/results/mi_experimento/annotations.csv

# → Abre CSV, anota: correctness (0-1), faithfulness (0-1), citation_precision, citation_recall
# → El CSV NO revela qué modo generó cada respuesta

# Importar anotaciones completadas
evidencialocal import-annotations \
  --results experiments/results/mi_experimento \
  --annotations experiments/results/mi_experimento/annotations_completed.csv

# Generar reporte Markdown con bootstrap pareado (IC95%)
evidencialocal report \
  --results experiments/results/mi_experimento \
  --output experiments/results/mi_experimento/report.md
```

### 6. Interfaz Web (Streamlit)
```bash
streamlit run -m evidencialocal.ui
# Se abre en http://localhost:8501
# - Pregunta única con selector de modo
# - Comparación lado a lado (baseline vs corrective)
# - Visualización de chunks recuperados + scores
# - Exportación de resultados a CSV/JSON
```

---

## Formato de dataset para evaluación

Archivo **JSONL** (una muestra por línea):

```jsonl
{"id": "q1", "question": "¿Qué es el aprendizaje supervisado?", "gold_answer": "Es un paradigma donde...", "gold_citations": ["doc1#chunk3", "doc2#chunk1"]}
{"id": "q2", "question": "¿Cuáles son los tipos de aprendizaje supervisado?", "gold_answer": "Clasificación y regresión...", "gold_citations": ["doc1#chunk5"]}
```

| Campo | Requerido | Descripción |
|-------|-----------|-------------|
| `id` | Sí | Identificador único |
| `question` | Sí | Pregunta en lenguaje natural |
| `gold_answer` | No | Respuesta de referencia (para correctness) |
| `gold_citations` | No | Lista de `doc_id#chunk_index` relevantes (para citation recall) |

> **Sin gold**: la evaluación reporta *faithfulness*, *citation precision*, *latencia*, *tokens*, *abstention rate* — **no** computa recall/precision reales ni "exactitud" con similitud textual arbitraria.

---

## Métricas de evaluación

| Métrica | Rango | Tipo | Descripción |
|---------|-------|------|-------------|
| **Correctness** | 0–1 | Humana | Precisión factual vs gold (si hay gold) |
| **Faithfulness** | 0–1 | Humana | Fidelidad al contexto recuperado (no alucina) |
| **Citation Precision** | 0–1 | Humana | Citas correctas / total citadas |
| **Citation Recall** | 0–1 | Humana | Citas correctas / citas relevantes en gold |
| **Latencia p50 / p95 / media** | ms | Automática | Tiempo pregunta→respuesta |
| **Tokens (prompt/completion/total)** | int | Automática | Uso de contexto y generación |
| **Abstention rate** | 0–1 | Automática | % de respuestas abstenidas |
| **Bootstrap pareado** | IC95% | Estadística | Diferencias entre modos con 10k resamples |

---

## Desarrollo

```bash
# Tests unitarios + integración (con fakes, sin Ollama real)
pytest tests/ -v

# Solo unitarios (rápidos)
pytest tests/unit/ -v

# Linting (ruff)
ruff check src/ tests/
ruff format --check src/ tests/

# Auto-fix linting
ruff check --fix src/ tests/
ruff format src/ tests/

# Type checking (mypy strict)
mypy src/

# Pre-commit (opcional)
pre-commit install
pre-commit run --all-files
```

### Estructura de tests
```
tests/
├── unit/
│   ├── test_models.py       # Pydantic models validation
│   └── test_config.py       # Settings loading/validation
└── integration/
    └── test_pipeline.py     # RAGPipeline con fakes (no Ollama real)
```

---

## Estructura del proyecto

```
Self-RAG/
├── src/evidencialocal/
│   ├── config/          # Pydantic Settings (YAML + env vars)
│   ├── core/            # Modelos de dominio, excepciones, contratos
│   ├── embeddings/      # Embedder E5 (prefijos, normalización, batch)
│   ├── vectorstore/     # ChromaDB wrapper (CRUD, búsqueda, persistencia)
│   ├── metadata/        # SQLiteStore (WAL, migraciones, trazas)
│   ├── llm/
│   │   ├── ollama_client.py   # Cliente tipado, streaming, métricas
│   │   └── llama_client.py    # (Alternativo) llama-cpp-python directo
│   ├── retrieval/
│   │   ├── document_processor.py  # PDF→texto, chunking tokenizer-E5
│   │   └── retriever.py           # Búsqueda vectorial + filtros
│   ├── pipeline/
│   │   ├── indexer.py           # Indexación idempotente + manifest
│   │   └── rag_pipeline.py      # BaselinePipeline / CorrectivePipeline
│   ├── cli/
│   │   ├── main.py              # Grupo principal Typer
│   │   ├── evaluate.py          # Evaluación + bootstrap
│   │   └── demo.py              # Generación datos sintéticos
│   ├── eval/                    # Métricas, plantillas ciegas, bootstrap
│   └── ui/                      # Streamlit app (páginas, componentes)
├── tests/
│   ├── unit/
│   └── integration/
├── configs/
│   └── default.yaml             # Configuración por defecto (versionada)
├── data/
│   ├── raw/                     # PDFs de entrada (gitignored)
│   └── demo_questions.jsonl     # Preguntas de ejemplo (versionada)
├── datasets/
│   └── demo/                    # Datasets de evaluación (gitignored)
├── experiments/
│   └── results/                 # Resultados de runs (gitignored)
├── docs/
│   ├── architecture.md          # Detalle técnico + diagramas
│   ├── defense.md               # Defensa metodológica del proyecto
│   ├── implementation_status.md # Checklist fase 1/2
│   ├── research_protocol.md     # Protocolo experimental
│   └── usage.md                 # Guía de uso extendida
├── .env.example                 # Plantilla variables de entorno
├── .gitignore                   # Exclusiones completas
├── CLAUDE.md                    # Instrucciones para Claude Code
├── pyproject.toml               # Proyecto, deps, entry-points
├── requirements.lock.txt        # Lock generado (gitignored)
└── README.md                    # ← ESTE ARCHIVO
```

---

## Reglas de diseño (no negociables)

1. **Pregunta original inmutable** — máximo 3 recuperaciones totales
2. **Citas trazables** — `chunk_id` existente en BD; `page` entero o `null`
3. **JSON válido ≠ evidencia** — el crítico evalúa suficiencia, no sintaxis
4. **Simetría de fidelidad** — baseline y correctivo tienen *idénticas* reglas de abstención
5. **Coseno ≠ probabilidad** — umbrales de recuperación son heurísticos, no bayesianos
6. **Score LLM ≠ certeza** — `confidence` es auto-reportado, no calibrado
7. **Sin gold → sin recall/precision reales** — métricas proxy se etiquetan como tales
8. **Trazabilidad completa** — semilla, hash config, modelo, commit, latencia, tokens, RAM/VRAM
9. **Nada inventado** — ni bibliografía, ni salidas de test, ni funciones `TODO` vendidas como hechas

---

## Documentación extendida

| Documento | Descripción |
|-----------|-------------|
| [Arquitectura](docs/architecture.md) | Diagramas de flujo, decisiones técnicas, trade-offs |
| [Guía de uso](docs/usage.md) | Workflows completos, troubleshooting, tips |
| [Protocolo de investigación](docs/research_protocol.md) | Diseño experimental, cegamiento, análisis estadístico |
| [Defensa del proyecto](docs/defense.md) | Justificación metodológica, limitaciones, trabajo futuro |
| [Estado de implementación](docs/implementation_status.md) | Checklist fase 1 (núcleo/CLI) y fase 2 (UI/experimentos) |

---

## Configuración avanzada (`configs/local.yaml`)

```yaml
# configs/local.yaml (copia de default.yaml)
embedding:
  model_name: "intfloat/multilingual-e5-small"
  device: "auto"              # auto/cpu/cuda/mps
  batch_size: 32
  normalize: true
  query_prefix: "query: "
  passage_prefix: "passage: "

retrieval:
  top_k: 5
  score_threshold: null       # null = sin umbral duro
  metadata_filter: null

chunking:
  chunk_size: 512             # tokens (tokenizer E5)
  chunk_overlap: 50
  respect_sentence_boundary: true

llm:
  model: "qwen2.5:3b"         # cualquier modelo en `ollama list`
  temperature: 0.1
  top_p: 0.9
  num_ctx: 4096               # ventana de contexto
  timeout_seconds: 120
  max_retries: 3

pipeline:
  baseline:
    allow_abstention: true
  corrective:
    max_retries: 2
    critic_temperature: 0.0
    deduplicate_by: ["doc_id", "chunk_index"]
    explicit_abstention: true

paths:
  chroma_dir: "data/chroma"
  sqlite_path: "data/sqlite/metadata.db"
  index_manifest: "data/index_manifest.json"

eval:
  bootstrap_resamples: 10000
  confidence_level: 0.95
  blind_annotation: true
```

Variables de entorno (`.env`) tienen prioridad sobre YAML:
```bash
OLLAMA_HOST=http://localhost:11434
LLM_MODEL=qwen2.5:3b
EMBEDDING_MODEL=intfloat/multilingual-e5-small
CHROMA_DIR=data/chroma
SQLITE_PATH=data/sqlite/metadata.db
```

---

## Reproducibilidad experimental

Cada `evaluate` registra automáticamente en SQLite:
- `run_id` (UUID), `timestamp`, `git_commit` (si hay repo)
- `config_hash` (SHA256 de YAML+env efectiva)
- `embedding_model`, `llm_model`, `llm_params`
- `seed` (Python, numpy, torch si aplica)
- `hardware`: CPU, RAM, GPU, VRAM, OS
- Por muestra: latencia, tokens, modo, respuesta, citas, abstención

Esto permite **re-ejecutar exactamente** el mismo experimento y auditar diferencias.

---

## Lo que NO hay (por diseño)

| Excluido | Razón |
|----------|-------|
| LangChain / LangGraph | Abstracciones opacas, difícil depuración, dependencias pesadas |
| Docker obligatorio | Barrera de entrada; Ollama ya aísla el modelo |
| OCR / PDF complejos | `pypdf` suficiente para PDFs de texto; OCR es otro proyecto |
| Fine-tuning / LoRA | Fuera de alcance: RAG ≠ entrenamiento |
| Frontend separado (React/Vue) | Streamlit cubre necesidades de demo/evaluación |
| APIs de pago (OpenAI, Anthropic) | 100% local por principio |
| Métricas automáticas de "calidad" (BLEU, ROUGE, BERTScore) | No correlacionan con factualidad en RAG; requieren gold |

---

## Roadmap

### Fase 1: Núcleo completo y CLI reproducible 
- [x] Embeddings E5 + ChromaDB + SQLite
- [x] Pipeline Baseline + Corrective
- [x] CLI Typer con 10 comandos
- [x] Evaluación con bootstrap pareado + plantillas ciegas
- [x] Tests unitarios + integración (fakes)
- [x] Configuración YAML + env vars

### Fase 2: Interfaz, experimentos y entrega 
- [x] Streamlit UI (comparación lado a lado)
- [ ] Dataset demo curado (20+ preguntas con gold)
- [ ] Experimentos piloto documentados
- [ ] Reporte automático con gráficos (matplotlib/seaborn)
- [ ] Empaquetado `pip install evidencialocal` (PyPI)
- [ ] Documentación en ReadTheDocs

### Fase 3: Extensiones de investigación 
- [ ] Retrieval híbrido (BM25 + denso) + RRF
- [ ] Reranker cross-encoder local (bge-reranker-small)
- [ ] Métricas de calibración (ECE, reliability diagrams)
- [ ] Soporte multi-documento / multi-hop
- [ ] Benchmark contra RAGAS / TruLens (solo métricas, sin LLMs juez)

---

## Contribuir

1. Fork → rama `feature/xyz` o `fix/xyz`
2. `pre-commit run --all-files` pasa
3. Tests verdes: `pytest tests/ -v`
4. Type-check limpio: `mypy src/`
5. PR con descripción del cambio + issue relacionado

**No se aceptan**: dependencias cloud, wrappers LLM genéricos, métricas proxy sin validación humana, código sin tests.

---

## Licencia

**MIT** — úsalo, modifícalo, distribúyelo. Atribución apreciada.

---

> **EvidenciaLocal**: *Donde la recuperación se somete a juicio crítico antes de generar.*