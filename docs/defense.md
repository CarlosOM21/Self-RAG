# Defensa del Proyecto - EvidenciaLocal

## Contexto académico

Este proyecto se desarrolla como trabajo de fin de grado / máster / proyecto de investigación educativa. El objetivo es demostrar competencias en:
- Ingeniería de software (arquitectura limpia, testing, documentación)
- Sistemas de información / RAG (embeddings, vector stores, retrieval)
- Evaluación experimental rigurosa (diseño within-subjects, ceguera, bootstrap)
- Comunicación técnica (docs, CLI, UI en español)

## Preguntas esperadas y respuestas

### 1. "¿Por qué no usar LangChain / LlamaIndex / Haystack?"

**Respuesta**: Requisito explícito del enunciado. Además:
- Control total sobre el flujo de datos (crítico para comparar modos fairly)
- Sin dependencias pesadas ni "magic" opaco
- Código propio auditable línea por línea
- Facilita entender qué hace cada componente (valor educativo)

### 2. "¿Por qué ChromaDB y no FAISS / Pinecone / Weaviate?"

**Respuesta**:
- ChromaDB: Python nativo, persistente, sin servidor separado, distancia coseno nativa
- FAISS: Requiere gestión manual de índices + metadatos aparte
- Pinecone/Weaviate: Servicios gestionados (cloud), violan "100% local"
- ChromaDB suficiente para escala educativa (<100k chunks)

### 3. "¿Por qué Sentence Transformers E5 small y no OpenAI embeddings / Cohere / BGE?"

**Respuesta**:
- **Local obligatorio**: Sin APIs de pago, sin nube
- **Multilingüe**: E5-small soporta español nativamente
- **Dimensión 384**: Balance velocidad/calidad, cabe en CPU
- **Prefijos query:/passage:**: Diseño E5 para retrieval asimétrico
- **Normalizado**: Coseno = dot product, eficiente en Chroma

### 4. "¿Por qué Ollama y no vLLM / TGI / llama.cpp directo?"

**Respuesta**:
- Ollama: Gestión de modelos simple, API OpenAI-compatible, multiplexing
- vLLM/TGI: Overkill para modelo 3B, requieren GPU CUDA
- llama.cpp: Más bajo nivel, más código boilerplate
- Requisito: "Ollama local con qwen2.5:3b como candidato inicial"

### 5. "¿El modo correctivo es solo 'retrieve de nuevo si no gusta'?"

**Respuesta**: No. Diferencias clave:
- **Crítica estructurada**: LLM evalúa *suficiencia* con esquema JSON (sufficient, reason, missing_aspects, retry_query)
- **Deduplicación**: No recupera chunks ya vistos (evita bucles)
- **Pregunta original inmutable**: Siempre se usa la misma pregunta, no se reformula
- **Máximo 2 reintentos**: Límite duro, no bucle infinito
- **Abstención explícita**: Si tras reintentos sigue insuficiente, se abstené con razón

### 6. "¿Cómo garantizan que la comparación es justa (fair)?"

**Respuesta**:
- **Mismo prompt generador** en ambos modos (solo cambia contexto)
- **Mismas instrucciones de abstención** en ambos modos
- **Mismo top-k, mismo embedding, mismo LLM, misma temperatura**
- **Orden alternado** por pregunta (evita sesgo orden)
- **Plantilla anotación ciega al modo** (anotadores no saben qué modo)
- **Configuración congelada** (manifiesto con hash)
- **Semilla fija** para reproducibilidad

### 7. "¿Por qué no miden 'accuracy' automática con ROUGE/BERTScore?"

**Respuesta**: Porque **no hay gold standard confiable para RAG abierto**.
- ROUGE/BERTScore miden similitud textual, no corrección factual
- Una respuesta correcta pero distinta redacción → score bajo
- Una respuesta incorrecta pero similar léxicamente → score alto
- **Solo anotación humana experta** valida correctness/faithfulness en RAG
- Sin gold → métricas automáticas = NA (no se inventan números)

### 8. "¿Qué validez tiene el experimento con n=50 preguntas?"

**Respuesta**: 
- Potencia 0.80 para d=0.3 (efecto pequeño-medio) con α=0.05 bilateral
- Within-subjects reduce varianza inter-sujeto → mayor potencia
- Bootstrap pareado (1000 muestras) da IC robustos sin normalidad
- **Limitación declarada**: No generaliza a otros modelos/corpus/dominios
- Objetivo: **Evidencia interna válida**, no claim de generalidad externa

### 9. "¿El sistema 'sabe' cuándo no sabe?"

**Respuesta**: 
- **Abstención operativa**: Ambos modos pueden responder `"abstained": true` con razón
- **No es confianza calibrada**: Score LLM ≠ probabilidad de acierto
- **Heurística**: "Contexto no contiene info suficiente para responder"
- **Validación**: Anotadores marcan "abstención apropiada" (sí/no)
- **Métrica**: Appropriate Abstention Rate

### 10. "¿Cómo manejan citas falsas / alucinación de UUIDs?"

**Respuesta**:
- **Validación post-generación**: Solo se aceptan citas que estén en `retrievals` de esa ronda
- **Filtrado automático**: `citations = [c for c in citations if c in valid_citation_ids]`
- **Trazabilidad**: Cada respuesta guarda `retrievals` completos para auditoría
- **Métrica**: Citation Precision mide % citas válidas

### 11. "¿Por qué SQLite y no JSON files / DuckDB / PostgreSQL?"

**Respuesta**:
- SQLite: Zero-config, ACID, WAL mode (concurrencia lectores), portable, estándar
- JSON: No transaccional, corruption risk, no queries eficientes
- DuckDB: OLAP, overkill para metadatos transaccionales
- PostgreSQL: Requiere servidor, violan "local simple"

### 12. "¿El chunking por tokenizer E5 es necesario?"

**Respuesta**: 
- **Sí**. Chunking por caracteres/palabras ≠ tokens del modelo embedding
- E5 tiene vocabulario propio; chunking por tokens garantiza:
  - Chunks caben en ventana de contexto (512 tokens)
  - Overlap real en espacio token (no carácter)
  - Consistencia embedding/retrieval/generación
- `transformers.AutoTokenizer` del mismo modelo asegura alineación

### 13. "¿Qué pasa si el PDF es escaneado (imagen)?"

**Respuesta**: 
- **Fuera de alcance**: Requisito explícito "No OCR"
- `pypdf` extrae solo texto nativo
- Documentos escaneados → 0 páginas extraídas → status FAILED con error
- Usuario debe pre-procesar con OCR externo si necesita

### 14. "¿Cómo versionan el índice?"

**Respuesta**:
- **IndexManifest** en SQLite (tabla key-value JSON)
- Contiene: config hash, embedding model+dim, chunk_size, tokenizer, docs con hash+status
- **Idempotencia**: `content_hash` (SHA256 PDF) → salta si idéntico y READY
- **Incompatibilidad**: Al indexar, verifica config vs manifiesto → error si difiere (salvo `--force-reindex`)

### 15. "¿Por qué Streamlit y no FastAPI + React / Gradio / Chainlit?"

**Respuesta**:
- **Requisito**: "Streamlit, sin frontend separado"
- Streamlit: Python puro, reactivo, deploy trivial, widgets nativos
- Gradio: Más orientado a demos ML, menos control layout
- FastAPI+React: Dos proyectos, build step, overkill para UI interna

### 16. "¿El sistema es 'privado' / 'confidencial'?"

**Respuesta**: 
- **Local ≠ Privado automático**: 
  - Código corre en tu máquina, datos no salen *por diseño*
  - PERO: Claude Code (agente desarrollo) ve tu código y datos si los pasas al chat
  - Ollama: Verifica que no telemetría (es open source)
  - ChromaDB: `anonymized_telemetry=False` en config
- **No garantía legal**: Para datos sensibles, revisar código completo y aislamiento de red

### 17. "¿Cuál es la contribución novel?"

**Respuesta**: 
- **No hay claim de novedad científica** (arquitecturas RAG correctivas existen en literatura)
- **Contribución ingenieril/educativa**: 
  - Implementación limpia, comparable, auditable de baseline vs correctivo
  - Protocolo evaluación riguroso (ciega, bootstrap, manifiesto)
  - Stack 100% local reproducible sin dependencias cloud
  - Documentación en español para contexto académico hispanohablante

### 18. "¿Qué harían con más tiempo/recursos?"

**Respuesta** (honestidad):
- Evaluar más modelos LLM (Llama 3.1, Phi-3, Gemma 2) y embeddings (BGE, E5-large)
- Añadir reranker (cross-encoder) entre retrieval y generación
- Implementar hybrid search (BM25 + vector)
- UI: historial conversacional, multi-documento, exportar conversación
- Tests: property-based (Hypothesis), mutation testing
- CI/CD: GitHub Actions con matrix Python/OS

## Checklist de entrega

- [x] Código fuente completo, instalable (`pip install -e .`)
- [x] Tests unitarios + integración (offline con fakes)
- [x] CLI con todos los comandos especificados
- [x] Streamlit UI funcional en español
- [x] Documentación: README, architecture, usage, research_protocol, defense, implementation_status
- [x] Configuración por defecto + .env.example
- [x] Requirements.lock.txt (tras `pip freeze` en entorno probado)
- [x] Dataset demo + preguntas sintéticas
- [x] Scripts de evaluación end-to-end

## Limitaciones declaradas (no ocultar)

1. Solo probado con `qwen2.5:3b` y `intfloat/multilingual-e5-small`
2. Solo corpus en español (PDFs de prueba)
3. Evaluación requiere anotación humana (costoso, subjetivo)
4. Sin reranker, sin hybrid search, sin query expansion
5. Chunking simple (sliding window), sin estructura semántica
6. No manejo de tablas/imágenes en PDFs
7. No autenticación / multi-usuario en UI
8. No métricas de confianza calibrada (conformal prediction, etc.)