# Arquitectura de EvidenciaLocal

## Visión general

EvidenciaLocal separa dos procesos: **indexación offline** del corpus y **consulta online** en dos modos. Toda la inferencia se ejecuta localmente.

```text
┌──────────────────────────────────────────────────────────────┐
│                    CLI Typer / Streamlit                     │
└──────────────────────────────┬───────────────────────────────┘
                               │ AskRequest
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                         RAGPipeline                          │
│                                                              │
│ Baseline:   retrieve → generate → validate → respond         │
│ Correctivo: retrieve → critique ↺ retrieve(new query)        │
│                           → generate once → validate         │
└───────────────┬───────────────────┬──────────────────┬───────┘
                │                   │                  │
                ▼                   ▼                  ▼
        ┌──────────────┐    ┌──────────────┐   ┌──────────────┐
        │ E5 Embedder  │    │  ChromaDB    │   │ llama.cpp    │
        │ query/passage│    │ vectors+text │   │ GGUF local   │
        └──────────────┘    └──────────────┘   └──────────────┘
                                   │                  │
                                   ▼                  ▼
                            ┌──────────────┐   ┌──────────────┐
                            │   SQLite     │   │ JSON schemas │
                            │ metadata +   │   │ answer +     │
                            │ traces/eval  │   │ critique     │
                            └──────────────┘   └──────────────┘
```

## Indexación

```text
PDF
  → Indexer.index_directory
  → SHA-256(bytes) para idempotencia
  → DocumentProcessor.extract_pdf_pages
  → chunk_text_by_tokens por página
  → Embedder.encode_passages en lote
  → ChromaStore.add_chunks
  → SQLiteStore.upsert_document + upsert_chunks_metadata
  → IndexManifest en SQLite
```

Cada chunk tiene un UUID propio, UUID del documento, nombre de archivo, página, índice, número de tokens, posiciones aproximadas de caracteres y hash del texto. El UUID del chunk es la clave que une el vector, el texto y las citas.

## Consulta Baseline

1. `AskRequest` valida y normaliza la pregunta.
2. `Retriever.retrieve_for_baseline` aplica el prefijo E5 `query:` y busca en Chroma.
3. Chroma devuelve distancia coseno; `ChromaStore` la transforma en similitud `1 - distancia`.
4. Se limita el contexto a `max_context_chunks` para proteger la ventana del LLM.
5. `LlamaClient.generate_answer` construye mensajes system/user y solicita JSON con esquema.
6. El pipeline extrae y filtra citas: solo se aceptan UUID presentes en el contexto enviado.
7. Si no hay evidencia o la respuesta no es válida, se produce una abstención explícita.
8. Se guarda una traza en SQLite y se devuelve `AskResponse`.

## Consulta Correctiva

```text
previous_queries = [pregunta original]

para intento = 0 .. max_retries:
    si intento = 0:
        recuperar(pregunta original)
    si no:
        recuperar(previous_queries[-1])
        ampliar candidatos, eliminar IDs ya vistos y cortar a top-k

    si no existe evidencia acumulada:
        abstenerse sin llamar al LLM
    si una ronda posterior no aporta chunks:
        salir del bucle y generar con la evidencia acumulada

    seleccionar contexto acotado con participación de todas las rondas
    critique_context(pregunta original, contexto)

    si suficiente o último intento:
        salir del bucle
    si no:
        añadir retry_query (o fallback con missing_aspects) a previous_queries

generate_answer(pregunta original, mejor contexto acumulado)
validar respuesta y citas
guardar traza
```

Diferencias importantes respecto a un bucle ingenuo:

- la reformulación modifica la **búsqueda**, nunca la pregunta usada para responder;
- se sobre-recuperan candidatos porque el top-k nuevo podría estar ocupado por duplicados;
- no se genera una respuesta tentativa en cada ronda;
- si no hay chunks nuevos, todavía se ejecuta la generación final con la evidencia disponible;
- una respuesta formada solo por `[DOC_N ID:...]` se considera inválida, no una respuesta visible.

## Selección y límite de contexto

`RAGPipeline._select_context` recorre las rondas por ranking de forma entrelazada: primer chunk de cada ronda, segundo de cada ronda, etc. Así evita que la última reformulación desplace toda la evidencia inicial o que la inicial impida usar los hallazgos nuevos.

`CORRECTIVE__MAX_CONTEXT_CHUNKS` limita el número de chunks enviados al modelo. El valor por defecto es 8; con chunks de hasta 512 tokens, `n_ctx=8192` y `max_tokens=512`, deja margen para instrucciones y metadatos. Los resultados completos siguen disponibles en la traza aunque el contexto generativo esté acotado.

## Contratos JSON

Respuesta:

```json
{
  "answer": "Texto completo en lenguaje natural",
  "citations": ["uuid-de-un-chunk"],
  "abstained": false,
  "abstention_reason": null
}
```

Crítica:

```json
{
  "sufficient": false,
  "reason": "Falta una contribución concreta",
  "missing_aspects": ["test de Turing"],
  "retry_query": "test de Turing contribución inteligencia artificial"
}
```

`LlamaClient` valida campos, tipos y contenido mínimo. Una salida malformada o una respuesta no abstenida formada solo por una cita provoca un nuevo intento con una instrucción de corrección. Después de los intentos se eleva `GenerationError`; no se presenta JSON crudo como si fuera una respuesta.

## Persistencia

### ChromaDB

- Colección persistente `evidencialocal`.
- Espacio coseno (`hnsw:space=cosine`).
- Documento: texto del chunk.
- Embedding: lista de 384 valores con el modelo por defecto.
- Metadata: procedencia y posición del chunk.

### SQLite

Tablas principales:

- `documents`: identidad, ruta, hash, estado y páginas;
- `chunks`: metadatos y relación con documentos;
- `index_manifest`: configuración y totales serializados;
- `traces`: consulta, respuesta, rondas, críticas, latencias y tokens;
- `evaluation_results`: resultados por muestra y modo;
- `experiment_manifest`: configuración y progreso de evaluaciones.

SQLite usa WAL si está habilitado y claves foráneas siempre. Las escrituras de metadatos de chunks de un documento se agrupan en una transacción.

## Configuración y rutas

- Defaults: `Settings` y clases anidadas en `config/settings.py`.
- YAML: solo cuando la CLI recibe `--config`.
- Entorno: `.env` y variables del proceso con separador `__`.
- Precedencia: defaults < YAML < `.env` < entorno.
- `data/chroma`: vectores persistentes.
- `data/sqlite/metadata.db`: metadatos y trazas.
- `data/raw`: corpus PDF local.
- `experiments/results`: resultados de evaluación.
- GGUF relativo: resuelto desde la raíz del proyecto.
