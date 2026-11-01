# Guía de uso de EvidenciaLocal

Esta guía describe el funcionamiento actual: PDFs reales, inferencia directa con `llama-cpp-python` y los modos Baseline/Correctivo. No es necesario instalar ni ejecutar Ollama.

## 1. Instalación

### Linux o macOS

```bash
cd Self-RAG
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### Windows (PowerShell)

```powershell
cd Self-RAG
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

La compilación de `llama-cpp-python` puede requerir las herramientas de compilación habituales del sistema si no existe una wheel para la versión de Python utilizada.

## 2. Configurar el modelo GGUF

Por defecto se busca este archivo en la raíz del proyecto:

```text
qwen2.5-3b-instruct-q4_k_m.gguf
```

Para usar otra ubicación:

```bash
export LLAMA_CPP__MODEL_PATH=/ruta/absoluta/al/modelo.gguf
```

En PowerShell:

```powershell
$env:LLAMA_CPP__MODEL_PATH="C:\ruta\al\modelo.gguf"
```

También puedes copiar `.env.example` a `.env` y editar allí los valores. El GGUF no se sube a Git.

## 3. Diagnóstico

```bash
evidencialocal doctor
```

Se comprueban:

1. versión de Python;
2. existencia del modelo GGUF;
3. acceso a ChromaDB y número de chunks;
4. apertura de SQLite;
5. carga del modelo de embeddings y su dimensión.

Con detalle de configuración:

```bash
evidencialocal doctor --verbose
```

## 4. Preparar e indexar el corpus

Solo se procesan PDFs con texto extraíble. No hay generación de PDFs de demostración ni OCR integrado.

```bash
mkdir -p data/raw
cp /ruta/a/tus/documentos/*.pdf data/raw/
evidencialocal index --input data/raw
```

También se puede indexar cualquier otro directorio:

```bash
evidencialocal index --input /ruta/a/corpus
```

La indexación incremental:

- ordena los PDF para que el proceso sea determinista;
- calcula SHA-256 de los bytes de cada archivo;
- omite documentos listos con el mismo hash;
- extrae texto por páginas;
- crea chunks con solape;
- genera embeddings por lote;
- guarda vectores en Chroma y metadatos en SQLite;
- actualiza el manifiesto.

Comandos relacionados:

```bash
evidencialocal list-docs
evidencialocal list-docs --status ready
evidencialocal list-docs --status failed
```

Reindexación completa:

```bash
evidencialocal index --input data/raw --force-reindex
```

Borrar todo el índice **sin** volver a indexar:

```bash
evidencialocal delete-index
# no interactivo:
evidencialocal delete-index --yes
```

## 5. Realizar preguntas

La pregunta se escribe como argumento posicional, no con `--question`:

```bash
evidencialocal ask "¿Qué hizo Turing para la IA?" --mode baseline
evidencialocal ask "¿Qué hizo Turing para la IA?" --mode corrective
```

Mostrar la evidencia:

```bash
evidencialocal ask "¿Qué es el test de Turing?" \
  --mode corrective \
  --top-k 5 \
  --show-context
```

### Qué esperar de Baseline

- una sola recuperación;
- una llamada de generación;
- ninguna crítica;
- abstención si no hay contexto o el modelo considera que no basta.

### Qué esperar de Correctivo

- una recuperación inicial;
- una crítica de suficiencia por ronda procesada;
- hasta dos recuperaciones adicionales;
- uso efectivo de `retry_query` para buscar evidencia nueva;
- deduplicación por `chunk_id`;
- una única generación final;
- abstención si finalmente la evidencia no basta.

El número de críticas puede ser menor que el máximo: si la primera evidencia ya es suficiente, no hay reintentos. Que el modo correctivo no repita una búsqueda en una pregunta concreta no significa que el bucle esté desactivado.

## 6. Interfaz web

```bash
streamlit run src/evidencialocal/ui/streamlit_app.py
```

Para otro puerto:

```bash
streamlit run src/evidencialocal/ui/streamlit_app.py --server.port 8502
```

Flujo recomendado:

1. abrir **Inicio** y comprobar que el GGUF está disponible;
2. entrar en **Documentos**;
3. subir PDFs o indicar un directorio;
4. indexar;
5. entrar en **Consultar**;
6. elegir modo y escribir la pregunta;
7. revisar respuesta, fuentes citadas y, si se desea, evidencia y críticas.

La última respuesta permanece visible durante la sesión de Streamlit para que cambiar los controles no obligue a repetir la inferencia.

## 7. Configuración YAML y entorno

Usa `configs/default.yaml` como plantilla:

```bash
cp configs/default.yaml configs/local.yaml
```

Ejemplo mínimo:

```yaml
embedding:
  device: cpu
  batch_size: 16

llama_cpp:
  n_threads: 8
  n_ctx: 8192
  max_tokens: 512

retrieval:
  top_k: 5

corrective:
  max_retries: 2
  max_context_chunks: 8
```

Aplicar el mismo archivo al indexar y consultar:

```bash
evidencialocal index --input data/raw --config configs/local.yaml
evidencialocal ask "¿Qué es RAG?" --mode corrective --config configs/local.yaml
```

Precedencia:

```text
defaults < YAML < .env < variables de entorno del proceso
```

Ejemplos de entorno:

```bash
export EMBEDDING__DEVICE=cpu
export EMBEDDING__BATCH_SIZE=16
export LLAMA_CPP__N_THREADS=8
export RETRIEVAL__TOP_K=5
export CORRECTIVE__MAX_RETRIES=2
```

Al cambiar embeddings, dimensión, tokenizer, tamaño o solape de chunks, utiliza `--force-reindex`. El indexador rechaza configuraciones incompatibles con un manifiesto existente para evitar mezclar vectores incompatibles.

## 8. Evaluación

Formato JSONL:

```jsonl
{"id":"q1","question":"¿Qué hizo Turing para la IA?","gold_answer":"Referencia opcional","gold_citations":[]}
{"id":"q2","question":"¿Qué es la máquina de Turing?","gold_answer":null,"gold_citations":[]}
```

Ejecutar:

```bash
evidencialocal evaluate \
  --dataset datasets/mi_dataset.jsonl \
  --modes baseline,corrective \
  --output experiments/results/mi_experimento
```

Limitar muestras durante una prueba:

```bash
evidencialocal evaluate \
  --dataset datasets/mi_dataset.jsonl \
  --modes baseline,corrective \
  --output experiments/results/prueba \
  --limit 2
```

Anotar y reportar:

```bash
evidencialocal export-annotations \
  --results experiments/results/mi_experimento \
  --output experiments/results/mi_experimento/annotations.csv

# Completar el CSV con valores entre 0 y 1 y conservar annotation_id.

evidencialocal import-annotations \
  --results experiments/results/mi_experimento \
  --annotations experiments/results/mi_experimento/annotations.csv

evidencialocal report \
  --results experiments/results/mi_experimento \
  --output experiments/results/mi_experimento/report.md
```

Métricas de anotación:

- `correctness`: corrección respecto a la referencia o conocimiento verificado;
- `faithfulness`: fidelidad de la respuesta a la evidencia;
- `citation_precision`: proporción de citas pertinentes entre las emitidas;
- `citation_recall`: cobertura de las citas relevantes disponibles o marcadas como gold.

El bootstrap pareado compara muestras con el mismo `sample_id` cuando ambos modos tienen valores para una métrica. No se inventan métricas si no hay anotaciones.

## 9. Resolución de problemas

### El GGUF no aparece como disponible

```bash
ls -l qwen2.5-3b-instruct-q4_k_m.gguf
export LLAMA_CPP__MODEL_PATH=/ruta/correcta/modelo.gguf
evidencialocal doctor
```

La ruta relativa se resuelve desde la raíz del proyecto, no desde el directorio arbitrario donde se haya abierto una terminal.

### Falla la carga del modelo de embeddings

La primera descarga necesita acceso a Hugging Face. Si el modelo ya está en caché, comprueba que la caché sea accesible. Para depurar permisos o espacio:

```bash
df -h .
export EMBEDDING__DEVICE=cpu
evidencialocal doctor
```

### Un PDF no produce chunks

Comprueba que tenga texto seleccionable. Un escaneo de imágenes necesita OCR externo antes de indexarlo. Los PDF criptográficamente protegidos o dañados también pueden fallar en `pypdf`.

### La respuesta es una abstención

Revisa `--show-context`:

- si no hay chunks, el corpus no tiene evidencia recuperable;
- si los chunks son irrelevantes, prueba otro `top_k`, revisa el corpus o ajusta el chunking y reindexa;
- en Correctivo, revisa las razones y `retry_query` de las críticas.

El sistema no utiliza conocimiento externo para rellenar lagunas del corpus.

### La generación es lenta

En Correctivo hay al menos una crítica y una generación, por lo que suele tardar más que Baseline. Opciones:

```bash
export LLAMA_CPP__N_THREADS=8
export LLAMA_CPP__MAX_TOKENS=384
export RETRIEVAL__TOP_K=3
export CORRECTIVE__MAX_RETRIES=1
export EMBEDDING__DEVICE=cuda   # solo si está disponible y configurada
```

Reduce valores si falta RAM/VRAM. El tiempo exacto depende del hardware, GGUF, longitud del contexto y número de rondas.

### Chroma o SQLite están en mal estado

Haz copia de cualquier resultado que necesites y reconstruye:

```bash
evidencialocal index --input data/raw --force-reindex
```

No borres manualmente solo una de las dos bases: Chroma y SQLite contienen información complementaria.

### La indexación no cambia después de editar un PDF

El hash se calcula sobre los bytes. Guarda realmente el nuevo archivo y vuelve a ejecutar `index`. Un archivo modificado recibe otro hash y se procesa como una nueva versión; si quieres reconstruir un corpus limpio, usa `--force-reindex`.

## 10. Comprobaciones para desarrollo

```bash
pytest -q
ruff check src tests
ruff format --check src tests
python -m compileall -q src tests
```

