# Guía de Uso de EvidenciaLocal

## Instalación paso a paso

### Linux (Ubuntu/Debian/Arch/CachyOS)

```bash
# 1. Instalar dependencias del sistema
# Ubuntu/Debian:
sudo apt update && sudo apt install -y python3.11 python3.11-venv python3.11-dev

# Arch/CachyOS:
sudo pacman -S python python-virtualenv

# 2. Instalar Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 3. Iniciar servicio Ollama (systemd)
sudo systemctl enable --now ollama

# 4. Descargar modelo por defecto
ollama pull qwen2.5:3b

# 5. Clonar/entrar al proyecto
cd Self-RAG

# 6. Entorno virtual
python3.11 -m venv .venv
source .venv/bin/activate

# 7. Instalar paquete
pip install --upgrade pip
pip install -e ".[dev]"

# 8. Verificar
evidencialocal doctor
```

### Windows (PowerShell)

```powershell
# 1. Instalar Python 3.11+ desde python.org o winget
winget install Python.Python.3.11

# 2. Instalar Ollama
winget install Ollama.Ollama

# 3. Descargar modelo
ollama pull qwen2.5:3b

# 4. Entrar al proyecto
cd Self-RAG

# 5. Entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 6. Instalar paquete
pip install --upgrade pip
pip install -e ".[dev]"

# 7. Verificar
evidencialocal doctor
```

### Verificación de instalación

```bash
# Debe mostrar todo en verde
evidencialocal doctor

# Salida esperada:
# Python: 3.11.x
# Ollama: ✓ Disponible (Modelo: qwen2.5:3b @ http://localhost:11434)
# ChromaDB: ✓ OK (X chunks indexados)
# SQLite: ✓ OK
# Embeddings: ✓ OK (intfloat/multilingual-e5-small dim=384)
```

## Flujo de trabajo típico

### 1. Preparar documentos

```bash
# Opción A: Usar documentos propios
mkdir -p data/raw
cp /ruta/a/tus/documentos/*.pdf data/raw/

# Opción B: Generar datos demo (sintéticos, educativos)
evidencialocal demo
```

### 2. Indexar

```bash
# Indexación incremental (idempotente - salta ya indexados)
evidencialocal index --input data/raw

# Con configuración personalizada
evidencialocal index --input data/raw --config configs/local.yaml

# Reindexación completa (borra todo lo anterior)
evidencialocal index --input data/raw --force-reindex
```

### 3. Consultar

```bash
# Pregunta simple baseline
evidencialocal ask --mode baseline --question "¿Qué es el aprendizaje supervisado?"

# Pregunta simple correctivo
evidencialocal ask --mode corrective --question "¿Qué es el aprendizaje supervisado?"

# Ver chunks recuperados
evidencialocal ask --mode corrective --question "¿Qué es ML?" --show-context

# Cambiar top-k
evidencialocal ask --mode baseline --question "¿Qué es overfitting?" --top-k 10
```

### 4. Evaluar (comparativa)

```bash
# Preparar dataset (JSONL)
cat > datasets/mi_eval.jsonl << 'EOF'
{"id": "q1", "question": "¿Qué es el aprendizaje supervisado?", "gold_answer": "Aprende de datos etiquetados..."}
{"id": "q2", "question": "Diferencia entre clasificación y regresión", "gold_answer": "Clasificación: categorías. Regresión: continuo."}
EOF

# Ejecutar evaluación
evidencialocal evaluate \
  --dataset datasets/mi_eval.jsonl \
  --modes baseline,corrective \
  --output experiments/results/mi_eval

# Exportar plantilla para anotación humana (ciega al modo)
evidencialocal export-annotations \
  --results experiments/results/mi_eval \
  --output experiments/results/mi_eval/annotations.csv

# Editar annotations.csv con anotaciones humanas (correctness, faithfulness, etc.)

# Importar anotaciones
evidencialocal import-annotations \
  --results experiments/results/mi_eval \
  --annotations experiments/results/mi_eval/annotations.csv

# Generar reporte final
evidencialocal report \
  --results experiments/results/mi_eval \
  --output experiments/results/mi_eval/report.md
```

### 5. Interfaz Web

```bash
# Iniciar servidor local (por defecto localhost:8501)
streamlit run -m evidencialocal.ui

# O con puerto personalizado
streamlit run -m evidencialocal.ui --server.port 8502 --server.address 0.0.0.0
```

## Configuración avanzada

### Variables de entorno (.env)

```bash
# Modelo LLM diferente
OLLAMA__MODEL=llama3.1:8b
OLLAMA__TEMPERATURE=0.0

# Embeddings diferentes
EMBEDDING__MODEL_NAME=intfloat/multilingual-e5-base
EMBEDDING__DIMENSION=768

# Directorios personalizados
CHROMA__PERSIST_DIRECTORY=/mnt/data/chroma
SQLITE__DATABASE_PATH=/mnt/data/sqlite/metadata.db
```

### Archivo YAML (configs/local.yaml)

```yaml
embedding:
  model_name: "intfloat/multilingual-e5-small"
  dimension: 384

ollama:
  model: "qwen2.5:3b"
  temperature: 0.1

indexing:
  chunk_size: 512
  chunk_overlap: 50

retrieval:
  top_k: 5

corrective:
  max_retries: 2
```

Uso:
```bash
evidencialocal index --input data/raw --config configs/local.yaml
evidencialocal ask --mode corrective --question "..." --config configs/local.yaml
```

## Solución de problemas comunes

### Error: "ModelNotFoundError: No se pudo cargar el modelo de embeddings"

```bash
# Verificar conexión a internet (primera descarga)
# O usar modelo cacheado
ls ~/.cache/huggingface/hub/models--intfloat--multilingual-e5-small/
```

### Error: "Ollama no disponible" / "Connection refused"

```bash
# Verificar servicio
systemctl status ollama  # Linux
# Windows: servicios.msc → Ollama

# Reiniciar
sudo systemctl restart ollama

# Verificar puerto
curl http://localhost:11434/api/tags
```

### Error: "ChromaDB: Error en recuperación"

```bash
# Revisar permisos directorio
ls -la data/chroma/

# Reindexar si corrupto
evidencialocal index --input data/raw --force-reindex
```

### Error: "SQLite: database is locked"

```bash
# Solo una instancia a la vez
# Verificar que no hay procesos zombie
pkill -f "evidencialocal"
pkill -f "streamlit"
```

### Out of Memory (OOM) en embeddings

```bash
# Reducir batch size
export EMBEDDING__BATCH_SIZE=16

# Forzar CPU
export EMBEDDING__DEVICE=cpu
```

### Modelo LLM muy lento

```bash
# Usar modelo más pequeño
ollama pull qwen2.5:1.5b
export OLLAMA__MODEL=qwen2.5:1.5b

# O reducir max_tokens
export OLLAMA__MAX_TOKENS=1024
```

## Comandos de mantenimiento

```bash
# Listar documentos indexados
evidencialocal list-docs

# Listar solo fallidos
evidencialocal list-docs --status failed

# Eliminar documento específico
evidencialocal delete-index --yes  # Elimina TODO

# Ver estadísticas índice
evidencialocal list-docs  # muestra total docs/chunks en header
```

## Formato de archivos

### Dataset evaluación (JSONL)

```jsonl
{"id": "unique_id", "question": "Pregunta en español", "gold_answer": "Respuesta de referencia opcional", "gold_citations": ["uuid1", "uuid2"]}
{"id": "q2", "question": "Otra pregunta", "gold_answer": null, "gold_citations": []}
```

### Plantilla anotaciones (CSV generado)

```csv
sample_id,question,predicted_answer,predicted_citations,abstained,latency_ms,retrieval_count,critique_count,correctness,faithfulness,citation_precision,citation_recall,annotator_notes
q1,"¿Qué es ML?","Respuesta del modelo","uuid1;uuid2",False,1250,1,0,,,
```

Valores para anotación:
- `correctness`: 0.0 a 1.0 (precisión factual)
- `faithfulness`: 0.0 a 1.0 (fidelidad al contexto)
- `citation_precision`: 0.0 a 1.0
- `citation_recall`: 0.0 a 1.0

## Mejores prácticas

1. **Usa siempre el mismo `--config`** para indexar y consultar
2. **Ejecuta `doctor`** antes de sesiones importantes
3. **Guarda el `config_hash`** del experimento (está en el manifiesto)
4. **Anota en ciego**: usa `export-annotations` que oculta el modo
5. **Registra semillas**: `EVALUATION__RANDOM_SEED=42` para reproducibilidad
6. **No mezcles corpus**: usa directorios separados por experimento