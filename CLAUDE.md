# EvidenciaLocal - CLAUDE.md

## Contrato del Proyecto

**EvidenciaLocal** es un sistema RAG educativo local con dos modos comparables:
- **Baseline**: RAG estándar con recuperación única y generación con permiso de abstención
- **Correctivo**: RAG con evaluación crítica del contexto recuperado, hasta 2 reintentos de recuperación, deduplicación y abstención explícita

### Especificaciones Técnicas

- **Embeddings**: Sentence Transformers `intfloat/multilingual-e5-small` (384 dims), prefijos `query:`/`passage:`, normalizados
- **Vector DB**: Chroma persistente con distancia coseno
- **Metadatos/Trazas**: SQLite propio
- **LLM Local**: Ollama con `qwen2.5:3b` como modelo inicial
- **Dependencias**: Pydantic, pypdf, Streamlit, Typer, pytest
- **Python**: 3.11
- **Prohibido**: LangChain, LangGraph, nube, APIs de pago, Docker obligatorio, OCR, fine-tuning, frontend separado
- **Arquitectura**: Controlador explícito, no agente que ejecuta comandos del usuario

### Reglas Fundamentales

1. Pregunta original inmutable, máximo 3 recuperaciones
2. Citas apuntan a IDs existentes; página real o null
3. JSON correcto no garantiza evidencia suficiente; solo razón breve, no cadena de pensamiento
4. Prohibido favorecer baseline quitándole reglas de fidelidad o abstención
5. Distancias coseno ≠ probabilidad, score LLM ≠ certeza
6. Sin gold: no computar recall/precision reales ni 'exactitud' con similitud textual arbitraria
7. No usar crítico operativo como único juez científico
8. Registrar semillas, hashes, modelo, recursos y latencia; NA para uso desconocido
9. No inventar bibliografía, mediciones, salidas de tests ni funciones TODO vendidas como terminadas

### Comandos CLI Exactos

```bash
evidencialocal doctor
evidencialocal demo
evidencialocal index --input data/raw --config configs/default.yaml
evidencialocal ask --mode baseline --question "¿Qué es el aprendizaje supervisado?"
evidencialocal ask --mode corrective --question "¿Qué es el aprendizaje supervisado?"
evidencialocal evaluate --dataset datasets/demo/questions.jsonl --modes baseline,corrective --output experiments/results/demo
evidencialocal export-annotations --results experiments/results/demo --output experiments/results/demo/annotations.csv
evidencialocal report --results experiments/results/demo --output experiments/results/demo/report.md
```

### Estado de Implementación

- [ ] Fase 1: Núcleo completo y CLI reproducible
- [ ] Fase 2: Interfaz, experimentos y entrega