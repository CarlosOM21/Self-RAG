# Protocolo de Investigación - EvidenciaLocal

## Objetivo

Comparar empíricamente dos arquitecturas RAG en un entorno educativo local:
- **Baseline**: RAG estándar (recuperación única + generación con abstención)
- **Correctivo**: RAG con evaluación crítica iterativa (hasta 2 reintentos, deduplicación, abstención explícita)

## Hipótesis

1. **H1 (Correctitud)**: El modo correctivo produce respuestas factualmente más correctas que baseline en preguntas que requieren síntesis de múltiples fuentes.
2. **H2 (Fidelidad)**: El modo correctivo produce respuestas más fieles al contexto recuperado (menos alucinaciones).
3. **H3 (Abstención útil)**: El modo correctivo se abstené más apropiadamente (cuando realmente falta información) que baseline.
4. **H4 (Coste)**: El modo correctivo tiene mayor latencia y uso de tokens, pero dentro de límites aceptables para uso educativo.

## Diseño experimental

### Sujetos
- Preguntas de un dataset educativo (STEM, humanidades, etc.)
- Minimum 50 preguntas, idealmente 100+
- Balanceadas por tipo: factual, conceptual, procedimental, comparativa

### Diseño
- **Within-subjects**: Cada pregunta se evalúa en AMBOS modos
- **Orden alternado**: Aleatorizado por pregunta para evitar sesgo de orden
- **Ciego al modo**: Anotadores no saben qué modo generó cada respuesta

### Variables independientes
- Modo: {baseline, corrective}
- Configuración fija (congelada por experimento): modelo embedding, chunk_size, top_k, modelo LLM, temperatura, seed

### Variables dependientes (métricas principales)
1. **Correctness** (0-1): Precisión factual según gold standard + juicio experto
2. **Faithfulness** (0-1): Fidelidad al contexto recuperado (juicio experto)
3. **Citation Precision**: Citas correctas / Total citadas
4. **Citation Recall**: Citas correctas / Citas relevantes en gold
5. **Abstention Rate**: % respuestas abstenidas
6. **Appropriate Abstention**: % abstenciones donde realmente faltaba info (juicio experto)
7. **Latencia**: p50, p95, media (ms)
8. **Tokens totales**: Promedio por consulta

### Variables de control (registradas, no manipuladas)
- Hash de configuración completa
- Commit git
- Semilla aleatoria
- Modelo LLM (nombre + digest si disponible)
- Modelo embedding (nombre + hash)
- Hardware: CPU/GPU, RAM
- Versiones de dependencias clave

## Procedimiento

### 1. Preparación
```bash
# Congelar configuración
cp configs/default.yaml configs/frozen_$(date +%Y%m%d).yaml
git commit -am "Freeze config for experiment EXP-001"

# Preparar dataset
# - 50+ preguntas JSONL con gold_answer cuando sea posible
# - Temas diversos, dificultad variada
```

### 2. Ejecución automática
```bash
# Semilla fija para reproducibilidad
export EVALUATION__RANDOM_SEED=42

evidencialocal evaluate \
  --dataset datasets/experiment/questions.jsonl \
  --modes baseline,corrective \
  --output experiments/results/EXP-001
```

### 3. Anotación humana (ciega)
```bash
# Generar plantilla (modo oculto)
evidencialocal export-annotations \
  --results experiments/results/EXP-001 \
  --output experiments/results/EXP-001/annotations.csv

# Distribuir a 2+ anotadores expertos
# Cada uno anota independientemente:
# - correctness: 0.0-1.0
# - faithfulness: 0.0-1.0
# - citation_precision: 0.0-1.0
# - citation_recall: 0.0-1.0
# - annotator_notes: texto libre
```

### 4. Consolidación y análisis
```bash
# Importar anotaciones (promediar si múltiples anotadores)
evidencialocal import-annotations ...

# Generar reporte
evidencialocal report \
  --results experiments/results/EXP-001 \
  --output experiments/results/EXP-001/report.md
```

### 5. Análisis estadístico
- **Test pareado**: Wilcoxon signed-rank (no paramétrico) para correctness, faithfulness
- **Bootstrap pareado**: 1000 remuestreos para IC 95% de diferencias de medias
- **McNemar**: Para tasas de abstención (binario)
- **Effect size**: Cohen's d para diferencias estandarizadas
- **Corrección múltiple**: Bonferroni si >3 métricas primarias

## Criterios de significancia

- α = 0.05 (bilateral)
- Potencia objetivo: 0.80
- Tamaño mínimo de efecto de interés: d = 0.3 (pequeño-medio)
- Tamaño muestral calculado: n ≥ 50 preguntas para d=0.3, α=0.05, power=0.80

## Ética y limitaciones

### Limitaciones conocidas
1. **Un solo modelo LLM**: Resultados no generalizan a otros modelos
2. **Un solo modelo embedding**: idem
3. **Corpus específico**: Resultados dependen del dominio de los PDFs
4. **Anotadores no ciegos a contenido**: Solo ciegos al modo
5. **Sin usuarios reales**: Métricas de utilidad percibida no medidas
6. **Escala pequeña**: No es evaluación a gran escala tipo TREC

### Qué NO se afirma
- ❌ "Correctivo es mejor en general"
- ❌ "Resultados generalizables a producción"
- ❌ "Métricas de confianza calibradas"
- ❌ "Sistema listo para despliegue educativo real"

### Qué SÍ se puede afirmar (si datos soportan)
- ✅ "En este corpus, con este modelo, correctivo muestra X% mayor correctness (p<0.05)"
- ✅ "Latencia media correctivo es Y ms mayor (IC 95%: [a,b])"
- ✅ "Tasa de abstención apropiada es Z% en correctivo vs W% en baseline"

## Entregables del experimento

1. **Manifiesto de experimento** (JSON): config_hash, seed, git_commit, versiones
2. **Resultados brutos** (JSONL): Todas las respuestas con metadatos completos
3. **Anotaciones** (CSV): Scores por anotador, por muestra, por modo
4. **Reporte** (Markdown): Tablas comparativas, intervalos de confianza, conclusiones
5. **Código exacto**: Commit git taggeado (ej. `exp/EXP-001`)

## Checklist de reproducibilidad

- [ ] Configuración congelada (YAML + .env) versionada
- [ ] Dataset de preguntas versionado
- [ ] Semilla aleatoria documentada y usada
- [ ] Versiones exactas de dependencias (requirements.lock.txt)
- [ ] Modelo LLM: nombre + digest Ollama (`ollama list --format json`)
- [ ] Modelo embedding: nombre + hash (get_model_hash())
- [ ] Hardware documentado (CPU, GPU, RAM)
- [ ] Anotadores: número, expertise, entrenamiento previo
- [ ] Acuerdo inter-anotador reportado (Krippendorff's α o ICC)
- [ ] Código etiquetado con tag git del experimento