"""Módulo de evaluación comparativa."""

import csv
import json
import logging
import random
import statistics
from datetime import datetime
from pathlib import Path
from uuid import UUID

from ...config.settings import Settings
from ...core.models import (
    AskRequest,
    EvaluationResult,
    EvaluationSample,
    ExperimentManifest,
    Mode,
)
from ...metadata.sqlite_store import get_sqlite_store
from ...pipeline.rag_pipeline import get_rag_pipeline

logger = logging.getLogger(__name__)


def load_dataset(dataset_path: Path) -> list[EvaluationSample]:
    """Carga dataset JSONL."""
    samples = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            samples.append(EvaluationSample(**data))
    return samples


def run_evaluation(
    dataset_path: Path,
    modes: list[Mode],
    output_dir: Path,
    settings: Settings,
    limit: int | None = None,
) -> None:
    """Ejecuta evaluación completa."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar dataset
    samples = load_dataset(dataset_path)
    if limit:
        samples = samples[:limit]

    logger.info(f"Evaluando {len(samples)} muestras en {len(modes)} modos")

    # Pipeline RAG
    pipeline = get_rag_pipeline(settings)

    # SQLite para persistir resultados
    sqlite = get_sqlite_store(settings)

    # Crear manifiesto de experimento
    import hashlib
    config_str = json.dumps(settings.model_dump(mode="json"), sort_keys=True, default=str)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

    manifest = ExperimentManifest(
        name=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config_hash=config_hash,
        settings_snapshot=settings.model_dump(mode="json"),
        random_seed=settings.evaluation.random_seed,
        modes_evaluated=modes,
        dataset_path=str(dataset_path),
        total_samples=len(samples),
    )
    sqlite.save_experiment_manifest(manifest)

    # Orden de ejecución alternado para evitar sesgo
    random.seed(settings.evaluation.random_seed)
    eval_order = []
    for sample in samples:
        mode_order = list(modes)
        random.shuffle(mode_order)
        for mode in mode_order:
            eval_order.append((sample, mode))

    # Ejecutar evaluaciones
    completed = 0
    for sample, mode in eval_order:
        try:
            request = AskRequest(question=sample.question, mode=mode)
            response = pipeline.ask(request)

            result = EvaluationResult(
                sample_id=sample.id,
                mode=mode,
                question=sample.question,
                predicted_answer=response.answer,
                predicted_citations=response.citations,
                abstained=response.abstained,
                latency_ms=response.total_latency_ms,
                total_tokens=response.total_tokens,
                retrieval_count=len(response.retrievals),
                critique_count=len(response.critiques),
            )

            sqlite.save_evaluation_result(result, manifest.id)
            completed += 1

            if completed % 10 == 0:
                logger.info(f"Progreso: {completed}/{len(eval_order)}")

        except Exception as e:
            logger.error(f"Error evaluando {sample.id} modo {mode}: {e}")

    sqlite.increment_completed_samples(manifest.id)
    logger.info(f"Evaluación completada: {completed} resultados guardados")

    # Exportar resultados brutos JSON
    export_results_json(sqlite, manifest.id, output_dir / "results_raw.json")


def export_results_json(sqlite: "SQLiteStore", experiment_id: UUID, output_path: Path) -> None:
    """Exporta resultados a JSON."""
    results = sqlite.get_evaluation_results(experiment_id)
    data = [r.model_dump(mode="json") for r in results]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def export_annotations_template(results_dir: Path, output_path: Path) -> None:
    """Exporta plantilla CSV para anotación humana ciega al modo."""

    # Buscar archivo de resultados
    result_files = list(results_dir.glob("results_raw.json"))
    if not result_files:
        raise FileNotFoundError(f"No se encontró results_raw.json en {results_dir}")

    with open(result_files[0], encoding="utf-8") as f:
        results = json.load(f)

    # Preparar filas para anotación (ciega al modo)
    rows = []
    for r in results:
        # Mezclar orden para ceguera (se guarda mapeo interno)
        rows.append({
            "sample_id": r["sample_id"],
            "question": r["question"],
            "predicted_answer": r["predicted_answer"],
            "predicted_citations": ";".join(r["predicted_citations"]),
            "abstained": r["abstained"],
            "latency_ms": r["latency_ms"],
            "retrieval_count": r["retrieval_count"],
            "critique_count": r["critique_count"],
            "mode": r["mode"],  # Se oculta en versión para anotadores
            "correctness": "",
            "faithfulness": "",
            "citation_precision": "",
            "citation_recall": "",
            "annotator_notes": "",
        })

    # Randomizar orden
    random.seed(42)
    random.shuffle(rows)

    # Escribir CSV
    fieldnames = [
        "sample_id", "question", "predicted_answer", "predicted_citations",
        "abstained", "latency_ms", "retrieval_count", "critique_count",
        "correctness", "faithfulness", "citation_precision", "citation_recall",
        "annotator_notes",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            # Ocultar mode en versión para anotadores
            public_row = {k: v for k, v in row.items() if k != "mode"}
            writer.writerow(public_row)


def import_annotations(results_dir: Path, annotations_path: Path) -> None:
    """Importa anotaciones humanas y calcula métricas."""

    result_files = list(results_dir.glob("results_raw.json"))
    if not result_files:
        raise FileNotFoundError(f"No se encontró results_raw.json en {results_dir}")

    with open(result_files[0], encoding="utf-8") as f:
        results = json.load(f)

    # Crear índice por sample_id + mode (oculto)
    results_idx = {(r["sample_id"], r["mode"]): r for r in results}

    # Leer anotaciones
    with open(annotations_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        annotations = list(reader)

    sqlite = get_sqlite_store(get_sqlite_store().settings)  # Reutilizar settings

    # Actualizar resultados con anotaciones
    for ann in annotations:
        # Necesitamos reconstruir mode - en versión real se guardaría mapping
        # Por ahora asumimos que hay un campo mode en anotaciones o se infiere
        pass  # Implementación completa requiere mapping guardado


def generate_report(results_dir: Path, output_path: Path) -> None:
    """Genera reporte Markdown con métricas comparativas."""

    result_files = list(results_dir.glob("results_raw.json"))
    if not result_files:
        raise FileNotFoundError(f"No se encontró results_raw.json en {results_dir}")

    with open(result_files[0], encoding="utf-8") as f:
        results = json.load(f)

    # Agrupar por modo
    by_mode: dict[str, list[dict]] = {}
    for r in results:
        mode = r["mode"]
        if mode not in by_mode:
            by_mode[mode] = []
        by_mode[mode].append(r)

    # Calcular métricas
    report_lines = [
        "# Reporte de Evaluación Comparativa",
        f"\nFecha: {datetime.now().isoformat()}",
        f"Muestras totales: {len(results)}",
        f"Modos evaluados: {', '.join(by_mode.keys())}",
        "\n## Métricas por Modo\n",
    ]

    for mode, mode_results in by_mode.items():
        report_lines.append(f"### {mode.upper()}\n")
        report_lines.append(f"- Muestras: {len(mode_results)}")

        # Latencia
        latencies = [r["latency_ms"] for r in mode_results if r["latency_ms"]]
        if latencies:
            report_lines.append(f"- Latencia media: {statistics.mean(latencies):.1f} ms")
            report_lines.append(f"- Latencia p50: {statistics.median(latencies):.1f} ms")
            if len(latencies) > 1:
                report_lines.append(f"- Latencia p95: {percentile(latencies, 95):.1f} ms")

        # Tokens
        tokens = [r["total_tokens"] for r in mode_results if r["total_tokens"]]
        if tokens:
            report_lines.append(f"- Tokens media: {statistics.mean(tokens):.0f}")

        # Abstenciones
        abstained = sum(1 for r in mode_results if r["abstained"])
        report_lines.append(f"- Abstenciones: {abstained} ({abstained/len(mode_results)*100:.1f}%)")

        # Recuperaciones
        ret_counts = [r["retrieval_count"] for r in mode_results]
        report_lines.append(f"- Recuperaciones media: {statistics.mean(ret_counts):.1f}")

        # Críticas (solo corrective)
        crit_counts = [r["critique_count"] for r in mode_results]
        report_lines.append(f"- Críticas media: {statistics.mean(crit_counts):.1f}")

        # Métricas anotadas (si disponibles)
        annotated = [r for r in mode_results if r.get("annotated")]
        if annotated:
            for metric in ["correctness", "faithfulness", "citation_precision", "citation_recall"]:
                vals = [r[metric] for r in annotated if r.get(metric) is not None]
                if vals:
                    report_lines.append(f"- {metric}: media={statistics.mean(vals):.3f}")

        report_lines.append("")

    # Comparación bootstrap pareada si ambos modos tienen anotaciones
    if len(by_mode) == 2 and all(any(r.get("annotated") for r in v) for v in by_mode.values()):
        modes_list = list(by_mode.keys())
        m1, m2 = modes_list[0], modes_list[1]
        report_lines.append("## Comparación Bootstrap Pareada\n")
        report_lines.append("*Requiere anotaciones completadas en ambos modos.*\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))


def percentile(data: list[float], p: float) -> float:
    """Percentil simple."""
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    if f == c:
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])
