"""Evaluación comparativa, anotación humana ciega y reportes Markdown."""

import csv
import hashlib
import json
import logging
import random
import statistics
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

from ..config.settings import Settings, get_settings
from ..core.exceptions import EvaluationError
from ..core.models import AskRequest, EvaluationResult, EvaluationSample, ExperimentManifest, Mode
from ..metadata.sqlite_store import SQLiteStore, get_sqlite_store
from ..pipeline.rag_pipeline import get_rag_pipeline

logger = logging.getLogger(__name__)
ANNOTATION_FIELDS = ("correctness", "faithfulness", "citation_precision", "citation_recall")


def load_dataset(dataset_path: Path) -> list[EvaluationSample]:
    """Carga un JSONL y valida cada muestra con Pydantic."""
    samples = []
    try:
        with dataset_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    samples.append(EvaluationSample(**json.loads(line)))
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise EvaluationError(
                        f"Muestra inválida en {dataset_path}:{line_number}: {exc}"
                    ) from exc
    except OSError as exc:
        raise EvaluationError(f"No se pudo leer el dataset {dataset_path}: {exc}") from exc
    return samples


def run_evaluation(
    dataset_path: Path,
    modes: list[Mode],
    output_dir: Path,
    settings: Settings,
    limit: int | None = None,
) -> None:
    """Ejecuta cada par muestra-modo y conserva resultados en SQLite y JSON."""
    if not modes:
        raise EvaluationError("Debe seleccionar al menos un modo")
    samples = load_dataset(dataset_path)
    if limit is not None:
        samples = samples[: max(0, limit)]
    if not samples:
        raise EvaluationError("El dataset no contiene muestras")

    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = get_rag_pipeline(settings)
    sqlite = get_sqlite_store(settings)
    config_str = json.dumps(settings.model_dump(mode="json"), sort_keys=True, default=str)
    manifest = ExperimentManifest(
        name=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config_hash=hashlib.sha256(config_str.encode()).hexdigest()[:16],
        settings_snapshot=settings.model_dump(mode="json"),
        random_seed=settings.evaluation.random_seed,
        modes_evaluated=modes,
        dataset_path=str(dataset_path),
        total_samples=len(samples) * len(modes),
    )
    sqlite.save_experiment_manifest(manifest)

    rng = random.Random(settings.evaluation.random_seed)
    eval_order = []
    for sample in samples:
        mode_order = list(modes)
        rng.shuffle(mode_order)
        eval_order.extend((sample, mode) for mode in mode_order)

    completed = 0
    for sample, mode in eval_order:
        try:
            response = pipeline.ask(AskRequest(question=sample.question, mode=mode))
            result = EvaluationResult(
                sample_id=sample.id,
                mode=mode,
                question=sample.question,
                predicted_answer=response.answer,
                predicted_citations=response.citations,
                abstained=response.abstained,
                latency_ms=response.total_latency_ms,
                tokens_used=response.total_tokens,
                retrieval_count=len(response.retrievals),
                critique_count=len(response.critiques),
            )
            sqlite.save_evaluation_result(result, manifest.id)
            completed += 1
            sqlite.increment_completed_samples(manifest.id)
            if completed % 10 == 0:
                logger.info("Progreso: %d/%d", completed, len(eval_order))
        except Exception as exc:
            logger.error("Error evaluando %s en modo %s: %s", sample.id, mode.value, exc)

    export_results_json(sqlite, manifest.id, output_dir / "results_raw.json")
    logger.info("Evaluación completada: %d/%d resultados", completed, len(eval_order))


def export_results_json(sqlite: SQLiteStore, experiment_id: UUID, output_path: Path) -> None:
    """Exporta resultados y añade el experimento para poder reimportar anotaciones."""
    data = []
    for result in sqlite.get_evaluation_results(experiment_id):
        item = result.model_dump(mode="json")
        item["experiment_id"] = str(experiment_id)
        data.append(item)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_results(results_dir: Path) -> tuple[Path, list[dict]]:
    path = results_dir / "results_raw.json"
    if not path.is_file():
        raise EvaluationError(f"No se encontró {path}")
    try:
        results = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"No se pudieron leer los resultados: {exc}") from exc
    if not isinstance(results, list):
        raise EvaluationError("results_raw.json debe contener una lista")
    return path, results


def export_annotations_template(results_dir: Path, output_path: Path) -> None:
    """Crea un CSV sin el modo y un mapeo privado para reimportarlo sin ambigüedad."""
    _, results = _read_results(results_dir)
    mapping = {}
    rows = []
    for result in results:
        annotation_id = uuid4().hex
        mapping[annotation_id] = {"sample_id": result["sample_id"], "mode": result["mode"]}
        rows.append(
            {
                "annotation_id": annotation_id,
                "sample_id": result["sample_id"],
                "question": result["question"],
                "predicted_answer": result["predicted_answer"],
                "predicted_citations": ";".join(result["predicted_citations"]),
                "abstained": result["abstained"],
                "latency_ms": result["latency_ms"],
                "retrieval_count": result["retrieval_count"],
                "critique_count": result["critique_count"],
                **dict.fromkeys(ANNOTATION_FIELDS, ""),
                "annotator_notes": "",
            }
        )

    random.Random(42).shuffle(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else ["annotation_id"])
        writer.writeheader()
        writer.writerows(rows)
    (results_dir / "annotations_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def import_annotations(results_dir: Path, annotations_path: Path) -> None:
    """Reincorpora métricas humanas al JSON y a SQLite mediante el mapeo ciego."""
    results_path, results = _read_results(results_dir)
    mapping_path = results_dir / "annotations_mapping.json"
    if not mapping_path.is_file():
        raise EvaluationError("Falta annotations_mapping.json; exporte primero la plantilla")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    results_index = {(item["sample_id"], item["mode"]): item for item in results}

    with annotations_path.open(encoding="utf-8", newline="") as file:
        annotations = list(csv.DictReader(file))
    sqlite = get_sqlite_store()
    imported = 0
    for annotation in annotations:
        annotation_id = annotation.get("annotation_id", "")
        if annotation_id not in mapping:
            raise EvaluationError(f"annotation_id desconocido: {annotation_id!r}")
        identity = mapping[annotation_id]
        key = (identity["sample_id"], identity["mode"])
        if key not in results_index:
            raise EvaluationError(f"No existe el resultado asociado a {annotation_id}")
        result = results_index[key]

        values = {}
        for field in ANNOTATION_FIELDS:
            raw_value = annotation.get(field, "").strip()
            if not raw_value:
                values[field] = None
                continue
            try:
                value = float(raw_value)
            except ValueError as exc:
                raise EvaluationError(f"{field} debe ser numérico: {raw_value!r}") from exc
            if not 0.0 <= value <= 1.0:
                raise EvaluationError(f"{field} debe estar entre 0 y 1")
            values[field] = value

        result.update(values)
        result["annotated"] = all(value is not None for value in values.values())
        result["annotator_notes"] = annotation.get("annotator_notes", "")
        experiment_id = result.get("experiment_id")
        if experiment_id and result["annotated"]:
            sqlite.update_evaluation_annotations(
                UUID(experiment_id),
                result["sample_id"],
                Mode(result["mode"]),
                values["correctness"],
                values["faithfulness"],
                values["citation_precision"],
                values["citation_recall"],
                result["annotator_notes"],
            )
        imported += 1

    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Anotaciones importadas: %d", imported)


def generate_report(results_dir: Path, output_path: Path) -> None:
    """Genera un reporte Markdown con métricas automáticas y humanas disponibles."""
    _, results = _read_results(results_dir)
    by_mode: dict[str, list[dict]] = {}
    for result in results:
        by_mode.setdefault(result["mode"], []).append(result)

    lines = [
        "# Reporte de evaluación comparativa",
        "",
        f"Fecha: {datetime.now().isoformat()}",
        f"Resultados: {len(results)}",
        f"Modos: {', '.join(by_mode)}",
        "",
        "## Métricas por modo",
        "",
    ]
    for mode, mode_results in by_mode.items():
        lines.extend([f"### {mode}", "", f"- Resultados: {len(mode_results)}"])
        latencies = [item["latency_ms"] for item in mode_results]
        if latencies:
            lines.append(f"- Latencia media: {statistics.mean(latencies):.1f} ms")
            lines.append(f"- Latencia p50: {statistics.median(latencies):.1f} ms")
            lines.append(f"- Latencia p95: {percentile(latencies, 95):.1f} ms")
        tokens = [
            item["tokens_used"] for item in mode_results if item.get("tokens_used") is not None
        ]
        if tokens:
            lines.append(f"- Tokens de salida medios: {statistics.mean(tokens):.1f}")
        abstained = sum(bool(item["abstained"]) for item in mode_results)
        lines.append(f"- Abstenciones: {abstained}/{len(mode_results)}")
        lines.append(
            f"- Recuperaciones medias: "
            f"{statistics.mean(item['retrieval_count'] for item in mode_results):.2f}"
        )
        for metric in ANNOTATION_FIELDS:
            values = [
                item[metric]
                for item in mode_results
                if item.get("annotated") and item.get(metric) is not None
            ]
            if values:
                lines.append(f"- {metric}: {statistics.mean(values):.3f} (n={len(values)})")
        lines.append("")

    if Mode.BASELINE.value in by_mode and Mode.CORRECTIVE.value in by_mode:
        evaluation = get_settings().evaluation
        lines.extend(
            _paired_bootstrap_section(
                by_mode,
                evaluation.bootstrap_samples,
                evaluation.confidence_level,
                evaluation.random_seed,
            )
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _paired_bootstrap_section(
    by_mode: dict[str, list[dict]],
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> list[str]:
    """Calcula intervalos bootstrap de diferencias corrective-baseline."""
    baseline = {item["sample_id"]: item for item in by_mode[Mode.BASELINE.value]}
    corrective = {item["sample_id"]: item for item in by_mode[Mode.CORRECTIVE.value]}
    confidence_label = f"IC{confidence_level * 100:g}%"
    lines = [f"## Comparación pareada (corrective - baseline; {confidence_label})", ""]
    rng = random.Random(seed)
    for metric in ANNOTATION_FIELDS:
        differences = [
            corrective[sample_id][metric] - baseline[sample_id][metric]
            for sample_id in baseline.keys() & corrective.keys()
            if baseline[sample_id].get(metric) is not None
            and corrective[sample_id].get(metric) is not None
        ]
        if not differences:
            continue
        observed = statistics.mean(differences)
        resampled = []
        for _ in range(bootstrap_samples):
            sample = [rng.choice(differences) for _ in differences]
            resampled.append(statistics.mean(sample))
        tail_probability = (1 - confidence_level) * 50
        low = percentile(resampled, tail_probability)
        high = percentile(resampled, 100 - tail_probability)
        lines.append(
            f"- **{metric}**: diferencia={observed:.3f}; "
            f"{confidence_label} bootstrap [{low:.3f}, {high:.3f}]; pares={len(differences)}"
        )
    if len(lines) == 2:
        lines.append("No hay pares anotados suficientes para comparar métricas humanas.")
    lines.append("")
    return lines


def percentile(data: list[float], p: float) -> float:
    """Calcula un percentil mediante interpolación lineal."""
    if not data:
        raise ValueError("No se puede calcular un percentil de una lista vacía")
    sorted_data = sorted(data)
    position = (len(sorted_data) - 1) * p / 100
    lower = int(position)
    upper = min(lower + 1, len(sorted_data) - 1)
    if lower == upper:
        return sorted_data[lower]
    fraction = position - lower
    return sorted_data[lower] + fraction * (sorted_data[upper] - sorted_data[lower])
