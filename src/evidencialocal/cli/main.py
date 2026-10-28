"""CLI principal de EvidenciaLocal con Typer."""

import json
import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ..config.settings import Settings, get_settings
from ..core.exceptions import (
    EvidenciaLocalError,
)
from ..core.models import AskRequest, IndexStatus, Mode
from ..pipeline.indexer import get_indexer
from ..pipeline.rag_pipeline import get_rag_pipeline

app = typer.Typer(
    name="evidencialocal",
    help="Sistema RAG educativo local con modos baseline y correctivo",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _get_settings(config: Path | None = None) -> Settings:
    """Carga configuración, opcionalmente desde YAML."""
    settings = get_settings()
    if config and config.exists():
        import yaml
        with open(config) as f:
            config_data = yaml.safe_load(f)
        # TODO: aplicar config_data a settings si necesario
    return settings


@app.command()
def doctor(
    config: Path | None = typer.Option(None, "--config", "-c", help="Archivo de configuración YAML"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Salida detallada"),
):
    """Verifica dependencias y modelos sin descargar nada."""
    settings = _get_settings(config)

    console.print("[bold]EvidenciaLocal - Diagnóstico del sistema[/bold]\n")

    # 1. Python version
    console.print(f"Python: {sys.version.split()[0]}")

    # 2. Verificar llama.cpp
    from ..llm.llama_client import get_llama_client
    llama_client = get_llama_client(settings)
    llama_available = llama_client.check_model_available()

    table = Table(title="Estado de componentes")
    table.add_column("Componente", style="cyan")
    table.add_column("Estado", style="green")
    table.add_column("Detalles")

    table.add_row("Python", "✓ OK", sys.version.split()[0])
    table.add_row(
        "llama.cpp",
        "✓ Disponible" if llama_available else "✗ No disponible",
        f"Modelo: {settings.llama_cpp.model_path}",
    )

    if not llama_available:
        table.add_row(
            "Acción requerida",
            "⚠ Modelo no encontrado",
            f"Verifique que el archivo {settings.llama_cpp.model_path} existe",
        )

    # 3. ChromaDB
    from ..vectorstore.chroma_store import get_chroma_store
    chroma = get_chroma_store(settings)
    try:
        count = chroma.get_chunk_count()
        table.add_row("ChromaDB", "✓ OK", f"{count} chunks indexados en {settings.chroma.persist_directory}")
    except Exception as e:
        table.add_row("ChromaDB", "✗ Error", str(e))

    # 4. SQLite
    from ..metadata.sqlite_store import get_sqlite_store
    sqlite = get_sqlite_store(settings)
    try:
        # Test query
        sqlite.conn.execute("SELECT 1").fetchone()
        table.add_row("SQLite", "✓ OK", f"BD en {settings.sqlite.database_path}")
    except Exception as e:
        table.add_row("SQLite", "✗ Error", str(e))

    # 5. Embeddings
    from ..embeddings.embedder import get_embedder
    try:
        embedder = get_embedder(settings)
        _ = embedder.model  # Trigger load
        table.add_row(
            "Embeddings",
            "✓ OK",
            f"{embedder.model_name} (dim={embedder.dimension})",
        )
    except Exception as e:
        table.add_row("Embeddings", "✗ Error", str(e))

    console.print(table)

    if verbose:
        console.print("\n[bold]Configuración completa:[/bold]")
        console.print(json.dumps(settings.model_dump(mode="json"), indent=2, default=str))

    if not llama_available:
        console.print("\n[bold yellow]⚠ Modelo llama.cpp no disponible.[/bold yellow]")
        console.print(f"  Verifique que el archivo {settings.llama_cpp.model_path} existe")
        raise typer.Exit(code=1)

    console.print("\n[green]✓ Todo listo para usar EvidenciaLocal[/green]")


@app.command()
def demo(
    output_dir: Path = typer.Option(Path("data/raw"), "--output", "-o", help="Directorio de salida"),
    force: bool = typer.Option(False, "--force", "-f", help="Sobrescribir si existe"),
):
    """Genera documentos y preguntas sintéticas para demostración (no toca corpus real)."""
    from .demo import generate_demo_data

    console.print("[bold]Generando datos de demostración...[/bold]")
    generate_demo_data(output_dir, force)
    console.print(f"[green]✓ Datos de demo generados en {output_dir}[/green]")
    console.print("\nPara indexar y probar:")
    console.print(f"  evidencialocal index --input {output_dir}")
    console.print('  evidencialocal ask --mode baseline --question "¿Qué es el aprendizaje supervisado?"')


@app.command()
def index(
    input_dir: Path = typer.Option(..., "--input", "-i", help="Directorio con PDFs"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Archivo de configuración YAML"),
    force_reindex: bool = typer.Option(False, "--force-reindex", help="Reindexar todo (borra índice anterior)"),
):
    """Indexa documentos PDF en el vector store."""
    settings = _get_settings(config)
    indexer = get_indexer(settings)

    if not input_dir.exists():
        console.print(f"[red]Error: Directorio no existe: {input_dir}[/red]")
        raise typer.Exit(code=1)

    if force_reindex:
        console.print("[yellow]Reindexación completa (borra índice anterior)...[/yellow]")
        manifest = indexer.reindex_all(input_dir)
    else:
        console.print(f"[bold]Indexando documentos en {input_dir}...[/bold]")
        manifest = indexer.index_directory(input_dir)

    stats = indexer.get_index_stats()
    console.print("\n[green]✓ Indexación completada[/green]")
    console.print(f"  Documentos: {stats['total_documents']}")
    console.print(f"  Chunks: {stats['total_chunks']}")
    console.print(f"  Modelo embedding: {stats['embedding_model']}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Pregunta a responder"),
    mode: Mode = typer.Option(Mode.BASELINE, "--mode", "-m", help="Modo: baseline o correctivo"),
    top_k: int | None = typer.Option(None, "--top-k", "-k", help="Número de chunks a recuperar"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Archivo de configuración YAML"),
    show_context: bool = typer.Option(False, "--show-context", help="Mostrar chunks recuperados"),
):
    """Responde una pregunta usando el modo especificado."""
    settings = _get_settings(config)
    pipeline = get_rag_pipeline(settings)

    console.print(f"[bold]Modo:[/bold] {mode.value}")
    console.print(f"[bold]Pregunta:[/bold] {question}\n")

    try:
        request = AskRequest(question=question, mode=mode, top_k=top_k)
        response = pipeline.ask(request)

        # Mostrar respuesta
        if response.abstained:
            console.print("[yellow]ABSTENCIÓN:[/yellow]")
            console.print(f"  Razón: {response.abstention_reason}")
        else:
            console.print("[green]RESPUESTA:[/green]")
            console.print(f"  {response.answer}")

        if response.citations:
            console.print(f"\n[bold]Citas ({len(response.citations)}):[/bold]")
            for cid in response.citations:
                console.print(f"  • {cid}")

        console.print("\n[bold]Métricas:[/bold]")
        console.print(f"  Latencia total: {response.total_latency_ms:.1f} ms")
        console.print(f"  Tokens: {response.total_tokens or 'N/A'}")
        console.print(f"  Recuperaciones: {len(response.retrievals)}")
        if response.critiques:
            console.print(f"  Críticas: {len(response.critiques)}")

        if show_context:
            console.print("\n[bold]Contexto recuperado:[/bold]")
            for i, round_results in enumerate(response.retrievals):
                console.print(f"  Ronda {i+1}:")
                for r in round_results:
                    page = f" (pág. {r.page_number})" if r.page_number else ""
                    console.print(f"    [{r.rank}] {r.document_filename}{page} - score: {r.score:.3f}")
                    console.print(f"        {r.text[:150]}...")

    except EvidenciaLocalError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def evaluate(
    dataset: Path = typer.Option(..., "--dataset", "-d", help="Archivo JSONL con preguntas"),
    modes: str = typer.Option("baseline,corrective", "--modes", help="Modos a evaluar (coma-separado)"),
    output: Path = typer.Option(..., "--output", "-o", help="Directorio de resultados"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Archivo de configuración YAML"),
    limit: int | None = typer.Option(None, "--limit", "-l", help="Limitar número de muestras"),
):
    """Ejecuta evaluación comparativa sobre dataset."""
    from .evaluate import run_evaluation

    settings = _get_settings(config)
    mode_list = [Mode(m.strip()) for m in modes.split(",")]

    console.print(f"[bold]Evaluando modos: {', '.join(m.value for m in mode_list)}[/bold]")
    console.print(f"Dataset: {dataset}")
    console.print(f"Salida: {output}")

    run_evaluation(dataset, mode_list, output, settings, limit)
    console.print("[green]✓ Evaluación completada[/green]")


@app.command()
def export_annotations(
    results_dir: Path = typer.Option(..., "--results", "-r", help="Directorio con resultados de evaluación"),
    output: Path = typer.Option(..., "--output", "-o", help="Archivo CSV de salida"),
):
    """Exporta plantilla CSV para anotación humana."""
    from .evaluate import export_annotations_template

    console.print("[bold]Exportando plantilla de anotación...[/bold]")
    export_annotations_template(results_dir, output)
    console.print(f"[green]✓ Plantilla exportada a {output}[/green]")


@app.command()
def report(
    results_dir: Path = typer.Option(..., "--results", "-r", help="Directorio con resultados de evaluación"),
    output: Path = typer.Option(..., "--output", "-o", help="Archivo Markdown de salida"),
):
    """Genera reporte Markdown comparativo."""
    from .evaluate import generate_report

    console.print("[bold]Generando reporte...[/bold]")
    generate_report(results_dir, output)
    console.print(f"[green]✓ Reporte generado en {output}[/green]")


@app.command()
def import_annotations(
    results_dir: Path = typer.Option(..., "--results", "-r", help="Directorio con resultados"),
    annotations: Path = typer.Option(..., "--annotations", "-a", help="Archivo CSV con anotaciones"),
):
    """Importa anotaciones humanas a resultados de evaluación."""
    from .evaluate import import_annotations

    console.print("[bold]Importando anotaciones...[/bold]")
    import_annotations(results_dir, annotations)
    console.print("[green]✓ Anotaciones importadas[/green]")


@app.command()
def delete_index(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Confirmar sin prompt"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Archivo de configuración YAML"),
):
    """Elimina todo el índice (Chroma + SQLite + manifiesto) con confirmación."""
    settings = _get_settings(config)
    indexer = get_indexer(settings)

    if not confirm:
        console.print("[yellow]⚠ Esto borrará TODO el índice (documentos, chunks, manifiesto).[/yellow]")
        confirm = typer.confirm("¿Continuar?")

    if confirm:
        indexer.reindex_all(Path("data/raw"))  # Esto borra y recrea vacío
        console.print("[green]✓ Índice eliminado[/green]")
    else:
        console.print("Cancelado.")


@app.command()
def list_docs(
    status: IndexStatus | None = typer.Option(None, "--status", "-s", help="Filtrar por estado"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Archivo de configuración YAML"),
):
    """Lista documentos indexados."""
    settings = _get_settings(config)
    indexer = get_indexer(settings)
    manifest = indexer.load_manifest()

    table = Table(title="Documentos indexados")
    table.add_column("ID", style="dim")
    table.add_column("Archivo")
    table.add_column("Páginas")
    table.add_column("Chunks")
    table.add_column("Estado")
    table.add_column("Hash")

    for doc in manifest.documents.values():
        if status and doc.status != status:
            continue
        table.add_row(
            str(doc.id)[:8],
            doc.filename,
            str(doc.page_count),
            str(len(indexer.sqlite.get_chunks_by_document(doc.id))),
            doc.status.value,
            doc.content_hash[:12] + "...",
        )

    console.print(table)
    console.print(f"\nTotal: {len(manifest.documents)} documentos, {manifest.total_chunks} chunks")


if __name__ == "__main__":
    app()
