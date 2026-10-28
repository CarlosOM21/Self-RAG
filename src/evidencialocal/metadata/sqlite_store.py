"""Almacén de metadatos y trazas con SQLite."""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from ..config.settings import get_settings
from ..core.models import (
    ChunkMetadata,
    Document,
    EvaluationResult,
    ExperimentManifest,
    IndexManifest,
    IndexStatus,
)

logger = logging.getLogger(__name__)


SCHEMA_VERSION = 1

DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    page_count INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    document_filename TEXT NOT NULL,
    page_number INTEGER,
    chunk_index INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

CREATE TABLE IF NOT EXISTS traces (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    question TEXT NOT NULL,
    mode TEXT NOT NULL,
    answer TEXT,
    citations TEXT,  -- JSON array of chunk IDs
    abstained INTEGER NOT NULL,
    abstention_reason TEXT,
    retrievals TEXT,  -- JSON array of retrieval rounds
    critiques TEXT,   -- JSON array of critiques
    total_latency_ms REAL NOT NULL,
    total_tokens INTEGER,
    model TEXT NOT NULL,
    seed INTEGER
);

CREATE INDEX IF NOT EXISTS idx_traces_timestamp ON traces(timestamp);
CREATE INDEX IF NOT EXISTS idx_traces_mode ON traces(mode);

CREATE TABLE IF NOT EXISTS index_manifest (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    question TEXT NOT NULL,
    predicted_answer TEXT NOT NULL,
    predicted_citations TEXT,  -- JSON array
    abstained INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    tokens_used INTEGER,
    retrieval_count INTEGER NOT NULL,
    critique_count INTEGER NOT NULL DEFAULT 0,
    correctness REAL,
    faithfulness REAL,
    citation_precision REAL,
    citation_recall REAL,
    annotated INTEGER NOT NULL DEFAULT 0,
    annotator_notes TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_eval_experiment ON evaluation_results(experiment_id);
CREATE INDEX IF NOT EXISTS idx_eval_sample ON evaluation_results(sample_id);

CREATE TABLE IF NOT EXISTS experiment_manifest (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    settings_snapshot TEXT NOT NULL,
    git_commit TEXT,
    random_seed INTEGER NOT NULL,
    modes_evaluated TEXT NOT NULL,
    dataset_path TEXT NOT NULL,
    total_samples INTEGER NOT NULL,
    completed_samples INTEGER NOT NULL DEFAULT 0
);
"""


class SQLiteStore:
    """Almacén SQLite para metadatos, trazas y resultados de evaluación."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.db_path = Path(self.settings.sqlite.database_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    @property
    def conn(self) -> sqlite3.Connection:
        """Conexión SQLite (lazy, thread-safe con check_same_thread=False)."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                isolation_level=None,  # autocommit mode
            )
            self._conn.row_factory = sqlite3.Row
            if self.settings.sqlite.enable_wal:
                self._conn.execute("PRAGMA journal_mode=WAL;")
                self._conn.execute("PRAGMA foreign_keys=ON;")
        return self._conn

    def _init_db(self) -> None:
        """Inicializa esquema y versionado."""
        with self._transaction() as cur:
            cur.executescript(DDL)
            # Verificar/establecer versión
            cur.execute("SELECT version FROM schema_version")
            row = cur.fetchone()
            if row is None:
                cur.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
            elif row[0] != SCHEMA_VERSION:
                logger.warning(f"Versión de esquema BD ({row[0]}) != código ({SCHEMA_VERSION})")

    @contextmanager
    def _transaction(self):
        """Context manager para transacciones."""
        cur = self.conn.cursor()
        try:
            yield cur
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # ========== Documentos ==========

    def upsert_document(self, doc: Document) -> None:
        """Inserta o actualiza documento."""
        with self._transaction() as cur:
            cur.execute(
                """
                INSERT INTO documents (id, source_path, filename, content_hash, page_count, created_at, status, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    source_path=excluded.source_path,
                    filename=excluded.filename,
                    content_hash=excluded.content_hash,
                    page_count=excluded.page_count,
                    status=excluded.status,
                    error_message=excluded.error_message
                """,
                (
                    str(doc.id),
                    str(doc.source_path),
                    doc.filename,
                    doc.content_hash,
                    doc.page_count,
                    doc.created_at.isoformat(),
                    doc.status.value,
                    doc.error_message,
                ),
            )

    def get_document(self, doc_id: UUID) -> Document | None:
        """Obtiene documento por ID."""
        cur = self.conn.execute("SELECT * FROM documents WHERE id = ?", (str(doc_id),))
        row = cur.fetchone()
        if row:
            return self._row_to_document(row)
        return None

    def get_document_by_hash(self, content_hash: str) -> Document | None:
        """Obtiene documento por hash de contenido (para idempotencia)."""
        cur = self.conn.execute("SELECT * FROM documents WHERE content_hash = ?", (content_hash,))
        row = cur.fetchone()
        if row:
            return self._row_to_document(row)
        return None

    def list_documents(self, status: IndexStatus | None = None) -> list[Document]:
        """Lista documentos, opcionalmente filtrados por estado."""
        if status:
            cur = self.conn.execute("SELECT * FROM documents WHERE status = ? ORDER BY created_at DESC", (status.value,))
        else:
            cur = self.conn.execute("SELECT * FROM documents ORDER BY created_at DESC")
        return [self._row_to_document(row) for row in cur.fetchall()]

    def delete_document(self, doc_id: UUID) -> bool:
        """Elimina documento (cascada elimina chunks)."""
        with self._transaction() as cur:
            cur.execute("DELETE FROM documents WHERE id = ?", (str(doc_id),))
            return cur.rowcount > 0

    def _row_to_document(self, row: sqlite3.Row) -> Document:
        return Document(
            id=UUID(row["id"]),
            source_path=Path(row["source_path"]),
            filename=row["filename"],
            content_hash=row["content_hash"],
            page_count=row["page_count"],
            created_at=datetime.fromisoformat(row["created_at"]),
            status=IndexStatus(row["status"]),
            error_message=row["error_message"],
        )

    # ========== Chunks ==========

    def upsert_chunk_metadata(self, meta: ChunkMetadata) -> None:
        """Inserta o actualiza metadatos de chunk."""
        with self._transaction() as cur:
            cur.execute(
                """
                INSERT INTO chunks (id, document_id, document_filename, page_number, chunk_index,
                                   token_count, char_start, char_end, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    document_id=excluded.document_id,
                    document_filename=excluded.document_filename,
                    page_number=excluded.page_number,
                    chunk_index=excluded.chunk_index,
                    token_count=excluded.token_count,
                    char_start=excluded.char_start,
                    char_end=excluded.char_end,
                    content_hash=excluded.content_hash
                """,
                (
                    str(meta.chunk_id),
                    str(meta.document_id),
                    meta.document_filename,
                    meta.page_number,
                    meta.chunk_index,
                    meta.token_count,
                    meta.char_start,
                    meta.char_end,
                    meta.content_hash,
                ),
            )

    def get_chunk_metadata(self, chunk_id: UUID) -> ChunkMetadata | None:
        """Obtiene metadatos de chunk."""
        cur = self.conn.execute("SELECT * FROM chunks WHERE id = ?", (str(chunk_id),))
        row = cur.fetchone()
        if row:
            return ChunkMetadata(
                chunk_id=UUID(row["id"]),
                document_id=UUID(row["document_id"]),
                document_filename=row["document_filename"],
                page_number=row["page_number"],
                chunk_index=row["chunk_index"],
                token_count=row["token_count"],
                char_start=row["char_start"],
                char_end=row["char_end"],
                content_hash=row["content_hash"],
            )
        return None

    def get_chunks_by_document(self, doc_id: UUID) -> list[ChunkMetadata]:
        """Obtiene todos los chunks de un documento."""
        cur = self.conn.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (str(doc_id),),
        )
        return [
            ChunkMetadata(
                chunk_id=UUID(row["id"]),
                document_id=UUID(row["document_id"]),
                document_filename=row["document_filename"],
                page_number=row["page_number"],
                chunk_index=row["chunk_index"],
                token_count=row["token_count"],
                char_start=row["char_start"],
                char_end=row["char_end"],
                content_hash=row["content_hash"],
            )
            for row in cur.fetchall()
        ]

    def delete_chunks_by_document(self, doc_id: UUID) -> int:
        """Elimina todos los chunks de un documento."""
        with self._transaction() as cur:
            cur.execute("DELETE FROM chunks WHERE document_id = ?", (str(doc_id),))
            return cur.rowcount

    # ========== Trazas ==========

    def save_trace(self, trace_data: dict[str, Any]) -> str:
        """Guarda traza de una consulta."""
        trace_id = str(UUID(int=int(datetime.now().timestamp() * 1_000_000) % (2**128)))
        with self._transaction() as cur:
            cur.execute(
                """
                INSERT INTO traces (id, timestamp, question, mode, answer, citations, abstained,
                                   abstention_reason, retrievals, critiques, total_latency_ms,
                                   total_tokens, model, seed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    trace_data["timestamp"],
                    trace_data["question"],
                    trace_data["mode"],
                    trace_data.get("answer"),
                    json.dumps(trace_data.get("citations", [])),
                    1 if trace_data.get("abstained") else 0,
                    trace_data.get("abstention_reason"),
                    json.dumps(trace_data.get("retrievals", [])),
                    json.dumps(trace_data.get("critiques", [])),
                    trace_data["total_latency_ms"],
                    trace_data.get("total_tokens"),
                    trace_data["model"],
                    trace_data.get("seed"),
                ),
            )
        return trace_id

    # ========== Manifiesto de índice ==========

    def save_manifest(self, manifest: IndexManifest) -> None:
        """Guarda manifiesto de índice (clave-valor JSON)."""
        data = manifest.model_dump(mode="json")
        with self._transaction() as cur:
            for key, value in data.items():
                cur.execute(
                    """
                    INSERT INTO index_manifest (key, value, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                    """,
                    (key, json.dumps(value), datetime.now().isoformat()),
                )

    def load_manifest(self) -> IndexManifest | None:
        """Carga manifiesto de índice."""
        cur = self.conn.execute("SELECT key, value FROM index_manifest")
        rows = cur.fetchall()
        if not rows:
            return None
        data = {row["key"]: json.loads(row["value"]) for row in rows}
        # Convertir documentos
        if "documents" in data:
            docs = {}
            for ch, doc_data in data["documents"].items():
                docs[ch] = Document(**doc_data)
            data["documents"] = docs
        return IndexManifest(**data)

    # ========== Evaluación ==========

    def save_experiment_manifest(self, manifest: ExperimentManifest) -> None:
        """Guarda manifiesto de experimento."""
        with self._transaction() as cur:
            cur.execute(
                """
                INSERT INTO experiment_manifest (id, name, created_at, config_hash, settings_snapshot,
                                                git_commit, random_seed, modes_evaluated, dataset_path,
                                                total_samples, completed_samples)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    completed_samples=excluded.completed_samples
                """,
                (
                    str(manifest.id),
                    manifest.name,
                    manifest.created_at.isoformat(),
                    manifest.config_hash,
                    json.dumps(manifest.settings_snapshot),
                    manifest.git_commit,
                    manifest.random_seed,
                    json.dumps([m.value for m in manifest.modes_evaluated]),
                    manifest.dataset_path,
                    manifest.total_samples,
                    manifest.completed_samples,
                ),
            )

    def get_experiment_manifest(self, exp_id: UUID) -> ExperimentManifest | None:
        """Obtiene manifiesto de experimento."""
        cur = self.conn.execute("SELECT * FROM experiment_manifest WHERE id = ?", (str(exp_id),))
        row = cur.fetchone()
        if row:
            return ExperimentManifest(
                id=UUID(row["id"]),
                name=row["name"],
                created_at=datetime.fromisoformat(row["created_at"]),
                config_hash=row["config_hash"],
                settings_snapshot=json.loads(row["settings_snapshot"]),
                git_commit=row["git_commit"],
                random_seed=row["random_seed"],
                modes_evaluated=[Mode(m) for m in json.loads(row["modes_evaluated"])],
                dataset_path=row["dataset_path"],
                total_samples=row["total_samples"],
                completed_samples=row["completed_samples"],
            )
        return None

    def save_evaluation_result(self, result: EvaluationResult, experiment_id: UUID) -> None:
        """Guarda resultado de evaluación."""
        with self._transaction() as cur:
            cur.execute(
                """
                INSERT INTO evaluation_results (id, experiment_id, sample_id, mode, question,
                                               predicted_answer, predicted_citations, abstained,
                                               latency_ms, tokens_used, retrieval_count, critique_count,
                                               correctness, faithfulness, citation_precision, citation_recall,
                                               annotated, annotator_notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    correctness=excluded.correctness,
                    faithfulness=excluded.faithfulness,
                    citation_precision=excluded.citation_precision,
                    citation_recall=excluded.citation_recall,
                    annotated=excluded.annotated,
                    annotator_notes=excluded.annotator_notes
                """,
                (
                    str(UUID(int=hash((str(experiment_id), result.sample_id, result.mode.value)) % (2**128))),
                    str(experiment_id),
                    result.sample_id,
                    result.mode.value,
                    result.question,
                    result.predicted_answer,
                    json.dumps([str(c) for c in result.predicted_citations]),
                    1 if result.abstained else 0,
                    result.latency_ms,
                    result.tokens_used,
                    result.retrieval_count,
                    result.critique_count,
                    result.correctness,
                    result.faithfulness,
                    result.citation_precision,
                    result.citation_recall,
                    1 if result.annotated else 0,
                    result.annotator_notes,
                    datetime.now().isoformat(),
                ),
            )

    def get_evaluation_results(self, experiment_id: UUID) -> list[EvaluationResult]:
        """Obtiene todos los resultados de un experimento."""
        cur = self.conn.execute(
            "SELECT * FROM evaluation_results WHERE experiment_id = ?", (str(experiment_id),)
        )
        results = []
        for row in cur.fetchall():
            results.append(
                EvaluationResult(
                    sample_id=row["sample_id"],
                    mode=Mode(row["mode"]),
                    question=row["question"],
                    predicted_answer=row["predicted_answer"],
                    predicted_citations=[UUID(c) for c in json.loads(row["predicted_citations"] or "[]")],
                    abstained=bool(row["abstained"]),
                    latency_ms=row["latency_ms"],
                    tokens_used=row["tokens_used"],
                    retrieval_count=row["retrieval_count"],
                    critique_count=row["critique_count"],
                    correctness=row["correctness"],
                    faithfulness=row["faithfulness"],
                    citation_precision=row["citation_precision"],
                    citation_recall=row["citation_recall"],
                    annotated=bool(row["annotated"]),
                    annotator_notes=row["annotator_notes"],
                )
            )
        return results

    def update_evaluation_annotations(
        self,
        experiment_id: UUID,
        sample_id: str,
        mode: Mode,
        correctness: float,
        faithfulness: float,
        citation_precision: float,
        citation_recall: float,
        annotator_notes: str | None = None,
    ) -> None:
        """Actualiza anotaciones humanas en resultado de evaluación."""
        with self._transaction() as cur:
            cur.execute(
                """
                UPDATE evaluation_results
                SET correctness=?, faithfulness=?, citation_precision=?, citation_recall=?,
                    annotated=1, annotator_notes=?
                WHERE experiment_id=? AND sample_id=? AND mode=?
                """,
                (
                    correctness,
                    faithfulness,
                    citation_precision,
                    citation_recall,
                    annotator_notes,
                    str(experiment_id),
                    sample_id,
                    mode.value,
                ),
            )

    def increment_completed_samples(self, experiment_id: UUID) -> None:
        """Incrementa contador de muestras completadas."""
        with self._transaction() as cur:
            cur.execute(
                "UPDATE experiment_manifest SET completed_samples = completed_samples + 1 WHERE id = ?",
                (str(experiment_id),),
            )

    def close(self) -> None:
        """Cierra conexión."""
        if self._conn:
            self._conn.close()
            self._conn = None


_sqlite_store_instance: SQLiteStore | None = None


def get_sqlite_store(settings=None) -> SQLiteStore:
    """Obtiene la instancia singleton del almacén SQLite."""
    global _sqlite_store_instance
    if _sqlite_store_instance is None or settings is not None:
        _sqlite_store_instance = SQLiteStore(settings)
    return _sqlite_store_instance


def reset_sqlite_store() -> None:
    """Resetea el almacén SQLite (útil para tests)."""
    global _sqlite_store_instance
    if _sqlite_store_instance is not None:
        _sqlite_store_instance.close()
    _sqlite_store_instance = None
