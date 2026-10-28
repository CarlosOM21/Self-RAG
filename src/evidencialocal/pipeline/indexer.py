"""Pipeline de indexación con idempotencia y manifiesto."""

import logging
from pathlib import Path

from ..config.settings import get_settings
from ..core.exceptions import IndexingError
from ..core.models import IndexManifest, IndexStatus
from ..metadata.sqlite_store import get_sqlite_store
from ..retrieval.document_processor import get_document_processor
from ..vectorstore.chroma_store import get_chroma_store

logger = logging.getLogger(__name__)


class Indexer:
    """Indexa documentos: extrae, chunkea, embedde, almacena en Chroma y SQLite con manifiesto."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.processor = get_document_processor(settings)
        self.chroma = get_chroma_store(settings)
        self.sqlite = get_sqlite_store(settings)

    def load_manifest(self) -> IndexManifest:
        """Carga manifiesto existente o crea uno nuevo."""
        manifest = self.sqlite.load_manifest()
        if manifest is None:
            manifest = IndexManifest(
                embedding_model=self.settings.embedding.model_name,
                embedding_dimension=self.settings.embedding.dimension,
                chunk_size=self.settings.indexing.chunk_size,
                chunk_overlap=self.settings.indexing.chunk_overlap,
                tokenizer_model=self.settings.indexing.tokenizer_model,
            )
        return manifest

    def save_manifest(self, manifest: IndexManifest) -> None:
        """Guarda manifiesto."""
        self.sqlite.save_manifest(manifest)

    def index_directory(
        self,
        input_dir: Path,
        config_path: Path | None = None,
        force_reindex: bool = False,
    ) -> IndexManifest:
        """
        Indexa todos los PDFs en un directorio.
        Idempotente: salta documentos ya indexados (mismo hash).
        """
        if not input_dir.exists():
            raise IndexingError(f"Directorio no existe: {input_dir}")

        pdf_files = list(input_dir.rglob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No se encontraron PDFs en {input_dir}")
            return self.load_manifest()

        logger.info(f"Encontrados {len(pdf_files)} archivos PDF")

        # Cargar manifiesto
        manifest = self.load_manifest()

        # Verificar compatibilidad de configuración
        if not force_reindex:
            self._check_config_compatibility(manifest)

        # Procesar documentos
        new_chunks = []
        processed_count = 0
        skipped_count = 0
        failed_count = 0

        for pdf_path in pdf_files:
            try:
                # Calcular hash del archivo completo para idempotencia
                # Leer y hashear
                import hashlib
                content_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()

                # Verificar si ya está indexado
                existing_doc = manifest.get_document_by_hash(content_hash)
                if existing_doc and existing_doc.status == IndexStatus.READY and not force_reindex:
                    logger.debug(f"Saltando {pdf_path.name} (ya indexado)")
                    skipped_count += 1
                    continue

                # Procesar documento
                doc, chunks = self.processor.process_document(pdf_path)

                if doc.status == IndexStatus.READY and chunks:
                    # Guardar en Chroma
                    self.chroma.add_chunks(chunks)
                    # Guardar metadatos en SQLite
                    self.sqlite.upsert_document(doc)
                    for chunk in chunks:
                        self.sqlite.upsert_chunk_metadata(chunk.metadata)

                    # Actualizar manifiesto
                    manifest.add_document(doc)
                    manifest.total_chunks += len(chunks)
                    new_chunks.extend(chunks)
                    processed_count += 1
                else:
                    # Documento fallido
                    self.sqlite.upsert_document(doc)
                    manifest.add_document(doc)
                    failed_count += 1

            except Exception as e:
                logger.error(f"Error indexando {pdf_path}: {e}")
                failed_count += 1

        # Guardar manifiesto actualizado
        self.save_manifest(manifest)

        logger.info(
            f"Indexación completada: {processed_count} nuevos, "
            f"{skipped_count} saltados, {failed_count} fallidos. "
            f"Total chunks: {manifest.total_chunks}"
        )

        return manifest

    def _check_config_compatibility(self, manifest: IndexManifest) -> None:
        """Verifica compatibilidad de configuración con manifiesto existente."""
        mismatches = []
        if manifest.embedding_model != self.settings.embedding.model_name:
            mismatches.append(f"modelo embedding: {manifest.embedding_model} vs {self.settings.embedding.model_name}")
        if manifest.embedding_dimension != self.settings.embedding.dimension:
            mismatches.append(f"dimensión: {manifest.embedding_dimension} vs {self.settings.embedding.dimension}")
        if manifest.chunk_size != self.settings.indexing.chunk_size:
            mismatches.append(f"chunk_size: {manifest.chunk_size} vs {self.settings.indexing.chunk_size}")
        if manifest.chunk_overlap != self.settings.indexing.chunk_overlap:
            mismatches.append(f"chunk_overlap: {manifest.chunk_overlap} vs {self.settings.indexing.chunk_overlap}")
        if manifest.tokenizer_model != self.settings.indexing.tokenizer_model:
            mismatches.append(f"tokenizer: {manifest.tokenizer_model} vs {self.settings.indexing.tokenizer_model}")

        if mismatches:
            raise IndexingError(
                "Configuración incompatible con índice existente. "
                "Use --force-reindex o corrija la configuración:\n" + "\n".join(mismatches)
            )

    def reindex_all(self, input_dir: Path) -> IndexManifest:
        """Reindexación completa: borra Chroma y SQLite, vuelve a indexar."""
        logger.warning("Iniciando reindexación completa (borra índice anterior)")
        self.chroma.reset_collection()
        # SQLite: borrar documentos y chunks
        with self.sqlite._transaction() as cur:
            cur.execute("DELETE FROM chunks")
            cur.execute("DELETE FROM documents")
            cur.execute("DELETE FROM index_manifest")
        return self.index_directory(input_dir, force_reindex=True)

    def delete_document(self, doc_id: str) -> bool:
        """Elimina un documento del índice (Chroma + SQLite + manifiesto)."""
        from uuid import UUID
        doc_uuid = UUID(doc_id)

        # Obtener chunks del documento
        chunk_metas = self.sqlite.get_chunks_by_document(doc_uuid)
        chunk_ids = [cm.chunk_id for cm in chunk_metas]

        # Borrar de Chroma
        if chunk_ids:
            self.chroma.delete_chunks(chunk_ids)

        # Borrar de SQLite
        chunks_deleted = self.sqlite.delete_chunks_by_document(doc_uuid)
        doc_deleted = self.sqlite.delete_document(doc_uuid)

        # Actualizar manifiesto
        manifest = self.load_manifest()
        # Buscar y remover por ID
        doc_to_remove = None
        for ch, doc in manifest.documents.items():
            if doc.id == doc_uuid:
                doc_to_remove = ch
                break
        if doc_to_remove:
            del manifest.documents[doc_to_remove]
            manifest.total_documents = len(manifest.documents)
            manifest.total_chunks -= chunks_deleted
            self.save_manifest(manifest)

        return doc_deleted

    def get_index_stats(self) -> dict:
        """Estadísticas del índice actual."""
        manifest = self.load_manifest()
        return {
            "total_documents": manifest.total_documents,
            "total_chunks": manifest.total_chunks,
            "embedding_model": manifest.embedding_model,
            "embedding_dimension": manifest.embedding_dimension,
            "chunk_size": manifest.chunk_size,
            "chunk_overlap": manifest.chunk_overlap,
            "tokenizer_model": manifest.tokenizer_model,
            "documents_by_status": {
                status.value: sum(1 for d in manifest.documents.values() if d.status == status)
                for status in IndexStatus
            },
        }


_indexer_instance = None


def get_indexer(settings=None):
    """Obtiene la instancia singleton del indexador."""
    global _indexer_instance
    if _indexer_instance is None or settings is not None:
        _indexer_instance = Indexer(settings)
    return _indexer_instance


def reset_indexer():
    """Resetea el indexador (útil para tests)."""
    global _indexer_instance
    _indexer_instance = None
