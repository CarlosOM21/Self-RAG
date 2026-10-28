"""Almacén vectorial con ChromaDB (distancia coseno)."""

import logging
from typing import Any
from uuid import UUID

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..config.settings import get_settings
from ..core.exceptions import RetrievalError
from ..core.models import Chunk, RetrievalResult

logger = logging.getLogger(__name__)


class ChromaStore:
    """Wrapper para ChromaDB con colección persistente y distancia coseno."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.chroma_settings = self.settings.chroma
        self._client: chromadb.PersistentClient | None = None
        self._collection: chromadb.Collection | None = None

    @property
    def client(self) -> chromadb.PersistentClient:
        """Cliente ChromaDB persistente (lazy)."""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=self.chroma_settings.persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        """Colección ChromaDB (lazy, crea si no existe)."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.chroma_settings.collection_name,
                metadata={"hnsw:space": self.chroma_settings.distance_metric},
            )
        return self._collection

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Añade chunks a la colección."""
        if not chunks:
            return

        try:
            ids = [str(c.id) for c in chunks]
            documents = [c.text for c in chunks]
            embeddings = [c.embedding for c in chunks]
            metadatas = [c.metadata.model_dump(mode="json") for c in chunks]

            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            logger.info(f"Añadidos {len(chunks)} chunks a ChromaDB")

        except Exception as e:
            logger.error(f"Error añadiendo chunks a ChromaDB: {e}")
            raise RetrievalError(f"Error añadiendo chunks: {e}")

    def delete_chunks(self, chunk_ids: list[UUID]) -> None:
        """Elimina chunks por IDs."""
        if not chunk_ids:
            return

        try:
            ids = [str(cid) for cid in chunk_ids]
            self.collection.delete(ids=ids)
            logger.info(f"Eliminados {len(chunk_ids)} chunks de ChromaDB")
        except Exception as e:
            logger.error(f"Error eliminando chunks: {e}")
            raise RetrievalError(f"Error eliminando chunks: {e}")

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Consulta por similitud coseno (mayor score = más similar)."""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            retrieval_results = []
            if results["ids"] and results["ids"][0]:
                for rank, (chunk_id, document, metadata, distance) in enumerate(
                    zip(
                        results["ids"][0],
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0],
                    ),
                    start=1,
                ):
                    # Chroma devuelve distancia coseno (0 = idéntico, 2 = opuesto)
                    # Convertir a similitud coseno: 1 - distance/2
                    # O más directamente: score = 1 - distance (para coseno normalizado)
                    # Con vectores normalizados: coseno = 1 - distancia/2
                    score = 1.0 - (distance / 2.0)

                    retrieval_results.append(
                        RetrievalResult(
                            chunk_id=UUID(chunk_id),
                            document_id=UUID(metadata["document_id"]),
                            document_filename=metadata["document_filename"],
                            page_number=metadata.get("page_number"),
                            text=document,
                            score=max(0.0, min(1.0, score)),  # clamp [0,1]
                            rank=rank,
                        )
                    )

            return retrieval_results

        except Exception as e:
            logger.error(f"Error consultando ChromaDB: {e}")
            raise RetrievalError(f"Error en recuperación: {e}")

    def get_chunk_count(self) -> int:
        """Número total de chunks en la colección."""
        try:
            return self.collection.count()
        except Exception:
            return 0

    def reset_collection(self) -> None:
        """Elimina y recrea la colección (para reindexación completa)."""
        try:
            self.client.delete_collection(name=self.chroma_settings.collection_name)
            self._collection = None
            # Se recreará en next access
            _ = self.collection
            logger.info("Colección ChromaDB reiniciada")
        except Exception as e:
            logger.error(f"Error reiniciando colección: {e}")
            raise RetrievalError(f"Error reiniciando colección: {e}")

    def close(self) -> None:
        """Cierra el cliente (ChromaDB persistente no requiere close explícito)."""
        self._client = None
        self._collection = None


_chroma_store_instance: ChromaStore | None = None


def get_chroma_store(settings=None) -> ChromaStore:
    """Obtiene la instancia singleton del vector store."""
    global _chroma_store_instance
    if _chroma_store_instance is None or settings is not None:
        _chroma_store_instance = ChromaStore(settings)
    return _chroma_store_instance


def reset_chroma_store() -> None:
    """Resetea el vector store (útil para tests)."""
    global _chroma_store_instance
    if _chroma_store_instance is not None:
        _chroma_store_instance.close()
    _chroma_store_instance = None
