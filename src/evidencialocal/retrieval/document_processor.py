"""Procesador de documentos: extracción PDF y chunking con tokenizer E5."""

import hashlib
import logging
from pathlib import Path
from uuid import uuid4

import pypdf
from transformers import AutoTokenizer

from ..config.settings import get_settings
from ..core.exceptions import IndexingError
from ..core.models import Chunk, ChunkMetadata, Document, IndexStatus
from ..embeddings.embedder import get_embedder

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Procesa documentos PDF: extrae texto por páginas, hace chunking con tokenizer E5, genera embeddings."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.indexing_settings = self.settings.indexing
        self.embedding_settings = self.settings.embedding
        self._tokenizer: AutoTokenizer | None = None
        self._embedder = get_embedder(settings)

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Tokenizer del modelo E5 (lazy)."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.indexing_settings.tokenizer_model,
                use_fast=True,
            )
        return self._tokenizer

    def compute_content_hash(self, content: str) -> str:
        """SHA256 del contenido para idempotencia."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def extract_pdf_pages(self, pdf_path: Path) -> list[tuple[int, str]]:
        """Extrae texto de cada página del PDF. Returns: [(page_num, text), ...]"""
        pages = []
        try:
            with open(pdf_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    # Limpiar texto
                    text = " ".join(text.split())
                    if text.strip():
                        pages.append((i + 1, text))  # 1-based page numbers
            logger.info(f"Extraídas {len(pages)} páginas de {pdf_path.name}")
        except Exception as e:
            logger.error(f"Error extrayendo PDF {pdf_path}: {e}")
            raise IndexingError(f"Error extrayendo PDF: {e}")
        return pages

    def chunk_text_by_tokens(
        self,
        text: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[tuple[str, int, int]]:
        """
        Divide texto en chunks por tokens (tokenizer E5).
        Returns: [(chunk_text, char_start, char_end), ...]
        """
        chunk_size = chunk_size or self.indexing_settings.chunk_size
        chunk_overlap = chunk_overlap or self.indexing_settings.chunk_overlap
        min_chunk_size = self.indexing_settings.min_chunk_size

        # Tokenizar todo el texto
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            return []

        chunks = []
        start_token = 0
        char_pos = 0

        # Mapear tokens a posiciones de caracteres (aproximado)
        # Usamos decode para obtener el texto de cada ventana de tokens
        while start_token < len(tokens):
            end_token = min(start_token + chunk_size, len(tokens))
            chunk_tokens = tokens[start_token:end_token]
            chunk_text = self.tokenizer.decode(chunk_tokens, clean_up_tokenization_spaces=True)

            # Calcular posiciones aproximadas de caracteres
            # Decodificar tokens previos para saber dónde empieza
            prev_text = self.tokenizer.decode(tokens[:start_token], clean_up_tokenization_spaces=True)
            char_start = len(prev_text)
            char_end = char_start + len(chunk_text)

            if len(chunk_tokens) >= min_chunk_size:
                chunks.append((chunk_text.strip(), char_start, char_end))

            if end_token >= len(tokens):
                break

            start_token += chunk_size - chunk_overlap

        return chunks

    def process_document(self, pdf_path: Path) -> Document:
        """Procesa un documento PDF completo: extrae, chunkea, embedde."""
        # Extraer páginas
        pages = self.extract_pdf_pages(pdf_path)
        if not pages:
            raise IndexingError(f"PDF sin texto extraíble: {pdf_path}")

        # Texto completo para hash
        full_text = "\n\n".join(page_text for _, page_text in pages)
        content_hash = self.compute_content_hash(full_text)

        # Crear documento base
        doc = Document(
            source_path=pdf_path,
            filename=pdf_path.name,
            content_hash=content_hash,
            page_count=len(pages),
            status=IndexStatus.PENDING,
        )

        # Chunking por páginas para mantener página real
        all_chunks = []
        chunk_index = 0

        for page_num, page_text in pages:
            page_chunks = self.chunk_text_by_tokens(page_text)
            for chunk_text, char_start, char_end in page_chunks:
                # Embedding del chunk
                embedding = self._embedder.encode_single_passage(chunk_text)

                # Metadatos
                chunk_meta = ChunkMetadata(
                    chunk_id=uuid4(),
                    document_id=doc.id,
                    document_filename=doc.filename,
                    page_number=page_num,
                    chunk_index=chunk_index,
                    token_count=len(self.tokenizer.encode(chunk_text, add_special_tokens=False)),
                    char_start=char_start,
                    char_end=char_end,
                    content_hash=self.compute_content_hash(chunk_text),
                )

                chunk = Chunk(
                    id=chunk_meta.chunk_id,
                    document_id=doc.id,
                    text=chunk_text,
                    embedding=embedding,
                    metadata=chunk_meta,
                )
                all_chunks.append(chunk)
                chunk_index += 1

        doc.status = IndexStatus.READY
        logger.info(f"Documento {pdf_path.name} procesado: {len(all_chunks)} chunks")
        return doc, all_chunks

    def process_documents_batch(self, pdf_paths: list[Path]) -> list[tuple[Document, list[Chunk]]]:
        """Procesa múltiples documentos."""
        results = []
        for pdf_path in pdf_paths:
            try:
                doc, chunks = self.process_document(pdf_path)
                results.append((doc, chunks))
            except Exception as e:
                logger.error(f"Error procesando {pdf_path}: {e}")
                # Crear documento fallido
                doc = Document(
                    source_path=pdf_path,
                    filename=pdf_path.name,
                    content_hash="",
                    page_count=0,
                    status=IndexStatus.FAILED,
                    error_message=str(e),
                )
                results.append((doc, []))
        return results


_document_processor_instance: DocumentProcessor | None = None


def get_document_processor(settings=None) -> DocumentProcessor:
    """Obtiene la instancia singleton del procesador de documentos."""
    global _document_processor_instance
    if _document_processor_instance is None or settings is not None:
        _document_processor_instance = DocumentProcessor(settings)
    return _document_processor_instance


def reset_document_processor() -> None:
    """Resetea el procesador (útil para tests)."""
    global _document_processor_instance
    _document_processor_instance = None
