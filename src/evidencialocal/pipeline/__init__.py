"""Pipeline principal: indexación y dos modos RAG."""

from .indexer import Indexer, get_indexer
from .rag_pipeline import RAGPipeline, get_rag_pipeline

__all__ = ["Indexer", "get_indexer", "RAGPipeline", "get_rag_pipeline"]
