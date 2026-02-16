"""Retrieval: embeddings and vector store (hybrid search: Chroma + BM25, RRF)."""

from src.retrieval.embeddings import get_embedding_model
from src.retrieval.vector_store import (
    get_vector_store,
    query_documents,
    invalidate_corpus_cache,
)

__all__ = ["get_embedding_model", "get_vector_store", "query_documents", "invalidate_corpus_cache"]
