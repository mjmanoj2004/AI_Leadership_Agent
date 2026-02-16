"""Sentence-transformers embedding model. Config-driven."""

import logging
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from config import get_settings

logger = logging.getLogger(__name__)

_embedding_model: HuggingFaceEmbeddings | None = None


def get_embedding_model() -> Embeddings:
    """Return singleton HuggingFace embeddings (sentence-transformers)."""
    global _embedding_model
    if _embedding_model is None:
        settings = get_settings()
        _embedding_model = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Loaded embedding model: %s", settings.embedding_model_name)
    return _embedding_model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts. Convenience wrapper."""
    return get_embedding_model().embed_documents(texts)
