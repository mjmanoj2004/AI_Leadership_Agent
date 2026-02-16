"""Chroma vector store with hybrid search: semantic (Chroma) + keyword (BM25) fused by RRF."""

import logging
import re
from typing import List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from config import get_settings
from src.models.schemas import Source
from src.retrieval.embeddings import get_embedding_model

logger = logging.getLogger(__name__)

# Hybrid search: top 5 from each system, RRF fusion → final top 5
HYBRID_TOP_K = 5
RRF_K = 60  # Reciprocal Rank Fusion constant

_vector_store: Optional[Chroma] = None
_bm25_corpus_cache: Optional[Tuple[List[str], List[dict], BM25Okapi]] = None  # (texts, metadatas, bm25)


def _normalize_text(text: str) -> str:
    """Normalize for matching across semantic and keyword results."""
    return re.sub(r"\s+", " ", (text or "").strip())


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer for BM25: lowercase, split on non-alphanumeric."""
    return re.findall(r"\w+", (text or "").lower())


def get_vector_store() -> Chroma:
    """Return singleton Chroma vector store."""
    global _vector_store
    if _vector_store is None:
        settings = get_settings()
        persist_dir = str(settings.chroma_persist_dir)
        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        _vector_store = Chroma(
            client=client,
            collection_name=settings.chroma_collection_name,
            embedding_function=get_embedding_model(),
        )
        logger.info("Chroma vector store ready: %s", persist_dir)
    return _vector_store


def _get_bm25_corpus() -> Tuple[List[str], List[dict], BM25Okapi]:
    """Get or build BM25 corpus from Chroma (all stored chunks). Returns (texts, metadatas, bm25)."""
    global _bm25_corpus_cache
    if _bm25_corpus_cache is not None:
        return _bm25_corpus_cache

    store = get_vector_store()
    coll = getattr(store, "_collection", None)
    if coll is None:
        # Fallback: get client and collection by name
        client = getattr(store, "_client", None)
        if client is not None:
            settings = get_settings()
            coll = client.get_collection(settings.chroma_collection_name)
    if coll is None:
        raise RuntimeError("Could not access Chroma collection for BM25 corpus")

    raw = coll.get(include=["documents", "metadatas"])
    ids_list = raw.get("ids") or []
    docs_list = raw.get("documents") or []
    metadatas_list = raw.get("metadatas") or [{}] * len(docs_list)
    if len(metadatas_list) != len(docs_list):
        metadatas_list = [{}] * len(docs_list)

    texts = [d or "" for d in docs_list]
    if not texts:
        tokenized = []
        bm25 = BM25Okapi(tokenized)
        _bm25_corpus_cache = (texts, metadatas_list, bm25)
        return _bm25_corpus_cache

    tokenized_corpus = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    _bm25_corpus_cache = (texts, metadatas_list, bm25)
    logger.debug("BM25 corpus built: %d chunks", len(texts))
    return _bm25_corpus_cache


def invalidate_corpus_cache() -> None:
    """Call after adding or removing documents so BM25 corpus is rebuilt on next query."""
    global _bm25_corpus_cache
    _bm25_corpus_cache = None
    logger.debug("BM25 corpus cache invalidated")


def _reciprocal_rank_fusion(
    chroma_results: List[Tuple[str, dict, float]],  # (content, metadata, chroma_relevance)
    bm25_results: List[Tuple[str, dict]],           # (content, metadata)
    k: int = RRF_K,
    top_n: int = HYBRID_TOP_K,
) -> List[Source]:
    """Merge semantic and keyword rankings with RRF; return top_n Sources."""
    # RRF score: sum over systems 1 / (k + rank), rank 1-based
    rrf_scores: dict[str, float] = {}
    content_to_meta: dict[str, dict] = {}

    for rank, (content, meta, _) in enumerate(chroma_results, start=1):
        key = _normalize_text(content)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
        content_to_meta[key] = meta

    for rank, (content, meta) in enumerate(bm25_results, start=1):
        key = _normalize_text(content)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
        if key not in content_to_meta:
            content_to_meta[key] = meta

    # Sort by RRF score descending, take top_n; we need original content for Source (use first seen)
    key_to_content: dict[str, str] = {}
    for content, meta, _ in chroma_results:
        key_to_content[_normalize_text(content)] = content
    for content, meta in bm25_results:
        key = _normalize_text(content)
        if key not in key_to_content:
            key_to_content[key] = content

    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: -rrf_scores[x])
    sources: List[Source] = []
    for key in sorted_keys[:top_n]:
        content = key_to_content.get(key, key)
        meta = content_to_meta.get(key, {})
        score = round(rrf_scores[key], 4)
        sources.append(Source(content=content, metadata=meta, score=score))
    return sources


def query_documents(
    query: str,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
    use_hybrid: bool = True,
) -> List[Source]:
    """
    Hybrid retrieval: semantic (Chroma top 5) + keyword (BM25 top 5) → RRF → final top 5.
    If use_hybrid is False or BM25 corpus is empty, falls back to Chroma-only.
    """
    store = get_vector_store()
    settings = get_settings()
    k = top_k or settings.top_k_retrieve
    threshold = score_threshold if score_threshold is not None else settings.retrieval_score_threshold
    n = min(k, HYBRID_TOP_K)  # per-system and final top

    # 1) Semantic: Chroma similarity search (top 5)
    chroma_results = store.similarity_search_with_score(query, k=n)
    chroma_list: List[Tuple[str, dict, float]] = []
    for doc, score in chroma_results:
        relevance = 1.0 - score if score <= 1.0 else 1.0 / (1.0 + score)
        if relevance >= threshold:
            chroma_list.append((doc.page_content, doc.metadata or {}, round(relevance, 4)))

    # 2) Keyword: BM25 over all stored chunks (top 5)
    bm25_list: List[Tuple[str, dict]] = []
    try:
        texts, metadatas_list, bm25 = _get_bm25_corpus()
        if texts and bm25 is not None:
            tokenized_query = _tokenize(query)
            if tokenized_query:
                doc_scores = bm25.get_scores(tokenized_query)
                # top n indices by score
                indices = sorted(range(len(doc_scores)), key=lambda i: -doc_scores[i])[:n]
                for i in indices:
                    if doc_scores[i] > 0:
                        bm25_list.append((texts[i], metadatas_list[i] if i < len(metadatas_list) else {}))
    except Exception as e:
        logger.warning("BM25 retrieval failed (%s), using semantic-only", e)

    # 3) RRF fusion → final top 5
    if use_hybrid and (chroma_list or bm25_list):
        return _reciprocal_rank_fusion(chroma_list, bm25_list, k=RRF_K, top_n=n)
    # Fallback: Chroma-only
    return [
        Source(content=content, metadata=meta, score=rel)
        for content, meta, rel in chroma_list
    ]
