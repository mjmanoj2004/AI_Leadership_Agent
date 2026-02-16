"""Internal Research node: query vector DB and gather company context."""

import logging
from typing import Any

from src.graph.state import DecisionGraphState
from src.retrieval.vector_store import query_documents

logger = logging.getLogger(__name__)


def internal_research_node(state: DecisionGraphState) -> dict[str, Any]:
    """Query Chroma for main question and sub-questions; aggregate context."""
    question = state.get("question") or ""
    sub_questions = state.get("sub_questions") or []
    trace = list(state.get("reasoning_trace") or [])
    trace.append({"node": "internal_research", "summary": "Retrieving internal company documents"})

    all_sources: list[dict[str, Any]] = []
    seen = set()

    for q in [question] + list(sub_questions):
        sources = query_documents(q)
        for s in sources:
            key = (s.content[:100], s.metadata.get("source_file", ""))
            if key not in seen:
                seen.add(key)
                all_sources.append({"content": s.content, "metadata": s.metadata, "score": s.score})

    context_parts = [f"[{i+1}] {s['content']}" for i, s in enumerate(all_sources)]
    internal_context = "\n\n".join(context_parts) if context_parts else "No relevant internal documents found."

    return {
        "internal_context": internal_context,
        "retrieved_sources": all_sources,
        "retrieval_count": len(all_sources),
        "reasoning_trace": trace,
    }
