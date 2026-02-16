"""Insight Agent: RAG-based, grounded in internal documents, with sources."""

import logging
import re
from typing import List

from src.models.schemas import AgentType, AskResponse, Source
from src.llm.factory import invoke_for_text
from src.prompts.insight_prompt import INSIGHT_SYSTEM_PROMPT, INSIGHT_USER_TEMPLATE
from src.retrieval.vector_store import query_documents

logger = logging.getLogger(__name__)


def _strip_source_artifacts_from_answer(text: str) -> str:
    """Remove any Source N, relevance %, and boilerplate that should not appear in the answer."""
    if not text or not text.strip():
        return text
    # Remove lines like "Source 1 (relevance: 3%)" or "Source 2"
    text = re.sub(r"(?m)^\s*Source\s+\d+\s*(?:\([^)]*relevance[^)]*\))?\s*\n?", "", text, flags=re.I)
    # Remove "(relevance: N%)" anywhere
    text = re.sub(r"\s*\(relevance:\s*[\d.]+%\)", "", text, flags=re.I)
    # Remove common boilerplate blocks (multi-line)
    blocks_to_remove = [
        r"Summary from internal documents\s*\n+",
        r"The following relevant information was found[^\n]*\n+",
        r"Use the Sources section below[^\n]*\n+",
        r"Answer\s*\(from internal documents\)\s*\n+",
        r"Based on the retrieved company documents[^\n]*\n+",
        r"here is the relevant information:\s*\n+",
    ]
    for pattern in blocks_to_remove:
        text = re.sub(pattern, "", text, flags=re.I)
    # Collapse multiple newlines and trim
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _format_answer_from_sources(question: str, sources_list: List[Source]) -> str:
    """When LLM fails, do not show raw content. Tell user a summary was not generated and point to Sources (file names)."""
    if not sources_list:
        return (
            "No relevant documents were found for this question. "
            "Try rephrasing or adding more documents to the knowledge base."
        )
    file_names = []
    for s in sources_list:
        name = (s.metadata or {}).get("source_file")
        if name and name not in file_names:
            file_names.append(name)
    files_line = ", ".join(file_names) if file_names else "the files listed under Sources"
    return (
        "A written summary could not be generated (model unavailable or error). "
        f"Relevant data was found in {files_line}. "
        "**To get AI summaries:** set `HF_TOKEN` in your `.env` file (HuggingFace token from https://huggingface.co/settings/tokens), then restart the API. See README for details."
    )


# Shorter prompt used when the main LLM call fails (often works with rate limits or smaller context)
_SIMPLE_SUMMARY_PROMPT = """Summarize the following for an executive in 3–5 bullet points. Be concise. Do not copy long passages.

Text:
{context}

Question: {question}

Bullet-point summary:"""


def run_insight_agent(question: str) -> AskResponse:
    """Retrieve relevant docs, build context, generate grounded answer with sources."""
    sources_list = query_documents(question)
    # Pass context without "Source 1/2" labels so the LLM does not echo them
    context = "\n\n---\n\n".join(
        (s.content or "").strip() for s in sources_list
    )
    if not context.strip():
        context = "No relevant internal documents were found for this question."

    user_prompt = INSIGHT_USER_TEMPLATE.format(context=context, question=question)
    full_prompt = f"{INSIGHT_SYSTEM_PROMPT}\n\n{user_prompt}"
    answer = None
    try:
        answer = invoke_for_text(full_prompt)
        if answer and answer.strip():
            answer = _strip_source_artifacts_from_answer(answer)
    except Exception as e:
        logger.warning("Main LLM call failed (%s), trying shorter fallback prompt", e)
        try:
            short_context = context[:2000].rstrip() + ("..." if len(context) > 2000 else "")
            simple_prompt = _SIMPLE_SUMMARY_PROMPT.format(context=short_context, question=question)
            answer = invoke_for_text(simple_prompt)
            if answer and answer.strip():
                answer = _strip_source_artifacts_from_answer(answer)
        except Exception as e2:
            logger.exception("Fallback LLM call also failed: %s", e2)
    if not (answer and answer.strip()):
        answer = _format_answer_from_sources(question, sources_list)

    return AskResponse(
        agent_type=AgentType.INSIGHT,
        answer=answer,
        sources=sources_list,
        reasoning_trace=None,
        risk_summary=None,
    )
