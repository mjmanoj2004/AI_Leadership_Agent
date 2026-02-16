"""Knowledge Gap node: detect missing information, list assumptions, decide if context sufficient."""

import json
import logging
from typing import Any

from src.graph.state import DecisionGraphState
from src.llm.factory import invoke_for_text

logger = logging.getLogger(__name__)

KNOWLEDGE_GAP_SYSTEM = """You are a research analyst. Given the strategic question and the internal context retrieved, determine:
1. What information is missing or incomplete to fully answer the question?
2. What assumptions would we need to make?
3. Is the internal context sufficient to proceed with strategic options? Answer "yes" or "no".

Output valid JSON only:
{
  "knowledge_gaps": ["gap1", "gap2"],
  "assumptions": ["assumption1", "assumption2"],
  "context_sufficient": true or false,
  "refined_sub_questions": ["optional refined sub-question if we need to re-query"]
}
"""


def knowledge_gap_node(state: DecisionGraphState) -> dict[str, Any]:
    """Detect gaps, list assumptions, set context_sufficient and optionally refined_sub_questions."""
    question = state.get("question") or ""
    intent = state.get("intent", "")
    internal_context = state.get("internal_context") or ""
    trace = list(state.get("reasoning_trace") or [])
    trace.append({"node": "knowledge_gap", "summary": "Assessing knowledge gaps and assumptions"})

    user = f"Question: {question}\nIntent: {intent}\n\nInternal context:\n{internal_context[:4000]}"
    full = f"{KNOWLEDGE_GAP_SYSTEM}\n\n{user}"
    try:
        raw = invoke_for_text(full)
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        data = json.loads(text)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Knowledge gap JSON parse failed: %s", e)
        data = {
            "knowledge_gaps": [],
            "assumptions": ["Limited internal data available."],
            "context_sufficient": True,
            "refined_sub_questions": [],
        }

    refined = data.get("refined_sub_questions", [])
    sufficient = bool(data.get("context_sufficient", True))
    out: dict[str, Any] = {
        "knowledge_gaps": data.get("knowledge_gaps", []),
        "assumptions": data.get("assumptions", []),
        "context_sufficient": sufficient,
        "refined_sub_questions": refined,
        "reasoning_trace": trace,
    }
    if not sufficient and refined:
        out["sub_questions"] = refined
        out["iteration_count"] = (state.get("iteration_count") or 0) + 1
    return out
