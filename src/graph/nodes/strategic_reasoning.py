"""Strategic Reasoning node: multi-step reasoning, generate 2-4 options, pros/cons."""

import logging
from typing import Any

from src.graph.state import DecisionGraphState
from src.llm.factory import invoke_for_text
from src.prompts.strategic_planner_prompt import (
    STRATEGIC_PLANNER_SYSTEM_PROMPT,
    STRATEGIC_PLANNER_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)


def strategic_reasoning_node(state: DecisionGraphState) -> dict[str, Any]:
    """Generate strategic options with pros/cons from context."""
    question = state.get("question") or ""
    sub_questions = state.get("sub_questions") or []
    internal_context = state.get("internal_context") or ""
    trace = list(state.get("reasoning_trace") or [])
    trace.append({"node": "strategic_reasoning", "summary": "Generating strategic options and trade-offs"})

    sub_q_str = "\n".join(f"- {q}" for q in sub_questions)
    user = STRATEGIC_PLANNER_USER_TEMPLATE.format(
        context=internal_context[:6000],
        question=question,
        sub_questions=sub_q_str or "None",
    )
    full = f"{STRATEGIC_PLANNER_SYSTEM_PROMPT}\n\n{user}"
    strategic_options = invoke_for_text(full)

    return {
        "strategic_options": strategic_options,
        "trade_offs": strategic_options,
        "reasoning_trace": trace,
    }
