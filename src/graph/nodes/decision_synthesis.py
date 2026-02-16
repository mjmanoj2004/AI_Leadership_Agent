"""Decision Synthesis node: recommend best option, rationale, structured output."""

import logging
from typing import Any

from src.graph.state import DecisionGraphState
from src.llm.factory import invoke_for_text
from src.prompts.synthesis_prompt import SYNTHESIS_SYSTEM_PROMPT, SYNTHESIS_USER_TEMPLATE

logger = logging.getLogger(__name__)


def decision_synthesis_node(state: DecisionGraphState) -> dict[str, Any]:
    """Synthesize final recommendation with Executive Summary, Options, Risk, Recommendation, Assumptions, Confidence."""
    question = state.get("question") or ""
    strategic_options = state.get("strategic_options") or ""
    risk_analysis = state.get("risk_analysis") or ""
    trace = list(state.get("reasoning_trace") or [])
    trace.append({"node": "decision_synthesis", "summary": "Synthesizing final recommendation"})

    user = SYNTHESIS_USER_TEMPLATE.format(
        question=question,
        strategic_options=strategic_options[:5000],
        risk_analysis=risk_analysis[:3000],
    )
    full = f"{SYNTHESIS_SYSTEM_PROMPT}\n\n{user}"
    final_answer = invoke_for_text(full)

    # Extract confidence from text if present
    confidence = "MEDIUM"
    if "confidence level" in final_answer.lower():
        for level in ("HIGH", "MEDIUM", "LOW"):
            if level in final_answer.upper():
                confidence = level
                break

    return {
        "final_answer": final_answer,
        "confidence_level": confidence,
        "recommended_action": final_answer,
        "reasoning_trace": trace,
    }
