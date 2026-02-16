"""Risk Assessment node: risks per option, score risk level."""

import logging
import re
from typing import Any

from src.graph.state import DecisionGraphState
from src.llm.factory import invoke_for_text
from src.prompts.risk_analysis_prompt import (
    RISK_ANALYSIS_SYSTEM_PROMPT,
    RISK_ANALYSIS_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)


def _parse_risk_scores(text: str) -> tuple[dict[str, float], dict[str, str]]:
    """Extract option -> score and option -> level from risk analysis text."""
    scores: dict[str, float] = {}
    levels: dict[str, str] = {}
    # Look for "Option X: name" and "score: N" or "score: N/10"
    option = None
    for line in text.split("\n"):
        m = re.match(r"##\s*Option\s+[A-Z]:\s*(.+)", line, re.I)
        if m:
            option = m.group(1).strip()
        if option:
            sm = re.search(r"score:\s*(\d+(?:\.\d+)?)", line, re.I)
            if sm:
                scores[option] = min(10.0, max(1.0, float(sm.group(1))))
            lm = re.search(r"Risk level:\s*(LOW|MEDIUM|HIGH)", line, re.I)
            if lm:
                levels[option] = lm.group(1).upper()
    return scores, levels


def risk_assessment_node(state: DecisionGraphState) -> dict[str, Any]:
    """Identify risks per option and score risk level."""
    strategic_options = state.get("strategic_options") or ""
    internal_context = state.get("internal_context") or ""
    trace = list(state.get("reasoning_trace") or [])
    trace.append({"node": "risk_assessment", "summary": "Assessing risks per strategic option"})

    user = RISK_ANALYSIS_USER_TEMPLATE.format(
        strategic_options=strategic_options[:5000],
        context=internal_context[:3000],
    )
    full = f"{RISK_ANALYSIS_SYSTEM_PROMPT}\n\n{user}"
    risk_analysis = invoke_for_text(full)
    risk_scores, risk_levels = _parse_risk_scores(risk_analysis)

    return {
        "risk_analysis": risk_analysis,
        "risk_scores": risk_scores,
        "risk_levels": risk_levels,
        "reasoning_trace": trace,
    }
