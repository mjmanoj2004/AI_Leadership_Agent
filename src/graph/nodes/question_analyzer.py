"""Question Analyzer node: classify, extract intent, generate sub-questions."""

import json
import logging
from typing import Any

from src.graph.state import DecisionGraphState
from src.llm.factory import get_llm, invoke_for_text
from src.prompts.question_analyzer_prompt import (
    QUESTION_ANALYZER_SYSTEM_PROMPT,
    QUESTION_ANALYZER_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)


def question_analyzer_node(state: DecisionGraphState) -> dict[str, Any]:
    """Classify question, extract intent, generate sub-questions. Updates state."""
    question = state.get("question") or ""
    trace = list(state.get("reasoning_trace") or [])
    trace.append({"node": "question_analyzer", "summary": "Analyzing question and generating sub-questions"})

    user_prompt = QUESTION_ANALYZER_USER_TEMPLATE.format(question=question)
    full_prompt = f"{QUESTION_ANALYZER_SYSTEM_PROMPT}\n\n{user_prompt}"
    try:
        raw = invoke_for_text(full_prompt)
        # Strip markdown code block if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        data = json.loads(text)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Question analyzer JSON parse failed: %s. Using defaults.", e)
        data = {
            "classification": "strategic",
            "intent": question,
            "sub_questions": [question],
        }

    return {
        "classification": data.get("classification", "strategic"),
        "intent": data.get("intent", question),
        "sub_questions": data.get("sub_questions", [question]),
        "reasoning_trace": trace,
    }
