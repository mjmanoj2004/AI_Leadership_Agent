"""Question classification and routing to Insight or Decision agent."""

import logging
from typing import Literal

from src.agents.insight_agent import run_insight_agent
from src.agents.decision_agent import run_decision_agent
from src.llm.factory import invoke_for_text
from src.prompts.question_classifier_prompt import CLASSIFIER_SYSTEM_PROMPT, CLASSIFIER_USER_TEMPLATE
from src.models.schemas import AskRequest, AskResponse, AgentType

logger = logging.getLogger(__name__)


def classify_question(question: str) -> Literal["insight", "strategic"]:
    """Use lightweight LLM prompt to classify as factual (insight) or strategic (decision agent)."""
    user = CLASSIFIER_USER_TEMPLATE.format(question=question)
    full = f"{CLASSIFIER_SYSTEM_PROMPT}\n\n{user}"
    try:
        raw = invoke_for_text(full).strip().lower()
        if "strategic" in raw:
            return "strategic"
        return "insight"
    except Exception as e:
        logger.warning("Classifier failed (%s), defaulting to strategic", e)
        return "strategic"


def route_and_answer(request: AskRequest) -> AskResponse:
    """Route by mode (or classify when auto) and return response."""
    question = request.question.strip()
    if not question:
        return AskResponse(
            agent_type=AgentType.INSIGHT,
            answer="Please provide a question.",
            sources=[],
        )

    mode = (request.mode or "auto").lower()
    if mode == "insight":
        return run_insight_agent(question)
    if mode == "strategic":
        return run_decision_agent(question)
    # auto
    kind = classify_question(question)
    if kind == "insight":
        return run_insight_agent(question)
    return run_decision_agent(question)
