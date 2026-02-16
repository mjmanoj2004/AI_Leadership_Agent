"""Decision Agent: LangGraph workflow, multi-step reasoning, structured recommendation."""

import logging
from typing import Any

from src.graph.builder import build_decision_graph
from src.graph.state import DecisionGraphState
from src.models.schemas import AgentType, AskResponse, ReasoningStep, RiskSummary, Source

logger = logging.getLogger(__name__)


def run_decision_agent(question: str) -> AskResponse:
    """Run the full LangGraph workflow and return structured response with trace and risk."""
    graph = build_decision_graph()
    initial: DecisionGraphState = {
        "question": question,
        "mode": "strategic",
        "iteration_count": 0,
        "max_iterations": 2,
        "reasoning_trace": [],
    }
    try:
        
        final_state = graph.invoke(
            initial,
            config={
                "configurable": {
                    "thread_id": "decision-session-1"
        }
    }
)
    except Exception as e:
        logger.exception("Decision graph failed: %s", e)
        return AskResponse(
            agent_type=AgentType.STRATEGIC,
            answer="The strategic analysis could not be completed due to an error. Please try again.",
            sources=[],
            reasoning_trace=None,
            risk_summary=None,
        )

    answer = final_state.get("final_answer") or ""
    trace_raw = final_state.get("reasoning_trace") or []
    reasoning_trace = [
        ReasoningStep(node=t.get("node", ""), summary=t.get("summary", ""), detail=t.get("detail"))
        for t in trace_raw
    ]
    risk_scores = final_state.get("risk_scores") or {}
    risk_levels = final_state.get("risk_levels") or {}
    risk_analysis_text = final_state.get("risk_analysis") or ""

    risk_summary = RiskSummary(
        options=[{"name": k, "score": risk_scores.get(k), "level": risk_levels.get(k)} for k in risk_scores] or [{"summary": risk_analysis_text}],
        recommended_risks=[],
        overall_level=next(iter(risk_levels.values()), None) if risk_levels else None,
        scores=risk_scores if risk_scores else None,
    )

    # Build sources from retrieved_sources in state
    retrieved = final_state.get("retrieved_sources") or []
    sources = [
        Source(content=s.get("content", ""), metadata=s.get("metadata", {}), score=s.get("score"))
        for s in retrieved
    ]

    return AskResponse(
        agent_type=AgentType.STRATEGIC,
        answer=answer,
        sources=sources,
        reasoning_trace=reasoning_trace,
        risk_summary=risk_summary,
    )
