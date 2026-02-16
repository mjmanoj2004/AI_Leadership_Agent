"""LangGraph workflow builder. Conditional edges and loop for re-query / refine."""

import logging
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import DecisionGraphState
from src.graph.nodes import (
    question_analyzer_node,
    internal_research_node,
    knowledge_gap_node,
    strategic_reasoning_node,
    risk_assessment_node,
    decision_synthesis_node,
)

logger = logging.getLogger(__name__)

# Max re-query loops to avoid infinite loops
DEFAULT_MAX_ITERATIONS = 2


def _route_after_knowledge_gap(state: DecisionGraphState) -> Literal["internal_research", "strategic_reasoning"]:
    """If context insufficient and we have refined sub-questions and under max iterations, re-query else proceed."""
    sufficient = state.get("context_sufficient", True)
    refined = state.get("refined_sub_questions") or []
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", DEFAULT_MAX_ITERATIONS)
    if not sufficient and refined and iteration < max_iter:
        return "internal_research"
    return "strategic_reasoning"


def build_decision_graph(checkpointer=None):
    """
    Build the Decision Agent LangGraph. Supports looping: insufficient context -> re-query.
    Design allows plugging new nodes via graph.add_node() and add_conditional_edges().
    """
    graph = StateGraph(DecisionGraphState)

    # Register nodes (extensible: add more nodes here)
    graph.add_node("question_analyzer", question_analyzer_node)
    graph.add_node("internal_research", internal_research_node)
    graph.add_node("knowledge_gap", knowledge_gap_node)
    graph.add_node("strategic_reasoning", strategic_reasoning_node)
    graph.add_node("risk_assessment", risk_assessment_node)
    graph.add_node("decision_synthesis", decision_synthesis_node)

    # Entry
    graph.set_entry_point("question_analyzer")

    # question_analyzer -> internal_research
    graph.add_edge("question_analyzer", "internal_research")

    graph.add_edge("internal_research", "knowledge_gap")

    # knowledge_gap -> conditional: re-query or strategic_reasoning
    graph.add_conditional_edges("knowledge_gap", _route_after_knowledge_gap)

    # strategic_reasoning -> risk_assessment -> decision_synthesis -> END
    graph.add_edge("strategic_reasoning", "risk_assessment")
    graph.add_edge("risk_assessment", "decision_synthesis")
    graph.add_edge("decision_synthesis", END)

    # Optional: when looping to internal_research, we need to pass refined_sub_questions into retrieval.
    # The internal_research node currently uses state["sub_questions"]; we can merge refined_sub_questions into sub_questions in knowledge_gap or in a small wrapper.
    # For simplicity, internal_research_node already uses sub_questions; we'll have knowledge_gap_node set sub_questions to refined_sub_questions when we loop (so state update in knowledge_gap should add "sub_questions": refined_sub_questions when context_sufficient is False). Let me add that in knowledge_gap return.
    # Actually when we loop, the state gets merged. So when we go back to internal_research, state["sub_questions"] will still be the original. We need to update sub_questions when we decide to re-query. So in the edge from knowledge_gap to internal_research we need the state to have sub_questions = refined_sub_questions. So in knowledge_gap_node return we add "sub_questions": state.get("refined_sub_questions") or state.get("sub_questions") when context_sufficient is False. Let me update knowledge_gap_node to return refined sub_questions as the new sub_questions when re-querying.
    # I'll add in knowledge_gap return: if not context_sufficient and refined_sub_questions: also return {"sub_questions": refined_sub_questions, "iteration_count": (state.get("iteration_count") or 0) + 1}.
    # And in initial state we pass iteration_count=0, max_iterations=DEFAULT_MAX_ITERATIONS.
    comp = graph.compile(checkpointer=checkpointer or MemorySaver())
    return comp
