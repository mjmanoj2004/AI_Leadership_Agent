"""Shared state for the Decision Agent LangGraph. TypedDict for extensibility."""

from typing import Any, Literal, TypedDict


class DecisionGraphState(TypedDict, total=False):
    """State passed between nodes. All fields optional for flexibility and looping."""

    # Input
    question: str
    mode: str

    # Question analyzer
    classification: Literal["factual", "strategic"]
    intent: str
    sub_questions: list[str]

    # Internal research
    internal_context: str
    retrieved_sources: list[dict[str, Any]]
    retrieval_count: int

    # Knowledge gap
    knowledge_gaps: list[str]
    assumptions: list[str]
    context_sufficient: bool
    refined_sub_questions: list[str]

    # Strategic reasoning
    strategic_options: str
    trade_offs: str

    # Risk assessment
    risk_analysis: str
    risk_scores: dict[str, float]
    risk_levels: dict[str, str]

    # Synthesis
    final_answer: str
    confidence_level: str
    recommended_action: str

    # Control / routing
    next_step: str
    iteration_count: int
    max_iterations: int

    # Trace for API response
    reasoning_trace: list[dict[str, str]]
