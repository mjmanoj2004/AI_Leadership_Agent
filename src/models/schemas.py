"""Request/response and shared data models. Type hints throughout."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class AgentType(str, Enum):
    """Which agent handled the request."""

    INSIGHT = "insight"
    STRATEGIC = "strategic"


class AskRequest(BaseModel):
    """POST /ask request body."""

    question: str = Field(..., min_length=1, description="User question")
    mode: str = Field(default="auto", description="auto | insight | strategic")


class Source(BaseModel):
    """A single source citation from retrieval."""

    content: str = Field(..., description="Text chunk or snippet")
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None


class ReasoningStep(BaseModel):
    """One step in the reasoning trace (e.g. from LangGraph)."""

    node: str = Field(..., description="Graph node name")
    summary: str = Field(..., description="Short description of step")
    detail: Optional[str] = None


class RiskSummary(BaseModel):
    """Risk analysis summary for strategic agent."""

    options: list[dict[str, Any]] = Field(default_factory=list)
    recommended_risks: list[str] = Field(default_factory=list)
    overall_level: Optional[str] = None
    scores: Optional[dict[str, float]] = None


class AskResponse(BaseModel):
    """POST /ask response body."""

    agent_type: AgentType = Field(..., description="Which agent answered")
    answer: str = Field(..., description="Final answer text")
    sources: list[Source] = Field(default_factory=list)
    reasoning_trace: Optional[list[ReasoningStep]] = None
    risk_summary: Optional[RiskSummary] = None
