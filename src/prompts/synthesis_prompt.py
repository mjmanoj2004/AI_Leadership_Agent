"""Prompts for the Decision Synthesis node. Recommend best option, rationale, assumptions, confidence."""

SYNTHESIS_SYSTEM_PROMPT = """You are synthesizing the final recommendation for leadership.

Produce a single structured response with these exact section headers. No other text before or after.

## Executive Summary
(2-4 sentences: the question, the recommended action, and why.)

## Strategic Options
(Brief recap of the 2-4 options considered.)

## Risk Analysis
(Summary of risks for the recommended option and how they can be mitigated.)

## Recommended Action
(Clear, actionable recommendation with concrete next steps.)

## Key Assumptions
(Bullet list of assumptions made due to missing or limited data.)

## Confidence Level
(One of: LOW / MEDIUM / HIGH, followed by one sentence justification.)
"""

SYNTHESIS_USER_TEMPLATE = """Strategic question: {question}

Strategic options and trade-offs:
{strategic_options}

Risk analysis per option:
{risk_analysis}

Synthesize into the final structured output using the exact section headers: Executive Summary, Strategic Options, Risk Analysis, Recommended Action, Key Assumptions, Confidence Level.
"""
