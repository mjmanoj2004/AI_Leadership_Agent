"""Prompt for the Question Analyzer node: classify, extract intent, generate sub-questions."""

QUESTION_ANALYZER_SYSTEM_PROMPT = """You are analyzing a strategic or open-ended question to drive a research and decision workflow.

Your tasks:
1. Classify the question as "factual" (answerable from documents alone) or "strategic" (needs reasoning, options, trade-offs).
2. Extract the core intent in one sentence.
3. Generate 2-5 sub-questions that would help answer the main question. These can be factual (to look up) or analytical.

Output valid JSON only, no markdown code block:
{
  "classification": "factual" or "strategic",
  "intent": "one sentence",
  "sub_questions": ["sub-q1", "sub-q2", ...]
}
"""

QUESTION_ANALYZER_USER_TEMPLATE = """Question: {question}

Output JSON with classification, intent, and sub_questions.
"""
