"""
HuggingFace LLM factory.
Config-driven singleton factory using HuggingFace Hub.
Compatible with latest langchain + langchain-huggingface.
"""

import logging
from typing import Optional, Union

from config import get_settings

from langchain_core.language_models import BaseChatModel, BaseLLM
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

logger = logging.getLogger(__name__)

_llm: Optional[Union[BaseLLM, BaseChatModel]] = None


def get_llm() -> Union[BaseLLM, BaseChatModel]:
    """
    Return singleton HuggingFace LLM (Hub API).
    Wrapped as ChatHuggingFace for chat-style invocation.
    """
    global _llm

    if _llm is not None:
        return _llm

    settings = get_settings()
    model_name = settings.llm_model_name
    token = settings.huggingface_hub_token

    if not token:
        raise ValueError(
            "HUGGINGFACE_HUB_TOKEN not found. "
            "Please set it in your .env file."
        )

    # Create base endpoint LLM
    base_llm = HuggingFaceEndpoint(
        repo_id=model_name,
        huggingfacehub_api_token=token,
        max_new_tokens=1024,
        temperature=0.3,
    )

    # Wrap endpoint as chat model
    _llm = ChatHuggingFace(llm=base_llm)

    logger.info("Using HuggingFace Hub LLM: %s", model_name)

    return _llm


def invoke_for_text(prompt: str, max_retries: int = 2) -> str:
    """
    Invoke LLM with a string prompt.
    Works with both ChatModel and BaseLLM.
    Retries automatically on failure.
    """
    llm = get_llm()
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            if isinstance(llm, BaseChatModel):
                response = llm.invoke([HumanMessage(content=prompt)])
                text = getattr(response, "content", str(response))
            else:
                text = llm.invoke(prompt)

            if text and str(text).strip():
                return str(text).strip()

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                logger.warning(
                    "LLM attempt %s failed (%s), retrying...",
                    attempt + 1,
                    e,
                )
            else:
                logger.exception(
                    "LLM failed after %s attempts",
                    max_retries + 1,
                )

    raise last_error or RuntimeError("LLM returned no response")
