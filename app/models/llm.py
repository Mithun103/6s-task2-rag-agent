"""Provides the application's primary language model instance.

This module contains a factory function for initializing and configuring the
ChatGroq language model client using settings from the application's
central configuration.
"""

from langchain_groq import ChatGroq
from pydantic import SkipValidation
from app.config import settings


def get_llm() -> SkipValidation[ChatGroq]:
    """Initializes and returns a configured ChatGroq LLM instance.

    This function reads model parameters such as the model name, temperature,
    and API key from the global settings object and uses them to create a
    ready-to-use instance of the ChatGroq client.

    Returns:
        A configured instance of the langchain_groq.ChatGroq class.
    """
    return ChatGroq(
        model=settings.GROQ_MODEL_NAME,
        temperature=settings.GROQ_TEMPERATURE,
        max_tokens=settings.GROQ_MAX_TOKENS,
        groq_api_key=settings.GROQ_API_KEY,
        model_kwargs={
            "top_p": settings.GROQ_TOP_P,
            "frequency_penalty": settings.GROQ_FREQUENCY_PENALTY,
            "presence_penalty": settings.GROQ_PRESENCE_PENALTY,
        },
    )