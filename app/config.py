"""Manages application-wide configuration settings.

This module uses Pydantic's BaseSettings to load configuration from environment
variables defined in a .env file. It provides a centralized, type-safe way to
manage settings for various services like API keys, LLM parameters, and the
vector database.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Defines and loads configuration settings for the application.

    Attributes:
        model_config: Pydantic configuration to load from a .env file.

        GROQ_API_KEY: API key for the Groq language model service.
        GOOGLE_API_KEY: API key for Google's Generative AI services (e.g., Gemini).
        LANGSMITH_API_KEY: API key for LangSmith tracing and observability.

        GROQ_MODEL_NAME: The specific Groq model to use for generation.
        GROQ_TEMPERATURE: Controls the randomness of the LLM's output.
        GROQ_MAX_TOKENS: The maximum number of tokens to generate.
        GROQ_TOP_P: The nucleus sampling parameter.
        GROQ_FREQUENCY_PENALTY: Penalty for repeating tokens.
        GROQ_PRESENCE_PENALTY: Penalty for introducing new topics.

        EMBEDDING_MODEL_NAME: The Gemini model used for text embeddings.
        CHROMA_PERSIST_DIR: The directory to persist the ChromaDB database.
        CHROMA_COLLECTION_NAME: The name of the collection within ChromaDB.
    """
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    GROQ_API_KEY: str
    GOOGLE_API_KEY: str
    LANGSMITH_API_KEY: str

    GROQ_MODEL_NAME: str = "llama3-70b-8192"
    GROQ_TEMPERATURE: float = 0.4
    GROQ_MAX_TOKENS: int = 2048
    GROQ_TOP_P: float = 0.95
    GROQ_FREQUENCY_PENALTY: float = 0.0
    GROQ_PRESENCE_PENALTY: float = 0.0

    EMBEDDING_MODEL_NAME: str = "models/embedding-001"
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "pdf_rag_collection"


settings = Settings()