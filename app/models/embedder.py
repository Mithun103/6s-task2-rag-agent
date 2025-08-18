"""Provides the embedding model client for vectorizing text.

This module contains the Embedder class, which serves as a wrapper around the
Google Generative AI embedding model client from LangChain. It is responsible
for converting text chunks and user queries into numerical vectors.
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List

class Embedder:
    """Generates vector embeddings for text using Google's Gemini model."""

    def __init__(self, model_name: str, google_api_key: str):
        """Initializes the Gemini embedding model.

        Args:
            model_name: The specific name of the Gemini embedding model to use
                (e.g., 'models/embedding-001').
            google_api_key: The API key for accessing Google's Generative AI services.
        """
        self.model = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=google_api_key
        )

    def embed_chunks(self, chunks: List[str]) -> List[List[float]]:
        """Creates vector embeddings for a list of text chunks.

        Args:
            chunks: A list of strings, where each string is a text chunk.

        Returns:
            A list of lists of floats, where each inner list is the vector
            embedding for a corresponding chunk.
        """
        return self.model.embed_documents(chunks)

    def embed_query(self, query: str) -> List[float]:
        """Creates a vector embedding for a single query string.

        Args:
            query: The user's query text.

        Returns:
            A list of floats representing the query's vector embedding.
        """
        return self.model.embed_query(query)