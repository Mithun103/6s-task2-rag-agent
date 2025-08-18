"""Defines Pydantic models for API request and response validation.

This module contains the data structures used by FastAPI to validate incoming
request bodies and to structure the arguments for agent tools, ensuring that
all data moving through the application is well-formed and type-safe.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class AgentChatRequest(BaseModel):
    """Defines the structure for a request to the agent chat endpoint."""
    question: str


class KnowledgeSearchInput(BaseModel):
    """Defines the structured input arguments for the KnowledgeSearchTool.

    This model is used as the 'args_schema' for the agent's primary search
    tool. It ensures that the agent provides all necessary parameters in the
    correct format when it decides to use the tool.

    Attributes:
        query: The natural language question or topic to search for.
        doc_ids: An optional list of document IDs to restrict the search to.
            If not provided, the search will span the entire knowledge base.
        top_k: The maximum number of relevant chunks to retrieve from the
            vector store.
    """
    query: str = Field(..., description="The natural language query to search in the uploaded documents.")
    doc_ids: Optional[List[str]] = Field(
        None, description="Optional list of specific document IDs to restrict the search. Example: ['fdeac49e-b311-4623-bdb5-57a7764736e5']"
    )
    top_k: int = Field(10, description="Number of top relevant chunks to retrieve.")