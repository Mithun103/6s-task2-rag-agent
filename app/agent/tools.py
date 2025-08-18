"""Defines the tools available to the ReAct agent for interacting with the RAG system.
This module contains the definitions for all tools that the LangChain agent
can use to perform its tasks. This includes searching the knowledge base and
performing administrative actions like clearing the vector store.
"""

from typing import List, Optional, Type
from pydantic import BaseModel
from langchain.tools import BaseTool
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from app.models.llm import get_llm
from app.schemas import KnowledgeSearchInput
from app.core.vector_store import VectorStore
from app.models.embedder import Embedder


class KnowledgeSearchTool(BaseTool):
    """A tool for searching and synthesizing information from the PDF knowledge base."""
    name: str = "knowledge_base_search"
    description: str = (
        "Use this tool to search the knowledge base. Be specific in your queries.\n"
        "For document-specific searches, include doc_ids.\n"
        "For topic searches, use clear search terms.\n"
        "Example: query='technical skills', doc_ids=['abc-123']\n"
        "Returns comprehensive answers with source citations."
    )
    args_schema: Type[BaseModel] = KnowledgeSearchInput

    vector_store: VectorStore
    embedder: Embedder
    llm: object

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.llm:
            self.llm = get_llm()

    def format_sources(self, sources: List[dict]) -> str:
        """Formats the source citations in a clean, readable way with deduplication."""
        # Create a set of tuples to deduplicate sources
        unique_sources = {(source['doc_id'], source['page']) for source in sources}
        # Convert back to sorted list for consistent output
        formatted_sources = [
            f"- Document {doc_id}, Page {page}"
            for doc_id, page in sorted(unique_sources)
        ]
        return "\n".join(formatted_sources) if formatted_sources else "No sources available"

    def _run(self, query: str, doc_ids: Optional[List[str]] = None, top_k: int = 10) -> str:
        """Executes the knowledge search.

        This method vectorizes the user's query, retrieves the most relevant
        document chunks from the VectorStore, and then uses an LLM to generate
        a final answer based on the retrieved context. It also formats a list
        of sources from the metadata of the retrieved chunks.

        Args:
            query: The user's question or search term.
            doc_ids: An optional list of document IDs to restrict the search to.
            top_k: The maximum number of chunks to retrieve.

        Returns:
            A string containing the LLM-generated answer and the sources it used.
        """
        try:
            if query and len(query) == 36 and query.count("-") == 4:
                doc_ids = [query]
                query = ""

            query_embedding = self.embedder.embed_query(query or " ")
            chunks = self.vector_store.query(query_embedding, doc_ids=doc_ids, top_k=top_k)

            if not chunks:
                return "No relevant information found in the uploaded documents."

            context = "\n\n".join([f"[Page {c['metadata']['page_number']}] {c['content']}" for c in chunks])
            sources_array = [
                {"doc_id": c["metadata"]["doc_id"], "page": c["metadata"]["page_number"]}
                for c in chunks
            ]

            prompt = PromptTemplate.from_template("""
                        You are a precise and thorough assistant. Analyze and answer using ONLY the provided context.

                        Context:
                        {context}

                        Question: {question}

                        Instructions:
                        1. Provide a clear, detailed answer based only on the context
                        2. If no relevant information is found, respond EXACTLY with:
                           "No relevant information found in the documents."
                        3. If information is found, respond EXACTLY in this format:
                           Based on the documents: [Your detailed answer here]
                           Sources: [List each source on a new line with exact doc_id and page]

                        Current sources to cite: {sources}

                        Response:
                        """)
            
            # Create the chain using the new pattern
            chain = prompt | self.llm
            
            # Get the LLM's response
            response = chain.invoke({
                "context": context,
                "question": query or "Summarize the document.",
                "sources": sources_array
            })
            
            # Extract the content from AIMessage
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract and clean up the response content
            response_content = response.content if hasattr(response, 'content') else str(response)
            response_content = response_content.strip()
            
            # If no relevant info was found, return standard message
            if not chunks or "no relevant information" in response_content.lower():
                return "No relevant information found in the documents."
                
            # Format the sources consistently
            formatted_sources = "\n".join([
                f"- Document {source['doc_id']}, Page {source['page']}"
                for source in sources_array
            ])
            
            # Format the final response in the standard format
            final_response = f"Based on the documents: {response_content}\nSources:\n{formatted_sources}"
            
            return final_response

        except Exception as e:
            return f"Error in knowledge search: {e}"

    async def _arun(self, query: str, doc_ids: Optional[List[str]] = None, top_k: int = 10) -> str:
        """Asynchronously executes the knowledge search.

        Args:
            query: The user's question or search term.
            doc_ids: An optional list of document IDs to restrict the search to.
            top_k: The maximum number of chunks to retrieve.

        Returns:
            A string containing the LLM-generated answer and the sources it used.
        """
        return self._run(query=query, doc_ids=doc_ids, top_k=top_k)


class WipeVectorStoreTool(BaseTool):
    """An administrative tool for clearing all data from the vector store."""
    name: str = "wipe_vector_store"
    description: str = (
        "Wipes the vector store completely and re-initializes it. "
        "Automatically triggered by the agent when requested; no confirmation needed."
    )

    vector_store: VectorStore

    def _run(self, *args, **kwargs) -> str:
        """Executes the wipe-and-reset operation on the VectorStore.

        Returns:
            A string indicating the success or failure of the operation.
        """
        try:
            self.vector_store.wipe_and_reset()
            return "Vector store wiped successfully."
        except Exception as e:
            return f"Failed to wipe vector store: {str(e)}"

    async def _arun(self, *args, **kwargs) -> str:
        """Asynchronously executes the wipe-and-reset operation.

        Returns:
            A string indicating the success or failure of the operation.
        """
        return self._run(*args, **kwargs)