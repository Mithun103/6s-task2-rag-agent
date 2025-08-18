"""Provides the main business logic and orchestration for the RAG application.

This module contains the RAGService class, which acts as the central hub,
initializing all core components and orchestrating the application's primary
workflows, such as document processing and agent invocation.
"""

import os
from app.agent.agent import create_rag_agent
from app.core.parser import DocumentParser
from app.core.splitter import TextSplitter
from app.core.vector_store import VectorStore
from app.models.embedder import Embedder
from app.models.llm import get_llm
from app.config import settings
from app.exceptions import DocumentProcessingError, AgentError


class RAGService:
    """The central service class for all RAG and agent operations."""

    def __init__(self):
        """Initializes all necessary components for the RAG service.

        This constructor sets up the core logic modules (parser, splitter,
        embedder, vector store), the LLM client, and creates the fully
        configured ReAct agent executor, making it ready to handle requests.
        """
        self.parser = DocumentParser()
        self.splitter = TextSplitter()
        self.embedder = Embedder(
            model_name=settings.EMBEDDING_MODEL_NAME,
            google_api_key=settings.GOOGLE_API_KEY
        )
        self.vector_store = VectorStore(
            persist_dir=settings.CHROMA_PERSIST_DIR,
            collection_name=settings.CHROMA_COLLECTION_NAME,
        )
        llm_client = get_llm()

        self.agent_executor = create_rag_agent(
            llm=llm_client,
            embedder=self.embedder,
            vector_store=self.vector_store
        )

    def process_document_background(self, doc_id: str, file_path: str):
        """Handles the asynchronous processing of an uploaded PDF file.

        This method orchestrates the full ingestion pipeline: parsing text,
        splitting it into chunks, generating embeddings, and storing them in
        the vector database. It includes debug prints for observability and
        is designed to be run as a background task.

        Args:
            doc_id: The unique identifier for the document being processed.
            file_path: The local path to the temporary PDF file.
        """
        try:
            print(f"\n[+] Processing doc_id: {doc_id}")
            parsed_content = self.parser.parse(file_path)
            chunks = self.splitter.split(parsed_content)

            print(f"\n--- Generated {len(chunks)} chunks for doc_id: {doc_id} ---")
            for i, chunk in enumerate(chunks[:3]):
                print(f"[Chunk {i+1}] Page: {chunk.get('page_number')}, Content: {chunk['content'][:250]}...")
            if len(chunks) > 3:
                print(f"...and {len(chunks)-3} more chunks")
            print("--------------------------------------------------\n")

            chunk_contents = [chunk["content"] for chunk in chunks]
            embeddings = self.embedder.embed_chunks(chunk_contents)

            self.vector_store.add(doc_id, chunks, embeddings)
            print(f"--- Stored {len(chunks)} chunks in VectorStore for doc_id: {doc_id} ---\n")

        except Exception as e:
            print(f"[-] Error during document processing for {doc_id}: {e}")
            raise DocumentProcessingError(f"Failed to process document {file_path}: {e}")
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    async def invoke_agent(self, question: str) -> dict:
        """Asynchronously invokes the ReAct agent with a user's question.

        Args:
            question: The user's input string/question for the agent.

        Returns:
            A dictionary containing the agent's complete response, typically
            including the final 'output' field.
        """
        try:

            response = await self.agent_executor.ainvoke({"input": question})
        except Exception as e:
            print(f"[-] Error during agent invocation: {e}")
            raise AgentError(f"Failed to invoke agent: {e}")
        return response


rag_service = RAGService()