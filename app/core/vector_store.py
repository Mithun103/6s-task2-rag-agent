"""Manages all interactions with the ChromaDB vector database.

This module provides the VectorStore class, which serves as a dedicated
interface for storing, querying, and managing document embeddings in ChromaDB.
It abstracts the low-level client operations into a clean, reusable service.
"""

import chromadb
import uuid
from typing import Optional, List


class VectorStore:
    """A wrapper for ChromaDB client operations."""

    def __init__(self, persist_dir: str, collection_name: str):
        """Initializes the VectorStore.

        Args:
            persist_dir: The local directory on disk where the database
                will be stored and persisted.
            collection_name: The name of the collection within ChromaDB to
                use for storing embeddings.
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._init_collection()

    def _init_collection(self):
        """Initializes the persistent ChromaDB client and collection."""
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add(self, doc_id: str, chunks: list[dict], embeddings: list[list[float]]):
        """Adds document chunks and their embeddings to the Chroma collection.

        Args:
            doc_id: The unique identifier for the source document.
            chunks: A list of chunk dictionaries, each expected to have a
                'content' key and an optional 'page_number' key.
            embeddings: A list of vector embeddings, where each embedding
                corresponds to a chunk in the 'chunks' list.
        """
        metadatas = [{"doc_id": doc_id, "page_number": c.get("page_number", None)} for c in chunks]
        self.collection.add(
            ids=[str(uuid.uuid4()) for _ in chunks],
            embeddings=embeddings,
            documents=[c.get("content", "") for c in chunks],
            metadatas=metadatas
        )

    def query(
        self,
        query_embedding: list[float],
        doc_ids: Optional[List[str]] = None,
        top_k: int = 10
    ) -> list[dict]:
        """Retrieves relevant chunks from the collection by embedding similarity.

        Args:
            query_embedding: The vector representation of the user's query.
            doc_ids: An optional list of document IDs to filter the search.
                If None, the search is performed across all documents.
            top_k: The maximum number of relevant chunks to retrieve.

        Returns:
            A list of dictionaries, where each dictionary represents a
            retrieved chunk and contains its 'content' and 'metadata'.
        """
        where_filter = {"doc_id": {"$in": doc_ids}} if doc_ids else {}
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter if where_filter else None
        )
        retrieved_chunks = []
        if results and results.get("documents"):
            docs_list = results.get("documents")[0]
            metadatas_list = results.get("metadatas")[0] if results.get("metadatas") else [{}]*len(docs_list)
            for i, doc in enumerate(docs_list):
                retrieved_chunks.append({
                    "content": doc if isinstance(doc, str) else str(doc),
                    "metadata": metadatas_list[i] if i < len(metadatas_list) else {}
                })
        return retrieved_chunks

    def wipe_and_reset(self):
        """Deletes all data in the collection and re-initializes it.

        This is a destructive operation that permanently removes the existing
        Chroma collection from disk and creates a new, empty one in its place.

        Returns:
            A string confirming that the collection has been wiped.
        """
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        return f"Collection '{self.collection_name}' wiped and re-initialized."