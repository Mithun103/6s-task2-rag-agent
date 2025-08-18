"""Handles the process of splitting large text content into smaller chunks.

This module provides a wrapper around LangChain's text splitting
functionalities to break down documents into sizes suitable for embedding.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextSplitter:
    """A wrapper for the RecursiveCharacterTextSplitter from LangChain.

    This class is responsible for taking large sections of text, often generated
    by the DocumentParser, and splitting them into smaller, overlapping chunks.
    This process is essential for ensuring the text segments fit within the
    context window of embedding models.
    """
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 300):
        """Initializes the TextSplitter.

        Args:
            chunk_size: The maximum number of characters for each chunk.
            chunk_overlap: The number of characters to overlap between
                consecutive chunks to maintain contextual continuity.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split(self, documents: list[dict]) -> list[dict]:
        """Splits a list of document sections into smaller chunks.

        This method iterates through each document section, splits its 'content'
        field into smaller texts, and creates a new list of chunk dictionaries,
        preserving the original page number for each new chunk.

        Args:
            documents: A list of dictionaries, where each dictionary
                represents a section of a document. Expected keys are
                'content' and 'page_number'.

        Returns:
            A list of dictionaries, where each dictionary represents a single,
            smaller chunk of text, ready for embedding.
        """
        chunks = []
        for doc in documents:
            split_content = self.splitter.split_text(doc["content"])
            for chunk in split_content:
                chunks.append({
                    "page_number": doc["page_number"],
                    "content": chunk,
                })
        return chunks