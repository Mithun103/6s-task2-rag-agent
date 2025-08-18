"""Defines custom exception classes for the application."""

class DocumentProcessingError(Exception):
    """Raised when an error occurs during PDF parsing, chunking, or embedding."""
    pass

class AgentError(Exception):
    """Raised when the LangChain agent fails to process a request."""
    pass