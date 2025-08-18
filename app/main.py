"""The main entry point for the FastAPI application.

This module is responsible for creating and configuring the FastAPI app instance.
It brings together the API routes defined in other modules and sets up the
application's root endpoint for health checks.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.api import router
from app.exceptions import DocumentProcessingError, AgentError


app = FastAPI(
    title="Intelligent PDF RAG API",
    description="An API with a ReAct agent for querying single or multiple documents.",
    version="2.0.0"
)

@app.exception_handler(DocumentProcessingError)
async def document_processing_exception_handler(request: Request, exc: DocumentProcessingError):
    """Handles errors that occur during the document ingestion process."""
    return JSONResponse(
        status_code=422, # Unprocessable Entity
        content={"message": f"Could not process the uploaded document: {exc}"},
    )

@app.exception_handler(AgentError)
async def agent_exception_handler(request: Request, exc: AgentError):
    """Handles errors that occur during the agent's execution."""
    return JSONResponse(
        status_code=500, # Internal Server Error
        content={"message": f"An error occurred in the agent: {exc}"},
    )


# Include all the routes from api.py, prefixed with /api
app.include_router(router, prefix="/api")


@app.get("/")
def read_root():
    """Provides a root endpoint for basic health checks.

    Returns:
        A dictionary indicating the server status and the URL for the
        interactive API documentation.
    """
    return {"status": "ok", "docs_url": "/docs"}