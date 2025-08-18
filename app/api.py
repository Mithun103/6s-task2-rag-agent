"""Defines the API endpoints for the RAG agent application.

This module creates a FastAPI APIRouter and defines the web routes for
interacting with the system. It includes an endpoint for uploading new
documents and an endpoint for chatting with the intelligent agent.
"""

import uuid
import os
import shutil
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from app.schemas import AgentChatRequest
from app.services import rag_service

router = APIRouter()

@router.post("/upload", status_code=201)
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Handles the upload of a new PDF document.

    This endpoint accepts a PDF file, saves it to a temporary location,
    and schedules a background task to process and index its content.
    It immediately returns a unique document ID for future reference.

    Args:
        background_tasks: FastAPI dependency to run tasks after returning a response.
        file: The PDF file uploaded by the user.

    Returns:
        A dictionary containing the new doc_id and a confirmation message.
    """
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")
    
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    doc_id = str(uuid.uuid4())
    background_tasks.add_task(rag_service.process_document_background, doc_id, temp_file_path)
    
    return {"doc_id": doc_id, "message": "Document upload successful. Processing has started."}

@router.post("/agent/chat")
async def agent_chat(request: AgentChatRequest):
    """Receives a question and passes it to the intelligent agent.

    This endpoint is the primary interface for querying the knowledge base.
    It takes a user's question, invokes the RAG agent, and returns the
    agent's final, synthesized answer.

    Args:
        request: A Pydantic model containing the user's question.

    Returns:
        A dictionary containing the agent's final answer.
    """
    try:
        response = await rag_service.invoke_agent(request.question)
        return {"answer": response['output']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))