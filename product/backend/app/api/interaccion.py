"""Synthetic User Interaction API routes."""
from fastapi import APIRouter, HTTPException
from typing import Optional
from pydantic import BaseModel

router = APIRouter()


class ChatMessage(BaseModel):
    """Schema for chat message."""
    arquetipo_id: str
    mensaje: str
    imagen_url: Optional[str] = None
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Schema for chat response."""
    respuesta: str
    session_id: str
    fricciones: Optional[list[str]] = None
    emociones: Optional[dict] = None


class EvaluacionRequest(BaseModel):
    """Schema for prototype evaluation request."""
    arquetipo_id: str
    imagen_url: str
    contexto: Optional[str] = None
    preguntas: Optional[list[str]] = None


class EvaluacionResponse(BaseModel):
    """Schema for evaluation response."""
    feedback: str
    puntuacion: Optional[float] = None
    fricciones: list[str]
    sugerencias: list[str]
    emociones: dict


@router.post("/chat", response_model=ChatResponse)
async def chat_con_usuario(message: ChatMessage):
    """Chat with a synthetic user."""
    # TODO: Implement with Gemini LLM
    raise HTTPException(status_code=501, detail="Not implemented")


@router.post("/evaluar", response_model=EvaluacionResponse)
async def evaluar_prototipo(request: EvaluacionRequest):
    """Evaluate a prototype with a synthetic user."""
    # TODO: Implement with Gemini Vision + LLM
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/historial/{session_id}")
async def get_historial(session_id: str):
    """Get conversation history for a session."""
    # TODO: Implement with Supabase
    raise HTTPException(status_code=404, detail="Session not found")


@router.get("/sesiones")
async def list_sesiones(arquetipo_id: Optional[str] = None):
    """List interaction sessions."""
    # TODO: Implement with Supabase
    return {"sesiones": [], "total": 0}
