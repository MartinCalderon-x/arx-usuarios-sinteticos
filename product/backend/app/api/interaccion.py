"""Synthetic User Interaction API routes."""
import uuid
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from pydantic import BaseModel

from app.core.security import require_auth
from app.services import supabase
from app.services.gemini import GeminiChatService

router = APIRouter()
chat_service = GeminiChatService()


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


def get_user_id(user: dict) -> str:
    """Extract user ID from JWT payload."""
    return user.get("sub")


@router.post("/chat", response_model=ChatResponse)
async def chat_con_usuario(message: ChatMessage, user: dict = Depends(require_auth)):
    """Chat with a synthetic user. Requires authentication."""
    user_id = get_user_id(user)

    # Get or create session
    session_id = message.session_id
    if not session_id:
        # Create new session
        session = await supabase.create_sesion(message.arquetipo_id, user_id)
        if not session:
            raise HTTPException(status_code=500, detail="Error al crear sesion")
        session_id = session["id"]
    else:
        # Verify session exists and belongs to user
        session = await supabase.get_sesion(session_id, user_id)
        if not session:
            raise HTTPException(status_code=404, detail="Sesion no encontrada")

    # Get archetype
    arquetipo = await supabase.get_arquetipo(message.arquetipo_id, user_id)
    if not arquetipo:
        raise HTTPException(status_code=404, detail="Arquetipo no encontrado")

    # Get conversation history
    historial = await supabase.get_historial(session_id)

    # Save user message
    await supabase.save_mensaje(session_id, {
        "rol": "usuario",
        "contenido": message.mensaje,
        "imagen_url": message.imagen_url
    })

    # Generate response using Gemini
    response = await chat_service.chat_as_synthetic_user(
        arquetipo=arquetipo,
        mensaje=message.mensaje,
        historial=historial,
        imagen_url=message.imagen_url
    )

    # Save synthetic user response
    await supabase.save_mensaje(session_id, {
        "rol": "sintetico",
        "contenido": response["respuesta"],
        "fricciones": response.get("fricciones", []),
        "emociones": response.get("emociones", {})
    })

    return ChatResponse(
        respuesta=response["respuesta"],
        session_id=session_id,
        fricciones=response.get("fricciones"),
        emociones=response.get("emociones")
    )


@router.post("/evaluar", response_model=EvaluacionResponse)
async def evaluar_prototipo(request: EvaluacionRequest, user: dict = Depends(require_auth)):
    """Evaluate a prototype with a synthetic user. Requires authentication."""
    user_id = get_user_id(user)

    # Get archetype
    arquetipo = await supabase.get_arquetipo(request.arquetipo_id, user_id)
    if not arquetipo:
        raise HTTPException(status_code=404, detail="Arquetipo no encontrado")

    # Evaluate prototype with Gemini
    result = await chat_service.evaluate_prototype(
        arquetipo=arquetipo,
        imagen_url=request.imagen_url,
        contexto=request.contexto,
        preguntas=request.preguntas
    )

    return EvaluacionResponse(
        feedback=result.get("feedback", ""),
        puntuacion=result.get("puntuacion"),
        fricciones=result.get("fricciones", []),
        sugerencias=result.get("sugerencias", []),
        emociones=result.get("emociones", {})
    )


@router.get("/historial/{session_id}")
async def get_historial(session_id: str, user: dict = Depends(require_auth)):
    """Get conversation history for a session. Requires authentication."""
    user_id = get_user_id(user)

    # Verify session exists and belongs to user
    session = await supabase.get_sesion(session_id, user_id)
    if not session:
        raise HTTPException(status_code=404, detail="Sesion no encontrada")

    historial = await supabase.get_historial(session_id)
    return {"session_id": session_id, "mensajes": historial, "total": len(historial)}


@router.get("/sesiones")
async def list_sesiones(arquetipo_id: Optional[str] = None, user: dict = Depends(require_auth)):
    """List interaction sessions. Requires authentication."""
    user_id = get_user_id(user)
    sesiones, total = await supabase.list_sesiones(user_id, arquetipo_id)
    return {"sesiones": sesiones, "total": total}
