"""Visual Analysis API routes."""
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from typing import Optional
from pydantic import BaseModel

from app.core.security import require_auth

router = APIRouter()


class AnalisisURLRequest(BaseModel):
    """Schema for URL analysis request."""
    url: str
    tipo_analisis: list[str] = ["heatmap", "focus_map", "aoi"]


class AnalisisResponse(BaseModel):
    """Schema for analysis response."""
    id: str
    imagen_url: str
    heatmap_url: Optional[str] = None
    focus_map_url: Optional[str] = None
    clarity_score: Optional[float] = None
    areas_interes: Optional[list[dict]] = None
    insights: Optional[list[str]] = None


@router.post("/imagen")
async def analizar_imagen(file: UploadFile = File(...), user: dict = Depends(require_auth)):
    """Analyze an uploaded image with Vision AI. Requires authentication."""
    # TODO: Implement with Gemini Vision
    raise HTTPException(status_code=501, detail="Not implemented")


@router.post("/url")
async def analizar_url(request: AnalisisURLRequest, user: dict = Depends(require_auth)):
    """Analyze an image from URL with Vision AI. Requires authentication."""
    # TODO: Implement with Gemini Vision
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/{analisis_id}", response_model=AnalisisResponse)
async def get_analisis(analisis_id: str, user: dict = Depends(require_auth)):
    """Get analysis results by ID. Requires authentication."""
    # TODO: Implement with Supabase
    raise HTTPException(status_code=404, detail="Analysis not found")


@router.get("/")
async def list_analisis(user: dict = Depends(require_auth)):
    """List recent analyses. Requires authentication."""
    # TODO: Implement with Supabase
    return {"analisis": [], "total": 0}


@router.post("/comparar")
async def comparar_disenos(imagen_a_url: str, imagen_b_url: str, user: dict = Depends(require_auth)):
    """Compare two designs side by side. Requires authentication."""
    # TODO: Implement A/B comparison
    raise HTTPException(status_code=501, detail="Not implemented")
