"""Visual Analysis API routes."""
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Optional
from pydantic import BaseModel

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
async def analizar_imagen(file: UploadFile = File(...)):
    """Analyze an uploaded image with Vision AI."""
    # TODO: Implement with Gemini Vision
    raise HTTPException(status_code=501, detail="Not implemented")


@router.post("/url")
async def analizar_url(request: AnalisisURLRequest):
    """Analyze an image from URL with Vision AI."""
    # TODO: Implement with Gemini Vision
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/{analisis_id}", response_model=AnalisisResponse)
async def get_analisis(analisis_id: str):
    """Get analysis results by ID."""
    # TODO: Implement with Supabase
    raise HTTPException(status_code=404, detail="Analysis not found")


@router.get("/")
async def list_analisis():
    """List recent analyses."""
    # TODO: Implement with Supabase
    return {"analisis": [], "total": 0}


@router.post("/comparar")
async def comparar_disenos(imagen_a_url: str, imagen_b_url: str):
    """Compare two designs side by side."""
    # TODO: Implement A/B comparison
    raise HTTPException(status_code=501, detail="Not implemented")
