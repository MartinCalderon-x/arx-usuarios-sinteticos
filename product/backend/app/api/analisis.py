"""Visual Analysis API routes."""
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from typing import Optional
from pydantic import BaseModel

from app.core.security import require_auth
from app.services import supabase
from app.services.gemini import GeminiVisionService

router = APIRouter()
gemini_service = GeminiVisionService()


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
    created_at: Optional[str] = None


def get_user_id(user: dict) -> str:
    """Extract user ID from JWT payload."""
    return user.get("sub")


@router.post("/imagen", response_model=AnalisisResponse)
async def analizar_imagen(file: UploadFile = File(...), user: dict = Depends(require_auth)):
    """Analyze an uploaded image with Vision AI. Requires authentication."""
    user_id = get_user_id(user)

    # Read image content
    content = await file.read()

    # Analyze with Gemini Vision
    analysis_result = await gemini_service.analyze_image(content, file.content_type)

    # Save to database
    data = {
        "imagen_url": f"upload://{file.filename}",
        "tipo_analisis": ["heatmap", "focus_map", "aoi"],
        "resultados": analysis_result,
        "clarity_score": analysis_result.get("clarity_score"),
        "areas_interes": analysis_result.get("areas_interes", []),
        "insights": analysis_result.get("insights", []),
    }
    result = await supabase.create_analisis(data, user_id)
    if not result:
        raise HTTPException(status_code=500, detail="Error al guardar analisis")
    return result


@router.post("/url", response_model=AnalisisResponse)
async def analizar_url(request: AnalisisURLRequest, user: dict = Depends(require_auth)):
    """Analyze an image from URL with Vision AI. Requires authentication."""
    user_id = get_user_id(user)

    # Analyze with Gemini Vision
    analysis_result = await gemini_service.analyze_url(request.url, request.tipo_analisis)

    # Save to database
    data = {
        "imagen_url": request.url,
        "tipo_analisis": request.tipo_analisis,
        "resultados": analysis_result,
        "clarity_score": analysis_result.get("clarity_score"),
        "areas_interes": analysis_result.get("areas_interes", []),
        "insights": analysis_result.get("insights", []),
    }
    result = await supabase.create_analisis(data, user_id)
    if not result:
        raise HTTPException(status_code=500, detail="Error al guardar analisis")
    return result


@router.get("/{analisis_id}", response_model=AnalisisResponse)
async def get_analisis(analisis_id: str, user: dict = Depends(require_auth)):
    """Get analysis results by ID. Requires authentication."""
    user_id = get_user_id(user)
    result = await supabase.get_analisis(analisis_id, user_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analisis no encontrado")
    return result


@router.get("/")
async def list_analisis(user: dict = Depends(require_auth)):
    """List recent analyses. Requires authentication."""
    user_id = get_user_id(user)
    analisis_list, total = await supabase.list_analisis(user_id)
    return {"analisis": analisis_list, "total": total}


@router.post("/comparar")
async def comparar_disenos(imagen_a_url: str, imagen_b_url: str, user: dict = Depends(require_auth)):
    """Compare two designs side by side. Requires authentication."""
    user_id = get_user_id(user)

    # Analyze both images
    result_a = await gemini_service.analyze_url(imagen_a_url, ["heatmap", "focus_map", "aoi"])
    result_b = await gemini_service.analyze_url(imagen_b_url, ["heatmap", "focus_map", "aoi"])

    # Generate comparison
    comparison = await gemini_service.compare_designs(result_a, result_b)

    return {
        "imagen_a": {"url": imagen_a_url, "analisis": result_a},
        "imagen_b": {"url": imagen_b_url, "analisis": result_b},
        "comparacion": comparison
    }
