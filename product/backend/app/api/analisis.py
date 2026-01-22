"""Visual Analysis API routes."""
import asyncio
import io
import time
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Query
from typing import Optional
from pydantic import BaseModel
import httpx
from PIL import Image
import numpy as np

from app.core.security import require_auth
from app.core.config import get_settings
from app.services import supabase
from app.services.gemini import GeminiVisionService
from app.services.heatmap import HybridHeatmapService, HeatmapComparisonService

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()
gemini_service = GeminiVisionService()
hybrid_service = HybridHeatmapService()
comparison_service = HeatmapComparisonService()


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


# ============================================================================
# Model Comparison Endpoint
# ============================================================================

class ModelResult(BaseModel):
    """Result from a single model."""
    heatmap_base64: str
    heatmap_overlay_base64: Optional[str] = None
    regions: list[dict]
    inference_time_ms: float
    model_name: str


class ComparisonMetrics(BaseModel):
    """Metrics comparing two models."""
    correlation_coefficient: float
    kl_divergence: float
    similarity: float
    nss: float
    alignment_percentage: float
    verdict: str


class ModelComparisonResponse(BaseModel):
    """Response for model comparison endpoint."""
    ml_model: Optional[ModelResult] = None
    hybrid_model: ModelResult
    comparison: Optional[ComparisonMetrics] = None
    ml_service_available: bool
    total_time_ms: float


async def call_ml_service(image_data: bytes) -> Optional[dict]:
    """Call ML Service to get DeepGaze prediction.

    Args:
        image_data: Raw image bytes.

    Returns:
        Dict with heatmap and metadata, or None if service unavailable.
    """
    if not settings.ml_service_enabled:
        logger.info("ML Service disabled in settings")
        return None

    try:
        async with httpx.AsyncClient(timeout=settings.ml_service_timeout) as client:
            files = {"file": ("image.png", image_data, "image/png")}
            response = await client.post(
                f"{settings.ml_service_url}/predict",
                files=files,
                params={"overlay": True, "threshold": 0.3},
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"ML Service returned {response.status_code}")
                return None

    except httpx.TimeoutException:
        logger.warning("ML Service timeout")
        return None
    except httpx.ConnectError:
        logger.warning("ML Service connection failed")
        return None
    except Exception as e:
        logger.error(f"ML Service error: {e}")
        return None


async def generate_hybrid_result(
    image_data: bytes,
    gemini_result: dict,
) -> dict:
    """Generate hybrid heatmap from Gemini AOI data.

    Args:
        image_data: Raw image bytes.
        gemini_result: Gemini Vision analysis result.

    Returns:
        Dict with heatmap and metadata.
    """
    start_time = time.time()

    # Get image dimensions
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    width, height = image.size

    # Get AOI data from Gemini result
    aoi_data = gemini_result.get("areas_interes", [])

    # Generate hybrid heatmap
    heatmap = hybrid_service.generate_from_aoi(
        aoi_data, width, height, include_center_bias=True
    )

    # Generate heatmap images
    heatmap_base64 = hybrid_service.heatmap_to_base64(heatmap, colormap="jet")
    overlay_base64 = hybrid_service.heatmap_to_base64(
        heatmap, colormap="jet", original_image=image, alpha=0.5
    )

    # Extract regions for consistency with ML Service format
    regions = []
    for i, aoi in enumerate(aoi_data):
        regions.append({
            "x": aoi.get("x", 50),
            "y": aoi.get("y", 50),
            "width": aoi.get("width", 10),
            "height": aoi.get("height", 10),
            "intensity": aoi.get("intensidad", 50),
            "area": int(aoi.get("width", 10) * aoi.get("height", 10) * width * height / 10000),
            "orden_visual": aoi.get("orden_visual", i + 1),
        })

    inference_time = (time.time() - start_time) * 1000

    return {
        "heatmap_base64": heatmap_base64,
        "heatmap_overlay_base64": overlay_base64,
        "regions": regions,
        "inference_time_ms": round(inference_time, 2),
        "heatmap_array": heatmap,  # Keep for comparison
    }


@router.post("/comparar-modelos", response_model=ModelComparisonResponse)
async def comparar_modelos(
    file: UploadFile = File(..., description="Image to analyze"),
    user: dict = Depends(require_auth),
):
    """Compare visual attention predictions from DeepGaze ML and Hybrid models.

    This endpoint runs both models in parallel and returns:
    - DeepGaze (ML Service) results if available
    - Hybrid (Gemini + Gaussian) results
    - Comparison metrics showing how well hybrid approximates ML

    The ML model (DeepGaze) is treated as ground truth for comparison.

    Args:
        file: Image file to analyze (PNG, JPEG, etc.)

    Returns:
        ModelComparisonResponse with results from both models and metrics.
    """
    total_start = time.time()
    user_id = get_user_id(user)

    # Read image data
    image_data = await file.read()

    # Run Gemini analysis first (needed for hybrid model)
    gemini_result = await gemini_service.analyze_image(
        image_data, file.content_type or "image/png"
    )

    # Run both models in parallel
    ml_task = call_ml_service(image_data)
    hybrid_task = generate_hybrid_result(image_data, gemini_result)

    ml_result, hybrid_result = await asyncio.gather(ml_task, hybrid_task)

    # Build response
    ml_model_response = None
    comparison_metrics = None
    ml_service_available = ml_result is not None

    if ml_result:
        ml_model_response = ModelResult(
            heatmap_base64=ml_result["heatmap_base64"],
            heatmap_overlay_base64=ml_result.get("heatmap_overlay_base64"),
            regions=ml_result["regions"],
            inference_time_ms=ml_result["metadata"]["inference_time_ms"],
            model_name=ml_result["metadata"]["model"],
        )

        # Decode ML heatmap for comparison
        import base64
        ml_heatmap_bytes = base64.b64decode(ml_result["heatmap_base64"])
        ml_heatmap_img = Image.open(io.BytesIO(ml_heatmap_bytes)).convert("L")
        ml_heatmap_array = np.array(ml_heatmap_img).astype(np.float32) / 255.0

        # Calculate comparison metrics
        metrics = comparison_service.compare(
            ml_heatmap_array,
            hybrid_result["heatmap_array"],
        )
        comparison_metrics = ComparisonMetrics(**metrics)

    hybrid_model_response = ModelResult(
        heatmap_base64=hybrid_result["heatmap_base64"],
        heatmap_overlay_base64=hybrid_result["heatmap_overlay_base64"],
        regions=hybrid_result["regions"],
        inference_time_ms=hybrid_result["inference_time_ms"],
        model_name="hybrid_gemini_gaussian",
    )

    total_time = (time.time() - total_start) * 1000

    return ModelComparisonResponse(
        ml_model=ml_model_response,
        hybrid_model=hybrid_model_response,
        comparison=comparison_metrics,
        ml_service_available=ml_service_available,
        total_time_ms=round(total_time, 2),
    )
