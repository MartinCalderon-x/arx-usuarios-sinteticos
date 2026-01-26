"""Flujos (User Journeys) API routes."""
import base64
import io
import logging
import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from typing import Optional
from pydantic import BaseModel
import httpx
from PIL import Image
import numpy as np

from app.core.security import require_auth
from app.core.config import get_settings
from app.services import supabase
from app.services.gemini import GeminiVisionService
from app.services.heatmap import HybridHeatmapService

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()
gemini_service = GeminiVisionService()
hybrid_service = HybridHeatmapService()


# ============================================
# Pydantic Schemas
# ============================================

class FlujoCreate(BaseModel):
    """Schema for creating a flow."""
    nombre: str
    descripcion: Optional[str] = None
    url_inicial: Optional[str] = None


class FlujoUpdate(BaseModel):
    """Schema for updating a flow."""
    nombre: Optional[str] = None
    descripcion: Optional[str] = None
    url_inicial: Optional[str] = None
    estado: Optional[str] = None
    configuracion: Optional[dict] = None


class FlujoResponse(BaseModel):
    """Schema for flow response."""
    id: str
    nombre: str
    descripcion: Optional[str] = None
    url_inicial: Optional[str] = None
    estado: str = "activo"
    total_pantallas: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class FlujoDetailResponse(FlujoResponse):
    """Schema for flow detail with screens."""
    pantallas: list[dict] = []


class PantallaResponse(BaseModel):
    """Schema for screen response."""
    id: str
    flujo_id: str
    orden: int
    origen: str
    url: Optional[str] = None
    titulo: Optional[str] = None
    screenshot_url: Optional[str] = None
    heatmap_url: Optional[str] = None
    overlay_url: Optional[str] = None
    clarity_score: Optional[float] = None
    areas_interes: Optional[list[dict]] = None
    insights: Optional[list[str]] = None
    modelo_usado: Optional[str] = None
    elementos_clickeables: Optional[list[dict]] = None
    created_at: Optional[str] = None


class PantallaURLRequest(BaseModel):
    """Schema for adding screen from URL."""
    url: str
    titulo: Optional[str] = None


class ReordenarRequest(BaseModel):
    """Schema for reordering screens."""
    orden_ids: list[str]


def get_user_id(user: dict) -> str:
    """Extract user ID from JWT payload."""
    return user.get("sub")


# ============================================
# Flujos CRUD
# ============================================

@router.get("/")
async def list_flujos(
    estado: Optional[str] = None,
    user: dict = Depends(require_auth)
):
    """List all flows for the authenticated user."""
    user_id = get_user_id(user)
    flujos, total = await supabase.list_flujos(user_id, estado=estado)
    return {"flujos": flujos, "total": total}


@router.post("/", response_model=FlujoResponse)
async def create_flujo(flujo: FlujoCreate, user: dict = Depends(require_auth)):
    """Create a new flow."""
    user_id = get_user_id(user)
    data = flujo.model_dump(exclude_none=True)
    result = await supabase.create_flujo(data, user_id)
    if not result:
        raise HTTPException(status_code=500, detail="Error al crear flujo")
    return result


@router.get("/{flujo_id}", response_model=FlujoDetailResponse)
async def get_flujo(flujo_id: str, user: dict = Depends(require_auth)):
    """Get a flow with all its screens."""
    user_id = get_user_id(user)
    result = await supabase.get_flujo(flujo_id, user_id)
    if not result:
        raise HTTPException(status_code=404, detail="Flujo no encontrado")

    # Generate fresh signed URLs for all screens
    pantallas_with_urls = []
    for pantalla in result.get("pantallas", []):
        signed_urls = supabase.get_signed_urls_for_pantalla(
            user_id, flujo_id, pantalla["id"]
        )
        pantalla_data = {
            **pantalla,
            "screenshot_url": signed_urls.get("screenshot"),
            "heatmap_url": signed_urls.get("heatmap"),
            "overlay_url": signed_urls.get("overlay"),
        }
        pantallas_with_urls.append(pantalla_data)

    result["pantallas"] = pantallas_with_urls
    return result


@router.put("/{flujo_id}", response_model=FlujoResponse)
async def update_flujo(
    flujo_id: str,
    flujo: FlujoUpdate,
    user: dict = Depends(require_auth)
):
    """Update a flow."""
    user_id = get_user_id(user)
    data = flujo.model_dump(exclude_none=True)
    if not data:
        raise HTTPException(status_code=400, detail="No hay datos para actualizar")
    result = await supabase.update_flujo(flujo_id, data, user_id)
    if not result:
        raise HTTPException(status_code=404, detail="Flujo no encontrado")
    return result


@router.delete("/{flujo_id}")
async def delete_flujo(flujo_id: str, user: dict = Depends(require_auth)):
    """Delete a flow and all its screens."""
    user_id = get_user_id(user)
    deleted = await supabase.delete_flujo(flujo_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Flujo no encontrado")
    return {"message": "Flujo eliminado"}


# ============================================
# Pantallas Management
# ============================================

@router.post("/{flujo_id}/pantallas/upload", response_model=PantallaResponse)
async def upload_pantalla(
    flujo_id: str,
    file: UploadFile = File(...),
    titulo: Optional[str] = None,
    user: dict = Depends(require_auth)
):
    """Upload an image as a new screen in the flow.

    This endpoint:
    1. Uploads the image to storage
    2. Analyzes with Gemini Vision for AOI
    3. Generates heatmap with DeepGaze/Hybrid
    4. Saves screen metadata to database
    """
    user_id = get_user_id(user)
    pantalla_id = str(uuid.uuid4())

    # Verify flow exists and belongs to user
    flujo = await supabase.get_flujo(flujo_id, user_id)
    if not flujo:
        raise HTTPException(status_code=404, detail="Flujo no encontrado")

    # Read image content
    content = await file.read()
    content_type = file.content_type or "image/png"

    # 1. Analyze with Gemini Vision
    try:
        analysis_result = await gemini_service.analyze_image(content, content_type)
        aoi_data = analysis_result.get("areas_interes", [])
    except Exception as e:
        logger.error(f"Gemini analysis failed: {e}")
        analysis_result = {}
        aoi_data = []

    # 2. Generate heatmap
    heatmap_bytes = None
    overlay_bytes = None
    modelo_usado = "hybrid_v2_ittikoch"

    # Try ML Service first
    if settings.ml_service_enabled:
        try:
            async with httpx.AsyncClient(timeout=settings.ml_service_timeout) as client:
                response = await client.post(
                    f"{settings.ml_service_url}/predict",
                    files={"file": (file.filename or "image.png", content, content_type)},
                    params={"overlay": True, "colormap": "jet"},
                )
                if response.status_code == 200:
                    ml_result = response.json()
                    heatmap_bytes = base64.b64decode(ml_result["heatmap_base64"])
                    if ml_result.get("heatmap_overlay_base64"):
                        overlay_bytes = base64.b64decode(ml_result["heatmap_overlay_base64"])
                    modelo_usado = ml_result.get("metadata", {}).get("model", "deepgaze3")
        except Exception as e:
            logger.warning(f"ML-Service unavailable: {e}")

    # Fallback to hybrid
    if heatmap_bytes is None:
        try:
            image = Image.open(io.BytesIO(content)).convert("RGB")
            image_array = np.array(image)
            heatmap, _ = hybrid_service.generate_hybrid(
                image_array, aoi_data, include_center_bias=True, include_bottom_up=True
            )
            heatmap_bytes = base64.b64decode(hybrid_service.heatmap_to_base64(heatmap, colormap="jet"))
            overlay_bytes = base64.b64decode(
                hybrid_service.heatmap_to_base64(heatmap, colormap="jet", original_image=image, alpha=0.5)
            )
        except Exception as e:
            logger.error(f"Hybrid heatmap generation failed: {e}")
            heatmap_bytes = None
            overlay_bytes = None

    # 3. Upload to storage
    try:
        screenshot_path = supabase.upload_pantalla_image(
            content, user_id, flujo_id, pantalla_id, "screenshot.png", content_type
        )
        heatmap_path = None
        overlay_path = None

        if heatmap_bytes:
            heatmap_path = supabase.upload_pantalla_image(
                heatmap_bytes, user_id, flujo_id, pantalla_id, "heatmap.png"
            )
        if overlay_bytes:
            overlay_path = supabase.upload_pantalla_image(
                overlay_bytes, user_id, flujo_id, pantalla_id, "overlay.png"
            )
    except Exception as e:
        logger.error(f"Storage upload failed: {e}")
        raise HTTPException(status_code=500, detail="Error al subir imagenes")

    # 4. Save to database
    pantalla_data = {
        "id": pantalla_id,
        "origen": "upload",
        "titulo": titulo or file.filename,
        "screenshot_path": screenshot_path,
        "heatmap_path": heatmap_path,
        "overlay_path": overlay_path,
        "clarity_score": analysis_result.get("clarity_score"),
        "areas_interes": aoi_data,
        "insights": analysis_result.get("insights", []),
        "modelo_usado": modelo_usado,
    }

    result = await supabase.create_pantalla(flujo_id, pantalla_data, user_id)
    if not result:
        raise HTTPException(status_code=500, detail="Error al guardar pantalla")

    # Generate signed URLs for response
    signed_urls = supabase.get_signed_urls_for_pantalla(user_id, flujo_id, pantalla_id)

    return PantallaResponse(
        id=pantalla_id,
        flujo_id=flujo_id,
        orden=result.get("orden", 1),
        origen="upload",
        titulo=titulo or file.filename,
        screenshot_url=signed_urls.get("screenshot"),
        heatmap_url=signed_urls.get("heatmap"),
        overlay_url=signed_urls.get("overlay"),
        clarity_score=analysis_result.get("clarity_score"),
        areas_interes=aoi_data,
        insights=analysis_result.get("insights", []),
        modelo_usado=modelo_usado,
        created_at=result.get("created_at"),
    )


@router.post("/{flujo_id}/pantallas/url", response_model=PantallaResponse)
async def add_pantalla_from_url(
    flujo_id: str,
    request: PantallaURLRequest,
    user: dict = Depends(require_auth)
):
    """Add a screen by capturing a URL (requires web-capture-service).

    Note: This is a placeholder that will be implemented when the
    web-capture-service is deployed. For now, it returns an error
    indicating the service is not available.
    """
    user_id = get_user_id(user)

    # Verify flow exists
    flujo = await supabase.get_flujo(flujo_id, user_id)
    if not flujo:
        raise HTTPException(status_code=404, detail="Flujo no encontrado")

    # Check if web capture service is configured
    web_capture_url = getattr(settings, 'web_capture_service_url', None)
    if not web_capture_url:
        raise HTTPException(
            status_code=503,
            detail="Servicio de captura web no disponible. Use la opcion de subir imagen."
        )

    # TODO: Implement web capture when service is deployed
    # This will call the web-capture-service to screenshot the URL
    raise HTTPException(
        status_code=501,
        detail="Captura de URL no implementada aun. Use la opcion de subir imagen."
    )


@router.get("/{flujo_id}/pantallas/{pantalla_id}", response_model=PantallaResponse)
async def get_pantalla(
    flujo_id: str,
    pantalla_id: str,
    user: dict = Depends(require_auth)
):
    """Get a specific screen with fresh signed URLs."""
    user_id = get_user_id(user)
    result = await supabase.get_pantalla(pantalla_id, user_id)
    if not result:
        raise HTTPException(status_code=404, detail="Pantalla no encontrada")

    # Verify it belongs to the flow
    if result.get("flujo_id") != flujo_id:
        raise HTTPException(status_code=404, detail="Pantalla no encontrada en este flujo")

    # Generate fresh signed URLs
    signed_urls = supabase.get_signed_urls_for_pantalla(user_id, flujo_id, pantalla_id)

    return PantallaResponse(
        id=result.get("id"),
        flujo_id=flujo_id,
        orden=result.get("orden", 1),
        origen=result.get("origen", "upload"),
        url=result.get("url"),
        titulo=result.get("titulo"),
        screenshot_url=signed_urls.get("screenshot"),
        heatmap_url=signed_urls.get("heatmap"),
        overlay_url=signed_urls.get("overlay"),
        clarity_score=result.get("clarity_score"),
        areas_interes=result.get("areas_interes"),
        insights=result.get("insights"),
        modelo_usado=result.get("modelo_usado"),
        elementos_clickeables=result.get("elementos_clickeables"),
        created_at=result.get("created_at"),
    )


@router.delete("/{flujo_id}/pantallas/{pantalla_id}")
async def delete_pantalla(
    flujo_id: str,
    pantalla_id: str,
    user: dict = Depends(require_auth)
):
    """Delete a screen from the flow."""
    user_id = get_user_id(user)

    # Verify pantalla exists and belongs to this flow
    pantalla = await supabase.get_pantalla(pantalla_id, user_id)
    if not pantalla or pantalla.get("flujo_id") != flujo_id:
        raise HTTPException(status_code=404, detail="Pantalla no encontrada")

    deleted = await supabase.delete_pantalla(pantalla_id, user_id)
    if not deleted:
        raise HTTPException(status_code=500, detail="Error al eliminar pantalla")

    return {"message": "Pantalla eliminada"}


@router.put("/{flujo_id}/pantallas/reordenar")
async def reordenar_pantallas(
    flujo_id: str,
    request: ReordenarRequest,
    user: dict = Depends(require_auth)
):
    """Reorder screens in a flow."""
    user_id = get_user_id(user)

    # Verify flow exists
    flujo = await supabase.get_flujo(flujo_id, user_id)
    if not flujo:
        raise HTTPException(status_code=404, detail="Flujo no encontrado")

    success = await supabase.reordenar_pantallas(flujo_id, request.orden_ids, user_id)
    if not success:
        raise HTTPException(status_code=500, detail="Error al reordenar pantallas")

    return {"message": "Pantallas reordenadas"}
