"""Usability Testing API routes."""
import io
import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from pydantic import BaseModel
import httpx

from app.core.security import require_auth
from app.services import supabase
from app.services.element_detection import ElementDetectionService
from app.services.usability_simulation import UsabilitySimulationService

logger = logging.getLogger(__name__)

router = APIRouter()
element_service = ElementDetectionService()
simulation_service = UsabilitySimulationService()


# ============================================
# Pydantic Schemas
# ============================================

class MisionCreate(BaseModel):
    """Schema for creating a mission."""
    nombre: str
    instrucciones: str
    pantalla_inicio_id: Optional[str] = None
    pantalla_objetivo_id: Optional[str] = None
    elemento_objetivo: Optional[dict] = None
    max_pasos: int = 10


class MisionUpdate(BaseModel):
    """Schema for updating a mission."""
    nombre: Optional[str] = None
    instrucciones: Optional[str] = None
    pantalla_inicio_id: Optional[str] = None
    pantalla_objetivo_id: Optional[str] = None
    elemento_objetivo: Optional[dict] = None
    max_pasos: Optional[int] = None
    estado: Optional[str] = None


class MisionResponse(BaseModel):
    """Schema for mission response."""
    id: str
    flujo_id: str
    nombre: str
    instrucciones: str
    pantalla_inicio_id: Optional[str] = None
    pantalla_objetivo_id: Optional[str] = None
    elemento_objetivo: Optional[dict] = None
    max_pasos: int = 10
    estado: str = "activa"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class MisionDetailResponse(MisionResponse):
    """Schema for mission detail with simulations."""
    simulaciones: list[dict] = []


class SimulacionRequest(BaseModel):
    """Schema for running a simulation."""
    arquetipo_ids: list[str]


class SimulacionResponse(BaseModel):
    """Schema for simulation response."""
    id: str
    mision_id: str
    arquetipo_id: str
    completada: bool = False
    exito: bool = False
    path_tomado: list[dict] = []
    pasos_totales: int = 0
    misclicks: int = 0
    tiempo_estimado_ms: Optional[int] = None
    fricciones: list[dict] = []
    feedback_arquetipo: Optional[str] = None
    emociones: dict = {}
    created_at: Optional[str] = None


class ElementosUpdateRequest(BaseModel):
    """Schema for updating elements manually."""
    elementos_clickeables: list[dict]


class MetricasResponse(BaseModel):
    """Schema for aggregated metrics."""
    total_simulaciones: int
    success_rate: float
    avg_misclicks: float
    avg_pasos: float
    avg_tiempo_ms: float
    total_fricciones: int
    fricciones_por_tipo: dict
    metrics_by_nivel: dict


def get_user_id(user: dict) -> str:
    """Extract user ID from JWT payload."""
    return user.get("sub")


# ============================================
# Element Detection
# ============================================

@router.post("/pantallas/{pantalla_id}/detectar-elementos")
async def detectar_elementos(
    pantalla_id: str,
    user: dict = Depends(require_auth)
):
    """
    Detect clickeable elements in a screen using Gemini Vision.

    Returns the detected elements and updates the screen record.
    """
    user_id = get_user_id(user)

    # Get pantalla
    pantalla = await supabase.get_pantalla(pantalla_id, user_id)
    if not pantalla:
        raise HTTPException(status_code=404, detail="Pantalla no encontrada")

    # Get screenshot URL
    flujo_id = pantalla.get("flujo_id")
    signed_urls = supabase.get_signed_urls_for_pantalla(user_id, flujo_id, pantalla_id)
    screenshot_url = signed_urls.get("screenshot")

    if not screenshot_url:
        raise HTTPException(status_code=400, detail="Pantalla no tiene screenshot")

    # Fetch image
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(screenshot_url, timeout=30.0)
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="Error al obtener imagen")
            image_data = response.content
            content_type = response.headers.get("content-type", "image/png")
    except Exception as e:
        logger.error(f"Error fetching screenshot: {e}")
        raise HTTPException(status_code=500, detail="Error al obtener screenshot")

    # Detect elements
    try:
        elementos = await element_service.detect_elements(image_data, content_type)
    except Exception as e:
        logger.error(f"Element detection failed: {e}")
        raise HTTPException(status_code=500, detail="Error en deteccion de elementos")

    # Update pantalla with detected elements
    await supabase.update_pantalla(
        pantalla_id,
        {"elementos_clickeables": elementos},
        user_id
    )

    return {
        "pantalla_id": pantalla_id,
        "elementos_detectados": len(elementos),
        "elementos": elementos
    }


@router.put("/pantallas/{pantalla_id}/elementos")
async def update_elementos(
    pantalla_id: str,
    request: ElementosUpdateRequest,
    user: dict = Depends(require_auth)
):
    """
    Manually update/correct detected elements for a screen.
    """
    user_id = get_user_id(user)

    # Verify pantalla exists
    pantalla = await supabase.get_pantalla(pantalla_id, user_id)
    if not pantalla:
        raise HTTPException(status_code=404, detail="Pantalla no encontrada")

    # Update elements
    result = await supabase.update_pantalla(
        pantalla_id,
        {"elementos_clickeables": request.elementos_clickeables},
        user_id
    )

    if not result:
        raise HTTPException(status_code=500, detail="Error al actualizar elementos")

    return {
        "pantalla_id": pantalla_id,
        "elementos_actualizados": len(request.elementos_clickeables)
    }


# ============================================
# Misiones CRUD
# ============================================

@router.get("/flujos/{flujo_id}/misiones")
async def list_misiones(
    flujo_id: str,
    user: dict = Depends(require_auth)
):
    """List all missions for a flow."""
    user_id = get_user_id(user)

    # Verify flow exists
    flujo = await supabase.get_flujo(flujo_id, user_id)
    if not flujo:
        raise HTTPException(status_code=404, detail="Flujo no encontrado")

    misiones = await supabase.list_misiones_flujo(flujo_id, user_id)
    return {"misiones": misiones, "total": len(misiones)}


@router.post("/flujos/{flujo_id}/misiones", response_model=MisionResponse)
async def create_mision(
    flujo_id: str,
    mision: MisionCreate,
    user: dict = Depends(require_auth)
):
    """Create a new usability testing mission."""
    user_id = get_user_id(user)

    # Verify flow exists
    flujo = await supabase.get_flujo(flujo_id, user_id)
    if not flujo:
        raise HTTPException(status_code=404, detail="Flujo no encontrado")

    # Set default start screen if not provided
    data = mision.model_dump(exclude_none=True)
    if not data.get("pantalla_inicio_id") and flujo.get("pantallas"):
        data["pantalla_inicio_id"] = flujo["pantallas"][0].get("id")

    result = await supabase.create_mision(flujo_id, data, user_id)
    if not result:
        raise HTTPException(status_code=500, detail="Error al crear mision")

    return MisionResponse(flujo_id=flujo_id, **result)


@router.get("/flujos/{flujo_id}/misiones/{mision_id}", response_model=MisionDetailResponse)
async def get_mision(
    flujo_id: str,
    mision_id: str,
    user: dict = Depends(require_auth)
):
    """Get a mission with all its simulations."""
    user_id = get_user_id(user)

    result = await supabase.get_mision_with_simulaciones(mision_id, user_id)
    if not result:
        raise HTTPException(status_code=404, detail="Mision no encontrada")

    if result.get("flujo_id") != flujo_id:
        raise HTTPException(status_code=404, detail="Mision no pertenece a este flujo")

    return result


@router.put("/flujos/{flujo_id}/misiones/{mision_id}", response_model=MisionResponse)
async def update_mision(
    flujo_id: str,
    mision_id: str,
    mision: MisionUpdate,
    user: dict = Depends(require_auth)
):
    """Update a mission."""
    user_id = get_user_id(user)

    # Verify mission exists and belongs to flow
    existing = await supabase.get_mision(mision_id, user_id)
    if not existing or existing.get("flujo_id") != flujo_id:
        raise HTTPException(status_code=404, detail="Mision no encontrada")

    data = mision.model_dump(exclude_none=True)
    if not data:
        raise HTTPException(status_code=400, detail="No hay datos para actualizar")

    result = await supabase.update_mision(mision_id, data, user_id)
    if not result:
        raise HTTPException(status_code=500, detail="Error al actualizar mision")

    return MisionResponse(flujo_id=flujo_id, **result)


@router.delete("/flujos/{flujo_id}/misiones/{mision_id}")
async def delete_mision(
    flujo_id: str,
    mision_id: str,
    user: dict = Depends(require_auth)
):
    """Delete a mission and all its simulations."""
    user_id = get_user_id(user)

    # Verify mission exists and belongs to flow
    existing = await supabase.get_mision(mision_id, user_id)
    if not existing or existing.get("flujo_id") != flujo_id:
        raise HTTPException(status_code=404, detail="Mision no encontrada")

    deleted = await supabase.delete_mision(mision_id, user_id)
    if not deleted:
        raise HTTPException(status_code=500, detail="Error al eliminar mision")

    return {"message": "Mision eliminada"}


# ============================================
# Simulaciones
# ============================================

@router.post("/misiones/{mision_id}/simular")
async def run_simulation(
    mision_id: str,
    request: SimulacionRequest,
    user: dict = Depends(require_auth)
):
    """
    Run usability simulations for a mission with specified archetypes.

    This will simulate each archetype navigating through the flow
    trying to complete the mission.
    """
    user_id = get_user_id(user)

    # Get mission
    mision = await supabase.get_mision(mision_id, user_id)
    if not mision:
        raise HTTPException(status_code=404, detail="Mision no encontrada")

    # Get flow with screens
    flujo = await supabase.get_flujo(mision.get("flujo_id"), user_id)
    if not flujo:
        raise HTTPException(status_code=404, detail="Flujo no encontrado")

    pantallas = flujo.get("pantallas", [])
    if not pantallas:
        raise HTTPException(status_code=400, detail="Flujo no tiene pantallas")

    # Get archetypes
    arquetipos = []
    for arq_id in request.arquetipo_ids:
        arq = await supabase.get_arquetipo(arq_id, user_id)
        if arq:
            arquetipos.append(arq)

    if not arquetipos:
        raise HTTPException(status_code=400, detail="No se encontraron arquetipos validos")

    # Fetch images for all screens
    pantalla_imagenes = {}
    async with httpx.AsyncClient() as client:
        for pantalla in pantallas:
            signed_urls = supabase.get_signed_urls_for_pantalla(
                user_id, flujo.get("id"), pantalla.get("id")
            )
            screenshot_url = signed_urls.get("screenshot")
            if screenshot_url:
                try:
                    response = await client.get(screenshot_url, timeout=30.0)
                    if response.status_code == 200:
                        pantalla_imagenes[pantalla.get("id")] = response.content
                except Exception as e:
                    logger.warning(f"Could not fetch screenshot for {pantalla.get('id')}: {e}")

    # Run simulations for each archetype
    resultados = []
    for arquetipo in arquetipos:
        try:
            sim_result = await simulation_service.simulate_mission(
                mision=mision,
                arquetipo=arquetipo,
                pantallas=pantallas,
                pantalla_imagenes=pantalla_imagenes
            )

            # Save simulation to database
            saved = await supabase.create_simulacion(
                mision_id=mision_id,
                arquetipo_id=arquetipo.get("id"),
                data=sim_result,
                user_id=user_id
            )

            if saved:
                saved["arquetipo_nombre"] = arquetipo.get("nombre")
                saved["arquetipo_nivel"] = arquetipo.get("nivel_digital")
                resultados.append(saved)
            else:
                resultados.append({
                    **sim_result,
                    "arquetipo_id": arquetipo.get("id"),
                    "arquetipo_nombre": arquetipo.get("nombre"),
                    "error": "No se pudo guardar simulacion"
                })

        except Exception as e:
            logger.error(f"Simulation failed for archetype {arquetipo.get('id')}: {e}")
            resultados.append({
                "arquetipo_id": arquetipo.get("id"),
                "arquetipo_nombre": arquetipo.get("nombre"),
                "completada": False,
                "exito": False,
                "error": str(e)
            })

    return {
        "mision_id": mision_id,
        "total_simulaciones": len(resultados),
        "exitosas": sum(1 for r in resultados if r.get("exito")),
        "resultados": resultados
    }


@router.get("/misiones/{mision_id}/simulaciones")
async def list_simulaciones(
    mision_id: str,
    user: dict = Depends(require_auth)
):
    """List all simulations for a mission."""
    user_id = get_user_id(user)

    # Verify mission exists
    mision = await supabase.get_mision(mision_id, user_id)
    if not mision:
        raise HTTPException(status_code=404, detail="Mision no encontrada")

    simulaciones = await supabase.get_simulaciones_with_arquetipos(mision_id, user_id)
    return {"simulaciones": simulaciones, "total": len(simulaciones)}


@router.get("/misiones/{mision_id}/metricas", response_model=MetricasResponse)
async def get_metricas(
    mision_id: str,
    user: dict = Depends(require_auth)
):
    """Get aggregated metrics for all simulations of a mission."""
    user_id = get_user_id(user)

    # Verify mission exists
    mision = await supabase.get_mision(mision_id, user_id)
    if not mision:
        raise HTTPException(status_code=404, detail="Mision no encontrada")

    # Get simulations with archetype info
    simulaciones = await supabase.get_simulaciones_with_arquetipos(mision_id, user_id)

    # Calculate metrics
    metricas = await simulation_service.calculate_aggregated_metrics(simulaciones)

    # Calculate metrics by nivel_digital
    metrics_by_nivel = {"bajo": [], "medio": [], "alto": []}
    for sim in simulaciones:
        arq = sim.get("arquetipo", {})
        nivel = arq.get("nivel_digital", "medio")
        if nivel in metrics_by_nivel:
            metrics_by_nivel[nivel].append(sim)

    # Aggregate by nivel
    for nivel, sims in metrics_by_nivel.items():
        if sims:
            total = len(sims)
            exitosas = sum(1 for s in sims if s.get("exito"))
            misclicks = sum(s.get("misclicks", 0) for s in sims)
            metricas["metrics_by_nivel"][nivel] = {
                "total": total,
                "exitos": exitosas,
                "success_rate": (exitosas / total) * 100 if total > 0 else 0,
                "avg_misclicks": misclicks / total if total > 0 else 0
            }

    return metricas
