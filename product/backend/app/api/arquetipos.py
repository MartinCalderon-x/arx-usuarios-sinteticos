"""Arquetipos (Synthetic Users) API routes."""
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from pydantic import BaseModel

from app.core.security import require_auth
from app.services import supabase

router = APIRouter()


class ArquetipoCreate(BaseModel):
    """Schema for creating an archetype."""
    nombre: str
    descripcion: str
    edad: Optional[int] = None
    genero: Optional[str] = None
    ocupacion: Optional[str] = None
    contexto: Optional[str] = None
    comportamiento: Optional[str] = None
    frustraciones: Optional[list[str]] = None
    objetivos: Optional[list[str]] = None
    template_id: Optional[str] = None


class ArquetipoResponse(BaseModel):
    """Schema for archetype response."""
    id: str
    nombre: str
    descripcion: Optional[str] = None
    edad: Optional[int] = None
    genero: Optional[str] = None
    ocupacion: Optional[str] = None
    contexto: Optional[str] = None
    comportamiento: Optional[str] = None
    frustraciones: Optional[list[str]] = None
    objetivos: Optional[list[str]] = None
    created_at: Optional[str] = None


def get_user_id(user: dict) -> str:
    """Extract user ID from JWT payload."""
    return user.get("sub")


@router.get("/")
async def list_arquetipos(user: dict = Depends(require_auth)):
    """List all archetypes. Requires authentication."""
    user_id = get_user_id(user)
    arquetipos, total = await supabase.list_arquetipos(user_id)
    return {"arquetipos": arquetipos, "total": total}


@router.post("/", response_model=ArquetipoResponse)
async def create_arquetipo(arquetipo: ArquetipoCreate, user: dict = Depends(require_auth)):
    """Create a new archetype. Requires authentication."""
    user_id = get_user_id(user)
    data = arquetipo.model_dump(exclude_none=True)
    result = await supabase.create_arquetipo(data, user_id)
    if not result:
        raise HTTPException(status_code=500, detail="Error al crear arquetipo")
    return result


@router.get("/{arquetipo_id}", response_model=ArquetipoResponse)
async def get_arquetipo(arquetipo_id: str, user: dict = Depends(require_auth)):
    """Get an archetype by ID. Requires authentication."""
    user_id = get_user_id(user)
    result = await supabase.get_arquetipo(arquetipo_id, user_id)
    if not result:
        raise HTTPException(status_code=404, detail="Arquetipo no encontrado")
    return result


@router.put("/{arquetipo_id}", response_model=ArquetipoResponse)
async def update_arquetipo(arquetipo_id: str, arquetipo: ArquetipoCreate, user: dict = Depends(require_auth)):
    """Update an archetype. Requires authentication."""
    user_id = get_user_id(user)
    data = arquetipo.model_dump(exclude_none=True)
    result = await supabase.update_arquetipo(arquetipo_id, data, user_id)
    if not result:
        raise HTTPException(status_code=404, detail="Arquetipo no encontrado")
    return result


@router.delete("/{arquetipo_id}")
async def delete_arquetipo(arquetipo_id: str, user: dict = Depends(require_auth)):
    """Delete an archetype. Requires authentication."""
    user_id = get_user_id(user)
    deleted = await supabase.delete_arquetipo(arquetipo_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Arquetipo no encontrado")
    return {"message": "Arquetipo eliminado"}


@router.get("/templates/")
async def list_templates():
    """List available archetype templates."""
    # Predefined templates
    templates = [
        {
            "id": "consumidor-digital",
            "nombre": "Consumidor Digital",
            "descripcion": "Usuario frecuente de apps y servicios digitales",
            "edad": 28,
            "ocupacion": "Profesional urbano",
        },
        {
            "id": "operario-industrial",
            "nombre": "Operario Industrial",
            "descripcion": "Trabajador de planta de producción",
            "edad": 35,
            "ocupacion": "Operador de maquinaria",
        },
        {
            "id": "adulto-mayor",
            "nombre": "Adulto Mayor",
            "descripcion": "Usuario con menor experiencia tecnológica",
            "edad": 65,
            "ocupacion": "Jubilado",
        },
    ]
    return {"templates": templates}
