"""Supabase Database Service."""
from functools import lru_cache
from typing import Optional
from supabase import create_client, Client
from app.core.config import get_settings

settings = get_settings()

# Table names with us_ prefix
TABLE_ARQUETIPOS = "us_arquetipos"
TABLE_ANALISIS = "us_analisis"
TABLE_SESIONES = "us_sesiones"
TABLE_MENSAJES = "us_mensajes"
TABLE_REPORTES = "us_reportes"


@lru_cache
def get_supabase_client() -> Client:
    """Get cached Supabase client."""
    if not settings.supabase_url or not settings.supabase_service_role_key:
        raise ValueError("Supabase not configured")
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


# ============================================
# Arquetipos
# ============================================
async def create_arquetipo(data: dict, user_id: str) -> dict:
    """Create a new archetype in the database."""
    client = get_supabase_client()
    data["user_id"] = user_id
    result = client.table(TABLE_ARQUETIPOS).insert(data).execute()
    return result.data[0] if result.data else None


async def get_arquetipo(arquetipo_id: str, user_id: str) -> Optional[dict]:
    """Get an archetype by ID (filtered by user)."""
    client = get_supabase_client()
    result = client.table(TABLE_ARQUETIPOS).select("*").eq("id", arquetipo_id).eq("user_id", user_id).execute()
    return result.data[0] if result.data else None


async def list_arquetipos(user_id: str, limit: int = 50, offset: int = 0) -> tuple[list[dict], int]:
    """List all archetypes for a user."""
    client = get_supabase_client()
    result = client.table(TABLE_ARQUETIPOS).select("*", count="exact").eq("user_id", user_id).range(offset, offset + limit - 1).order("created_at", desc=True).execute()
    return result.data, result.count or 0


async def update_arquetipo(arquetipo_id: str, data: dict, user_id: str) -> Optional[dict]:
    """Update an archetype (filtered by user)."""
    client = get_supabase_client()
    result = client.table(TABLE_ARQUETIPOS).update(data).eq("id", arquetipo_id).eq("user_id", user_id).execute()
    return result.data[0] if result.data else None


async def delete_arquetipo(arquetipo_id: str, user_id: str) -> bool:
    """Delete an archetype (filtered by user)."""
    client = get_supabase_client()
    result = client.table(TABLE_ARQUETIPOS).delete().eq("id", arquetipo_id).eq("user_id", user_id).execute()
    return len(result.data) > 0 if result.data else False


# ============================================
# Analisis
# ============================================
async def create_analisis(data: dict, user_id: str) -> dict:
    """Create a new analysis in the database."""
    client = get_supabase_client()
    data["user_id"] = user_id
    result = client.table(TABLE_ANALISIS).insert(data).execute()
    return result.data[0] if result.data else None


async def get_analisis(analisis_id: str, user_id: str) -> Optional[dict]:
    """Get analysis by ID (filtered by user)."""
    client = get_supabase_client()
    result = client.table(TABLE_ANALISIS).select("*").eq("id", analisis_id).eq("user_id", user_id).execute()
    return result.data[0] if result.data else None


async def list_analisis(user_id: str, limit: int = 50, offset: int = 0) -> tuple[list[dict], int]:
    """List all analyses for a user."""
    client = get_supabase_client()
    result = client.table(TABLE_ANALISIS).select("*", count="exact").eq("user_id", user_id).range(offset, offset + limit - 1).order("created_at", desc=True).execute()
    return result.data, result.count or 0


# ============================================
# Sesiones de interacción
# ============================================
async def create_sesion(arquetipo_id: str, user_id: str, contexto: Optional[str] = None) -> dict:
    """Create a new interaction session."""
    client = get_supabase_client()
    data = {
        "arquetipo_id": arquetipo_id,
        "user_id": user_id,
        "contexto": contexto,
        "estado": "activa"
    }
    result = client.table(TABLE_SESIONES).insert(data).execute()
    return result.data[0] if result.data else None


async def get_sesion(session_id: str, user_id: str) -> Optional[dict]:
    """Get a session by ID (filtered by user)."""
    client = get_supabase_client()
    result = client.table(TABLE_SESIONES).select("*").eq("id", session_id).eq("user_id", user_id).execute()
    return result.data[0] if result.data else None


async def list_sesiones(user_id: str, arquetipo_id: Optional[str] = None, limit: int = 50, offset: int = 0) -> tuple[list[dict], int]:
    """List all sessions for a user, optionally filtered by archetype."""
    client = get_supabase_client()
    query = client.table(TABLE_SESIONES).select("*", count="exact").eq("user_id", user_id)
    if arquetipo_id:
        query = query.eq("arquetipo_id", arquetipo_id)
    result = query.range(offset, offset + limit - 1).order("created_at", desc=True).execute()
    return result.data, result.count or 0


async def save_mensaje(session_id: str, data: dict) -> dict:
    """Save a chat message."""
    client = get_supabase_client()
    data["session_id"] = session_id
    result = client.table(TABLE_MENSAJES).insert(data).execute()
    return result.data[0] if result.data else None


async def get_historial(session_id: str) -> list[dict]:
    """Get conversation history for a session."""
    client = get_supabase_client()
    result = client.table(TABLE_MENSAJES).select("*").eq("session_id", session_id).order("created_at").execute()
    return result.data


# ============================================
# Reportes
# ============================================
async def create_reporte(data: dict, user_id: str) -> dict:
    """Create a new report in the database."""
    client = get_supabase_client()
    data["user_id"] = user_id
    result = client.table(TABLE_REPORTES).insert(data).execute()
    return result.data[0] if result.data else None


async def get_reporte(reporte_id: str, user_id: str) -> Optional[dict]:
    """Get report by ID (filtered by user)."""
    client = get_supabase_client()
    result = client.table(TABLE_REPORTES).select("*").eq("id", reporte_id).eq("user_id", user_id).execute()
    return result.data[0] if result.data else None


async def list_reportes(user_id: str, limit: int = 50, offset: int = 0) -> tuple[list[dict], int]:
    """List all reports for a user."""
    client = get_supabase_client()
    result = client.table(TABLE_REPORTES).select("*", count="exact").eq("user_id", user_id).range(offset, offset + limit - 1).order("created_at", desc=True).execute()
    return result.data, result.count or 0


async def delete_reporte(reporte_id: str, user_id: str) -> bool:
    """Delete a report (filtered by user)."""
    client = get_supabase_client()
    result = client.table(TABLE_REPORTES).delete().eq("id", reporte_id).eq("user_id", user_id).execute()
    return len(result.data) > 0 if result.data else False
