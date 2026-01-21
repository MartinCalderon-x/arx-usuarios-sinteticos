"""Supabase Database Service."""
from functools import lru_cache
from supabase import create_client, Client
from app.core.config import get_settings

settings = get_settings()


@lru_cache
def get_supabase_client() -> Client:
    """Get cached Supabase client."""
    if not settings.supabase_url or not settings.supabase_service_role_key:
        raise ValueError("Supabase not configured")
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


# Arquetipos
async def create_arquetipo(data: dict) -> dict:
    """Create a new archetype in the database."""
    client = get_supabase_client()
    result = client.table("arquetipos").insert(data).execute()
    return result.data[0] if result.data else None


async def get_arquetipo(arquetipo_id: str) -> dict:
    """Get an archetype by ID."""
    client = get_supabase_client()
    result = client.table("arquetipos").select("*").eq("id", arquetipo_id).execute()
    return result.data[0] if result.data else None


async def list_arquetipos(limit: int = 50, offset: int = 0) -> list[dict]:
    """List all archetypes."""
    client = get_supabase_client()
    result = client.table("arquetipos").select("*").range(offset, offset + limit - 1).execute()
    return result.data


async def update_arquetipo(arquetipo_id: str, data: dict) -> dict:
    """Update an archetype."""
    client = get_supabase_client()
    result = client.table("arquetipos").update(data).eq("id", arquetipo_id).execute()
    return result.data[0] if result.data else None


async def delete_arquetipo(arquetipo_id: str) -> bool:
    """Delete an archetype."""
    client = get_supabase_client()
    client.table("arquetipos").delete().eq("id", arquetipo_id).execute()
    return True


# Analisis
async def save_analisis(data: dict) -> dict:
    """Save analysis results."""
    client = get_supabase_client()
    result = client.table("analisis").insert(data).execute()
    return result.data[0] if result.data else None


async def get_analisis(analisis_id: str) -> dict:
    """Get analysis by ID."""
    client = get_supabase_client()
    result = client.table("analisis").select("*").eq("id", analisis_id).execute()
    return result.data[0] if result.data else None


# Sesiones de interacción
async def save_mensaje(session_id: str, data: dict) -> dict:
    """Save a chat message."""
    client = get_supabase_client()
    data["session_id"] = session_id
    result = client.table("mensajes").insert(data).execute()
    return result.data[0] if result.data else None


async def get_historial(session_id: str) -> list[dict]:
    """Get conversation history."""
    client = get_supabase_client()
    result = client.table("mensajes").select("*").eq("session_id", session_id).order("created_at").execute()
    return result.data
