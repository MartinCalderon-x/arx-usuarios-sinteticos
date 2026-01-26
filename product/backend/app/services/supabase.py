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
TABLE_FLUJOS = "us_flujos"
TABLE_PANTALLAS = "us_pantallas"
TABLE_TRANSICIONES = "us_transiciones"


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


# ============================================
# Storage (Bucket Privado)
# ============================================
BUCKET_ANALISIS = "us-analisis-images"
SIGNED_URL_EXPIRY = 3600  # 1 hora


def upload_image_to_storage(
    image_data: bytes,
    user_id: str,
    analisis_id: str,
    filename: str,
    content_type: str = "image/png"
) -> str:
    """Upload image to private Supabase Storage.

    Args:
        image_data: Raw image bytes.
        user_id: User ID for folder structure.
        analisis_id: Analysis ID for folder structure.
        filename: Filename (e.g., 'original.png', 'heatmap.png').
        content_type: MIME type of the image.

    Returns:
        Storage path (not URL) - bucket is private.
    """
    client = get_supabase_client()
    path = f"{user_id}/{analisis_id}/{filename}"

    client.storage.from_(BUCKET_ANALISIS).upload(
        path=path,
        file=image_data,
        file_options={"content-type": content_type}
    )

    return path


def get_signed_url(storage_path: str, expires_in: int = SIGNED_URL_EXPIRY) -> Optional[str]:
    """Generate signed URL for private file access.

    Args:
        storage_path: Path to file in storage bucket.
        expires_in: URL expiration time in seconds (default: 1 hour).

    Returns:
        Signed URL string or None if failed.
    """
    client = get_supabase_client()
    try:
        response = client.storage.from_(BUCKET_ANALISIS).create_signed_url(
            storage_path, expires_in
        )
        # Supabase SDK returns different key names depending on version
        return response.get("signedURL") or response.get("signedUrl")
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to create signed URL: {e}")
        return None


def get_signed_urls_for_analisis(user_id: str, analisis_id: str) -> dict:
    """Get signed URLs for all images of an analysis.

    Args:
        user_id: User ID.
        analisis_id: Analysis ID.

    Returns:
        Dict with signed URLs for 'original', 'heatmap', and 'overlay'.
    """
    paths = {
        "original": f"{user_id}/{analisis_id}/original.png",
        "heatmap": f"{user_id}/{analisis_id}/heatmap.png",
        "overlay": f"{user_id}/{analisis_id}/heatmap_overlay.png",
    }
    return {key: get_signed_url(path) for key, path in paths.items()}


def delete_analisis_images(user_id: str, analisis_id: str) -> bool:
    """Delete all images for an analysis from storage.

    Args:
        user_id: User ID.
        analisis_id: Analysis ID.

    Returns:
        True if successful, False otherwise.
    """
    client = get_supabase_client()
    folder_path = f"{user_id}/{analisis_id}"
    try:
        files = client.storage.from_(BUCKET_ANALISIS).list(folder_path)
        if files:
            paths = [f"{folder_path}/{f['name']}" for f in files]
            client.storage.from_(BUCKET_ANALISIS).remove(paths)
        return True
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to delete analysis images: {e}")
        return False


# ============================================
# Flujos
# ============================================
BUCKET_FLUJOS = "us-flujos-images"


async def create_flujo(data: dict, user_id: str) -> dict:
    """Create a new flow in the database."""
    client = get_supabase_client()
    data["user_id"] = user_id
    result = client.table(TABLE_FLUJOS).insert(data).execute()
    return result.data[0] if result.data else None


async def get_flujo(flujo_id: str, user_id: str) -> Optional[dict]:
    """Get a flow by ID with its screens (filtered by user)."""
    client = get_supabase_client()
    result = client.table(TABLE_FLUJOS).select("*").eq("id", flujo_id).eq("user_id", user_id).execute()
    if not result.data:
        return None

    flujo = result.data[0]

    # Get all screens for this flow
    pantallas_result = client.table(TABLE_PANTALLAS).select("*").eq("flujo_id", flujo_id).order("orden").execute()
    flujo["pantallas"] = pantallas_result.data or []

    return flujo


async def list_flujos(user_id: str, estado: Optional[str] = None, limit: int = 50, offset: int = 0) -> tuple[list[dict], int]:
    """List all flows for a user."""
    client = get_supabase_client()
    query = client.table(TABLE_FLUJOS).select("*", count="exact").eq("user_id", user_id)
    if estado:
        query = query.eq("estado", estado)
    result = query.range(offset, offset + limit - 1).order("created_at", desc=True).execute()
    return result.data, result.count or 0


async def update_flujo(flujo_id: str, data: dict, user_id: str) -> Optional[dict]:
    """Update a flow (filtered by user)."""
    client = get_supabase_client()
    result = client.table(TABLE_FLUJOS).update(data).eq("id", flujo_id).eq("user_id", user_id).execute()
    return result.data[0] if result.data else None


async def delete_flujo(flujo_id: str, user_id: str) -> bool:
    """Delete a flow and all its screens (filtered by user)."""
    client = get_supabase_client()
    # Delete images from storage first
    pantallas = client.table(TABLE_PANTALLAS).select("id").eq("flujo_id", flujo_id).execute()
    if pantallas.data:
        for pantalla in pantallas.data:
            delete_pantalla_images(user_id, flujo_id, pantalla["id"])

    # Delete flow (cascade will delete pantallas and transiciones)
    result = client.table(TABLE_FLUJOS).delete().eq("id", flujo_id).eq("user_id", user_id).execute()
    return len(result.data) > 0 if result.data else False


# ============================================
# Pantallas
# ============================================
async def create_pantalla(flujo_id: str, data: dict, user_id: str) -> dict:
    """Create a new screen in a flow."""
    client = get_supabase_client()
    data["flujo_id"] = flujo_id
    data["user_id"] = user_id

    # Get next order number if not specified
    if "orden" not in data:
        max_orden = client.table(TABLE_PANTALLAS).select("orden").eq("flujo_id", flujo_id).order("orden", desc=True).limit(1).execute()
        data["orden"] = (max_orden.data[0]["orden"] + 1) if max_orden.data else 1

    result = client.table(TABLE_PANTALLAS).insert(data).execute()
    return result.data[0] if result.data else None


async def get_pantalla(pantalla_id: str, user_id: str) -> Optional[dict]:
    """Get a screen by ID (filtered by user)."""
    client = get_supabase_client()
    result = client.table(TABLE_PANTALLAS).select("*").eq("id", pantalla_id).eq("user_id", user_id).execute()
    return result.data[0] if result.data else None


async def update_pantalla(pantalla_id: str, data: dict, user_id: str) -> Optional[dict]:
    """Update a screen (filtered by user)."""
    client = get_supabase_client()
    result = client.table(TABLE_PANTALLAS).update(data).eq("id", pantalla_id).eq("user_id", user_id).execute()
    return result.data[0] if result.data else None


async def delete_pantalla(pantalla_id: str, user_id: str) -> bool:
    """Delete a screen (filtered by user)."""
    client = get_supabase_client()
    # Get pantalla info for storage cleanup
    pantalla = client.table(TABLE_PANTALLAS).select("flujo_id").eq("id", pantalla_id).eq("user_id", user_id).execute()
    if pantalla.data:
        flujo_id = pantalla.data[0]["flujo_id"]
        delete_pantalla_images(user_id, flujo_id, pantalla_id)

    result = client.table(TABLE_PANTALLAS).delete().eq("id", pantalla_id).eq("user_id", user_id).execute()
    return len(result.data) > 0 if result.data else False


async def reordenar_pantallas(flujo_id: str, orden_ids: list[str], user_id: str) -> bool:
    """Reorder screens in a flow."""
    client = get_supabase_client()
    # Verify flow ownership
    flujo = client.table(TABLE_FLUJOS).select("id").eq("id", flujo_id).eq("user_id", user_id).execute()
    if not flujo.data:
        return False

    # Update order for each screen
    for idx, pantalla_id in enumerate(orden_ids, start=1):
        client.table(TABLE_PANTALLAS).update({"orden": idx}).eq("id", pantalla_id).eq("flujo_id", flujo_id).execute()

    return True


# ============================================
# Storage for Flujos
# ============================================
def upload_pantalla_image(
    image_data: bytes,
    user_id: str,
    flujo_id: str,
    pantalla_id: str,
    filename: str,
    content_type: str = "image/png"
) -> str:
    """Upload screen image to storage.

    Path structure: {user_id}/{flujo_id}/{pantalla_id}/{filename}
    """
    client = get_supabase_client()
    path = f"{user_id}/{flujo_id}/{pantalla_id}/{filename}"

    client.storage.from_(BUCKET_FLUJOS).upload(
        path=path,
        file=image_data,
        file_options={"content-type": content_type}
    )

    return path


def get_pantalla_signed_url(storage_path: str, expires_in: int = SIGNED_URL_EXPIRY) -> Optional[str]:
    """Generate signed URL for private pantalla file access."""
    client = get_supabase_client()
    try:
        response = client.storage.from_(BUCKET_FLUJOS).create_signed_url(
            storage_path, expires_in
        )
        return response.get("signedURL") or response.get("signedUrl")
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to create signed URL for pantalla: {e}")
        return None


def get_signed_urls_for_pantalla(user_id: str, flujo_id: str, pantalla_id: str) -> dict:
    """Get signed URLs for all images of a screen."""
    paths = {
        "screenshot": f"{user_id}/{flujo_id}/{pantalla_id}/screenshot.png",
        "heatmap": f"{user_id}/{flujo_id}/{pantalla_id}/heatmap.png",
        "overlay": f"{user_id}/{flujo_id}/{pantalla_id}/overlay.png",
    }
    return {key: get_pantalla_signed_url(path) for key, path in paths.items()}


def delete_pantalla_images(user_id: str, flujo_id: str, pantalla_id: str) -> bool:
    """Delete all images for a screen from storage."""
    client = get_supabase_client()
    folder_path = f"{user_id}/{flujo_id}/{pantalla_id}"
    try:
        files = client.storage.from_(BUCKET_FLUJOS).list(folder_path)
        if files:
            paths = [f"{folder_path}/{f['name']}" for f in files]
            client.storage.from_(BUCKET_FLUJOS).remove(paths)
        return True
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to delete pantalla images: {e}")
        return False


# ============================================
# Transiciones
# ============================================
async def create_transicion(flujo_id: str, data: dict) -> dict:
    """Create a transition between screens."""
    client = get_supabase_client()
    data["flujo_id"] = flujo_id
    result = client.table(TABLE_TRANSICIONES).insert(data).execute()
    return result.data[0] if result.data else None


async def get_transiciones_flujo(flujo_id: str) -> list[dict]:
    """Get all transitions for a flow."""
    client = get_supabase_client()
    result = client.table(TABLE_TRANSICIONES).select("*").eq("flujo_id", flujo_id).execute()
    return result.data or []
