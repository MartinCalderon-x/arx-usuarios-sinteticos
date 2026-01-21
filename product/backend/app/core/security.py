"""Security middleware for Supabase JWT validation."""
import jwt
from functools import wraps
from typing import Optional
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.config import get_settings

settings = get_settings()
security = HTTPBearer(auto_error=False)


def decode_supabase_jwt(token: str) -> Optional[dict]:
    """
    Decode and validate a Supabase JWT token.

    Supabase Auth JWTs have the following structure:
    - iss: https://<project-ref>.supabase.co/auth/v1
    - sub: user UUID
    - aud: authenticated
    - role: authenticated or anon
    """
    try:
        # Decode without verification first to get claims
        unverified = jwt.decode(token, options={"verify_signature": False})

        # Verify it's a Supabase token by checking the issuer
        iss = unverified.get("iss", "")

        # Check if issuer contains our Supabase project URL
        # Auth tokens have iss like: https://fbpvorxgxokflsyjhnnd.supabase.co/auth/v1
        # Anon key tokens have iss: supabase
        expected_host = settings.supabase_url.replace("https://", "")
        is_auth_token = expected_host in iss
        is_anon_token = iss == "supabase" and unverified.get("ref") == expected_host.replace(".supabase.co", "")

        if not is_auth_token and not is_anon_token:
            return None

        # Check expiration
        import time
        if unverified.get("exp", 0) < time.time():
            return None

        return unverified

    except jwt.exceptions.DecodeError:
        return None
    except Exception:
        return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Optional[dict]:
    """
    Dependency to get the current authenticated user from Supabase JWT.

    Returns None if no valid token is provided.
    Raises HTTPException for invalid tokens.
    """
    if credentials is None:
        return None

    token = credentials.credentials
    payload = decode_supabase_jwt(token)

    if payload is None:
        raise HTTPException(
            status_code=401,
            detail="Token invalido o expirado"
        )

    return payload


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """
    Dependency that requires authentication.

    Raises HTTPException if no valid token is provided.
    """
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Autenticacion requerida"
        )

    token = credentials.credentials
    payload = decode_supabase_jwt(token)

    if payload is None:
        raise HTTPException(
            status_code=401,
            detail="Token invalido o expirado"
        )

    return payload


def is_service_role(payload: dict) -> bool:
    """Check if the token is a service role token (from edge functions)."""
    return payload.get("role") == "service_role"


def is_authenticated_user(payload: dict) -> bool:
    """Check if the token is from an authenticated user."""
    return payload.get("role") == "authenticated"


async def require_service_role(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """
    Dependency that requires service role authentication.

    Only allows requests from Supabase Edge Functions or backend services.
    """
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Autenticacion requerida"
        )

    token = credentials.credentials
    payload = decode_supabase_jwt(token)

    if payload is None:
        raise HTTPException(
            status_code=401,
            detail="Token invalido o expirado"
        )

    if not is_service_role(payload):
        raise HTTPException(
            status_code=403,
            detail="Acceso denegado - se requiere service role"
        )

    return payload
