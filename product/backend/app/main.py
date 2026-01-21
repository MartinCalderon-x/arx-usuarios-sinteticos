"""FastAPI application entry point."""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import traceback

from app.core.config import get_settings
from app.api import arquetipos, analisis, interaccion, reportes

settings = get_settings()

app = FastAPI(
    title="Usuarios Sintéticos API",
    description="API para análisis visual y simulación de usuarios sintéticos",
    version="0.1.0",
    docs_url="/docs" if settings.cors_origins == "*" else None,  # Disable docs in prod
    redoc_url=None,
)

# CORS middleware - Must be added FIRST
# This ensures CORS headers are added even on error responses
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler to ensure all errors return proper JSON
    and CORS headers are applied by the middleware.
    """
    print(f"Unhandled exception: {exc}")
    print(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "Error interno del servidor"}
    )

# Include routers
app.include_router(arquetipos.router, prefix="/api/arquetipos", tags=["arquetipos"])
app.include_router(analisis.router, prefix="/api/analisis", tags=["analisis"])
app.include_router(interaccion.router, prefix="/api/interaccion", tags=["interaccion"])
app.include_router(reportes.router, prefix="/api/reportes", tags=["reportes"])


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "usuarios-sinteticos"}


@app.get("/health")
async def health():
    """Health check for Cloud Run."""
    return {"status": "healthy"}
