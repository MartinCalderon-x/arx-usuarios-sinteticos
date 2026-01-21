"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.api import arquetipos, analisis, interaccion, reportes

settings = get_settings()

app = FastAPI(
    title="Usuarios Sintéticos API",
    description="API para análisis visual y simulación de usuarios sintéticos",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
