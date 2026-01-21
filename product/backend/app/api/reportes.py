"""Reports API routes."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import Optional
from pydantic import BaseModel

router = APIRouter()


class ReporteRequest(BaseModel):
    """Schema for report generation request."""
    titulo: str
    formato: str = "pdf"  # pdf or pptx
    arquetipos_ids: Optional[list[str]] = None
    analisis_ids: Optional[list[str]] = None
    sesiones_ids: Optional[list[str]] = None
    incluir_resumen: bool = True
    incluir_recomendaciones: bool = True


class ReporteResponse(BaseModel):
    """Schema for report response."""
    id: str
    titulo: str
    formato: str
    url: str
    created_at: str


@router.post("/generar", response_model=ReporteResponse)
async def generar_reporte(request: ReporteRequest):
    """Generate a report from analysis and interaction data."""
    # TODO: Implement with ReportLab/python-pptx
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/{reporte_id}/descargar")
async def descargar_reporte(reporte_id: str):
    """Download a generated report."""
    # TODO: Implement with Supabase Storage
    raise HTTPException(status_code=404, detail="Report not found")


@router.get("/")
async def list_reportes():
    """List generated reports."""
    # TODO: Implement with Supabase
    return {"reportes": [], "total": 0}


@router.delete("/{reporte_id}")
async def delete_reporte(reporte_id: str):
    """Delete a report."""
    # TODO: Implement with Supabase
    raise HTTPException(status_code=501, detail="Not implemented")
