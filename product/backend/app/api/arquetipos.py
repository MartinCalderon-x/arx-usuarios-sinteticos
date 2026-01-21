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
    nivel_digital: Optional[str] = None  # bajo, medio, alto
    industria: Optional[str] = None  # tech, salud, retail, finanzas, manufactura, educacion
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
    nivel_digital: Optional[str] = None
    industria: Optional[str] = None
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
    templates = [
        # Consumidor Digital
        {
            "id": "early-adopter",
            "nombre": "Early Adopter",
            "categoria": "consumidor",
            "descripcion": "Usuario entusiasta de nuevas tecnologías, tolera bugs a cambio de probar primero",
            "edad": 28,
            "ocupacion": "Profesional tech",
            "nivel_digital": "alto",
            "industria": "tech",
            "comportamiento": "Explora todas las features nuevas, da feedback detallado, comparte en redes",
            "frustraciones": ["Falta de innovación", "Features que tardan en llegar", "UX anticuada"],
            "objetivos": ["Ser el primero en probar", "Optimizar su productividad", "Estar a la vanguardia"],
        },
        {
            "id": "usuario-promedio",
            "nombre": "Usuario Promedio",
            "categoria": "consumidor",
            "descripcion": "Usuario típico que busca que las cosas funcionen sin complicaciones",
            "edad": 35,
            "ocupacion": "Empleado de oficina",
            "nivel_digital": "medio",
            "industria": "servicios",
            "comportamiento": "Usa las funciones básicas, evita configuraciones complejas, prefiere lo familiar",
            "frustraciones": ["Cambios frecuentes en la UI", "Demasiadas opciones", "Errores inesperados"],
            "objetivos": ["Completar tareas rápidamente", "No perder tiempo aprendiendo", "Que todo funcione"],
        },
        {
            "id": "esceptico-digital",
            "nombre": "Escéptico Digital",
            "categoria": "consumidor",
            "descripcion": "Desconfía de la tecnología, prefiere métodos tradicionales",
            "edad": 50,
            "ocupacion": "Comerciante",
            "nivel_digital": "bajo",
            "industria": "retail",
            "comportamiento": "Evita apps nuevas, prefiere llamar por teléfono, necesita mucha ayuda",
            "frustraciones": ["Procesos solo digitales", "Falta de atención humana", "Complejidad innecesaria"],
            "objetivos": ["Resolver sin tecnología si es posible", "Hablar con una persona real", "Seguridad"],
        },
        {
            "id": "senior-tradicional",
            "nombre": "Senior Tradicional",
            "categoria": "consumidor",
            "descripcion": "Adulto mayor con experiencia tecnológica limitada",
            "edad": 68,
            "ocupacion": "Jubilado",
            "nivel_digital": "bajo",
            "industria": "salud",
            "comportamiento": "Necesita instrucciones claras, texto grande, procesos simples",
            "frustraciones": ["Letra pequeña", "Muchos pasos", "Términos técnicos", "Tiempo limitado"],
            "objetivos": ["Mantenerse conectado con familia", "Gestionar su salud", "Independencia"],
        },
        # Industria/Manufactura
        {
            "id": "operario-linea",
            "nombre": "Operario de Línea",
            "categoria": "manufactura",
            "descripcion": "Trabajador de planta de producción con tareas repetitivas",
            "edad": 32,
            "ocupacion": "Operador de maquinaria",
            "nivel_digital": "bajo",
            "industria": "manufactura",
            "comportamiento": "Sigue procedimientos estrictos, reporta incidencias, trabaja en turnos",
            "frustraciones": ["Sistemas lentos", "Formularios complejos", "Falta de capacitación"],
            "objetivos": ["Cumplir cuotas", "Evitar accidentes", "Reportar rápido"],
        },
        {
            "id": "supervisor-planta",
            "nombre": "Supervisor de Planta",
            "categoria": "manufactura",
            "descripcion": "Encargado de supervisar operaciones y personal en planta",
            "edad": 42,
            "ocupacion": "Supervisor de producción",
            "nivel_digital": "medio",
            "industria": "manufactura",
            "comportamiento": "Monitorea métricas, gestiona equipos, resuelve problemas urgentes",
            "frustraciones": ["Falta de visibilidad en tiempo real", "Reportes manuales", "Coordinación"],
            "objetivos": ["Cumplir metas de producción", "Reducir tiempos muertos", "Seguridad del equipo"],
        },
        {
            "id": "ingeniero-procesos",
            "nombre": "Ingeniero de Procesos",
            "categoria": "manufactura",
            "descripcion": "Profesional técnico que optimiza procesos productivos",
            "edad": 35,
            "ocupacion": "Ingeniero industrial",
            "nivel_digital": "alto",
            "industria": "manufactura",
            "comportamiento": "Analiza datos, implementa mejoras, documenta procedimientos",
            "frustraciones": ["Datos inconsistentes", "Resistencia al cambio", "Herramientas legacy"],
            "objetivos": ["Optimizar eficiencia", "Reducir costos", "Implementar innovación"],
        },
        # Retail
        {
            "id": "comprador-impulsivo",
            "nombre": "Comprador Impulsivo",
            "categoria": "retail",
            "descripcion": "Compra por emoción, sensible a ofertas y urgencia",
            "edad": 28,
            "ocupacion": "Profesional joven",
            "nivel_digital": "alto",
            "industria": "retail",
            "comportamiento": "Compra rápido, influenciado por redes sociales, busca gratificación inmediata",
            "frustraciones": ["Procesos de checkout largos", "Falta de stock", "Envío lento"],
            "objetivos": ["Encontrar ofertas", "Comprar rápido", "Recibir pronto"],
        },
        {
            "id": "comparador-precios",
            "nombre": "Comparador de Precios",
            "categoria": "retail",
            "descripcion": "Investiga exhaustivamente antes de comprar",
            "edad": 40,
            "ocupacion": "Contador",
            "nivel_digital": "medio",
            "industria": "retail",
            "comportamiento": "Compara en múltiples sitios, lee reviews, espera descuentos",
            "frustraciones": ["Precios ocultos", "Información incompleta", "Políticas confusas"],
            "objetivos": ["Mejor precio posible", "Evitar errores de compra", "Valor por dinero"],
        },
        {
            "id": "cliente-fiel",
            "nombre": "Cliente Fiel de Marca",
            "categoria": "retail",
            "descripcion": "Leal a marcas específicas, valora la relación",
            "edad": 45,
            "ocupacion": "Gerente",
            "nivel_digital": "medio",
            "industria": "retail",
            "comportamiento": "Repite compras, participa en programas de lealtad, recomienda",
            "frustraciones": ["Cambios en productos favoritos", "Mal servicio", "No ser reconocido"],
            "objetivos": ["Beneficios exclusivos", "Trato preferencial", "Calidad consistente"],
        },
        # Servicios Financieros
        {
            "id": "inversionista-novato",
            "nombre": "Inversionista Novato",
            "categoria": "finanzas",
            "descripcion": "Nuevo en inversiones, busca aprender y crecer",
            "edad": 30,
            "ocupacion": "Profesional",
            "nivel_digital": "alto",
            "industria": "finanzas",
            "comportamiento": "Consume contenido educativo, empieza con montos pequeños, pregunta mucho",
            "frustraciones": ["Jerga financiera", "Riesgo no explicado", "Comisiones ocultas"],
            "objetivos": ["Aprender a invertir", "Hacer crecer ahorros", "Entender el mercado"],
        },
        {
            "id": "usuario-fintech",
            "nombre": "Usuario Fintech",
            "categoria": "finanzas",
            "descripcion": "Prefiere soluciones digitales sobre banca tradicional",
            "edad": 27,
            "ocupacion": "Freelancer",
            "nivel_digital": "alto",
            "industria": "finanzas",
            "comportamiento": "Todo desde el celular, múltiples apps financieras, busca agilidad",
            "frustraciones": ["Ir a sucursal", "Procesos lentos", "Requisitos burocráticos"],
            "objetivos": ["Gestión 100% digital", "Transferencias instantáneas", "Control total"],
        },
        {
            "id": "cliente-banca-tradicional",
            "nombre": "Cliente Banca Tradicional",
            "categoria": "finanzas",
            "descripcion": "Prefiere la banca presencial y relación con ejecutivo",
            "edad": 55,
            "ocupacion": "Empresario",
            "nivel_digital": "bajo",
            "industria": "finanzas",
            "comportamiento": "Visita sucursal, prefiere hablar con personas, desconfía de lo digital",
            "frustraciones": ["Forzar uso de app", "Reducción de sucursales", "Atención impersonal"],
            "objetivos": ["Asesoría personal", "Seguridad", "Relación de confianza"],
        },
        # Salud
        {
            "id": "paciente-cronico",
            "nombre": "Paciente Crónico",
            "categoria": "salud",
            "descripcion": "Gestiona una condición de salud permanente",
            "edad": 52,
            "ocupacion": "Varios",
            "nivel_digital": "medio",
            "industria": "salud",
            "comportamiento": "Monitorea síntomas, agenda citas frecuentes, busca información",
            "frustraciones": ["Tiempos de espera", "Falta de coordinación médica", "Repetir historia"],
            "objetivos": ["Control de su condición", "Acceso fácil a médicos", "Historial unificado"],
        },
        {
            "id": "cuidador-familiar",
            "nombre": "Cuidador Familiar",
            "categoria": "salud",
            "descripcion": "Cuida de un familiar con necesidades de salud",
            "edad": 45,
            "ocupacion": "Profesional/cuidador",
            "nivel_digital": "medio",
            "industria": "salud",
            "comportamiento": "Gestiona citas de otros, coordina tratamientos, busca recursos",
            "frustraciones": ["Falta de acceso a información", "Burocracia", "Estrés"],
            "objetivos": ["Mejor cuidado para su familiar", "Simplificar gestión", "Apoyo"],
        },
    ]
    return {"templates": templates}
