"""Element Detection Service using Gemini Vision."""
import json
import base64
import uuid
from typing import Optional
import google.generativeai as genai
from app.core.config import get_settings

settings = get_settings()


def get_gemini_client():
    """Initialize Gemini client."""
    if not settings.google_api_key:
        raise ValueError("GOOGLE_API_KEY not configured")
    genai.configure(api_key=settings.google_api_key)
    return genai


class ElementDetectionService:
    """Service for detecting interactive elements in UI screenshots using Gemini Vision."""

    def __init__(self):
        self.client = get_gemini_client()
        self.model = self.client.GenerativeModel(settings.gemini_vision_model)

    async def detect_elements(
        self,
        image_data: bytes,
        content_type: str = "image/png"
    ) -> list[dict]:
        """
        Detect interactive/clickeable elements in a UI screenshot.

        Args:
            image_data: Raw image bytes.
            content_type: MIME type of the image.

        Returns:
            List of detected elements with bounding boxes and metadata.
        """
        prompt = self._get_detection_prompt()

        image_part = {
            "mime_type": content_type,
            "data": base64.b64encode(image_data).decode("utf-8")
        }

        try:
            response = self.model.generate_content([prompt, image_part])
            elements = self._parse_response(response.text)
            return self._validate_and_enrich_elements(elements)
        except Exception as e:
            return [{
                "id": str(uuid.uuid4()),
                "tipo": "error",
                "texto": str(e),
                "bbox": {"x": 0, "y": 0, "width": 0, "height": 0},
                "confianza": 0,
                "error": True
            }]

    def _get_detection_prompt(self) -> str:
        """Get the prompt for element detection."""
        return """Analiza esta captura de pantalla de una interfaz de usuario.

TAREA: Detectar TODOS los elementos interactivos/clickeables visibles.

TIPOS DE ELEMENTOS A DETECTAR:
- button: Botones (primarios, secundarios, iconicos)
- link: Enlaces de texto o imagenes
- input: Campos de entrada de texto, checkbox, radio, dropdown
- tab: Pestanas de navegacion
- menu: Items de menu o navegacion
- icon: Iconos clickeables (hamburger, cerrar, configuracion, etc.)
- card: Tarjetas clickeables
- image: Imagenes que parecen links (banners, productos)

PARA CADA ELEMENTO DETECTADO, proporciona:
1. tipo: Tipo del elemento (button, link, input, tab, menu, icon, card, image)
2. texto: Texto visible del elemento (si tiene)
3. descripcion: Descripcion breve de la funcion del elemento
4. bbox: Bounding box en porcentajes relativos (0-100):
   - x: Posicion horizontal del borde izquierdo (0=izquierda, 100=derecha)
   - y: Posicion vertical del borde superior (0=arriba, 100=abajo)
   - width: Ancho del elemento (0-100)
   - height: Alto del elemento (0-100)
5. confianza: Nivel de confianza 0.0-1.0 de que es clickeable
6. es_cta_principal: true si es un Call-To-Action principal (boton de accion destacado)
7. accesibilidad: Descripcion para lectores de pantalla

EJEMPLO DE RESPUESTA:
```json
[
  {
    "tipo": "button",
    "texto": "Comprar ahora",
    "descripcion": "Boton primario azul para iniciar compra",
    "bbox": {"x": 60, "y": 75, "width": 20, "height": 5},
    "confianza": 0.95,
    "es_cta_principal": true,
    "accesibilidad": "Boton de compra, proceder al checkout"
  },
  {
    "tipo": "link",
    "texto": "Ver mas",
    "descripcion": "Enlace para expandir informacion",
    "bbox": {"x": 70, "y": 60, "width": 10, "height": 3},
    "confianza": 0.85,
    "es_cta_principal": false,
    "accesibilidad": "Enlace para ver mas detalles del producto"
  }
]
```

NOTAS IMPORTANTES:
- Las coordenadas son PORCENTAJES de la imagen (0-100), no pixeles
- El punto (0,0) esta en la esquina superior izquierda
- Incluye TODOS los elementos visibles, no solo los principales
- Si hay navegacion, incluye cada item por separado
- Para formularios, incluye cada campo y boton

Responde SOLO con el JSON array, sin texto adicional."""

    def _parse_response(self, text: str) -> list[dict]:
        """Parse Gemini response to extract elements JSON."""
        try:
            # Remove markdown code blocks if present
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            result = json.loads(text.strip())

            # Ensure it's a list
            if isinstance(result, dict):
                result = [result]

            return result
        except json.JSONDecodeError:
            return []

    def _validate_and_enrich_elements(self, elements: list[dict]) -> list[dict]:
        """Validate elements and add UUIDs."""
        validated = []
        for elem in elements:
            # Skip invalid elements
            if not isinstance(elem, dict):
                continue
            if "bbox" not in elem:
                continue

            # Add UUID if not present
            if "id" not in elem:
                elem["id"] = str(uuid.uuid4())

            # Ensure all required fields
            validated_elem = {
                "id": elem.get("id", str(uuid.uuid4())),
                "tipo": elem.get("tipo", "unknown"),
                "texto": elem.get("texto", ""),
                "descripcion": elem.get("descripcion", ""),
                "bbox": self._validate_bbox(elem.get("bbox", {})),
                "confianza": min(1.0, max(0.0, float(elem.get("confianza", 0.5)))),
                "es_cta_principal": bool(elem.get("es_cta_principal", False)),
                "accesibilidad": elem.get("accesibilidad", elem.get("descripcion", ""))
            }

            validated.append(validated_elem)

        return validated

    def _validate_bbox(self, bbox: dict) -> dict:
        """Validate and normalize bounding box values."""
        return {
            "x": min(100, max(0, float(bbox.get("x", 0)))),
            "y": min(100, max(0, float(bbox.get("y", 0)))),
            "width": min(100, max(0, float(bbox.get("width", 0)))),
            "height": min(100, max(0, float(bbox.get("height", 0))))
        }

    async def find_element_by_text(
        self,
        elements: list[dict],
        target_text: str
    ) -> Optional[dict]:
        """Find an element by its text content (case-insensitive partial match)."""
        target_lower = target_text.lower()
        for elem in elements:
            elem_text = elem.get("texto", "").lower()
            elem_desc = elem.get("descripcion", "").lower()
            if target_lower in elem_text or target_lower in elem_desc:
                return elem
        return None

    async def find_element_by_type_and_text(
        self,
        elements: list[dict],
        element_type: str,
        target_text: Optional[str] = None
    ) -> Optional[dict]:
        """Find an element by type and optionally text."""
        matching = [e for e in elements if e.get("tipo") == element_type]

        if target_text and matching:
            target_lower = target_text.lower()
            for elem in matching:
                elem_text = elem.get("texto", "").lower()
                if target_lower in elem_text:
                    return elem

        # Return first match if no text specified or no text match
        return matching[0] if matching else None

    async def get_cta_elements(self, elements: list[dict]) -> list[dict]:
        """Get all Call-To-Action (CTA) elements."""
        return [e for e in elements if e.get("es_cta_principal", False)]

    async def get_navigation_elements(self, elements: list[dict]) -> list[dict]:
        """Get all navigation-related elements."""
        nav_types = {"link", "tab", "menu"}
        return [e for e in elements if e.get("tipo") in nav_types]
