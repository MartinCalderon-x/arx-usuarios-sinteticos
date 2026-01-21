"""Gemini Vision AI Service."""
import json
import base64
import httpx
import google.generativeai as genai
from typing import Optional
from app.core.config import get_settings

settings = get_settings()


def get_gemini_client():
    """Initialize Gemini client."""
    if not settings.google_api_key:
        raise ValueError("GOOGLE_API_KEY not configured")
    genai.configure(api_key=settings.google_api_key)
    return genai


class GeminiVisionService:
    """Service for visual analysis using Gemini Vision."""

    def __init__(self):
        self.client = get_gemini_client()
        self.model = self.client.GenerativeModel(settings.gemini_vision_model)

    async def analyze_image(self, image_data: bytes, content_type: str = "image/png") -> dict:
        """Analyze image bytes for attention patterns."""
        prompt = self._get_analysis_prompt()

        # Convert to base64 for Gemini
        image_part = {
            "mime_type": content_type,
            "data": base64.b64encode(image_data).decode("utf-8")
        }

        try:
            response = self.model.generate_content([prompt, image_part])
            return self._parse_response(response.text)
        except Exception as e:
            return self._default_response(str(e))

    async def analyze_url(self, url: str, tipo_analisis: list[str]) -> dict:
        """Analyze image from URL for attention patterns."""
        # Fetch image from URL
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            if response.status_code != 200:
                return self._default_response(f"Error fetching image: {response.status_code}")
            image_data = response.content
            content_type = response.headers.get("content-type", "image/png")

        return await self.analyze_image(image_data, content_type)

    async def compare_designs(self, result_a: dict, result_b: dict) -> dict:
        """Compare two design analysis results."""
        prompt = f"""Compara estos dos análisis de diseño y proporciona recomendaciones:

        Diseño A:
        - Clarity Score: {result_a.get('clarity_score', 'N/A')}
        - Areas de interés: {len(result_a.get('areas_interes', []))}
        - Insights: {result_a.get('insights', [])}

        Diseño B:
        - Clarity Score: {result_b.get('clarity_score', 'N/A')}
        - Areas de interés: {len(result_b.get('areas_interes', []))}
        - Insights: {result_b.get('insights', [])}

        Proporciona en JSON:
        1. ganador: "A" o "B" basado en métricas UX
        2. diferencias_clave: lista de diferencias importantes
        3. recomendaciones: sugerencias para mejorar ambos diseños
        """

        try:
            response = self.model.generate_content(prompt)
            text = response.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            return json.loads(text)
        except Exception as e:
            return {
                "ganador": "empate",
                "diferencias_clave": [],
                "recomendaciones": [f"Error en comparación: {str(e)}"]
            }

    def _get_analysis_prompt(self) -> str:
        """Get the prompt for visual analysis."""
        return """Analiza esta imagen de UI/diseño como un investigador UX experto.

        Proporciona un análisis detallado en formato JSON con:
        1. clarity_score: Puntuación de claridad visual 0-100
        2. areas_interes: Lista de áreas que atraen atención visual, cada una con:
           - nombre: descripción del elemento
           - x: posición horizontal relativa (0-100)
           - y: posición vertical relativa (0-100)
           - width: ancho relativo (0-100)
           - height: alto relativo (0-100)
           - intensidad: nivel de atención 0-100
           - orden_visual: orden en que el usuario lo vería (1, 2, 3...)
        3. focus_map: Descripción de dónde se enfoca la atención en los primeros 3-5 segundos
        4. insights: Lista de observaciones clave sobre el diseño
        5. fricciones: Posibles puntos de fricción o confusión para el usuario
        6. sugerencias: Recomendaciones para mejorar el diseño

        Responde SOLO con JSON válido, sin texto adicional."""

    def _parse_response(self, text: str) -> dict:
        """Parse Gemini response to extract JSON."""
        try:
            # Remove markdown code blocks if present
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return self._default_response("Error parsing response")

    def _default_response(self, error: Optional[str] = None) -> dict:
        """Return default response structure."""
        return {
            "clarity_score": 0,
            "areas_interes": [],
            "focus_map": "",
            "insights": [],
            "fricciones": [],
            "sugerencias": [],
            "error": error
        }


class GeminiChatService:
    """Service for synthetic user chat using Gemini."""

    def __init__(self):
        self.client = get_gemini_client()
        self.model = self.client.GenerativeModel(settings.gemini_model)

    async def chat_as_synthetic_user(
        self,
        arquetipo: dict,
        mensaje: str,
        historial: list[dict] = None,
        imagen_url: Optional[str] = None
    ) -> dict:
        """Generate a response as a synthetic user."""

        system_prompt = f"""Eres un usuario sintético con las siguientes características:
        - Nombre: {arquetipo.get('nombre', 'Usuario')}
        - Edad: {arquetipo.get('edad', 'No especificada')}
        - Género: {arquetipo.get('genero', 'No especificado')}
        - Ocupación: {arquetipo.get('ocupacion', 'No especificada')}
        - Contexto: {arquetipo.get('contexto', 'Usuario general')}
        - Comportamiento: {arquetipo.get('comportamiento', 'Neutral')}
        - Frustraciones: {arquetipo.get('frustraciones', [])}
        - Objetivos: {arquetipo.get('objetivos', [])}

        INSTRUCCIONES:
        - Responde como este usuario respondería de forma auténtica
        - Mantén coherencia con su perfil demográfico y psicográfico
        - Si se te muestra una imagen o prototipo, evalúalo desde su perspectiva
        - Identifica posibles fricciones, confusiones o aspectos positivos
        - Expresa emociones y reacciones realistas

        Responde en JSON con:
        1. respuesta: Tu respuesta como este usuario
        2. fricciones: Lista de puntos de fricción identificados
        3. emociones: Dict con sentimiento y emociones (ej: {{"sentiment": "positivo", "confianza": 0.7, "frustracion": 0.2}})
        """

        # Build conversation context
        messages = [system_prompt]
        if historial:
            for msg in historial[-10:]:  # Last 10 messages for context
                role = "Usuario" if msg.get("rol") == "usuario" else "Tú"
                messages.append(f"{role}: {msg.get('contenido', '')}")

        messages.append(f"Usuario: {mensaje}")

        try:
            # If there's an image, include it
            if imagen_url:
                async with httpx.AsyncClient() as client:
                    img_response = await client.get(imagen_url, timeout=30.0)
                    if img_response.status_code == 200:
                        image_part = {
                            "mime_type": img_response.headers.get("content-type", "image/png"),
                            "data": base64.b64encode(img_response.content).decode("utf-8")
                        }
                        response = self.model.generate_content(["\n".join(messages), image_part])
                    else:
                        response = self.model.generate_content("\n".join(messages))
            else:
                response = self.model.generate_content("\n".join(messages))

            return self._parse_chat_response(response.text)
        except Exception as e:
            return {
                "respuesta": f"Error al generar respuesta: {str(e)}",
                "fricciones": [],
                "emociones": {"sentiment": "neutral", "error": True}
            }

    async def evaluate_prototype(
        self,
        arquetipo: dict,
        imagen_url: str,
        contexto: Optional[str] = None,
        preguntas: Optional[list[str]] = None
    ) -> dict:
        """Evaluate a prototype from the perspective of a synthetic user."""

        prompt = f"""Como usuario sintético con estas características:
        - Nombre: {arquetipo.get('nombre', 'Usuario')}
        - Edad: {arquetipo.get('edad', 'No especificada')}
        - Ocupación: {arquetipo.get('ocupacion', 'No especificada')}
        - Contexto: {arquetipo.get('contexto', 'Usuario general')}
        - Frustraciones: {arquetipo.get('frustraciones', [])}
        - Objetivos: {arquetipo.get('objetivos', [])}

        Contexto de la evaluación: {contexto or 'Evaluación general del prototipo'}

        Evalúa esta interfaz/prototipo y responde en JSON:
        1. feedback: Tu opinión general como este usuario
        2. puntuacion: Puntuación del 1-10 desde tu perspectiva
        3. fricciones: Lista de puntos de fricción que experimentarías
        4. sugerencias: Lista de mejoras que te gustaría ver
        5. emociones: Dict con tu reacción emocional

        {"Preguntas específicas a responder: " + str(preguntas) if preguntas else ""}
        """

        try:
            async with httpx.AsyncClient() as client:
                img_response = await client.get(imagen_url, timeout=30.0)
                if img_response.status_code != 200:
                    raise Exception(f"Error fetching image: {img_response.status_code}")

                image_part = {
                    "mime_type": img_response.headers.get("content-type", "image/png"),
                    "data": base64.b64encode(img_response.content).decode("utf-8")
                }

            response = self.model.generate_content([prompt, image_part])
            return self._parse_eval_response(response.text)
        except Exception as e:
            return {
                "feedback": f"Error al evaluar: {str(e)}",
                "puntuacion": 0,
                "fricciones": [],
                "sugerencias": [],
                "emociones": {"error": True}
            }

    def _parse_chat_response(self, text: str) -> dict:
        """Parse chat response."""
        try:
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return {
                "respuesta": text,
                "fricciones": [],
                "emociones": {"sentiment": "neutral"}
            }

    def _parse_eval_response(self, text: str) -> dict:
        """Parse evaluation response."""
        try:
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return {
                "feedback": text,
                "puntuacion": 5,
                "fricciones": [],
                "sugerencias": [],
                "emociones": {"sentiment": "neutral"}
            }
