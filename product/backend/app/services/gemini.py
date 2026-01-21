"""Gemini Vision AI Service."""
import google.generativeai as genai
from app.core.config import get_settings

settings = get_settings()


def get_gemini_client():
    """Initialize Gemini client."""
    if not settings.google_api_key:
        raise ValueError("GOOGLE_API_KEY not configured")
    genai.configure(api_key=settings.google_api_key)
    return genai


async def analyze_image_attention(image_data: bytes) -> dict:
    """
    Analyze image for attention patterns using Gemini Vision.

    Returns predicted attention areas, focus points, and clarity score.
    """
    client = get_gemini_client()
    model = client.GenerativeModel(settings.gemini_vision_model)

    prompt = """Analyze this UI/design image as a UX researcher would.

    Provide a detailed analysis in JSON format with:
    1. attention_areas: List of areas that draw visual attention (x, y, width, height, intensity 0-100)
    2. focus_sequence: Order in which users would likely look at elements (first 5 seconds)
    3. clarity_score: Overall visual clarity rating 0-100
    4. insights: Key observations about the design
    5. potential_issues: Any UX issues or friction points

    Be specific about coordinates and provide actionable insights."""

    # TODO: Implement actual image analysis
    # response = model.generate_content([prompt, image_data])

    return {
        "attention_areas": [],
        "focus_sequence": [],
        "clarity_score": 0,
        "insights": [],
        "potential_issues": []
    }


async def chat_as_synthetic_user(
    arquetipo: dict,
    mensaje: str,
    historial: list[dict] = None,
    imagen_url: str = None
) -> dict:
    """
    Generate a response as a synthetic user.

    Uses the archetype's characteristics to simulate realistic user behavior.
    """
    client = get_gemini_client()
    model = client.GenerativeModel(settings.gemini_model)

    system_prompt = f"""Eres un usuario sintético con las siguientes características:
    - Nombre: {arquetipo.get('nombre', 'Usuario')}
    - Edad: {arquetipo.get('edad', 'No especificada')}
    - Ocupación: {arquetipo.get('ocupacion', 'No especificada')}
    - Contexto: {arquetipo.get('contexto', 'Usuario general')}
    - Comportamiento: {arquetipo.get('comportamiento', 'Neutral')}
    - Frustraciones: {arquetipo.get('frustraciones', [])}
    - Objetivos: {arquetipo.get('objetivos', [])}

    Responde como este usuario respondería. Sé auténtico a sus características.
    Si se te muestra una imagen o prototipo, evalúalo desde la perspectiva de este usuario.
    Identifica posibles fricciones, confusiones o aspectos positivos.
    """

    # TODO: Implement actual chat with context
    # response = model.generate_content([system_prompt, mensaje])

    return {
        "respuesta": "Implementación pendiente",
        "fricciones": [],
        "emociones": {"sentiment": "neutral"}
    }
