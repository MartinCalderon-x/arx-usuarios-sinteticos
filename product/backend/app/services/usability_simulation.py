"""Usability Simulation Service using synthetic user archetypes."""
import json
import base64
import random
from typing import Optional
import httpx
import google.generativeai as genai
from app.core.config import get_settings

settings = get_settings()


def get_gemini_client():
    """Initialize Gemini client."""
    if not settings.google_api_key:
        raise ValueError("GOOGLE_API_KEY not configured")
    genai.configure(api_key=settings.google_api_key)
    return genai


# Behavior configuration by digital level
NIVEL_DIGITAL_CONFIG = {
    "bajo": {
        "misclick_probability": 0.30,  # 30% chance of misclick
        "confusion_threshold": 0.60,   # Gets confused with >60% complexity
        "avg_time_per_step_ms": 8000,  # 8 seconds per step
        "expected_success_rate": (0.40, 0.60),  # 40-60%
        "expected_misclicks": (2, 4),
        "preferences": [
            "Prefiere elementos grandes y visibles",
            "Evita opciones complejas o con mucho texto",
            "Se confunde con iconos sin etiquetas",
            "Busca botones con colores llamativos"
        ]
    },
    "medio": {
        "misclick_probability": 0.15,  # 15% chance of misclick
        "confusion_threshold": 0.80,   # Gets confused with >80% complexity
        "avg_time_per_step_ms": 4000,  # 4 seconds per step
        "expected_success_rate": (0.70, 0.85),  # 70-85%
        "expected_misclicks": (1, 2),
        "preferences": [
            "Navegacion normal, prefiere opciones claras",
            "Puede usar iconos conocidos",
            "Busca textos descriptivos",
            "Prefiere flujos convencionales"
        ]
    },
    "alto": {
        "misclick_probability": 0.05,  # 5% chance of misclick
        "confusion_threshold": 0.95,   # Almost never confused
        "avg_time_per_step_ms": 2000,  # 2 seconds per step
        "expected_success_rate": (0.90, 0.98),  # 90-98%
        "expected_misclicks": (0, 1),
        "preferences": [
            "Navegacion eficiente y directa",
            "Identifica CTAs rapidamente",
            "Usa atajos cuando estan disponibles",
            "Tolera interfaces complejas"
        ]
    }
}


class UsabilitySimulationService:
    """Service for simulating usability testing with synthetic user archetypes."""

    def __init__(self):
        self.client = get_gemini_client()
        self.model = self.client.GenerativeModel(settings.gemini_model)

    async def simulate_mission(
        self,
        mision: dict,
        arquetipo: dict,
        pantallas: list[dict],
        pantalla_imagenes: dict[str, bytes]
    ) -> dict:
        """
        Simulate a complete usability mission with an archetype.

        Args:
            mision: Mission definition with instructions and target element.
            arquetipo: Archetype with user characteristics.
            pantallas: List of screens in the flow with detected elements.
            pantalla_imagenes: Dict mapping pantalla_id to image bytes.

        Returns:
            Simulation result with path, metrics, and feedback.
        """
        nivel_digital = arquetipo.get("nivel_digital", "medio")
        config = NIVEL_DIGITAL_CONFIG.get(nivel_digital, NIVEL_DIGITAL_CONFIG["medio"])

        # Initialize simulation state
        path_tomado = []
        misclicks = 0
        fricciones = []
        tiempo_total_ms = 0
        pantalla_actual_id = mision.get("pantalla_inicio_id")
        max_pasos = mision.get("max_pasos", 10)
        elemento_objetivo = mision.get("elemento_objetivo", {})

        # Find starting screen
        pantalla_actual = next(
            (p for p in pantallas if p.get("id") == pantalla_actual_id),
            pantallas[0] if pantallas else None
        )

        if not pantalla_actual:
            return self._error_result("No se encontro pantalla de inicio")

        # Simulation loop
        for paso in range(max_pasos):
            elementos = pantalla_actual.get("elementos_clickeables", [])

            if not elementos:
                fricciones.append({
                    "tipo": "sin_elementos",
                    "pantalla_id": pantalla_actual.get("id"),
                    "descripcion": "Pantalla sin elementos detectados"
                })
                break

            # Get archetype decision
            imagen_bytes = pantalla_imagenes.get(pantalla_actual.get("id"))
            decision = await self._get_archetype_decision(
                arquetipo=arquetipo,
                mision=mision,
                elementos=elementos,
                imagen_bytes=imagen_bytes,
                config=config,
                historial_path=path_tomado
            )

            # Apply misclick probability based on nivel_digital
            is_misclick = self._should_misclick(config, decision)
            if is_misclick:
                misclicks += 1
                decision = self._generate_misclick(elementos, decision, config)

            # Record step
            tiempo_paso = self._calculate_step_time(config, decision)
            tiempo_total_ms += tiempo_paso

            step_record = {
                "paso": paso + 1,
                "pantalla_id": pantalla_actual.get("id"),
                "pantalla_titulo": pantalla_actual.get("titulo"),
                "elemento_clickeado": decision.get("elemento_clickeado"),
                "razonamiento": decision.get("razonamiento"),
                "confusion": decision.get("confusion", False),
                "emocion": decision.get("emocion", "neutral"),
                "es_misclick": is_misclick,
                "tiempo_ms": tiempo_paso
            }
            path_tomado.append(step_record)

            # Check for friction
            if decision.get("confusion"):
                fricciones.append({
                    "tipo": "confusion",
                    "pantalla_id": pantalla_actual.get("id"),
                    "elemento_id": decision.get("elemento_clickeado", {}).get("id"),
                    "descripcion": decision.get("razonamiento", "Usuario confundido")
                })

            # Check if objective reached
            elemento_seleccionado = decision.get("elemento_clickeado", {})
            if self._is_objective_reached(elemento_seleccionado, elemento_objetivo):
                return {
                    "completada": True,
                    "exito": True,
                    "path_tomado": path_tomado,
                    "pasos_totales": paso + 1,
                    "misclicks": misclicks,
                    "tiempo_estimado_ms": tiempo_total_ms,
                    "fricciones": fricciones,
                    "feedback_arquetipo": decision.get("feedback_final", "Mision completada exitosamente"),
                    "emociones": {
                        "satisfaccion": 0.8 if misclicks < 2 else 0.5,
                        "frustracion": min(0.9, misclicks * 0.2),
                        "confianza": 0.9 if paso < 3 else 0.6
                    }
                }

            # Determine next screen (if element leads to another screen)
            next_pantalla_id = self._get_next_screen(
                pantalla_actual.get("id"),
                elemento_seleccionado,
                pantallas,
                mision
            )

            if next_pantalla_id and next_pantalla_id != pantalla_actual.get("id"):
                pantalla_actual = next(
                    (p for p in pantallas if p.get("id") == next_pantalla_id),
                    pantalla_actual
                )

        # Mission failed (exceeded max steps)
        return {
            "completada": True,
            "exito": False,
            "path_tomado": path_tomado,
            "pasos_totales": len(path_tomado),
            "misclicks": misclicks,
            "tiempo_estimado_ms": tiempo_total_ms,
            "fricciones": fricciones,
            "feedback_arquetipo": self._generate_failure_feedback(arquetipo, fricciones),
            "emociones": {
                "satisfaccion": 0.2,
                "frustracion": 0.8,
                "confianza": 0.3
            }
        }

    async def _get_archetype_decision(
        self,
        arquetipo: dict,
        mision: dict,
        elementos: list[dict],
        imagen_bytes: Optional[bytes],
        config: dict,
        historial_path: list[dict]
    ) -> dict:
        """Get the archetype's decision on which element to click."""

        nivel_digital = arquetipo.get("nivel_digital", "medio")
        preferencias = config.get("preferences", [])

        # Build context for the prompt
        elementos_simplificados = [
            {
                "id": e.get("id"),
                "tipo": e.get("tipo"),
                "texto": e.get("texto", ""),
                "descripcion": e.get("descripcion", ""),
                "es_cta_principal": e.get("es_cta_principal", False)
            }
            for e in elementos
        ]

        historial_texto = ""
        if historial_path:
            pasos = [f"- Paso {p['paso']}: Click en '{p['elemento_clickeado'].get('texto', 'elemento')}'"
                     for p in historial_path[-3:]]  # Last 3 steps
            historial_texto = f"\nHistorial reciente:\n" + "\n".join(pasos)

        prompt = f"""Eres {arquetipo.get('nombre', 'un usuario sintetico')}, {arquetipo.get('edad', 35)} anios.
Nivel de habilidad digital: {nivel_digital.upper()}
Ocupacion: {arquetipo.get('ocupacion', 'No especificada')}
Tus frustraciones: {arquetipo.get('frustraciones', [])}
Tus objetivos: {arquetipo.get('objetivos', [])}

COMO USUARIO DE NIVEL {nivel_digital.upper()}, tienes estas caracteristicas:
{chr(10).join('- ' + p for p in preferencias)}

MISION ACTUAL: "{mision.get('instrucciones', 'Completar la tarea')}"

ELEMENTO OBJETIVO: {mision.get('elemento_objetivo', {}).get('texto', 'No especificado')}
{historial_texto}

ELEMENTOS DISPONIBLES EN PANTALLA:
{json.dumps(elementos_simplificados, indent=2, ensure_ascii=False)}

DECIDE como este usuario especifico:
1. Que elemento clickearias?
2. Por que elegiste ese elemento (desde tu perspectiva como este usuario)?
3. Sientes confusion o dificultad?
4. Cual es tu estado emocional?

Responde en JSON:
{{
    "elemento_clickeado": {{"id": "uuid-del-elemento", "texto": "texto del elemento"}},
    "razonamiento": "Explicacion desde la perspectiva del usuario",
    "confusion": true/false,
    "emocion": "neutral/frustrado/satisfecho/confundido/ansioso",
    "confianza_decision": 0.0-1.0
}}
"""

        try:
            # Include image if available
            if imagen_bytes:
                image_part = {
                    "mime_type": "image/png",
                    "data": base64.b64encode(imagen_bytes).decode("utf-8")
                }
                response = self.model.generate_content([prompt, image_part])
            else:
                response = self.model.generate_content(prompt)

            return self._parse_decision_response(response.text, elementos)

        except Exception as e:
            # Fallback to random element selection
            random_elem = random.choice(elementos) if elementos else {}
            return {
                "elemento_clickeado": {
                    "id": random_elem.get("id"),
                    "texto": random_elem.get("texto", "")
                },
                "razonamiento": f"Seleccion automatica (error: {str(e)})",
                "confusion": True,
                "emocion": "confundido",
                "confianza_decision": 0.3
            }

    def _parse_decision_response(self, text: str, elementos: list[dict]) -> dict:
        """Parse the decision response from Gemini."""
        try:
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            result = json.loads(text.strip())

            # Validate element exists
            elem_id = result.get("elemento_clickeado", {}).get("id")
            found_elem = next((e for e in elementos if e.get("id") == elem_id), None)

            if not found_elem and elementos:
                # If element not found, pick first CTA or first element
                ctas = [e for e in elementos if e.get("es_cta_principal")]
                found_elem = ctas[0] if ctas else elementos[0]
                result["elemento_clickeado"] = {
                    "id": found_elem.get("id"),
                    "texto": found_elem.get("texto", "")
                }

            return result

        except json.JSONDecodeError:
            # Return first CTA or first element
            ctas = [e for e in elementos if e.get("es_cta_principal")]
            elem = ctas[0] if ctas else (elementos[0] if elementos else {})
            return {
                "elemento_clickeado": {
                    "id": elem.get("id"),
                    "texto": elem.get("texto", "")
                },
                "razonamiento": "Seleccion del elemento mas visible",
                "confusion": False,
                "emocion": "neutral",
                "confianza_decision": 0.5
            }

    def _should_misclick(self, config: dict, decision: dict) -> bool:
        """Determine if a misclick should occur based on probability."""
        base_prob = config.get("misclick_probability", 0.15)

        # Increase probability if user is confused
        if decision.get("confusion"):
            base_prob += 0.15

        # Increase if low confidence
        confidence = decision.get("confianza_decision", 0.5)
        if confidence < 0.4:
            base_prob += 0.10

        return random.random() < base_prob

    def _generate_misclick(
        self,
        elementos: list[dict],
        original_decision: dict,
        config: dict
    ) -> dict:
        """Generate a misclick by selecting a wrong element."""
        original_id = original_decision.get("elemento_clickeado", {}).get("id")

        # Filter out the correct element
        other_elements = [e for e in elementos if e.get("id") != original_id]

        if not other_elements:
            return original_decision

        # Pick a random wrong element
        wrong_elem = random.choice(other_elements)

        return {
            "elemento_clickeado": {
                "id": wrong_elem.get("id"),
                "texto": wrong_elem.get("texto", "")
            },
            "razonamiento": "Click accidental en elemento cercano",
            "confusion": True,
            "emocion": "frustrado",
            "confianza_decision": original_decision.get("confianza_decision", 0.5)
        }

    def _calculate_step_time(self, config: dict, decision: dict) -> int:
        """Calculate time for this step based on user level and confusion."""
        base_time = config.get("avg_time_per_step_ms", 4000)

        # Add time if confused
        if decision.get("confusion"):
            base_time *= 1.5

        # Add variance (+-30%)
        variance = random.uniform(0.7, 1.3)

        return int(base_time * variance)

    def _is_objective_reached(
        self,
        elemento_seleccionado: dict,
        elemento_objetivo: dict
    ) -> bool:
        """Check if the selected element matches the objective."""
        if not elemento_objetivo:
            return False

        # Match by text (case-insensitive partial match)
        objetivo_texto = elemento_objetivo.get("texto", "").lower()
        seleccionado_texto = elemento_seleccionado.get("texto", "").lower()

        if objetivo_texto and seleccionado_texto:
            if objetivo_texto in seleccionado_texto or seleccionado_texto in objetivo_texto:
                return True

        # Match by type if specified
        objetivo_tipo = elemento_objetivo.get("tipo")
        if objetivo_tipo and elemento_seleccionado.get("tipo") == objetivo_tipo:
            # If type matches and text partially matches
            if objetivo_texto in seleccionado_texto:
                return True

        return False

    def _get_next_screen(
        self,
        current_screen_id: str,
        elemento_clickeado: dict,
        pantallas: list[dict],
        mision: dict
    ) -> Optional[str]:
        """Determine the next screen based on element clicked."""
        # For now, simple progression to next screen in order
        current_idx = next(
            (i for i, p in enumerate(pantallas) if p.get("id") == current_screen_id),
            0
        )

        # Check if element should lead to specific screen (via href or similar)
        # This would be enhanced with actual transition mapping

        if current_idx < len(pantallas) - 1:
            return pantallas[current_idx + 1].get("id")

        return None

    def _generate_failure_feedback(
        self,
        arquetipo: dict,
        fricciones: list[dict]
    ) -> str:
        """Generate failure feedback from archetype perspective."""
        nivel = arquetipo.get("nivel_digital", "medio")
        nombre = arquetipo.get("nombre", "Usuario")

        if nivel == "bajo":
            return f"Como {nombre}, me senti perdido. La interfaz era confusa y no encontre lo que buscaba."
        elif nivel == "medio":
            return f"Como {nombre}, tuve dificultades. Algunos elementos no eran claros."
        else:
            return f"Como {nombre}, no logre completar la tarea a pesar de mi experiencia."

    def _error_result(self, message: str) -> dict:
        """Return an error result structure."""
        return {
            "completada": False,
            "exito": False,
            "path_tomado": [],
            "pasos_totales": 0,
            "misclicks": 0,
            "tiempo_estimado_ms": 0,
            "fricciones": [{"tipo": "error", "descripcion": message}],
            "feedback_arquetipo": message,
            "emociones": {"error": True}
        }

    async def calculate_aggregated_metrics(
        self,
        simulaciones: list[dict]
    ) -> dict:
        """Calculate aggregated metrics from multiple simulations."""
        if not simulaciones:
            return self._empty_metrics()

        total = len(simulaciones)
        exitosas = sum(1 for s in simulaciones if s.get("exito"))
        total_misclicks = sum(s.get("misclicks", 0) for s in simulaciones)
        total_pasos = sum(s.get("pasos_totales", 0) for s in simulaciones)
        total_tiempo = sum(s.get("tiempo_estimado_ms", 0) for s in simulaciones)

        # Collect all frictions
        todas_fricciones = []
        for s in simulaciones:
            todas_fricciones.extend(s.get("fricciones", []))

        # Count friction types
        friction_counts = {}
        for f in todas_fricciones:
            tipo = f.get("tipo", "unknown")
            friction_counts[tipo] = friction_counts.get(tipo, 0) + 1

        # Calculate by nivel_digital (if arquetipo info available)
        metrics_by_level = {
            "bajo": {"exitos": 0, "total": 0, "misclicks": 0},
            "medio": {"exitos": 0, "total": 0, "misclicks": 0},
            "alto": {"exitos": 0, "total": 0, "misclicks": 0}
        }

        # This would require arquetipo info to be included in simulaciones
        # For now, return overall metrics

        return {
            "total_simulaciones": total,
            "success_rate": (exitosas / total) * 100 if total > 0 else 0,
            "avg_misclicks": total_misclicks / total if total > 0 else 0,
            "avg_pasos": total_pasos / total if total > 0 else 0,
            "avg_tiempo_ms": total_tiempo / total if total > 0 else 0,
            "total_fricciones": len(todas_fricciones),
            "fricciones_por_tipo": friction_counts,
            "metrics_by_nivel": metrics_by_level
        }

    def _empty_metrics(self) -> dict:
        """Return empty metrics structure."""
        return {
            "total_simulaciones": 0,
            "success_rate": 0,
            "avg_misclicks": 0,
            "avg_pasos": 0,
            "avg_tiempo_ms": 0,
            "total_fricciones": 0,
            "fricciones_por_tipo": {},
            "metrics_by_nivel": {}
        }
