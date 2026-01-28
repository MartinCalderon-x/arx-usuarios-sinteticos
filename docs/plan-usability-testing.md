# Plan: Usability Testing con Usuarios Sinteticos

> **Estado:** Planificado
> **Fecha:** 2026-01-26
> **Relacionado:** Issue #15 (Flujos Multi-Pantalla)

## Vision General

Sistema estilo Maze para simular usability testing usando arquetipos de usuarios sinteticos + IA, sin necesidad de usuarios reales.

```
┌─────────────────────────────────────────────────────────────┐
│                    FLUJO DE TESTING                          │
├─────────────────────────────────────────────────────────────┤
│  1. DEFINIR MISION                                          │
│     "Encuentra el boton de contacto"                        │
│                                                              │
│  2. DETECTAR ELEMENTOS (Gemini Vision)                      │
│     [Boton CTA] [Link Nav] [Form Input] [Logo]              │
│                                                              │
│  3. SIMULAR USUARIOS (Arquetipos + LLM)                     │
│     Early Adopter → Click directo, exito                    │
│     Senior Tradicional → 3 misclicks, confusion             │
│                                                              │
│  4. GENERAR METRICAS                                        │
│     Success Rate: 75%                                        │
│     Avg Misclicks: 2.3                                       │
│     Friction Points: [Nav confusa, CTA poco visible]        │
└─────────────────────────────────────────────────────────────┘
```

---

## Modelo de Datos

### Nueva Tabla: `us_misiones`
```sql
CREATE TABLE us_misiones (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    flujo_id UUID REFERENCES us_flujos(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    nombre VARCHAR(255) NOT NULL,
    instrucciones TEXT NOT NULL,  -- "Encuentra el boton de contacto"
    pantalla_inicio_id UUID REFERENCES us_pantallas(id),
    pantalla_objetivo_id UUID REFERENCES us_pantallas(id),
    elemento_objetivo JSONB,  -- {tipo: "button", texto: "Contacto"}
    max_pasos INTEGER DEFAULT 10,
    estado VARCHAR(50) DEFAULT 'activa',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Nueva Tabla: `us_simulaciones`
```sql
CREATE TABLE us_simulaciones (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mision_id UUID REFERENCES us_misiones(id) ON DELETE CASCADE,
    arquetipo_id UUID REFERENCES us_arquetipos(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id),
    completada BOOLEAN DEFAULT FALSE,
    exito BOOLEAN DEFAULT FALSE,
    path_tomado JSONB DEFAULT '[]',
    pasos_totales INTEGER DEFAULT 0,
    misclicks INTEGER DEFAULT 0,
    tiempo_estimado_ms INTEGER,
    fricciones JSONB DEFAULT '[]',
    feedback_arquetipo TEXT,
    emociones JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Mejorar: `us_pantallas.elementos_clickeables`
Estructura para elementos detectados:
```json
[{
  "id": "uuid",
  "tipo": "button | link | input | tab | menu | icon",
  "texto": "Enviar",
  "descripcion": "Boton azul para enviar formulario",
  "bbox": {"x": 0-100, "y": 0-100, "width": 0-100, "height": 0-100},
  "confianza": 0.95,
  "es_cta_principal": true
}]
```

---

## Endpoints Backend

### Deteccion de Elementos
| Endpoint | Metodo | Descripcion |
|----------|--------|-------------|
| `/api/pantallas/{id}/detectar-elementos` | POST | Detectar elementos clickeables con Gemini Vision |
| `/api/pantallas/{id}/elementos` | PUT | Corregir elementos manualmente |

### Misiones CRUD
| Endpoint | Metodo | Descripcion |
|----------|--------|-------------|
| `/api/flujos/{id}/misiones` | GET | Listar misiones del flujo |
| `/api/flujos/{id}/misiones` | POST | Crear mision |
| `/api/flujos/{id}/misiones/{mid}` | GET | Obtener mision con simulaciones |
| `/api/flujos/{id}/misiones/{mid}` | PUT | Actualizar mision |
| `/api/flujos/{id}/misiones/{mid}` | DELETE | Eliminar mision |

### Simulaciones
| Endpoint | Metodo | Descripcion |
|----------|--------|-------------|
| `/api/misiones/{id}/simular` | POST | Ejecutar simulacion con arquetipos |
| `/api/misiones/{id}/simulaciones` | GET | Listar simulaciones |
| `/api/misiones/{id}/metricas` | GET | Obtener metricas agregadas |

---

## Logica de Simulacion

### Diferenciacion por nivel_digital

| Nivel | Comportamiento | Success Rate Esperado | Misclicks |
|-------|---------------|----------------------|-----------|
| **bajo** | Confusion con iconos, evita opciones complejas | 40-60% | 2-4 |
| **medio** | Navegacion normal, prefiere opciones claras | 70-85% | 1-2 |
| **alto** | Navegacion eficiente, identifica CTAs rapido | 90-98% | 0-1 |

### Factores que afectan la simulacion

1. **nivel_digital** → Probabilidad de misclick
2. **frustraciones** → Sensibilidad a patrones especificos
3. **objetivos** → Priorizacion de elementos
4. **edad** → Preferencia por elementos mas grandes/claros

### Prompt de Simulacion (resumen)
```
Eres {arquetipo.nombre}, {arquetipo.edad} años, nivel digital {nivel_digital}.
Tus frustraciones: {frustraciones}
Tus objetivos: {objetivos}

TAREA: "{mision.instrucciones}"

[Imagen de pantalla actual]

ELEMENTOS DISPONIBLES: {lista de elementos}

Decide que elemento clickear como este usuario especifico.
Si nivel_digital es bajo, tienes 30% de probabilidad de misclick.

Responde: {elemento_id, razonamiento, confusion, emocion}
```

---

## Componentes Frontend

### Paginas Nuevas
| Componente | Ruta | Descripcion |
|------------|------|-------------|
| `MisionesTab.tsx` | `/flujos/:id` (tab) | Lista de misiones en flujo |
| `MisionForm.tsx` | Modal | Crear/editar mision |
| `MisionDetail.tsx` | `/flujos/:id/misiones/:mid` | Detalle con simulaciones |
| `UsabilityDashboard.tsx` | `/flujos/:id/usability` | Dashboard de metricas |

### Componentes Reutilizables
| Componente | Funcion |
|------------|---------|
| `ElementOverlay.tsx` | Mostrar elementos detectados sobre imagen |
| `SimulationPlayer.tsx` | Reproductor visual del path tomado |
| `MetricsCard.tsx` | Card de metricas (success rate, misclicks) |
| `FrictionList.tsx` | Lista de fricciones detectadas |

### Integracion con FlujoDetail
Agregar tabs:
```
[Pantallas] [Misiones] [Usability]
```

---

## Archivos a Crear

### Backend
```
product/backend/app/services/element_detection.py   # Detectar elementos
product/backend/app/services/usability_simulation.py # Motor de simulacion
product/backend/app/api/usability.py                # Router de endpoints
```

### Frontend
```
product/frontend/src/components/flujos/ElementOverlay.tsx
product/frontend/src/components/flujos/MisionForm.tsx
product/frontend/src/components/flujos/MisionesTab.tsx
product/frontend/src/components/flujos/SimulationPlayer.tsx
product/frontend/src/pages/MisionDetail.tsx
product/frontend/src/pages/UsabilityDashboard.tsx
```

### Database
```
supabase/migrations/20260127000000_add_usability_testing.sql
```

## Archivos a Modificar

```
product/backend/app/main.py                    # Registrar router usability
product/backend/app/api/flujos.py              # Integrar deteccion de elementos
product/backend/app/services/supabase.py       # CRUD misiones/simulaciones
product/frontend/src/lib/api.ts                # usabilityApi + types
product/frontend/src/pages/FlujoDetail.tsx     # Agregar tabs Misiones/Usability
product/frontend/src/App.tsx                   # Nuevas rutas
```

---

## Fases de Implementacion

### Fase 1: Deteccion de Elementos
- [ ] Migracion SQL (solo tablas nuevas)
- [ ] `ElementDetectionService` con Gemini Vision
- [ ] Endpoint POST `/pantallas/{id}/detectar-elementos`
- [ ] Actualizar `upload_pantalla` para detectar automaticamente
- [ ] `ElementOverlay.tsx` para visualizar elementos
- [ ] UI para correccion manual

### Fase 2: Sistema de Misiones
- [ ] CRUD backend de misiones
- [ ] `MisionesTab.tsx` en FlujoDetail
- [ ] `MisionForm.tsx` modal
- [ ] Selector visual de elemento objetivo

### Fase 3: Motor de Simulacion
- [ ] `UsabilitySimulationService`
- [ ] Prompts diferenciados por nivel_digital
- [ ] Endpoint POST `/misiones/{id}/simular`
- [ ] Logica de transiciones entre pantallas
- [ ] Calculo de metricas individuales

### Fase 4: Dashboard y Visualizacion
- [ ] `MisionDetail.tsx` con path tomado
- [ ] `SimulationPlayer.tsx` reproductor visual
- [ ] `UsabilityDashboard.tsx` con graficos
- [ ] Metricas agregadas por nivel_digital

---

## Verificacion

1. **Subir pantalla** → Verificar que detecta elementos automaticamente
2. **Crear mision** → Verificar selector de elemento objetivo funciona
3. **Simular con arquetipo alto** → Success rate >90%, misclicks ~0
4. **Simular con arquetipo bajo** → Success rate <60%, misclicks >2
5. **Dashboard** → Metricas diferenciadas por nivel visible
6. **Player** → Reproduce path paso a paso con razonamiento

---

## Metricas de Exito

- Deteccion de elementos: >85% precision en UI estandar
- Diferenciacion clara entre niveles digitales en metricas
- Usuario puede crear mision y ver resultados en <5 minutos
- Fricciones detectadas son accionables para mejorar UX
