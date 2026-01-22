# Sistema Dual de Predicción de Atención Visual

## Resumen Ejecutivo

Este documento describe la implementación de un sistema dual de predicción de atención visual para la plataforma de Usuarios Sintéticos. El sistema combina dos enfoques complementarios:

1. **DeepGaze III**: Modelo de deep learning basado en redes neuronales convolucionales, entrenado con datos reales de eye-tracking.
2. **Modelo Híbrido**: Enfoque que combina análisis de Gemini Vision con interpolación gaussiana para generar mapas de atención.

El objetivo es proporcionar análisis de atención visual de alta calidad mientras se evalúa la viabilidad de un modelo más ligero (híbrido) comparado con el estado del arte (DeepGaze).

---

## 1. Fundamentos Teóricos

### 1.1 Atención Visual y Saliency

La **atención visual** es el proceso cognitivo mediante el cual el sistema visual humano selecciona información relevante del entorno. Los **mapas de saliencia** (saliency maps) representan la probabilidad de que una región de la imagen capture la atención visual.

#### Tipos de Atención Visual
- **Bottom-up (exógena)**: Guiada por características visuales (color, contraste, movimiento)
- **Top-down (endógena)**: Guiada por objetivos y conocimiento previo

### 1.2 Métricas de Evaluación

| Métrica | Descripción | Rango | Interpretación |
|---------|-------------|-------|----------------|
| **CC (Correlation Coefficient)** | Correlación de Pearson entre mapas | [-1, 1] | Mayor = mejor |
| **KL Divergence** | Divergencia Kullback-Leibler | [0, ∞) | Menor = mejor |
| **NSS (Normalized Scanpath Saliency)** | Saliencia normalizada en fijaciones | (-∞, ∞) | Mayor = mejor |
| **AUC-Judd** | Área bajo curva ROC | [0, 1] | Mayor = mejor |
| **SIM (Similarity)** | Similitud entre distribuciones | [0, 1] | Mayor = mejor |

---

## 2. Modelo 1: DeepGaze III

### 2.1 Origen y Desarrollo

**DeepGaze** es una familia de modelos desarrollada por el grupo de investigación de Matthias Kümmerer en la Universidad de Tübingen, Alemania.

#### Evolución de la Familia DeepGaze
| Versión | Año | Backbone | Innovación Principal |
|---------|-----|----------|---------------------|
| DeepGaze I | 2015 | AlexNet | Primera integración CNN para saliency |
| DeepGaze II | 2016 | VGG-19 | Features de múltiples capas |
| DeepGaze IIE | 2021 | VGG-19 | Ensemble de readouts |
| DeepGaze III | 2022 | ResNet-50 | Integración de fixation history |

### 2.2 Arquitectura de DeepGaze III

```
┌─────────────────────────────────────────────────────────────┐
│                      DeepGaze III                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────────┐   │
│  │  Input   │───▶│  ResNet-50  │───▶│  Multi-scale     │   │
│  │  Image   │    │  Backbone   │    │  Feature Maps    │   │
│  └──────────┘    └─────────────┘    └────────┬─────────┘   │
│                                               │              │
│                                               ▼              │
│                                    ┌──────────────────┐     │
│                                    │  Readout Network │     │
│                                    │  (1x1 Conv + BN) │     │
│                                    └────────┬─────────┘     │
│                                               │              │
│                                               ▼              │
│  ┌──────────┐                     ┌──────────────────┐     │
│  │ Center   │────────────────────▶│  Log-Density     │     │
│  │  Bias    │                     │  Combination     │     │
│  └──────────┘                     └────────┬─────────┘     │
│                                               │              │
│                                               ▼              │
│                                    ┌──────────────────┐     │
│                                    │  Saliency Map    │     │
│                                    │  (Probability)   │     │
│                                    └──────────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Características Técnicas

- **Input**: Imagen RGB de resolución variable (recomendado: 1024px lado largo)
- **Output**: Mapa de probabilidad de fijación (misma resolución que input)
- **Backbone**: ResNet-50 pre-entrenado en ImageNet
- **Training Data**: MIT1003 dataset (1003 imágenes con datos de eye-tracking de 15 observadores)
- **Loss Function**: Negative log-likelihood sobre fijaciones reales

### 2.4 Performance en Benchmarks

| Dataset | CC | KL | NSS | AUC-J |
|---------|-----|-----|------|-------|
| MIT300 | 0.87 | 0.32 | 2.45 | 0.88 |
| SALICON | 0.89 | 0.28 | 2.51 | 0.87 |
| CAT2000 | 0.84 | 0.41 | 2.38 | 0.86 |

### 2.5 Referencias Académicas

```bibtex
@article{kummerer2022deepgaze,
  title={DeepGaze III: Modeling free-viewing human scanpaths with deep learning},
  author={K{\"u}mmerer, Matthias and Bethge, Matthias},
  journal={Journal of Vision},
  volume={22},
  number={5},
  pages={7--7},
  year={2022},
  publisher={The Association for Research in Vision and Ophthalmology}
}

@inproceedings{kummerer2016deepgaze,
  title={DeepGaze II: Reading fixations from deep features trained on object recognition},
  author={K{\"u}mmerer, Matthias and Wallis, Thomas SA and Bethge, Matthias},
  booktitle={Journal of Vision},
  year={2016}
}
```

---

## 3. Modelo 2: Híbrido (Gemini + Gaussian)

### 3.1 Concepto

El modelo híbrido combina las capacidades de análisis visual de Gemini Vision con técnicas clásicas de procesamiento de imágenes para generar mapas de atención.

### 3.2 Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                    Modelo Híbrido                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌─────────────────┐    ┌───────────────┐  │
│  │  Input   │───▶│  Gemini Vision  │───▶│  Structured   │  │
│  │  Image   │    │  Analysis       │    │  AOI Data     │  │
│  └──────────┘    └─────────────────┘    └───────┬───────┘  │
│                                                   │          │
│                                                   ▼          │
│                                        ┌─────────────────┐  │
│                                        │  AOI to Points  │  │
│                                        │  Conversion     │  │
│                                        └───────┬─────────┘  │
│                                                   │          │
│                                                   ▼          │
│  ┌──────────┐                         ┌─────────────────┐  │
│  │  Image   │────────────────────────▶│  Gaussian       │  │
│  │  Size    │                         │  Interpolation  │  │
│  └──────────┘                         └───────┬─────────┘  │
│                                                   │          │
│                                                   ▼          │
│                                        ┌─────────────────┐  │
│                                        │  Gaussian Blur  │  │
│                                        │  σ = f(AOI)     │  │
│                                        └───────┬─────────┘  │
│                                                   │          │
│                                                   ▼          │
│                                        ┌─────────────────┐  │
│                                        │  Normalize &    │  │
│                                        │  Color Map      │  │
│                                        └───────┬─────────┘  │
│                                                   │          │
│                                                   ▼          │
│                                        ┌─────────────────┐  │
│                                        │  Saliency Map   │  │
│                                        └─────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Algoritmo de Generación

```python
def generate_hybrid_heatmap(image, aoi_data):
    """
    Genera heatmap híbrido a partir de AOIs de Gemini.

    Args:
        image: Imagen original (numpy array)
        aoi_data: Lista de AOIs con coordenadas e intensidad

    Returns:
        heatmap: Mapa de calor normalizado
    """
    h, w = image.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    for aoi in aoi_data:
        # Centro del AOI
        cx = int(aoi['x'] * w / 100)
        cy = int(aoi['y'] * h / 100)

        # Tamaño del AOI determina sigma
        aoi_w = int(aoi['width'] * w / 100)
        aoi_h = int(aoi['height'] * h / 100)
        sigma = max(aoi_w, aoi_h) / 2

        # Intensidad basada en orden visual
        intensity = aoi['intensidad'] / 100

        # Crear gaussian kernel
        y, x = np.ogrid[:h, :w]
        gaussian = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))

        # Acumular con peso de intensidad
        heatmap += gaussian * intensity

    # Normalizar a [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    return heatmap
```

### 3.4 Ventajas del Modelo Híbrido

| Aspecto | Beneficio |
|---------|-----------|
| **Latencia** | ~300ms vs ~800ms (DeepGaze) |
| **Recursos** | Sin GPU, <100MB memoria |
| **Interpretabilidad** | AOIs explícitos de Gemini |
| **Flexibilidad** | Funciona con cualquier tipo de imagen |
| **Costo** | Solo costo de API Gemini |

### 3.5 Limitaciones

- Depende de la calidad del análisis de Gemini
- No captura saliency bottom-up pura (contraste, bordes)
- Aproximación, no predicción basada en datos de eye-tracking

---

## 4. Sistema de Comparación

### 4.1 Arquitectura del Sistema

```
┌─────────────┐
│   Cliente   │
│  (Frontend) │
└──────┬──────┘
       │ POST /api/analisis/comparar-modelos
       ▼
┌──────────────────────────────────────────────────────────┐
│                     Backend (FastAPI)                     │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────────┐         ┌─────────────────────────┐ │
│  │ Request Handler │────────▶│ Parallel Execution      │ │
│  └─────────────────┘         │                         │ │
│                              │  ┌───────────────────┐  │ │
│                              │  │ ML Service Call   │  │ │
│                              │  │ (DeepGaze)        │  │ │
│                              │  └─────────┬─────────┘  │ │
│                              │            │            │ │
│                              │  ┌───────────────────┐  │ │
│                              │  │ Hybrid Generator  │  │ │
│                              │  │ (Gemini+Gaussian) │  │ │
│                              │  └─────────┬─────────┘  │ │
│                              │            │            │ │
│                              └────────────┼────────────┘ │
│                                           │              │
│                              ┌────────────▼────────────┐ │
│                              │  Metrics Calculator     │ │
│                              │  (CC, KL, SIM, etc.)    │ │
│                              └────────────┬────────────┘ │
│                                           │              │
│                              ┌────────────▼────────────┐ │
│                              │  Response Aggregator    │ │
│                              └────────────┬────────────┘ │
│                                           │              │
└───────────────────────────────────────────┼──────────────┘
                                            │
                                            ▼
                               ┌─────────────────────────┐
                               │      ML Service         │
                               │   (Cloud Run - GPU)     │
                               │                         │
                               │  ┌───────────────────┐  │
                               │  │    DeepGaze III   │  │
                               │  │    PyTorch        │  │
                               │  └───────────────────┘  │
                               └─────────────────────────┘
```

### 4.2 Cálculo de Métricas

```python
def calculate_comparison_metrics(heatmap_ml, heatmap_hybrid):
    """
    Calcula métricas de similitud entre dos heatmaps.

    Args:
        heatmap_ml: Heatmap de DeepGaze (ground truth)
        heatmap_hybrid: Heatmap del modelo híbrido

    Returns:
        dict con métricas de comparación
    """
    # Normalizar ambos mapas
    ml_norm = normalize_map(heatmap_ml)
    hybrid_norm = normalize_map(heatmap_hybrid)

    # Correlation Coefficient
    cc = np.corrcoef(ml_norm.flatten(), hybrid_norm.flatten())[0, 1]

    # KL Divergence (añadir epsilon para estabilidad)
    eps = 1e-7
    ml_prob = ml_norm / (ml_norm.sum() + eps)
    hybrid_prob = hybrid_norm / (hybrid_norm.sum() + eps)
    kl = np.sum(ml_prob * np.log((ml_prob + eps) / (hybrid_prob + eps)))

    # Similarity (histograma intersection)
    sim = np.minimum(ml_prob, hybrid_prob).sum()

    # Earth Mover's Distance (opcional, más costoso)
    # emd = wasserstein_distance(ml_norm.flatten(), hybrid_norm.flatten())

    return {
        'correlation_coefficient': float(cc),
        'kl_divergence': float(kl),
        'similarity': float(sim),
        'alignment_percentage': float(cc * 100)
    }
```

---

## 5. Implementación Técnica

### 5.1 ML Service (DeepGaze)

#### Estructura de Archivos
```
product/ml-service/
├── Dockerfile
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app
│   ├── config.py            # Configuración
│   └── services/
│       ├── __init__.py
│       └── deepgaze.py      # Servicio DeepGaze
└── models/
    └── .gitkeep             # Modelos se descargan en runtime
```

#### Dependencias
```
torch>=2.0.0
torchvision>=0.15.0
deepgaze-pytorch>=1.0.0
fastapi>=0.100.0
uvicorn>=0.22.0
pillow>=9.0.0
numpy>=1.24.0
```

### 5.2 Backend Updates

#### Nuevo Endpoint
```python
@router.post("/comparar-modelos")
async def comparar_modelos(
    file: UploadFile = File(...),
    user: dict = Depends(require_auth)
):
    """
    Ejecuta análisis con ambos modelos y retorna comparación.
    """
    # Ejecutar en paralelo
    ml_task = call_ml_service(file)
    hybrid_task = generate_hybrid_heatmap(file)

    ml_result, hybrid_result = await asyncio.gather(ml_task, hybrid_task)

    # Calcular métricas
    metrics = calculate_comparison_metrics(
        ml_result['heatmap'],
        hybrid_result['heatmap']
    )

    return {
        'ml_model': ml_result,
        'hybrid_model': hybrid_result,
        'comparison': metrics
    }
```

---

## 6. Configuración de Despliegue

### 6.1 ML Service en Cloud Run

```yaml
# cloudbuild-ml.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/ml-service', './product/ml-service']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/ml-service']
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'ml-service'
      - '--image=gcr.io/$PROJECT_ID/ml-service'
      - '--region=us-central1'
      - '--memory=4Gi'
      - '--cpu=2'
      - '--timeout=300'
      - '--concurrency=10'
```

### 6.2 Variables de Entorno

```bash
# ML Service
ML_SERVICE_URL=https://ml-service-xxxxx-uc.a.run.app
ML_SERVICE_TIMEOUT=30

# Modelo
DEEPGAZE_MODEL=deepgaze3
DEEPGAZE_DEVICE=cpu  # o cuda si hay GPU
```

---

## 7. Resultados Esperados

### 7.1 Hipótesis

- El modelo híbrido debería alcanzar **>70% de correlación** con DeepGaze en imágenes de UI/diseño
- El modelo híbrido será **2-3x más rápido** que DeepGaze
- La diferencia será mayor en imágenes con elementos no-UI (fotografías naturales)

### 7.2 Casos de Uso Recomendados

| Caso de Uso | Modelo Recomendado | Razón |
|-------------|-------------------|-------|
| Validación rápida de diseño | Híbrido | Velocidad, costo |
| Análisis detallado para cliente | DeepGaze | Precisión |
| A/B testing masivo | Híbrido | Escala |
| Reporte académico/científico | DeepGaze | Validación |

---

## 8. Trabajo Futuro

1. **Fine-tuning de DeepGaze** en imágenes de UI específicas
2. **Ensemble de modelos** combinando predicciones
3. **Modelo híbrido v2** con saliency bottom-up (contraste, bordes)
4. **Calibración** del modelo híbrido basada en comparaciones

---

## Apéndice A: Glosario

| Término | Definición |
|---------|------------|
| **Saliency Map** | Mapa que indica probabilidad de atención en cada píxel |
| **Fixation** | Punto donde el ojo se detiene para procesar información |
| **Saccade** | Movimiento rápido del ojo entre fijaciones |
| **AOI (Area of Interest)** | Región definida para análisis de atención |
| **Heatmap** | Visualización de saliency con colores cálidos/fríos |
| **Ground Truth** | Datos reales de eye-tracking usados como referencia |

---

## Apéndice B: Referencias

1. Kümmerer, M., & Bethge, M. (2022). DeepGaze III: Modeling free-viewing human scanpaths with deep learning. Journal of Vision.
2. Bylinskii, Z., et al. (2019). What do different evaluation metrics tell us about saliency models? IEEE TPAMI.
3. Itti, L., & Koch, C. (2001). Computational modelling of visual attention. Nature Reviews Neuroscience.
4. Google. (2024). Gemini Vision API Documentation.

---

*Documento generado para el proyecto Usuarios Sintéticos*
*Última actualización: 2026-01-21*
