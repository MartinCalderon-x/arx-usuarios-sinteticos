"""ML Service API for visual attention prediction.

This service provides REST endpoints for generating saliency maps
using DeepGaze deep learning models.

The service is designed to run as a standalone microservice,
typically deployed on Cloud Run with higher memory/CPU allocation
than the main backend.
"""
import io
import base64
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from PIL import Image

from app.config import get_settings
from app.services.deepgaze import DeepGazeService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

settings = get_settings()

# Global service instance
deepgaze_service: Optional[DeepGazeService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    global deepgaze_service

    logger.info(f"Starting ML Service: {settings.service_name}")
    logger.info(f"Model: {settings.deepgaze_model}, Device: {settings.device}")

    # Initialize service (model loaded lazily on first request)
    deepgaze_service = DeepGazeService()

    yield

    logger.info("Shutting down ML Service")


app = FastAPI(
    title="Visual Attention ML Service",
    description="Saliency prediction using DeepGaze models",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Response Models
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    model: str
    device: str


class AttentionRegion(BaseModel):
    """Detected attention region."""
    x: float
    y: float
    width: float
    height: float
    intensity: float
    area: int
    orden_visual: int


class PredictionMetadata(BaseModel):
    """Prediction metadata."""
    model: str
    inference_time_ms: float
    original_size: dict
    processed_size: dict
    device: str


class PredictionResponse(BaseModel):
    """Full prediction response."""
    heatmap_base64: str
    heatmap_overlay_base64: Optional[str] = None
    regions: list[AttentionRegion]
    metadata: PredictionMetadata


class CompactPredictionResponse(BaseModel):
    """Compact response with raw heatmap data."""
    heatmap_data: list[list[float]]
    regions: list[AttentionRegion]
    metadata: PredictionMetadata


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return HealthResponse(
        status="healthy",
        service=settings.service_name,
        model=settings.deepgaze_model,
        device=settings.device,
    )


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "Visual Attention ML Service",
        "version": "0.2.0",
        "model": settings.deepgaze_model,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_compact": "/predict/compact",
            "predict_image": "/predict/image",
            "predict_gazeplot": "/predict/gazeplot",
            "warmup": "/warmup",
        },
    }


@app.post("/warmup")
async def warmup():
    """Warmup endpoint to pre-load model.

    Call this endpoint to load the model into memory before
    handling actual requests. Useful for reducing cold start latency.
    """
    global deepgaze_service

    if deepgaze_service is None:
        deepgaze_service = DeepGazeService()

    start_time = time.time()

    # Create small test image and run prediction
    test_image = Image.new("RGB", (100, 100), color="white")
    img_bytes = io.BytesIO()
    test_image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    try:
        await deepgaze_service.predict(img_bytes.read())
        warmup_time = time.time() - start_time
        return {
            "status": "warmed_up",
            "warmup_time_seconds": round(warmup_time, 2),
            "model": settings.deepgaze_model,
        }
    except Exception as e:
        logger.error(f"Warmup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Warmup failed: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(..., description="Image file to analyze"),
    colormap: str = Query("jet", description="Colormap for heatmap (jet, hot, viridis)"),
    overlay: bool = Query(True, description="Include overlay on original image"),
    threshold: float = Query(0.3, description="Threshold for region detection"),
):
    """Generate saliency prediction for an uploaded image.

    This endpoint returns a full prediction with:
    - Base64-encoded heatmap image
    - Optional overlay on original image
    - Detected high-attention regions
    - Prediction metadata

    Args:
        file: Image file (PNG, JPEG, etc.)
        colormap: Matplotlib colormap name for visualization
        overlay: Whether to include overlay version
        threshold: Attention threshold for region detection (0-1)

    Returns:
        PredictionResponse with heatmap and metadata.
    """
    global deepgaze_service

    if deepgaze_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read image data
        image_data = await file.read()

        # Run prediction
        heatmap, metadata = await deepgaze_service.predict(image_data)

        # Load original image for overlay
        original_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Generate heatmap image
        heatmap_img = deepgaze_service.generate_heatmap_image(
            heatmap, colormap=colormap
        )

        # Encode heatmap to base64
        heatmap_buffer = io.BytesIO()
        heatmap_img.save(heatmap_buffer, format="PNG")
        heatmap_base64 = base64.b64encode(heatmap_buffer.getvalue()).decode()

        # Generate overlay if requested
        overlay_base64 = None
        if overlay:
            overlay_img = deepgaze_service.generate_heatmap_image(
                heatmap, colormap=colormap, alpha=0.5, original_image=original_image
            )
            overlay_buffer = io.BytesIO()
            overlay_img.save(overlay_buffer, format="PNG")
            overlay_base64 = base64.b64encode(overlay_buffer.getvalue()).decode()

        # Extract attention regions
        regions = deepgaze_service.extract_attention_regions(
            heatmap, threshold=threshold
        )

        return PredictionResponse(
            heatmap_base64=heatmap_base64,
            heatmap_overlay_base64=overlay_base64,
            regions=[AttentionRegion(**r) for r in regions],
            metadata=PredictionMetadata(**metadata),
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/compact", response_model=CompactPredictionResponse)
async def predict_compact(
    file: UploadFile = File(..., description="Image file to analyze"),
    threshold: float = Query(0.3, description="Threshold for region detection"),
    downsample: int = Query(4, description="Downsample factor for heatmap data"),
):
    """Generate compact saliency prediction.

    Returns raw heatmap data as a 2D array instead of images.
    Useful for custom visualization or further processing.

    Args:
        file: Image file (PNG, JPEG, etc.)
        threshold: Attention threshold for region detection (0-1)
        downsample: Factor to reduce heatmap resolution (1=full, 4=quarter)

    Returns:
        CompactPredictionResponse with raw heatmap array.
    """
    global deepgaze_service

    if deepgaze_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_data = await file.read()
        heatmap, metadata = await deepgaze_service.predict(image_data)

        # Downsample heatmap for smaller response
        if downsample > 1:
            h, w = heatmap.shape
            new_h, new_w = h // downsample, w // downsample
            heatmap_small = np.array(
                Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
                    (new_w, new_h), Image.Resampling.BILINEAR
                )
            ).astype(np.float32) / 255.0
        else:
            heatmap_small = heatmap

        # Convert to nested list for JSON
        heatmap_data = heatmap_small.round(3).tolist()

        # Extract regions from full-resolution heatmap
        regions = deepgaze_service.extract_attention_regions(
            heatmap, threshold=threshold
        )

        return CompactPredictionResponse(
            heatmap_data=heatmap_data,
            regions=[AttentionRegion(**r) for r in regions],
            metadata=PredictionMetadata(**metadata),
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/image")
async def predict_image(
    file: UploadFile = File(..., description="Image file to analyze"),
    colormap: str = Query("jet", description="Colormap for heatmap"),
    overlay: bool = Query(True, description="Overlay on original image"),
    alpha: float = Query(0.5, description="Overlay transparency"),
):
    """Generate saliency prediction and return as PNG image directly.

    Useful for direct embedding or when you only need the visualization.

    Args:
        file: Image file (PNG, JPEG, etc.)
        colormap: Matplotlib colormap name
        overlay: Whether to overlay on original
        alpha: Overlay transparency (0-1)

    Returns:
        PNG image response.
    """
    global deepgaze_service

    if deepgaze_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_data = await file.read()
        heatmap, _ = await deepgaze_service.predict(image_data)

        original_image = None
        if overlay:
            original_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        heatmap_img = deepgaze_service.generate_heatmap_image(
            heatmap,
            colormap=colormap,
            alpha=alpha,
            original_image=original_image,
        )

        img_buffer = io.BytesIO()
        heatmap_img.save(img_buffer, format="PNG")
        img_buffer.seek(0)

        return Response(
            content=img_buffer.read(),
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=heatmap.png"},
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ============================================================================
# Gaze Plot Endpoint
# ============================================================================

class Fixation(BaseModel):
    """A single fixation point in the gaze plot."""
    x: float
    y: float
    order: int
    intensity: float
    duration_estimate: int
    pixel_x: int
    pixel_y: int


class GazePlotResponse(BaseModel):
    """Response for gaze plot endpoint."""
    fixations: list[Fixation]
    scanpath_image_base64: Optional[str] = None
    heatmap_base64: str
    metadata: PredictionMetadata


@app.post("/predict/gazeplot", response_model=GazePlotResponse)
async def predict_gazeplot(
    file: UploadFile = File(..., description="Image file to analyze"),
    num_fixations: int = Query(10, ge=1, le=30, description="Number of fixations to generate"),
    colormap: str = Query("jet", description="Colormap for heatmap visualization"),
    include_scanpath: bool = Query(True, description="Include scanpath visualization image"),
):
    """Generate Gaze Plot (scanpath) prediction using Winner-Take-All + IoR.

    This endpoint simulates human visual attention by generating a sequence
    of fixation points showing where the eye would look and in what order.

    The algorithm:
    1. Generates a saliency map using DeepGaze
    2. Finds maximum saliency point (Winner-Take-All) = Fixation #1
    3. Applies Inhibition of Return (IoR) to suppress that area
    4. Repeats to find subsequent fixations

    Args:
        file: Image file (PNG, JPEG, etc.)
        num_fixations: Maximum number of fixations to generate (1-30)
        colormap: Matplotlib colormap for visualization
        include_scanpath: Whether to include visual scanpath image

    Returns:
        GazePlotResponse with fixation sequence and optional visualization.

    Example response:
        {
            "fixations": [
                {"x": 45.2, "y": 32.1, "order": 1, "intensity": 85.3, ...},
                {"x": 68.7, "y": 51.4, "order": 2, "intensity": 72.1, ...},
                ...
            ],
            "scanpath_image_base64": "...",
            "heatmap_base64": "...",
            "metadata": {...}
        }
    """
    global deepgaze_service

    if deepgaze_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read image data
        image_data = await file.read()

        # Run saliency prediction
        heatmap, metadata = await deepgaze_service.predict(image_data)

        # Generate gaze plot (sequence of fixations)
        fixations = deepgaze_service.generate_gaze_plot(
            heatmap,
            num_fixations=num_fixations,
        )

        # Generate heatmap image
        heatmap_img = deepgaze_service.generate_heatmap_image(heatmap, colormap=colormap)
        heatmap_buffer = io.BytesIO()
        heatmap_img.save(heatmap_buffer, format="PNG")
        heatmap_base64 = base64.b64encode(heatmap_buffer.getvalue()).decode()

        # Generate scanpath visualization if requested
        scanpath_base64 = None
        if include_scanpath and fixations:
            original_image = Image.open(io.BytesIO(image_data)).convert("RGB")
            scanpath_img = deepgaze_service.generate_scanpath_image(
                heatmap,
                fixations,
                original_image=original_image,
                colormap=colormap,
                alpha=0.4,
            )
            scanpath_buffer = io.BytesIO()
            scanpath_img.save(scanpath_buffer, format="PNG")
            scanpath_base64 = base64.b64encode(scanpath_buffer.getvalue()).decode()

        return GazePlotResponse(
            fixations=[Fixation(**f) for f in fixations],
            scanpath_image_base64=scanpath_base64,
            heatmap_base64=heatmap_base64,
            metadata=PredictionMetadata(**metadata),
        )

    except Exception as e:
        logger.error(f"Gaze plot prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8080,
        reload=settings.debug,
    )
