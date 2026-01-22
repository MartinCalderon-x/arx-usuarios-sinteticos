"""DeepGaze service for visual attention prediction.

DeepGaze is a family of deep learning models for saliency prediction developed
by Matthias Kümmerer at the University of Tübingen.

References:
    - DeepGaze III: Kümmerer, M., & Bethge, M. (2022). Journal of Vision.
    - DeepGaze II: Kümmerer, M., Wallis, T.S.A., & Bethge, M. (2016).
    - GitHub: https://github.com/matthias-k/DeepGaze
"""
import io
import logging
import os
import time
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom
from scipy.special import logsumexp

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# URL for MIT1003 centerbias from DeepGaze releases
CENTERBIAS_URL = "https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy"
CENTERBIAS_PATH = "/app/models/centerbias_mit1003.npy"


class DeepGazeService:
    """Service for generating saliency maps using DeepGaze models.

    This service provides visual attention prediction using pre-trained
    DeepGaze models. The models are trained on real eye-tracking data
    from the MIT1003 dataset.

    Attributes:
        model_name: Name of the DeepGaze model variant to use.
        device: Torch device (cpu or cuda).
        model: Loaded PyTorch model instance.

    Example:
        >>> service = DeepGazeService()
        >>> heatmap, metadata = await service.predict(image_bytes)
    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """Initialize DeepGaze service.

        Args:
            model_name: DeepGaze variant ('deepgaze2', 'deepgaze2e', 'deepgaze3').
                       Defaults to settings.deepgaze_model.
            device: Torch device ('cpu' or 'cuda').
                   Defaults to settings.device.
        """
        self.model_name = model_name or settings.deepgaze_model
        self.device = device or settings.device
        self.model = None
        self._center_bias = None

    def _download_centerbias(self):
        """Download centerbias file from GitHub releases if not present."""
        if os.path.exists(CENTERBIAS_PATH):
            return

        logger.info(f"Downloading centerbias from {CENTERBIAS_URL}")
        os.makedirs(os.path.dirname(CENTERBIAS_PATH), exist_ok=True)

        try:
            import urllib.request
            urllib.request.urlretrieve(CENTERBIAS_URL, CENTERBIAS_PATH)
            logger.info("Centerbias downloaded successfully")
        except Exception as e:
            logger.warning(f"Failed to download centerbias: {e}. Using fallback.")

    def _load_model(self):
        """Load DeepGaze model lazily on first prediction.

        The model is loaded on-demand to reduce startup time and memory
        usage when the service is not immediately needed.
        """
        if self.model is not None:
            return

        logger.info(f"Loading DeepGaze model: {self.model_name} on {self.device}")
        start_time = time.time()

        try:
            import deepgaze_pytorch

            # Use DeepGazeIIE which doesn't require scanpath history
            # DeepGazeIII requires fixation history which we don't have
            if self.model_name == "deepgaze2e":
                self.model = deepgaze_pytorch.DeepGazeIIE(pretrained=True)
            elif self.model_name in ["deepgaze3", "deepgaze2"]:
                # Fall back to DeepGazeIIE for simplicity
                logger.info(f"Using DeepGazeIIE instead of {self.model_name} (simpler API)")
                self.model = deepgaze_pytorch.DeepGazeIIE(pretrained=True)
            else:
                raise ValueError(f"Unknown model: {self.model_name}")

            self.model = self.model.to(self.device)
            self.model.eval()

            # Download and load centerbias from MIT1003 dataset
            self._download_centerbias()
            if os.path.exists(CENTERBIAS_PATH):
                self._center_bias = np.load(CENTERBIAS_PATH)
                logger.info(f"Centerbias loaded: shape {self._center_bias.shape}")
            else:
                # Fallback: create uniform centerbias
                self._center_bias = np.zeros((1024, 1024))
                logger.warning("Using uniform centerbias as fallback")

            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f}s")

        except ImportError as e:
            logger.error(f"Failed to import deepgaze_pytorch: {e}")
            raise RuntimeError(
                "DeepGaze library not installed. "
                "Install with: pip install deepgaze-pytorch"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _preprocess_image(self, image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Preprocess image for model input.

        Args:
            image: PIL Image to process.

        Returns:
            Tuple of (preprocessed tensor, original size).

        Notes:
            - Images are resized to have max dimension of max_image_size
            - DeepGaze expects images normalized to [0, 1] range
            - Original size is returned for proper heatmap resizing
        """
        original_size = image.size  # (width, height)

        # Resize if needed, maintaining aspect ratio
        max_size = settings.max_image_size
        if max(original_size) > max_size:
            ratio = max_size / max(original_size)
            new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Convert to numpy and normalize
        img_array = np.array(image).astype(np.float32) / 255.0

        # Ensure RGB (3 channels)
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]

        # Convert to torch tensor: (H, W, C) -> (1, C, H, W)
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

        return tensor.to(self.device), original_size

    def _get_center_bias(self, height: int, width: int) -> torch.Tensor:
        """Get center bias tensor for given dimensions.

        Center bias represents the prior probability distribution of
        human fixations, which tend to be biased toward the center
        of images.

        Args:
            height: Image height.
            width: Image width.

        Returns:
            Center bias tensor of shape (1, 1, height, width).
        """
        try:
            if self._center_bias is None or self._center_bias.size == 0:
                raise ValueError("No centerbias loaded")

            # Ensure centerbias is 2D
            cb = self._center_bias
            if cb.ndim != 2:
                logger.warning(f"Centerbias has unexpected shape: {cb.shape}")
                raise ValueError("Centerbias is not 2D")

            # Resize MIT1003 center bias to target size using PIL
            # Convert to image, resize, convert back
            # Note: centerbias values are log densities, can be negative
            cb_min, cb_max = cb.min(), cb.max()
            cb_normalized = (cb - cb_min) / (cb_max - cb_min + 1e-8)

            cb_img = Image.fromarray((cb_normalized * 255).astype(np.uint8))
            cb_resized = cb_img.resize((width, height), Image.Resampling.BILINEAR)
            center_bias = np.array(cb_resized).astype(np.float32) / 255.0

            # Restore to original scale
            center_bias = center_bias * (cb_max - cb_min) + cb_min

            # Renormalize log density after rescaling
            center_bias = center_bias - logsumexp(center_bias)

        except Exception as e:
            logger.warning(f"Using fallback centerbias: {e}")
            # Fallback: create simple gaussian center bias in log space
            y = np.linspace(-1, 1, height)
            x = np.linspace(-1, 1, width)
            xx, yy = np.meshgrid(x, y)
            center_bias = np.exp(-(xx**2 + yy**2) / 0.5)
            center_bias = np.log(center_bias + 1e-8)
            center_bias = center_bias - logsumexp(center_bias)

        # Ensure correct shape: (1, 1, H, W)
        center_bias = np.asarray(center_bias, dtype=np.float32)
        if center_bias.shape != (height, width):
            logger.error(f"Centerbias shape mismatch: {center_bias.shape} vs expected ({height}, {width})")
            # Force reshape if needed
            center_bias = np.resize(center_bias, (height, width)).astype(np.float32)

        tensor = torch.from_numpy(center_bias).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    async def predict(
        self,
        image_data: bytes,
        return_raw: bool = False
    ) -> Tuple[np.ndarray, dict]:
        """Generate saliency prediction for an image.

        Args:
            image_data: Raw image bytes (PNG, JPEG, etc.).
            return_raw: If True, return raw log-density; else return normalized heatmap.

        Returns:
            Tuple of (heatmap array, metadata dict).

            heatmap: numpy array of shape (H, W) with values in [0, 1].
            metadata: dict containing:
                - model: Model name used
                - inference_time_ms: Prediction time in milliseconds
                - original_size: Original image dimensions
                - processed_size: Size used for prediction

        Example:
            >>> heatmap, meta = await service.predict(image_bytes)
            >>> print(f"Inference took {meta['inference_time_ms']}ms")
        """
        self._load_model()

        start_time = time.time()

        # Load and preprocess image
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        tensor, original_size = self._preprocess_image(image)

        # Get dimensions
        _, _, height, width = tensor.shape

        # Get center bias
        center_bias = self._get_center_bias(height, width)

        # Run inference
        with torch.no_grad():
            log_density = self.model(tensor, center_bias)

        # Convert to numpy
        log_density_np = log_density.cpu().numpy().squeeze()

        # Convert log-density to probability and normalize
        if return_raw:
            heatmap = log_density_np
        else:
            # Softmax to convert log-density to probability
            density = np.exp(log_density_np)
            # Normalize to [0, 1]
            heatmap = (density - density.min()) / (density.max() - density.min() + 1e-8)

        # Resize to original size if different
        if heatmap.shape != (original_size[1], original_size[0]):
            heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
            heatmap_img = heatmap_img.resize(original_size, Image.Resampling.BILINEAR)
            heatmap = np.array(heatmap_img).astype(np.float32) / 255.0

        inference_time = (time.time() - start_time) * 1000

        metadata = {
            "model": self.model_name,
            "inference_time_ms": round(inference_time, 2),
            "original_size": {"width": original_size[0], "height": original_size[1]},
            "processed_size": {"width": width, "height": height},
            "device": self.device,
        }

        logger.info(f"Prediction completed in {inference_time:.2f}ms")

        return heatmap, metadata

    def generate_heatmap_image(
        self,
        heatmap: np.ndarray,
        colormap: str = "jet",
        alpha: float = 0.6,
        original_image: Optional[Image.Image] = None
    ) -> Image.Image:
        """Generate colored heatmap image, optionally overlaid on original.

        Args:
            heatmap: Saliency map array of shape (H, W) with values in [0, 1].
            colormap: Matplotlib colormap name ('jet', 'hot', 'viridis', etc.).
            alpha: Transparency for overlay (0=transparent, 1=opaque).
            original_image: If provided, overlay heatmap on this image.

        Returns:
            PIL Image with colored heatmap (or overlay).

        Example:
            >>> heatmap_img = service.generate_heatmap_image(
            ...     heatmap, colormap='jet', alpha=0.5, original_image=orig
            ... )
            >>> heatmap_img.save('output.png')
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm

        # Get colormap
        cmap = cm.get_cmap(colormap)

        # Apply colormap (returns RGBA)
        heatmap_colored = cmap(heatmap)

        # Convert to uint8
        heatmap_rgba = (heatmap_colored * 255).astype(np.uint8)
        heatmap_img = Image.fromarray(heatmap_rgba, mode="RGBA")

        if original_image is not None:
            # Ensure same size
            if heatmap_img.size != original_image.size:
                heatmap_img = heatmap_img.resize(
                    original_image.size, Image.Resampling.BILINEAR
                )

            # Convert original to RGBA
            original_rgba = original_image.convert("RGBA")

            # Blend images
            blended = Image.blend(original_rgba, heatmap_img, alpha)
            return blended

        return heatmap_img.convert("RGB")

    def extract_attention_regions(
        self,
        heatmap: np.ndarray,
        threshold: float = 0.5,
        min_area: int = 100
    ) -> list[dict]:
        """Extract high-attention regions from heatmap.

        Args:
            heatmap: Saliency map array of shape (H, W) with values in [0, 1].
            threshold: Minimum attention value to consider (0-1).
            min_area: Minimum region area in pixels.

        Returns:
            List of region dicts with:
                - x, y: Center coordinates (relative 0-100)
                - width, height: Bounding box size (relative 0-100)
                - intensity: Mean attention value in region
                - area: Region area in pixels
        """
        import cv2

        h, w = heatmap.shape

        # Threshold heatmap
        binary = (heatmap > threshold).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # Get bounding box
            x, y, bw, bh = cv2.boundingRect(contour)

            # Calculate mean intensity in region
            mask = np.zeros_like(heatmap, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            intensity = float(heatmap[mask > 0].mean())

            regions.append({
                "x": round((x + bw / 2) / w * 100, 1),
                "y": round((y + bh / 2) / h * 100, 1),
                "width": round(bw / w * 100, 1),
                "height": round(bh / h * 100, 1),
                "intensity": round(intensity * 100, 1),
                "area": int(area),
            })

        # Sort by intensity (highest first)
        regions.sort(key=lambda r: r["intensity"], reverse=True)

        # Add visual order
        for i, region in enumerate(regions):
            region["orden_visual"] = i + 1

        return regions
