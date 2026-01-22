"""Hybrid Heatmap Generation Service.

This module implements a hybrid approach for generating visual attention
heatmaps by combining Gemini Vision's semantic analysis with gaussian
interpolation techniques.

The hybrid model offers a lightweight alternative to deep learning models
like DeepGaze, trading some accuracy for significantly reduced latency
and infrastructure requirements.

Architecture:
    1. Gemini Vision analyzes the image and identifies Areas of Interest (AOI)
    2. Each AOI is converted to a weighted gaussian distribution
    3. Gaussians are combined and normalized to produce the final heatmap

References:
    - Gaussian mixture models for saliency: Itti & Koch (2001)
    - Center bias in fixations: Tatler (2007)
"""
import io
import base64
import logging
import time
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class HybridHeatmapService:
    """Service for generating heatmaps using Gemini AOI + Gaussian interpolation.

    This service provides a fast, lightweight alternative to deep learning
    saliency models by leveraging Gemini's visual understanding capabilities
    combined with classical gaussian blending.

    Attributes:
        default_sigma_factor: Factor to compute sigma from AOI size.
        center_bias_strength: Strength of center prior (0-1).
        min_sigma: Minimum gaussian sigma in pixels.

    Example:
        >>> service = HybridHeatmapService()
        >>> heatmap = service.generate_from_aoi(aoi_data, width=1024, height=768)
    """

    def __init__(
        self,
        default_sigma_factor: float = 0.5,
        center_bias_strength: float = 0.2,
        min_sigma: int = 20,
    ):
        """Initialize hybrid heatmap service.

        Args:
            default_sigma_factor: Multiplier for AOI size to get sigma.
            center_bias_strength: Weight of center gaussian prior (0-1).
            min_sigma: Minimum sigma value to prevent too sharp peaks.
        """
        self.default_sigma_factor = default_sigma_factor
        self.center_bias_strength = center_bias_strength
        self.min_sigma = min_sigma

    def generate_from_aoi(
        self,
        aoi_data: list[dict],
        width: int,
        height: int,
        include_center_bias: bool = True,
    ) -> np.ndarray:
        """Generate heatmap from Areas of Interest data.

        Takes AOI coordinates and intensities from Gemini Vision analysis
        and generates a smooth heatmap using gaussian interpolation.

        Args:
            aoi_data: List of AOI dicts with keys:
                - x: Center x position (0-100 relative)
                - y: Center y position (0-100 relative)
                - width: AOI width (0-100 relative)
                - height: AOI height (0-100 relative)
                - intensidad: Attention intensity (0-100)
                - orden_visual: Visual scan order (optional)
            width: Output heatmap width in pixels.
            height: Output heatmap height in pixels.
            include_center_bias: Whether to add center viewing bias.

        Returns:
            numpy array of shape (height, width) with values in [0, 1].

        Example:
            >>> aoi_data = [
            ...     {"x": 50, "y": 30, "width": 20, "height": 10, "intensidad": 80},
            ...     {"x": 25, "y": 60, "width": 15, "height": 15, "intensidad": 60},
            ... ]
            >>> heatmap = service.generate_from_aoi(aoi_data, 1024, 768)
        """
        start_time = time.time()

        # Initialize empty heatmap
        heatmap = np.zeros((height, width), dtype=np.float32)

        # Add center bias if requested
        if include_center_bias and self.center_bias_strength > 0:
            center_bias = self._generate_center_bias(width, height)
            heatmap += center_bias * self.center_bias_strength

        # Process each AOI
        for aoi in aoi_data:
            gaussian = self._aoi_to_gaussian(aoi, width, height)
            heatmap += gaussian

        # Normalize to [0, 1]
        if heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        elapsed = (time.time() - start_time) * 1000
        logger.debug(f"Hybrid heatmap generated in {elapsed:.2f}ms")

        return heatmap

    def _aoi_to_gaussian(
        self,
        aoi: dict,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Convert single AOI to gaussian distribution.

        Args:
            aoi: AOI dict with position and intensity.
            width: Image width.
            height: Image height.

        Returns:
            2D gaussian array of shape (height, width).
        """
        # Convert relative coordinates (0-100) to pixels
        cx = int(aoi.get("x", 50) * width / 100)
        cy = int(aoi.get("y", 50) * height / 100)

        # Calculate sigma from AOI size
        aoi_w = aoi.get("width", 10) * width / 100
        aoi_h = aoi.get("height", 10) * height / 100
        sigma = max(self.min_sigma, max(aoi_w, aoi_h) * self.default_sigma_factor)

        # Get intensity (0-100 -> 0-1)
        intensity = aoi.get("intensidad", 50) / 100

        # Apply order-based decay if visual order is provided
        orden = aoi.get("orden_visual", 1)
        order_decay = 1.0 / (1 + 0.1 * (orden - 1))  # First items get higher weight
        intensity *= order_decay

        # Generate gaussian
        y, x = np.ogrid[:height, :width]
        gaussian = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))

        return gaussian * intensity

    def _generate_center_bias(self, width: int, height: int) -> np.ndarray:
        """Generate center-biased prior distribution.

        Human eye fixations tend to be biased toward the center of images.
        This adds a mild gaussian centered on the image.

        Args:
            width: Image width.
            height: Image height.

        Returns:
            Center bias array of shape (height, width).
        """
        cx, cy = width / 2, height / 2
        sigma = min(width, height) * 0.4  # Wide sigma for subtle effect

        y, x = np.ogrid[:height, :width]
        center_bias = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))

        return center_bias

    def generate_heatmap_image(
        self,
        heatmap: np.ndarray,
        colormap: str = "jet",
        alpha: float = 0.6,
        original_image: Optional[Image.Image] = None,
    ) -> Image.Image:
        """Generate colored heatmap image, optionally overlaid on original.

        Args:
            heatmap: Saliency map array of shape (H, W) with values in [0, 1].
            colormap: Matplotlib colormap name ('jet', 'hot', 'viridis').
            alpha: Transparency for overlay (0=transparent, 1=opaque).
            original_image: If provided, overlay heatmap on this image.

        Returns:
            PIL Image with colored heatmap (or overlay).
        """
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

    def heatmap_to_base64(
        self,
        heatmap: np.ndarray,
        colormap: str = "jet",
        original_image: Optional[Image.Image] = None,
        alpha: float = 0.5,
    ) -> str:
        """Convert heatmap to base64-encoded PNG.

        Args:
            heatmap: Heatmap array.
            colormap: Colormap name.
            original_image: Optional image for overlay.
            alpha: Overlay transparency.

        Returns:
            Base64-encoded PNG string.
        """
        img = self.generate_heatmap_image(
            heatmap, colormap=colormap, alpha=alpha, original_image=original_image
        )

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()


class HeatmapComparisonService:
    """Service for comparing heatmaps from different models.

    Provides metrics to evaluate how well the hybrid model
    approximates the DeepGaze ground truth.
    """

    @staticmethod
    def normalize_map(heatmap: np.ndarray) -> np.ndarray:
        """Normalize heatmap to probability distribution.

        Args:
            heatmap: Input heatmap array.

        Returns:
            Normalized array that sums to 1.
        """
        eps = 1e-8
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.sum() + eps)
        return heatmap

    @staticmethod
    def correlation_coefficient(map1: np.ndarray, map2: np.ndarray) -> float:
        """Calculate Pearson correlation coefficient between maps.

        CC measures linear relationship between two distributions.
        Range: [-1, 1], higher is better.

        Args:
            map1: First heatmap.
            map2: Second heatmap.

        Returns:
            Correlation coefficient.
        """
        return float(np.corrcoef(map1.flatten(), map2.flatten())[0, 1])

    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """Calculate Kullback-Leibler divergence.

        KL(P||Q) measures information lost when Q approximates P.
        Range: [0, inf), lower is better.

        Args:
            p: Ground truth distribution (DeepGaze).
            q: Approximation (Hybrid).

        Returns:
            KL divergence value.
        """
        eps = 1e-8
        p_norm = HeatmapComparisonService.normalize_map(p)
        q_norm = HeatmapComparisonService.normalize_map(q)

        # KL divergence
        kl = np.sum(p_norm * np.log((p_norm + eps) / (q_norm + eps)))
        return float(kl)

    @staticmethod
    def similarity(map1: np.ndarray, map2: np.ndarray) -> float:
        """Calculate histogram intersection similarity.

        SIM measures overlap between two distributions.
        Range: [0, 1], higher is better.

        Args:
            map1: First heatmap.
            map2: Second heatmap.

        Returns:
            Similarity score.
        """
        p = HeatmapComparisonService.normalize_map(map1)
        q = HeatmapComparisonService.normalize_map(map2)
        return float(np.minimum(p, q).sum())

    @staticmethod
    def nss(saliency_map: np.ndarray, fixation_map: np.ndarray) -> float:
        """Calculate Normalized Scanpath Saliency.

        NSS evaluates saliency values at fixation locations.
        Higher values indicate better prediction of fixations.

        Args:
            saliency_map: Predicted saliency.
            fixation_map: Binary fixation map (or continuous attention).

        Returns:
            NSS score.
        """
        # Normalize saliency to zero mean and unit std
        s = (saliency_map - saliency_map.mean()) / (saliency_map.std() + 1e-8)

        # Get fixation points (threshold if continuous)
        if fixation_map.max() > 1:
            fixation_map = fixation_map / fixation_map.max()
        fixations = fixation_map > 0.5

        if fixations.sum() == 0:
            return 0.0

        # NSS is mean of normalized saliency at fixation points
        return float(s[fixations].mean())

    def compare(
        self,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
    ) -> dict:
        """Compare two heatmaps and return all metrics.

        Args:
            ground_truth: Reference heatmap (DeepGaze).
            prediction: Predicted heatmap (Hybrid).

        Returns:
            Dict with all comparison metrics.
        """
        # Ensure same shape
        if ground_truth.shape != prediction.shape:
            prediction_img = Image.fromarray((prediction * 255).astype(np.uint8))
            prediction_img = prediction_img.resize(
                (ground_truth.shape[1], ground_truth.shape[0]),
                Image.Resampling.BILINEAR,
            )
            prediction = np.array(prediction_img).astype(np.float32) / 255.0

        cc = self.correlation_coefficient(ground_truth, prediction)
        kl = self.kl_divergence(ground_truth, prediction)
        sim = self.similarity(ground_truth, prediction)
        nss = self.nss(prediction, ground_truth)

        # Calculate alignment percentage (CC-based, 0-100)
        alignment = max(0, cc) * 100

        return {
            "correlation_coefficient": round(cc, 4),
            "kl_divergence": round(kl, 4),
            "similarity": round(sim, 4),
            "nss": round(nss, 4),
            "alignment_percentage": round(alignment, 1),
            "verdict": self._generate_verdict(cc, kl, sim),
        }

    def _generate_verdict(self, cc: float, kl: float, sim: float) -> str:
        """Generate human-readable verdict based on metrics.

        Args:
            cc: Correlation coefficient.
            kl: KL divergence.
            sim: Similarity score.

        Returns:
            Verdict string.
        """
        if cc >= 0.8 and sim >= 0.7:
            return "Excelente: El modelo híbrido está muy alineado con DeepGaze"
        elif cc >= 0.6 and sim >= 0.5:
            return "Bueno: El modelo híbrido captura las principales áreas de atención"
        elif cc >= 0.4:
            return "Moderado: Hay diferencias significativas entre los modelos"
        else:
            return "Bajo: Los modelos difieren considerablemente"
