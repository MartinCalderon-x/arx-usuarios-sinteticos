"""Hybrid Heatmap Generation Service.

This module implements a hybrid approach for generating visual attention
heatmaps by combining Gemini Vision's semantic analysis with gaussian
interpolation techniques.

The hybrid model offers a lightweight alternative to deep learning models
like DeepGaze, trading some accuracy for significantly reduced latency
and infrastructure requirements.

Architecture v2 (Improved):
    1. Itti-Koch Bottom-Up Saliency:
       - Intensity channel (luminance)
       - Color channels (R-G, B-Y opponent)
       - Orientation channels (Gabor filters 0°, 45°, 90°, 135°)
       - Center-surround operations for contrast detection
    2. Top-Down Semantic Analysis:
       - Gemini Vision identifies Areas of Interest (AOI)
       - Each AOI is converted to a weighted gaussian distribution
    3. Specialized Detectors:
       - Face detection with evolutionary bias
       - Text detection with cognitive priority
    4. Fusion Layer:
       - Weighted combination: α·bottom_up + β·top_down + γ·detectors

References:
    - Itti, Koch, Niebur (1998): A model of saliency-based visual attention
    - Gaussian mixture models for saliency: Itti & Koch (2001)
    - Center bias in fixations: Tatler (2007)
    - Face detection bias: Cerf et al. (2009)
"""
import io
import base64
import logging
import time
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

logger = logging.getLogger(__name__)


class IttiKochSaliency:
    """Itti-Koch-Niebur (1998) bottom-up saliency model.

    Implements the classic computational model of visual attention that
    extracts low-level features and computes saliency through center-surround
    operations.

    Channels:
        - Intensity: Luminance (grayscale)
        - Color: R-G and B-Y opponent channels
        - Orientation: Gabor filters at 0°, 45°, 90°, 135°

    References:
        Itti, L., Koch, C., & Niebur, E. (1998). A model of saliency-based
        visual attention for rapid scene analysis. IEEE TPAMI.
    """

    def __init__(
        self,
        num_scales: int = 4,
        gabor_ksize: int = 31,
        gabor_sigma: float = 4.0,
        gabor_lambda: float = 10.0,
        gabor_gamma: float = 0.5,
    ):
        """Initialize Itti-Koch saliency extractor.

        Args:
            num_scales: Number of pyramid scales for center-surround.
            gabor_ksize: Gabor kernel size.
            gabor_sigma: Gabor sigma parameter.
            gabor_lambda: Gabor wavelength.
            gabor_gamma: Gabor aspect ratio.
        """
        self.num_scales = num_scales
        self.gabor_ksize = gabor_ksize
        self.gabor_sigma = gabor_sigma
        self.gabor_lambda = gabor_lambda
        self.gabor_gamma = gabor_gamma

    def compute(self, image: np.ndarray) -> np.ndarray:
        """Compute full Itti-Koch saliency map.

        Args:
            image: RGB image as numpy array (H, W, 3) with values in [0, 255].

        Returns:
            Saliency map normalized to [0, 1].
        """
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available, returning zero saliency")
            return np.zeros(image.shape[:2], dtype=np.float32)

        # Ensure float32 and normalize to [0, 1]
        img = image.astype(np.float32) / 255.0

        # Extract channels
        intensity_map = self._intensity_channel(img)
        rg_map, by_map = self._color_channels(img)
        orientation_maps = self._orientation_channels(img)

        # Combine all feature maps
        combined = intensity_map + rg_map + by_map
        for o_map in orientation_maps:
            combined += o_map

        # Normalize to [0, 1]
        combined = self._normalize(combined)

        return combined

    def _intensity_channel(self, img: np.ndarray) -> np.ndarray:
        """Extract intensity (luminance) channel.

        Args:
            img: RGB image normalized to [0, 1].

        Returns:
            Intensity saliency map.
        """
        # Convert to grayscale using standard weights
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        return self._center_surround(gray)

    def _color_channels(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract color opponent channels (R-G and B-Y).

        Args:
            img: RGB image normalized to [0, 1].

        Returns:
            Tuple of (R-G saliency, B-Y saliency).
        """
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        intensity = (r + g + b) / 3.0 + 1e-8

        # Normalize colors by intensity (as in original paper)
        r_norm = r / intensity
        g_norm = g / intensity
        b_norm = b / intensity

        # Red-Green opponent
        rg = np.abs(r_norm - g_norm)

        # Blue-Yellow opponent
        by = np.abs(b_norm - (r_norm + g_norm) / 2.0)

        # Apply center-surround to each
        rg_saliency = self._center_surround(rg)
        by_saliency = self._center_surround(by)

        return rg_saliency, by_saliency

    def _orientation_channels(self, img: np.ndarray) -> List[np.ndarray]:
        """Extract orientation channels using Gabor filters.

        Args:
            img: RGB image normalized to [0, 1].

        Returns:
            List of orientation saliency maps for 0°, 45°, 90°, 135°.
        """
        # Convert to grayscale
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray = gray.astype(np.float32) / 255.0

        orientations = [0, 45, 90, 135]
        orientation_maps = []

        for theta_deg in orientations:
            theta = np.deg2rad(theta_deg)
            gabor_kernel = cv2.getGaborKernel(
                ksize=(self.gabor_ksize, self.gabor_ksize),
                sigma=self.gabor_sigma,
                theta=theta,
                lambd=self.gabor_lambda,
                gamma=self.gabor_gamma,
                psi=0,
                ktype=cv2.CV_32F,
            )
            response = cv2.filter2D(gray, cv2.CV_32F, gabor_kernel)
            response = np.abs(response)
            orientation_maps.append(self._center_surround(response))

        return orientation_maps

    def _center_surround(self, feature_map: np.ndarray) -> np.ndarray:
        """Compute center-surround contrast at multiple scales.

        The center-surround operation detects local contrasts by comparing
        fine (center) and coarse (surround) scale representations.

        Args:
            feature_map: Single channel feature map.

        Returns:
            Center-surround saliency map.
        """
        # Build Gaussian pyramid
        pyramid = [feature_map]
        current = feature_map
        for _ in range(self.num_scales):
            current = cv2.pyrDown(current)
            pyramid.append(current)

        # Compute center-surround differences
        cs_maps = []
        for c in range(2, min(5, len(pyramid))):  # Center scales
            for delta in [3, 4]:  # Surround offset
                s = c + delta
                if s >= len(pyramid):
                    continue

                center = pyramid[c]
                surround = pyramid[s]

                # Resize surround to center size
                surround_resized = cv2.resize(
                    surround, (center.shape[1], center.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )

                # Absolute difference
                cs_map = np.abs(center - surround_resized)
                cs_maps.append(cs_map)

        if not cs_maps:
            return self._normalize(feature_map)

        # Resize all to original size and combine
        result = np.zeros_like(feature_map)
        for cs_map in cs_maps:
            resized = cv2.resize(
                cs_map, (feature_map.shape[1], feature_map.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
            result += resized

        return self._normalize(result)

    def _normalize(self, feature_map: np.ndarray) -> np.ndarray:
        """Normalize feature map to [0, 1] with N(.) operator.

        Uses iterative normalization as described in Itti & Koch (2001).

        Args:
            feature_map: Input feature map.

        Returns:
            Normalized feature map in [0, 1].
        """
        if feature_map.max() == feature_map.min():
            return np.zeros_like(feature_map)

        # Standard min-max normalization
        normalized = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)

        # Apply N(.) operator: promote maps with few strong peaks
        # by multiplying by (max - mean)^2
        mean_val = normalized.mean()
        max_val = normalized.max()
        normalized *= (max_val - mean_val) ** 2

        # Re-normalize to [0, 1]
        if normalized.max() > 0:
            normalized = normalized / normalized.max()

        return normalized.astype(np.float32)


class FaceTextDetector:
    """Detector for faces and text regions with attention weights.

    Human attention has evolutionary bias toward faces and cognitive
    priority for text/symbols. This detector identifies these regions
    and provides attention weight masks.

    Uses MediaPipe Face Detection as primary detector (more accurate),
    with OpenCV Haar Cascades as fallback.

    References:
        - Cerf, M., et al. (2009). Faces and text attract gaze.
        - Rayner, K. (1998). Eye movements in reading and information processing.
    """

    def __init__(
        self,
        face_weight: float = 1.8,
        text_weight: float = 1.4,
        face_scale_factor: float = 1.1,
        face_min_neighbors: int = 5,
        min_detection_confidence: float = 0.5,
    ):
        """Initialize face and text detector.

        Args:
            face_weight: Attention multiplier for face regions.
            text_weight: Attention multiplier for text regions.
            face_scale_factor: Scale factor for Haar cascade (fallback).
            face_min_neighbors: Minimum neighbors for Haar cascade.
            min_detection_confidence: Minimum confidence for MediaPipe detection.
        """
        self.face_weight = face_weight
        self.text_weight = text_weight
        self.face_scale_factor = face_scale_factor
        self.face_min_neighbors = face_min_neighbors
        self.min_detection_confidence = min_detection_confidence

        # Initialize MediaPipe Face Detection (primary)
        self._mp_face_detection = None
        if MEDIAPIPE_AVAILABLE:
            try:
                self._mp_face_detection = mp.solutions.face_detection.FaceDetection(
                    model_selection=1,  # 1 = full range model (better for distant faces)
                    min_detection_confidence=min_detection_confidence,
                )
                logger.info("MediaPipe Face Detection initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize MediaPipe: {e}")

        # Load Haar cascade classifier (fallback)
        self._face_cascade = None
        if CV2_AVAILABLE:
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self._face_cascade = cv2.CascadeClassifier(cascade_path)
            except Exception as e:
                logger.warning(f"Failed to load face cascade: {e}")

    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """Detect faces in image using MediaPipe (primary) or Haar (fallback).

        Args:
            image: RGB image as numpy array.

        Returns:
            List of face regions with bounding boxes and confidence.
        """
        # Try MediaPipe first (more accurate)
        if self._mp_face_detection is not None:
            faces = self._detect_faces_mediapipe(image)
            if faces:
                logger.debug(f"MediaPipe detected {len(faces)} face(s)")
                return faces

        # Fallback to Haar Cascade
        if self._face_cascade is not None:
            faces = self._detect_faces_haar(image)
            if faces:
                logger.debug(f"Haar Cascade detected {len(faces)} face(s)")
                return faces

        return []

    def _detect_faces_mediapipe(self, image: np.ndarray) -> List[dict]:
        """Detect faces using MediaPipe Face Detection.

        MediaPipe provides:
        - Higher accuracy than Haar cascades
        - Confidence scores
        - Key facial landmarks (eyes, nose, mouth, ears)

        Args:
            image: RGB image as numpy array.

        Returns:
            List of face dicts with bounding box, confidence, and landmarks.
        """
        if self._mp_face_detection is None:
            return []

        try:
            h, w = image.shape[:2]
            results = self._mp_face_detection.process(image.astype(np.uint8))

            if not results.detections:
                return []

            faces = []
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box

                # Convert relative coordinates to pixels
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                # Ensure bounds are within image
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)

                # Get confidence score
                confidence = detection.score[0] if detection.score else 0.5

                # Extract key landmarks (relative to bounding box)
                landmarks = {}
                if detection.location_data.relative_keypoints:
                    keypoint_names = ['right_eye', 'left_eye', 'nose_tip',
                                     'mouth_center', 'right_ear', 'left_ear']
                    for i, kp in enumerate(detection.location_data.relative_keypoints):
                        if i < len(keypoint_names):
                            landmarks[keypoint_names[i]] = {
                                'x': int(kp.x * w),
                                'y': int(kp.y * h)
                            }

                faces.append({
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "confidence": round(confidence, 3),
                    "detector": "mediapipe",
                    "landmarks": landmarks,
                })

            return faces

        except Exception as e:
            logger.warning(f"MediaPipe face detection failed: {e}")
            return []

    def _detect_faces_haar(self, image: np.ndarray) -> List[dict]:
        """Detect faces using OpenCV Haar Cascade (fallback).

        Args:
            image: RGB image as numpy array.

        Returns:
            List of face regions with bounding boxes.
        """
        if not CV2_AVAILABLE or self._face_cascade is None:
            return []

        try:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)

            faces = self._face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.face_scale_factor,
                minNeighbors=self.face_min_neighbors,
                minSize=(30, 30),
            )

            return [
                {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "confidence": 0.7,  # Haar doesn't provide confidence
                    "detector": "haar",
                }
                for (x, y, w, h) in faces
            ]

        except Exception as e:
            logger.warning(f"Haar face detection failed: {e}")
            return []

    def detect_text(self, image: np.ndarray) -> List[dict]:
        """Detect text regions in image using OCR.

        Args:
            image: RGB image as numpy array.

        Returns:
            List of text regions with bounding boxes.
        """
        if not TESSERACT_AVAILABLE:
            logger.debug("Tesseract not available, skipping text detection")
            return []

        try:
            # Get bounding boxes from Tesseract
            data = pytesseract.image_to_data(
                Image.fromarray(image.astype(np.uint8)),
                output_type=pytesseract.Output.DICT,
            )

            text_regions = []
            for i, conf in enumerate(data["conf"]):
                # Filter low confidence and empty text
                if int(conf) < 30 or not data["text"][i].strip():
                    continue

                x, y = data["left"][i], data["top"][i]
                w, h = data["width"][i], data["height"][i]

                if w > 5 and h > 5:  # Filter tiny boxes
                    text_regions.append({
                        "x": x, "y": y,
                        "width": w, "height": h,
                        "text": data["text"][i],
                        "confidence": int(conf),
                    })

            return text_regions

        except Exception as e:
            logger.warning(f"Text detection failed: {e}")
            return []

    def generate_attention_mask(
        self,
        image: np.ndarray,
        detect_faces: bool = True,
        detect_text: bool = True,
    ) -> np.ndarray:
        """Generate attention weight mask for faces and text.

        When MediaPipe is available, also adds extra attention to:
        - Eyes (highest attention - humans look at eyes first)
        - Nose and mouth (secondary facial features)

        Args:
            image: RGB image as numpy array.
            detect_faces: Whether to detect and weight faces.
            detect_text: Whether to detect and weight text.

        Returns:
            Attention multiplier mask (values >= 1.0).
        """
        height, width = image.shape[:2]
        mask = np.ones((height, width), dtype=np.float32)
        yy, xx = np.ogrid[:height, :width]

        if detect_faces:
            faces = self.detect_faces(image)
            for face in faces:
                x, y, w, h = face["x"], face["y"], face["width"], face["height"]
                confidence = face.get("confidence", 0.7)

                # Scale face weight by detection confidence
                effective_weight = 1.0 + (self.face_weight - 1.0) * confidence

                # Expand face region slightly (faces attract attention around them)
                expand = int(min(w, h) * 0.2)
                x1 = max(0, x - expand)
                y1 = max(0, y - expand)
                x2 = min(width, x + w + expand)
                y2 = min(height, y + h + expand)

                # Apply gaussian-weighted face attention
                cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
                sigma = max(w, h) * 0.5
                gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
                mask += gaussian * (effective_weight - 1.0)

                # If MediaPipe landmarks available, add extra weight to eyes
                # Eyes are the most attention-grabbing facial feature
                landmarks = face.get("landmarks", {})
                if landmarks:
                    eye_weight = 2.2  # Eyes get highest attention
                    eye_sigma = max(w, h) * 0.15  # Smaller sigma for precise eye focus

                    for eye_key in ["left_eye", "right_eye"]:
                        if eye_key in landmarks:
                            ex, ey = landmarks[eye_key]["x"], landmarks[eye_key]["y"]
                            eye_gaussian = np.exp(-((xx - ex) ** 2 + (yy - ey) ** 2) / (2 * eye_sigma ** 2))
                            mask += eye_gaussian * (eye_weight - 1.0)

                    # Nose and mouth get moderate extra attention
                    mouth_weight = 1.5
                    mouth_sigma = max(w, h) * 0.2

                    for feature_key in ["nose_tip", "mouth_center"]:
                        if feature_key in landmarks:
                            fx, fy = landmarks[feature_key]["x"], landmarks[feature_key]["y"]
                            feature_gaussian = np.exp(-((xx - fx) ** 2 + (yy - fy) ** 2) / (2 * mouth_sigma ** 2))
                            mask += feature_gaussian * (mouth_weight - 1.0)

            if faces:
                detector_used = faces[0].get("detector", "unknown") if faces else "none"
                logger.debug(f"Detected {len(faces)} face(s) using {detector_used}")

        if detect_text:
            text_regions = self.detect_text(image)

            # Group nearby text regions
            for region in text_regions:
                x, y, w, h = region["x"], region["y"], region["width"], region["height"]
                x2, y2 = min(width, x + w), min(height, y + h)
                x, y = max(0, x), max(0, y)

                # Apply text attention
                mask[y:y2, x:x2] = np.maximum(
                    mask[y:y2, x:x2],
                    self.text_weight
                )

            if text_regions:
                logger.debug(f"Detected {len(text_regions)} text region(s)")

        return mask


class HybridHeatmapService:
    """Service for generating heatmaps using Gemini AOI + Gaussian interpolation.

    This service provides a fast, lightweight alternative to deep learning
    saliency models by leveraging Gemini's visual understanding capabilities
    combined with classical gaussian blending.

    Version 2.0 adds:
        - Itti-Koch bottom-up saliency (intensity, color, orientation)
        - Face detection with evolutionary attention bias
        - Text/OCR detection with cognitive priority
        - Weighted fusion of bottom-up and top-down features

    Attributes:
        default_sigma_factor: Factor to compute sigma from AOI size.
        center_bias_strength: Strength of center prior (0-1).
        min_sigma: Minimum gaussian sigma in pixels.
        bottom_up_weight: Weight for Itti-Koch saliency (0-1).
        top_down_weight: Weight for Gemini AOI saliency (0-1).

    Example:
        >>> service = HybridHeatmapService()
        >>> # V1: AOI-only mode (backward compatible)
        >>> heatmap = service.generate_from_aoi(aoi_data, width=1024, height=768)
        >>> # V2: Full hybrid mode with image
        >>> heatmap = service.generate_hybrid(image, aoi_data)
    """

    def __init__(
        self,
        default_sigma_factor: float = 0.5,
        center_bias_strength: float = 0.15,
        min_sigma: int = 20,
        bottom_up_weight: float = 0.35,
        top_down_weight: float = 0.50,
        detector_weight: float = 0.15,
        enable_face_detection: bool = True,
        enable_text_detection: bool = True,
    ):
        """Initialize hybrid heatmap service.

        Args:
            default_sigma_factor: Multiplier for AOI size to get sigma.
            center_bias_strength: Weight of center gaussian prior (0-1).
            min_sigma: Minimum sigma value to prevent too sharp peaks.
            bottom_up_weight: Weight for Itti-Koch bottom-up saliency (0-1).
            top_down_weight: Weight for Gemini AOI top-down saliency (0-1).
            detector_weight: Weight for face/text detectors (0-1).
            enable_face_detection: Whether to detect faces.
            enable_text_detection: Whether to detect text (requires Tesseract).
        """
        self.default_sigma_factor = default_sigma_factor
        self.center_bias_strength = center_bias_strength
        self.min_sigma = min_sigma

        # Fusion weights (should sum to ~1.0 for balanced contribution)
        self.bottom_up_weight = bottom_up_weight
        self.top_down_weight = top_down_weight
        self.detector_weight = detector_weight

        # Feature extractors
        self.enable_face_detection = enable_face_detection
        self.enable_text_detection = enable_text_detection

        # Initialize sub-components lazily
        self._itti_koch: Optional[IttiKochSaliency] = None
        self._face_text_detector: Optional[FaceTextDetector] = None

    def _get_itti_koch(self) -> IttiKochSaliency:
        """Get or create Itti-Koch saliency extractor."""
        if self._itti_koch is None:
            self._itti_koch = IttiKochSaliency()
        return self._itti_koch

    def _get_face_text_detector(self) -> FaceTextDetector:
        """Get or create face/text detector."""
        if self._face_text_detector is None:
            self._face_text_detector = FaceTextDetector()
        return self._face_text_detector

    def generate_hybrid(
        self,
        image: np.ndarray,
        aoi_data: list[dict],
        include_center_bias: bool = True,
        include_bottom_up: bool = True,
        include_detectors: bool = True,
    ) -> Tuple[np.ndarray, dict]:
        """Generate heatmap using full hybrid model (v2).

        Combines:
            1. Itti-Koch bottom-up saliency (low-level features)
            2. Gemini AOI top-down attention (semantic understanding)
            3. Face/text detectors (specialized attention biases)
            4. Center bias prior

        Args:
            image: RGB image as numpy array (H, W, 3).
            aoi_data: List of AOI dicts from Gemini Vision.
            include_center_bias: Whether to add center viewing bias.
            include_bottom_up: Whether to compute Itti-Koch saliency.
            include_detectors: Whether to run face/text detection.

        Returns:
            Tuple of (heatmap array, metadata dict).
            heatmap: numpy array of shape (H, W) with values in [0, 1].
            metadata: dict with timing and component information.

        Example:
            >>> image = np.array(Image.open("test.jpg"))
            >>> aoi_data = gemini_service.analyze(image)
            >>> heatmap, meta = service.generate_hybrid(image, aoi_data)
        """
        start_time = time.time()
        height, width = image.shape[:2]

        metadata = {
            "mode": "hybrid_v2",
            "components": [],
            "weights": {
                "bottom_up": self.bottom_up_weight,
                "top_down": self.top_down_weight,
                "detectors": self.detector_weight,
            },
        }

        # Initialize components
        combined = np.zeros((height, width), dtype=np.float32)

        # 1. Bottom-up: Itti-Koch saliency
        bottom_up_map = None
        if include_bottom_up and self.bottom_up_weight > 0:
            itti_start = time.time()
            itti_koch = self._get_itti_koch()
            bottom_up_map = itti_koch.compute(image)
            metadata["components"].append("itti_koch")
            metadata["itti_koch_ms"] = round((time.time() - itti_start) * 1000, 2)

        # 2. Top-down: Gemini AOI gaussians
        top_down_map = None
        if aoi_data and self.top_down_weight > 0:
            aoi_start = time.time()
            top_down_map = self._generate_top_down_map(aoi_data, width, height)
            metadata["components"].append("gemini_aoi")
            metadata["aoi_count"] = len(aoi_data)
            metadata["aoi_ms"] = round((time.time() - aoi_start) * 1000, 2)

        # 3. Face/text detectors
        detector_mask = None
        if include_detectors and self.detector_weight > 0:
            detector_start = time.time()
            detector = self._get_face_text_detector()
            detector_mask = detector.generate_attention_mask(
                image,
                detect_faces=self.enable_face_detection,
                detect_text=self.enable_text_detection,
            )
            # Convert mask (multiplier) to saliency contribution
            # Only add where mask > 1 (detected regions)
            detector_saliency = np.maximum(0, detector_mask - 1.0)
            if detector_saliency.max() > 0:
                detector_saliency = detector_saliency / detector_saliency.max()

            metadata["components"].append("face_text_detector")
            metadata["detector_ms"] = round((time.time() - detector_start) * 1000, 2)
        else:
            detector_saliency = np.zeros((height, width), dtype=np.float32)

        # 4. Weighted fusion
        total_weight = 0.0

        if bottom_up_map is not None:
            combined += bottom_up_map * self.bottom_up_weight
            total_weight += self.bottom_up_weight

        if top_down_map is not None:
            combined += top_down_map * self.top_down_weight
            total_weight += self.top_down_weight

        if detector_saliency is not None and detector_saliency.max() > 0:
            combined += detector_saliency * self.detector_weight
            total_weight += self.detector_weight

        # Normalize by total weight
        if total_weight > 0:
            combined = combined / total_weight

        # 5. Apply center bias
        if include_center_bias and self.center_bias_strength > 0:
            center_bias = self._generate_center_bias(width, height)
            combined = combined * (1 - self.center_bias_strength) + center_bias * self.center_bias_strength
            metadata["components"].append("center_bias")

        # 6. Apply detector mask as multiplier (boost detected regions)
        if detector_mask is not None and include_detectors:
            combined = combined * detector_mask

        # Final normalization to [0, 1]
        if combined.max() > combined.min():
            combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)

        elapsed = (time.time() - start_time) * 1000
        metadata["total_ms"] = round(elapsed, 2)

        logger.info(
            f"Hybrid v2 heatmap generated in {elapsed:.2f}ms "
            f"(components: {', '.join(metadata['components'])})"
        )

        return combined, metadata

    def _generate_top_down_map(
        self,
        aoi_data: list[dict],
        width: int,
        height: int,
    ) -> np.ndarray:
        """Generate top-down attention map from AOI data.

        Args:
            aoi_data: List of AOI dicts from Gemini Vision.
            width: Image width.
            height: Image height.

        Returns:
            Top-down attention map normalized to [0, 1].
        """
        heatmap = np.zeros((height, width), dtype=np.float32)

        for aoi in aoi_data:
            gaussian = self._aoi_to_gaussian(aoi, width, height)
            heatmap += gaussian

        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap

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
