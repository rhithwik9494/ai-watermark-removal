"""
IOPaint API client for AI-based image inpainting.
Handles multipart form submission and raw binary response decoding.
"""

import io
import time
import requests
import numpy as np
import cv2
from typing import Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.utils.logger import get_logger

logger = get_logger(__name__)


class IOPaintClient:
    """
    Client for IOPaint inpainting API.
    
    Features:
    - Multipart form submission (image + mask)
    - Raw binary response handling
    - Automatic retry with exponential backoff
    - Connection pooling for performance
    """
    
    DEFAULT_URL = "http://127.0.0.1:8080/api/v1/inpaint"
    DEFAULT_MODEL = "lama"
    DEFAULT_TIMEOUT = 120
    MAX_RETRIES = 3
    
    def __init__(self, 
                 api_url: str = DEFAULT_URL,
                 model: str = DEFAULT_MODEL,
                 timeout: int = DEFAULT_TIMEOUT):
        """
        Initialize IOPaint client.
        
        Args:
            api_url: IOPaint API endpoint URL
            model: Model name to use (e.g., 'lama', 'ldm', 'mat')
            timeout: Request timeout in seconds
        """
        self.api_url = api_url
        self.model = model
        self.timeout = timeout
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=self.MAX_RETRIES,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=20)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        logger.info(f"IOPaint client initialized: url={api_url}, model={model}")
    
    def inpaint(self, 
                image: np.ndarray, 
                mask: np.ndarray,
                resize: Optional[tuple] = None,
                dilate_mask: int = 0) -> Optional[np.ndarray]:
        """
        Inpaint image using mask via IOPaint API.
        
        Args:
            image: Input image (BGR, uint8)
            mask: Binary mask (uint8, 255 = inpaint region)
            resize: Optional (width, height) to resize before sending
            dilate_mask: Pixels to dilate mask before inpainting (ensures full coverage)
            
        Returns:
            Inpainted image (BGR, uint8) or None on failure
        """
        start_time = time.time()
        
        # Optionally resize for faster processing
        original_shape = image.shape[:2]
        if resize is not None:
            image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, resize, interpolation=cv2.INTER_NEAREST)
        
        # Dilate mask to ensure full watermark coverage
        if dilate_mask > 0:
            kernel = np.ones((dilate_mask, dilate_mask), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Encode image to PNG bytes
        success, img_encoded = cv2.imencode(".png", image)
        if not success:
            logger.error("Failed to encode image to PNG")
            return None
        
        # Encode mask to PNG bytes
        success, mask_encoded = cv2.imencode(".png", mask)
        if not success:
            logger.error("Failed to encode mask to PNG")
            return None
        
        # Try multipart form data first, fallback to base64 JSON
        result_img = self._inpaint_multipart(image, mask, img_encoded, mask_encoded)
        
        if result_img is None:
            logger.warning("Multipart upload failed, trying base64 JSON fallback")
            result_img = self._inpaint_base64(img_encoded, mask_encoded)
        
        if result_img is None:
            return None
        
        # Resize back to original if needed
        if resize is not None:
            result_img = cv2.resize(result_img, (original_shape[1], original_shape[0]), 
                                    interpolation=cv2.INTER_LANCZOS4)
        
        elapsed = time.time() - start_time
        logger.debug(f"Inpaint completed in {elapsed:.2f}s")
        
        return result_img
    
    def _inpaint_multipart(self, image, mask, img_encoded, mask_encoded) -> Optional[np.ndarray]:
        """Try multipart form-data upload."""
        files = {
            "image": ("image.png", io.BytesIO(img_encoded.tobytes()), "image/png"),
            "mask": ("mask.png", io.BytesIO(mask_encoded.tobytes()), "image/png")
        }
        
        data = {
            "model": self.model
        }
        
        try:
            logger.debug(f"Sending multipart inpaint request to {self.api_url}")
            
            response = self.session.post(
                self.api_url,
                files=files,
                data=data,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                logger.warning(f"Multipart API error: status={response.status_code}, body={response.text[:200]}")
                return None
            
            # Decode raw binary response
            result = np.frombuffer(response.content, np.uint8)
            result_img = cv2.imdecode(result, cv2.IMREAD_COLOR)
            
            if result_img is None:
                logger.warning("Failed to decode multipart API response image")
                return None
            
            return result_img
            
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Multipart connection error: {e}")
            return None
        except requests.exceptions.Timeout as e:
            logger.warning(f"Multipart timeout after {self.timeout}s: {e}")
            return None
        except Exception as e:
            logger.warning(f"Multipart unexpected error: {e}")
            return None
    
    def _inpaint_base64(self, img_encoded, mask_encoded) -> Optional[np.ndarray]:
        """Fallback: base64 JSON upload using pre-encoded buffers."""
        try:
            import base64
            
            img_b64 = base64.b64encode(img_encoded).decode("utf-8")
            mask_b64 = base64.b64encode(mask_encoded).decode("utf-8")
            
            payload = {
                "image": img_b64,
                "mask": mask_b64,
                "model": self.model
            }
            
            logger.debug(f"Sending base64 JSON inpaint request to {self.api_url}")
            
            response = self.session.post(
                self.api_url,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                logger.warning(f"Base64 API error: status={response.status_code}, body={response.text[:200]}")
                return None
            
            # Decode response
            result = np.frombuffer(response.content, np.uint8)
            result_img = cv2.imdecode(result, cv2.IMREAD_COLOR)
            
            if result_img is None:
                logger.warning("Failed to decode base64 API response image")
                return None
            
            return result_img
            
        except Exception as e:
            logger.warning(f"Base64 fallback error: {e}")
            return None
    
    def health_check(self) -> bool:
        """
        Check if IOPaint API is reachable.
        
        Returns:
            True if API is available
        """
        try:
            # Try a simple GET to base URL or just check if port is open
            response = self.session.get(
                self.api_url.replace("/api/v1/inpaint", "/"),
                timeout=5
            )
            return response.status_code < 500
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    def close(self):
        """Close HTTP session."""
        self.session.close()
        logger.info("IOPaint client session closed")

