"""
Python Client SDK for YOLO-World Large API Service.
Allows agents and pipelines to query open-vocabulary detections with 1 line of code.
"""

import io
import base64
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from PIL import Image
import requests

class YoloWorldClient:
    """Client for querying the YOLO-World Object Detection API."""

    def __init__(self, base_url: str = "http://localhost:8005"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> Dict[str, Any]:
        """Check server status, model details, and GPU memory."""
        res = requests.get(f"{self.base_url}/health", timeout=10)
        res.raise_for_status()
        return res.json()

    def set_classes(self, classes: List[str]) -> Dict[str, Any]:
        """Update persistent open-vocabulary categories on the server."""
        res = requests.post(
            f"{self.base_url}/set_classes",
            json={"classes": classes},
            timeout=15
        )
        res.raise_for_status()
        return res.json()

    def detect(
        self,
        image: Union[str, Path, bytes, Image.Image],
        classes: Optional[List[str]] = None,
        confidence: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        imgsz: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run open-vocabulary detection on an image.
        
        Args:
            image: Path to image file, raw image bytes, or PIL Image.
            classes: Optional list of prompt categories (e.g. ['candlestick', 'support line']).
            confidence: Optional confidence threshold (e.g. 0.20).
            iou_threshold: Optional NMS IoU threshold (e.g. 0.45).
            imgsz: Optional inference resolution (e.g. 640 or 1280).
            
        Returns:
            Dict containing 'detections' (with pixel box and normalized box [ymin, xmin, ymax, xmax]),
            'image_width', 'image_height', 'inference_time_ms', and 'active_classes'.
        """
        # Prepare multipart file upload
        if isinstance(image, (str, Path)):
            with open(image, "rb") as f:
                img_bytes = f.read()
        elif isinstance(image, Image.Image):
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            img_bytes = buf.getvalue()
        elif isinstance(image, bytes):
            img_bytes = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
        data = {}
        if classes:
            data["classes"] = ",".join(classes)
        if confidence is not None:
            data["confidence"] = str(confidence)
        if iou_threshold is not None:
            data["iou_threshold"] = str(iou_threshold)
        if imgsz is not None:
            data["imgsz"] = str(imgsz)

        res = requests.post(
            f"{self.base_url}/detect",
            files=files,
            data=data,
            timeout=30
        )
        res.raise_for_status()
        return res.json()

    def detect_and_save_visual(
        self,
        image: Union[str, Path, bytes, Image.Image],
        output_path: Union[str, Path],
        classes: Optional[List[str]] = None,
        confidence: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        imgsz: Optional[int] = None
    ) -> Path:
        """Runs detection and saves annotated image with bounding boxes to disk."""
        if isinstance(image, (str, Path)):
            with open(image, "rb") as f:
                img_bytes = f.read()
        elif isinstance(image, Image.Image):
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            img_bytes = buf.getvalue()
        elif isinstance(image, bytes):
            img_bytes = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
        data = {}
        if classes:
            data["classes"] = ",".join(classes)
        if confidence is not None:
            data["confidence"] = str(confidence)
        if iou_threshold is not None:
            data["iou_threshold"] = str(iou_threshold)
        if imgsz is not None:
            data["imgsz"] = str(imgsz)

        res = requests.post(
            f"{self.base_url}/detect/visualize",
            files=files,
            data=data,
            timeout=30
        )
        res.raise_for_status()
        
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_bytes(res.content)
        return out_p
