"""
Production FastAPI service for YOLO-World Open-Vocabulary Object Detection.
Supports zero-shot open-vocabulary detection, dynamic class setting,
bounding box normalization for VLM/agent pipelines, and visualization.
"""

import io
import os
import time
import base64
import logging
from typing import List, Optional, Dict, Any, Tuple
from contextlib import asynccontextmanager

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from ultralytics import YOLOWorld

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("yolo_world_service")

# Configuration via environment variables
MODEL_NAME = os.environ.get("YOLO_MODEL_NAME", "yolov8l-worldv2.pt")
DEVICE = os.environ.get("YOLO_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CONF = float(os.environ.get("YOLO_DEFAULT_CONF", "0.20"))
DEFAULT_IOU = float(os.environ.get("YOLO_DEFAULT_IOU", "0.45"))
DEFAULT_IMGSZ = int(os.environ.get("YOLO_DEFAULT_IMGSZ", "640"))

DEFAULT_CLASSES = [
    "candlestick",
    "chart",
    "price axis",
    "timeline axis",
    "volume bar",
    "indicator pane",
    "trendline",
    "support level",
    "resistance level",
    "breakout candle",
    "arrow",
    "text label"
]

# Global model state
class ModelManager:
    def __init__(self):
        self.model: Optional[YOLOWorld] = None
        self.current_classes: List[str] = DEFAULT_CLASSES
        self.device: str = DEVICE

    def load(self):
        logger.info(f"Loading YOLO-World model: {MODEL_NAME} onto {self.device}...")
        self.model = YOLOWorld(MODEL_NAME)
        self.model.to(self.device)
        self.set_classes(self.current_classes)
        logger.info(f"Model loaded successfully with {len(self.current_classes)} default classes.")

    def set_classes(self, classes: List[str]):
        if not classes:
            return
        self.current_classes = [c.strip() for c in classes if c.strip()]
        self.model.set_classes(self.current_classes)
        logger.info(f"YOLO-World vocabulary set to: {self.current_classes}")

manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    manager.load()
    yield

app = FastAPI(
    title="YOLO-World Large API",
    description="Open-Vocabulary Object Detection Microservice using YOLO-World Large",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schemas
class SetClassesRequest(BaseModel):
    classes: List[str] = Field(..., description="List of category names or open-vocabulary prompts to detect")

class DetectionItem(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    box_xyxy: List[float] = Field(..., description="Pixel coordinates [x1, y1, x2, y2]")
    normalized_box: List[float] = Field(..., description="Normalized coordinates [ymin, xmin, ymax, xmax] (0.0 - 1.0)")
    area: float

class DetectResponse(BaseModel):
    detections: List[DetectionItem]
    image_width: int
    image_height: int
    active_classes: List[str]
    inference_time_ms: float

class DetectBase64Request(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded JPEG or PNG image")
    classes: Optional[List[str]] = Field(None, description="Optional custom classes for this prediction")
    confidence: Optional[float] = Field(None, description="Confidence threshold (0.0 - 1.0)")
    iou_threshold: Optional[float] = Field(None, description="IoU NMS threshold")
    imgsz: Optional[int] = Field(None, description="Inference resolution (default 640)")

def _run_prediction(
    pil_image: Image.Image,
    custom_classes: Optional[List[str]] = None,
    conf: Optional[float] = None,
    iou: Optional[float] = None,
    imgsz: Optional[int] = None
) -> Tuple[List[DetectionItem], float, Any]:
    t0 = time.perf_counter()
    w, h = pil_image.size
    
    # If custom classes are passed for this single call, temporarily update model classes
    if custom_classes and len(custom_classes) > 0:
        cleaned = [c.strip() for c in custom_classes if c.strip()]
        manager.set_classes(cleaned)
    
    confidence = conf if conf is not None else DEFAULT_CONF
    iou_thresh = iou if iou is not None else DEFAULT_IOU
    resolution = imgsz if imgsz is not None else DEFAULT_IMGSZ

    results = manager.model.predict(
        source=pil_image,
        conf=confidence,
        iou=iou_thresh,
        imgsz=resolution,
        device=manager.device,
        verbose=False
    )
    
    latency_ms = (time.perf_counter() - t0) * 1000.0
    res = results[0]
    
    detections: List[DetectionItem] = []
    if res.boxes is not None and len(res.boxes) > 0:
        boxes_xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        names = res.names

        for box, score, cid in zip(boxes_xyxy, confs, cls_ids):
            x1, y1, x2, y2 = [float(v) for v in box]
            ymin = max(0.0, min(1.0, y1 / h))
            xmin = max(0.0, min(1.0, x1 / w))
            ymax = max(0.0, min(1.0, y2 / h))
            xmax = max(0.0, min(1.0, x2 / w))
            
            cname = names.get(cid, str(cid)) if isinstance(names, dict) else (names[cid] if cid < len(names) else str(cid))
            area = (x2 - x1) * (y2 - y1)
            
            detections.append(DetectionItem(
                class_id=int(cid),
                class_name=cname,
                confidence=round(float(score), 4),
                box_xyxy=[round(v, 2) for v in [x1, y1, x2, y2]],
                normalized_box=[round(v, 4) for v in [ymin, xmin, ymax, xmax]],
                area=round(float(area), 2)
            ))
            
    return detections, latency_ms, res

@app.get("/health")
def health():
    gpu_mem = None
    if torch.cuda.is_available():
        gpu_mem = {
            "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 2),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
            "device_name": torch.cuda.get_device_name(0)
        }
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": manager.device,
        "active_classes_count": len(manager.current_classes),
        "active_classes": manager.current_classes,
        "gpu_memory": gpu_mem
    }

@app.post("/set_classes")
def set_classes(req: SetClassesRequest):
    manager.set_classes(req.classes)
    return {
        "status": "success",
        "active_classes": manager.current_classes,
        "count": len(manager.current_classes)
    }

@app.post("/detect", response_model=DetectResponse)
async def detect(
    file: UploadFile = File(..., description="Image file to detect"),
    classes: Optional[str] = Form(None, description="Comma-separated custom classes (e.g. 'candlestick, support line, person')"),
    confidence: Optional[float] = Form(None),
    iou_threshold: Optional[float] = Form(None),
    imgsz: Optional[int] = Form(None)
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    custom_cls_list = [c.strip() for c in classes.split(",")] if classes else None
    detections, latency, _ = _run_prediction(image, custom_cls_list, confidence, iou_threshold, imgsz)

    return DetectResponse(
        detections=detections,
        image_width=image.width,
        image_height=image.height,
        active_classes=manager.current_classes,
        inference_time_ms=round(latency, 2)
    )

@app.post("/detect/json", response_model=DetectResponse)
def detect_base64(req: DetectBase64Request):
    try:
        raw_b64 = req.image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(raw_b64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

    detections, latency, _ = _run_prediction(
        image,
        req.classes,
        req.confidence,
        req.iou_threshold,
        req.imgsz
    )

    return DetectResponse(
        detections=detections,
        image_width=image.width,
        image_height=image.height,
        active_classes=manager.current_classes,
        inference_time_ms=round(latency, 2)
    )

@app.post("/detect/visualize")
async def detect_visualize(
    file: UploadFile = File(...),
    classes: Optional[str] = Form(None),
    confidence: Optional[float] = Form(None),
    iou_threshold: Optional[float] = Form(None),
    imgsz: Optional[int] = Form(None)
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    custom_cls_list = [c.strip() for c in classes.split(",")] if classes else None
    _, _, result_obj = _run_prediction(image, custom_cls_list, confidence, iou_threshold, imgsz)

    # Plot bounding boxes on image
    plot_bgr = result_obj.plot()
    plot_rgb = plot_bgr[:, :, ::-1]  # BGR to RGB
    vis_pil = Image.fromarray(plot_rgb)

    buf = io.BytesIO()
    vis_pil.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8005"))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
