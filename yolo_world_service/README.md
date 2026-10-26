# 🌍 YOLO-World Large Microservice & API

High-performance, open-vocabulary object detection microservice powered by **YOLO-World Large** (`yolov8l-worldv2.pt`). Enables zero-shot open-vocabulary detection without model retraining by compiling natural-language text prompts directly into real-time detection heads.

---

## 🧠 1. How YOLO-World Works

Traditional object detectors (like YOLOv8 or Faster R-CNN) are trained on fixed vocabularies (e.g., COCO's 80 classes) and can never detect anything outside those categories without full fine-tuning.

**YOLO-World** introduces real-time open-vocabulary detection via a decoupled Vision-Language architecture:
1. **Visual Backbone**: High-efficiency YOLOv8 DarkNet backbone with Multi-Scale Path Aggregation Network (PAN).
2. **Text Encoder**: Frozen or fine-tuned CLIP text encoder transforms arbitrary open-vocabulary prompts (e.g. `"candlestick"`, `"support level"`, `"price axis"`, `"breakout candle"`, `"person"`) into dense text embeddings.
3. **Cross-Modality Fusion (RepVL-PAN)**: Re-parameterizable Vision-Language Path Aggregation Network fuses multi-scale visual feature maps with text feature embeddings using cross-attention.
4. **Offline Vocabulary Compilation ("Prompt-then-Detect")**:
   - For recurring queries, YOLO-World pre-computes text embeddings once and re-parameterizes them directly into the convolutional detection head weights.
   - During inference, the text encoder is bypassed entirely, achieving standard YOLO inference latencies (**15–30 ms per image**) with open-vocabulary flexibility!

---

## ⚡ 2. GPU Hardware & VRAM Requirements

| Metric | Measurement (YOLO-World Large v2) |
|---|---|
| **Model Parameter Count** | ~43.8 Million parameters |
| **Model Weights File Size** | 89.9 MB (`yolov8l-worldv2.pt`) |
| **Idle GPU VRAM Usage** | **~558 MB** |
| **Inference VRAM (640x640)** | **~750 MB – 1.0 GB** |
| **Inference VRAM (2K–4K High-Res)** | **~1.6 GB – 2.0 GB** |
| **Inference Latency (GPU)** | **15 ms – 45 ms** (batch size 1) |
| **Recommended Minimum GPU** | Any NVIDIA GPU with **≥ 2 GB VRAM** (RTX 3060, GTX 1660, T4, L4, A10, Jetson Orin, NVIDIA GB10 / DGX Spark) |
| **CPU Fallback Supported?** | Yes, runs on CPU at ~80–180 ms per frame |

---

## 🚀 3. Deployment on a Separate System

### Option A: Quickstart with Docker Compose (Recommended)

1. Clone or copy the `yolo_world_service/` directory to the target system.
2. Start the service:
```bash
docker compose up -d
```
3. Verify health:
```bash
curl http://localhost:8005/health
```

### Option B: Standalone Docker Run

```bash
docker run -d \
  --name yolo-world-large \
  --restart unless-stopped \
  --gpus all \
  -p 8005:8005 \
  -e PORT=8005 \
  -e YOLO_DEVICE=cuda \
  -e YOLO_MODEL_NAME=yolov8l-worldv2.pt \
  yolo-world-large:latest
```

*(Note: On systems without GPU pass-through, omit `--gpus all` and set `-e YOLO_DEVICE=cpu`.)*

### Option C: Native Python Environment

```bash
pip install -r requirements.txt
python3 server.py
```

---

## 📡 4. Agent API Specification & Usage

### 4.1 Health & Capabilities (`GET /health`)
Returns model info, device, GPU memory allocation, and active classes.

```bash
curl -s http://<HOST>:8005/health | jq .
```
**Response:**
```json
{
  "status": "healthy",
  "model": "yolov8l-worldv2.pt",
  "device": "cuda",
  "active_classes_count": 12,
  "active_classes": ["candlestick", "chart", "price axis", "support level", ...],
  "gpu_memory": {
    "allocated_mb": 558.02,
    "max_allocated_mb": 1701.42,
    "device_name": "NVIDIA GB10"
  }
}
```

---

### 4.2 Detect Objects with Custom Vocabulary (`POST /detect`)
Send an image with arbitrary open-vocabulary prompts.

```bash
curl -s -X POST "http://<HOST>:8005/detect" \
  -F "file=@sample_frame.jpg" \
  -F "classes=candlestick,price axis,support level,resistance level,person" \
  -F "confidence=0.20" | jq .
```

**Response Format:**
```json
{
  "detections": [
    {
      "class_id": 0,
      "class_name": "candlestick",
      "confidence": 0.684,
      "box_xyxy": [120.5, 340.2, 145.0, 410.8],
      "normalized_box": [0.3402, 0.1205, 0.4108, 0.145],
      "area": 1729.7
    }
  ],
  "image_width": 1920,
  "image_height": 1080,
  "active_classes": ["candlestick", "price axis", ...],
  "inference_time_ms": 24.5
}
```
> **Normalized Bounding Box Convention**: `[ymin, xmin, ymax, xmax]` scaled between `0.0` and `1.0`. Compatible with NVIDIA VSS, Cosmos-3 Nano, Qwen-VL, and standard VLM grounding frameworks.

---

### 4.3 Base64 / JSON Detection (`POST /detect/json`)
Ideal for agents passing images in memory without writing to disk:

```bash
curl -X POST "http://<HOST>:8005/detect/json" \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "<BASE64_DATA>",
    "classes": ["candlestick", "trendline"],
    "confidence": 0.25
  }'
```

---

### 4.4 Visual Bounding Box Rendering (`POST /detect/visualize`)
Renders predicted bounding boxes, labels, and confidence tags directly onto the image:

```bash
curl -X POST "http://<HOST>:8005/detect/visualize" \
  -F "file=@chart.jpg" \
  -F "classes=candlestick,resistance level,support level" \
  -F "confidence=0.20" \
  -o annotated_chart.jpg
```

---

### 4.5 Set Persistent Vocabulary (`POST /set_classes`)
Bakes a fixed vocabulary into the detector so future calls execute at maximum speed:

```bash
curl -X POST "http://<HOST>:8005/set_classes" \
  -H "Content-Type: application/json" \
  -d '{"classes": ["candlestick", "price axis", "volume pane", "trendline", "breakout"]}'
```

---

## 🐍 5. Python Client Example for Downstream Agents

```python
from yolo_world_service.client import YoloWorldClient

# Initialize client pointing to your deployed YOLO-World instance
client = YoloWorldClient(base_url="http://localhost:8005")

# Run open-vocabulary detection
results = client.detect(
    image="path/to/frame.jpg",
    classes=["candlestick", "resistance level", "price axis"],
    confidence=0.20
)

print(f"Found {len(results['detections'])} objects in {results['inference_time_ms']} ms")
for det in results["detections"]:
    print(f"Detected {det['class_name']} ({det['confidence']:.2f}) at normalized box {det['normalized_box']}")

# Save visualized boxes to disk
client.detect_and_save_visual(
    image="path/to/frame.jpg",
    output_path="debug_boxes.jpg",
    classes=["candlestick", "resistance level"]
)
```
