FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    YOLO_MODEL_NAME=yolov8l-worldv2.pt \
    PORT=8005 \
    HOST=0.0.0.0

WORKDIR /app

# Install system graphics libraries required by OpenCV/Ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download weights during build so startup is instant with zero runtime download delays
RUN python3 -c "from ultralytics import YOLOWorld; YOLOWorld('yolov8l-worldv2.pt')"

COPY server.py client.py ./

EXPOSE 8005

CMD ["python3", "server.py"]
