FROM vllm/vllm-openai:latest

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies for trading video understanding
RUN pip install --no-cache-dir \
    ultralytics \
    qwen_vl_utils \
    easyocr \
    wandb \
    sentence-transformers \
    fastapi \
    uvicorn \
    jinja2 \
    python-multipart

ENTRYPOINT ["python3"]
