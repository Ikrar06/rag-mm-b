# Base image: CUDA 12.0 + cuDNN 8 untuk paddlepaddle-gpu cu120 channel.
# L40S driver 595.x (CUDA 13.2) backward-compatible dengan CUDA 12.0 runtime.
# Combo CUDA 12.0.1 + paddle 2.6.2 cu120 = paling stabil & terverifikasi untuk PaddleOCR 2.9.x.
# Kalau butuh CUDA versi lain, periksa tag tersedia di https://hub.docker.com/r/nvidia/cuda/tags
FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Python 3.11 + system libs untuk OCR/PDF + cleanup.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    zlib1g-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    poppler-utils \
    tesseract-ocr \
    curl \
    ca-certificates \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install paddlepaddle-gpu dari Paddle cu120 channel.
# Combo paddlepaddle-gpu 2.6.2 + paddleocr 2.9.x = stable. Paddle 3.0.x belum punya
# wheel cu120 (cuma cu118 dan cu126), jadi pin ke 2.6.2.
RUN pip install --upgrade pip && \
    pip install paddlepaddle-gpu==2.6.2.post120 \
        -i https://www.paddlepaddle.org.cn/packages/stable/cu120/

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY users.json ./users.json

RUN mkdir -p /app/models /app/logs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
