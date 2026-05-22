# Base image: CUDA 12.4 runtime + cuDNN untuk paddlepaddle-gpu.
# L40S driver 595.x mendukung CUDA 13.2 — backward compatible dengan CUDA 12.4 runtime di container.
# Kalau host driver lebih tua (< 545.x), turunkan base ke 12.0 atau 11.8.
FROM nvidia/cuda:12.4.0-cudnn-runtime-ubuntu22.04

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

# Install paddlepaddle-gpu dari Paddle CUDA-12 channel.
# Index URL ini official dari paddlepaddle.org.cn — wheel pre-built untuk CUDA 12.
RUN pip install --upgrade pip && \
    pip install paddlepaddle-gpu==3.0.0 \
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
