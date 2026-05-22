# Base image: CUDA 12.4.1 + cuDNN 9 — stabil untuk L40S driver 595.x (CUDA 13.2).
# Backward-compatible: CUDA 12.4 runtime < CUDA 13.2 max driver.
# Ubuntu 24.04 hadir dengan Python 3.12.x final (bukan RC seperti Ubuntu 22.04).
# cu124 wheel tersedia untuk torch 2.5.x dan paddlepaddle-gpu 3.0.x.
FROM nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu24.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Ubuntu 24.04 default = Python 3.12.x final — tidak perlu PPA atau symlink manual.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-dev \
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
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PaddlePaddle 3.0.0 cu124 — versi pertama di branch 3.x yang punya wheel resmi CUDA 12.4.
# Referensi: https://www.paddlepaddle.org.cn/packages/stable/cu124/
RUN pip install --upgrade pip && \
    pip install paddlepaddle-gpu==3.0.0 \
        -i https://www.paddlepaddle.org.cn/packages/stable/cu124/

COPY requirements.txt .

# Torch cu124 wheel harus di-install sebelum requirements.txt karena
# beberapa package (sentence-transformers, FlagEmbedding) akan narik torch
# dari PyPI (CPU wheel) jika torch belum ada di environment.
RUN pip install \
    torch==2.5.1+cu124 \
    torchvision==0.20.1+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

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