FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Ubuntu 24.04 = Python 3.12.x final (bukan RC).
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

# Pakai "python3 -m pip" agar tidak conflict dengan debian-managed pip binary.
# PaddlePaddle 3.0.0 cu124 — wheel resmi CUDA 12.4, kompatibel dengan runtime 12.6.
# Referensi: https://www.paddlepaddle.org.cn/packages/stable/cu124/
RUN python3 -m pip install paddlepaddle-gpu==3.0.0 \
        -i https://www.paddlepaddle.org.cn/packages/stable/cu124/

COPY requirements.txt .

# Torch cu124 wheel di-install sebelum requirements.txt agar sentence-transformers
# dan FlagEmbedding tidak narik CPU wheel dari PyPI.
RUN python3 -m pip install \
    torch==2.5.1+cu124 \
    torchvision==0.20.1+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

RUN python3 -m pip install -r requirements.txt

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