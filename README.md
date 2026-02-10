# RAG Chatbot UNHAS — Prototype

RAG (Retrieval-Augmented Generation) chatbot untuk informasi akademik Universitas Hasanuddin.

## Tech Stack

| Komponen | Teknologi |
|----------|-----------|
| LLM | Qwen2.5-7B via Ollama |
| Embedding | Qwen3-Embedding-0.6B |
| Re-ranker | BGE-Reranker-v2-M3 |
| Vector DB | Qdrant (Docker) |
| Framework | LlamaIndex |
| Backend | FastAPI |
| Preprocessing / OCR | PaddleOCR + PyMuPDF |
| Chunking | Unstructured.io (chunk_by_title) |
| Frontend | HTML/JS sederhana |

## Hardware Requirements

- GPU: NVIDIA RTX 3060 12GB VRAM (atau setara)
- RAM: minimal 32 GB
- Total VRAM usage: ~7.4 GB (LLM + Embedding + Re-ranker)

## Setup

### 1. Clone & Virtual Environment
```bash
git clone <repo-url>
cd rag-prototype
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt

# Install PyTorch dengan CUDA (wajib untuk GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Jalankan Qdrant (Docker)
```bash
docker compose up -d
```
Dashboard: http://localhost:6333/dashboard

### 3. Install & Setup Ollama
```bash
# Install Ollama (lihat https://ollama.com)
ollama pull qwen2.5:7b
```

### 4. Download Model (First Time)
Saat pertama kali, embedding (~1.2GB) dan re-ranker (~2.3GB) perlu didownload dari HuggingFace:
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Embedding-0.6B')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-reranker-v2-m3')"
```

> **Tips download cepat:** Untuk file besar yang lambat/stuck, gunakan [aria2](https://github.com/aria2/aria2/releases) (multi-connection, resume support):
> ```bash
> aria2c -x 16 -s 16 -k 1M -o model.safetensors "https://huggingface.co/BAAI/bge-reranker-v2-m3/resolve/main/model.safetensors"
> ```
> Atau set HuggingFace token untuk rate limit lebih tinggi:
> ```bash
> # Windows PowerShell
> $env:HF_TOKEN="hf_your_token_here"
> $env:HF_HUB_ENABLE_HF_TRANSFER="1"
> pip install hf_transfer
> ```

### 5. Index Dokumen PDF
Taruh file PDF di `data/pdfs/`, lalu jalankan di terminal terpisah:
```bash
python -m scripts.index_documents
```
> Catatan: Jika VRAM penuh saat indexing, matikan Ollama dulu: `ollama stop qwen2.5:7b`

### 6. Jalankan Backend + Frontend
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
Chat UI: http://localhost:8000

## API Endpoints

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| POST | `/api/chat` | Kirim pertanyaan, dapat jawaban RAG + sumber |
| GET | `/api/health` | Cek status Ollama & Qdrant |
| POST | `/api/index` | Trigger indexing dokumen PDF |

Docs: http://localhost:8000/docs

## Project Structure

```
rag-prototype/
├── backend/
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Settings & env vars
│   ├── routers/             # API route handlers
│   ├── services/            # Business logic (RAG pipeline, indexing, OCR preprocessing)
│   ├── models/              # Pydantic schemas
│   └── prompts/             # Prompt templates
├── frontend/
│   ├── index.html           # Chat UI
│   ├── style.css            # Styling
│   └── app.js               # API calls + chat logic
├── data/
│   ├── pdfs/                # PDF dokumen UNHAS
│   └── images/              # Gambar hasil extract dari PDF (multimodal)
├── scripts/                 # Utility scripts (indexing, evaluasi)
├── tests/                   # Test files
├── docker-compose.yml       # Qdrant service
├── requirements.txt         # Python dependencies
└── .env                     # Environment variables
```

## Progress

- [x] Project setup (venv, dependencies, folder structure, docker-compose, config)
- [x] Backend services (RAG pipeline, indexing, prompt templates, schemas)
- [x] Preprocessing pipeline (PaddleOCR + Unstructured.io + image extraction)
- [x] API endpoints (FastAPI routes: /api/chat, /api/health, /api/index)
- [x] Frontend UI (chat interface dengan health check + sources)
- [ ] Evaluasi
