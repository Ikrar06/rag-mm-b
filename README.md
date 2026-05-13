# RAG Core — Chatbot Akademik UNHAS

**RAG Core** adalah komponen inti retrieval-augmented generation untuk chatbot akademik Universitas Hasanuddin. Repo ini hanya mencakup **pipeline RAG + API-nya** — bukan frontend produksi maupun BE utama.

```
Tim 1 (BE)  ──HTTP──▶  /api/query  ──▶  RAG Core (repo ini)  ──▶  Qdrant + LLM
Tim 3 (FE)  ──────────────────────────▶  (via BE Tim 1)
```

---

## Daftar Isi

1. [Arsitektur Pipeline](#arsitektur-pipeline)
2. [Tech Stack](#tech-stack)
3. [Struktur Folder](#struktur-folder)
4. [Setup Dev (RTX 3060)](#setup-dev-rtx-3060)
5. [Setup POC (L40S 48GB — IOH)](#setup-poc-l40s-48gb--ioh)
6. [Data Pipeline: JSON → Narasi → Qdrant](#data-pipeline-json--narasi--qdrant)
7. [API Reference](#api-reference)
8. [Environment Variables](#environment-variables)
9. [Untuk Tim 1 (BE)](#untuk-tim-1-be)
10. [Progress](#progress)

---

## Arsitektur Pipeline

Setiap pesan user melewati 6 lapisan sebelum dijawab:

```
User Message
    │
    ▼
[L1] Keyword Filter     — blokir konten berbahaya (senjata, narkoba, hack, dll.)
    │
    ▼
[L2] Moderation Model   — Llama Guard 3 1B via Ollama (passthrough di dev)
    │
    ▼
[L3] Intent Classifier  — IndoBERT 4-kelas:
    │                       chitchat → jawab langsung (tanpa RAG)
    │                       out_of_scope → tolak sopan
    │                       get_info_public → lanjut ke RAG
    │                       get_info_private → panggil API UNHAS (L4)
    ▼
[L4] Private API Handler — function calling ke endpoint UNHAS (jika dikonfigurasi)
    │                       fallback ke RAG jika UNHAS_API_BASE_URL kosong
    ▼
[L5] RAG Pipeline        — query condensation → retrieval → rerank → generation
    │
    ▼
[L6] Output Filter       — redact nama AI, stack teknologi, NIM pola regex, JWT, URL internal
    │
    ▼
Response
```

---

## Tech Stack

| Komponen | Dev (RTX 3060) | POC (L40S IOH) |
|---|---|---|
| LLM | Qwen2.5:7b via Ollama | Qwen3-VL-8B via vLLM |
| Embedding | Qwen3-Embedding-0.6B (in-process, GPU) | Qwen3-Embedding-0.6B via TEI |
| Re-ranker | BGE-Reranker-v2-M3 (in-process) | BGE-Reranker-v2-M3 via TEI |
| Moderation | passthrough (skip) | Llama Guard 3 1B via Ollama |
| Intent Classifier | IndoBERT (in-process) | IndoBERT (in-process) |
| Vector DB | Qdrant (Docker) | Qdrant (Docker) |
| Framework | LlamaIndex | LlamaIndex |
| Backend | FastAPI + SQLAlchemy | FastAPI + SQLAlchemy |
| Database | PostgreSQL (Docker) | PostgreSQL (Docker) |
| Cache | Redis (Docker) | Redis (Docker) |
| Storage | — | MinIO (untuk gambar) |
| Auth | JWT (bcrypt) | JWT (bcrypt) |
| Rate Limiting | slowapi | slowapi |
| Logging | structlog (JSON) | structlog (JSON) |

---

## Struktur Folder

```
rag-prototype/
├── backend/
│   ├── main.py                  # FastAPI entry point, middleware, startup
│   ├── config.py                # Semua env var, dengan nilai default
│   ├── limiter.py               # slowapi singleton
│   ├── logging_config.py        # structlog JSON setup
│   ├── routers/
│   │   ├── auth.py              # POST /api/auth/login, /api/auth/logout
│   │   ├── chat.py              # POST /api/chat, /api/chat/stream, GET /api/sessions
│   │   └── query.py             # POST /api/query, /api/query/stream  ← untuk Tim 1
│   ├── services/
│   │   ├── rag_pipeline.py      # Pipeline utama (L1–L6, query, query_stream)
│   │   ├── auth.py              # bcrypt verify, JWT encode/decode
│   │   ├── moderation.py        # Layer 2 — Llama Guard via Ollama
│   │   ├── intent_classifier.py # Layer 3 — IndoBERT 4-kelas
│   │   ├── output_filter.py     # Layer 6 — regex redaction
│   │   ├── private_api.py       # Layer 4 — function calling ke API UNHAS
│   │   ├── cache.py             # Redis semantic cache
│   │   ├── session.py           # CRUD session & message history
│   │   ├── indexing.py          # PDF indexing (PaddleOCR + Unstructured)
│   │   ├── index_narratives.py  # JSON narrative indexing ke Qdrant
│   │   └── llm_factory.py       # LLM provider abstraction (Ollama/vLLM/OpenAI)
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response schemas
│   ├── db/
│   │   ├── database.py          # SQLAlchemy engine, session, init_db
│   │   └── models.py            # ORM: User, Session, Message
│   └── prompts/
│       └── templates.py         # RAG, chitchat, condense, vision prompts
├── data/
│   ├── json/                    # File JSON dari API UNHAS (input narasi)
│   │   ├── fakultas.json
│   │   ├── prodi.json
│   │   ├── mahasiswa.json
│   │   └── ...
│   ├── narratives/              # Output teks narasi (generated, gitignore)
│   │   ├── fakultas/
│   │   ├── prodi/
│   │   └── ...
│   └── pdfs/                    # Dokumen PDF UNHAS (SOP, peraturan, dll.)
├── models/                      # Gitignored — download via scripts/download_models.py
│   ├── intent_classifier/       # IndoBERT fine-tuned sendiri untuk UNHAS (4-kelas)
│   │                            # Tersedia di HF: ikrarrr/rag-unhas-intent-classifier
│   └── bge-reranker-v2-m3/      # Pre-trained reranker (HF: BAAI/bge-reranker-v2-m3)
├── scripts/
│   ├── preprocess-template.py   # JSON → teks narasi (.txt)
│   ├── download_models.py       # Download semua model dari HuggingFace
│   └── start_demo.ps1           # Script demo (Windows)
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── docker-compose.yml           # Base compose
├── docker-compose.dev.yml       # Dev: Qdrant + Redis + PostgreSQL
├── docker-compose.poc.yml       # POC: semua service termasuk vLLM + TEI
├── .env.dev                     # Template env untuk dev
├── .env.poc                     # Template env untuk POC/production
├── requirements.txt
└── CLAUDE.md
```

---

## Setup Dev (RTX 3060)

### Prasyarat Hardware
- GPU: NVIDIA RTX 3060 12GB VRAM (atau setara)
- RAM: minimal 16 GB
- VRAM yang terpakai saat runtime: ~8–10 GB (LLM + Embedding + Re-ranker + IndoBERT)

> **Catatan:** Moderation model (Llama Guard 3) **tidak dijalankan** di dev karena VRAM tidak cukup. `MODERATION_BACKEND=passthrough` di `.env.dev`.

### 1. Clone & Virtual Environment

```powershell
git clone https://github.com/Ikrar06/rag-prototype
cd rag-prototype

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

# Install PyTorch dengan CUDA (wajib untuk GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Konfigurasi Environment

```powershell
copy .env.dev .env
```

Edit `.env` jika perlu, tapi default `.env.dev` sudah siap untuk dev lokal.

### 3. Jalankan Services (Docker)

```powershell
docker compose -f docker-compose.dev.yml up -d
```

Service yang jalan:
- Qdrant: http://localhost:6333/dashboard
- PostgreSQL: localhost:5432
- Redis: localhost:6379

### 4. Install & Setup Ollama

```powershell
# Install Ollama dari https://ollama.com, lalu:
ollama pull qwen2.5:7b
```

### 5. Download Model dari HuggingFace (pertama kali)

```powershell
# Download intent classifier (fine-tuned, ~500MB) + reranker (~2.3GB)
python scripts/download_models.py

# Atau download satu per satu:
python scripts/download_models.py --model intent    # hanya intent classifier
python scripts/download_models.py --model reranker  # hanya reranker

# Embedding (~600MB) — download terpisah karena dikelola LlamaIndex
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Embedding-0.6B')"
```

> **Download lambat?** Aktifkan hf_transfer:
> ```powershell
> pip install hf_transfer
> $env:HF_HUB_ENABLE_HF_TRANSFER="1"
> python scripts/download_models.py
> ```

### 6. Index Data dari JSON API UNHAS

```powershell
# 1. Taruh file JSON di data/json/
# 2. Generate narasi teks
python scripts/preprocess-template.py

# 3. Index ke Qdrant
python backend/services/index_narratives.py

# Force re-index (jika ada perubahan narasi):
python backend/services/index_narratives.py --force

# Re-index endpoint tertentu saja:
python backend/services/index_narratives.py --force --endpoint fakultas,prodi
```

> **Endpoint JSON yang didukung:** `fakultas`, `prodi`, `jenjang`, `kurikulum`, `mata-kuliah`, `prasyarat`, `rps`, `kelas`, `jadwal`, `fasilitas`, `pmb`, `pengumuman`, `mahasiswa`

> **Index dari PDF** (opsional, untuk SOP/peraturan): panggil `POST /api/index` setelah server jalan.

### 7. Jalankan Server

```powershell
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

- Chat UI: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## Setup POC (L40S 48GB — IOH)

### 1. Konfigurasi Environment

```bash
cp .env.poc .env
# Edit CHANGE_ME values:
# - MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_USER, MINIO_PASSWORD
# - DATABASE_URL (password postgres)
# - JWT_SECRET (gunakan string random panjang)
# - UNHAS_API_BASE_URL (isi jika endpoint UNHAS sudah siap)
```

### 2. Jalankan Semua Service

```bash
docker compose -f docker-compose.poc.yml up -d
```

Service yang jalan: vLLM, TEI Embedding, TEI Reranker, Qdrant, PostgreSQL, Redis, MinIO.

### 3. Index Data

```bash
python scripts/preprocess-template.py
python backend/services/index_narratives.py --force
```

---

## Data Pipeline: JSON → Narasi → Qdrant

Data akademik UNHAS dari API berbentuk tabular (JSON). Sebelum di-embed ke Qdrant, data dikonversi ke teks narasi bahasa Indonesia agar bisa di-retrieve secara semantik.

```
data/json/fakultas.json
        │
        ▼ scripts/preprocess-template.py
data/narratives/fakultas/
    ├── 0000_FT.txt          ← "Fakultas Teknik (FT, kode: 01) adalah..."
    ├── 0001_FMIPA.txt
    ├── ...
    └── _ringkasan.txt       ← "UNHAS memiliki 5 fakultas, 11 prodi, 45 gedung..."
        │
        ▼ backend/services/index_narratives.py
Qdrant collection: unhas_docs
```

**File JSON yang didukung** (taruh di `data/json/`):

| File | Visibilitas | Isi |
|---|---|---|
| `fakultas.json` | publik | Data 5 fakultas UNHAS |
| `prodi.json` | publik | 11 program studi |
| `jenjang.json` | publik | D3, D4, S1, Profesi, S2, S3 |
| `kurikulum.json` | publik | Kurikulum per prodi |
| `mata-kuliah.json` | publik | Daftar mata kuliah |
| `prasyarat.json` | publik | Prasyarat akademik MK |
| `rps.json` | publik | Rencana Pembelajaran Semester |
| `kelas.json` | publik | Kelas aktif + kuota |
| `jadwal.json` | publik | Jadwal kuliah |
| `fasilitas.json` | publik | Gedung & ruangan |
| `pmb.json` | publik | Penerimaan mahasiswa baru |
| `pengumuman.json` | publik | Pengumuman resmi |
| `mahasiswa.json` | publik | NIM, nama, prodi, status |

---

## API Reference

Semua endpoint tersedia di `http://localhost:8000/docs` (Swagger UI).

### Untuk Integrasi BE Eksternal (Tim 1)

**`POST /api/query`** — RAG query tanpa session management

```json
// Request
{
  "question": "apa syarat cuti akademik?",
  "history": [
    {"role": "user", "content": "halo"},
    {"role": "assistant", "content": "Halo! Ada yang bisa dibantu?"}
  ],
  "role": "public"
}

// Response
{
  "answer": "Untuk mengajukan cuti akademik, berikut langkahnya:\n- ...",
  "sources": [
    {
      "file_name": "SOP_Cuti_Akademik.pdf",
      "page": 3,
      "score": 0.87,
      "text_preview": "..."
    }
  ],
  "condensed_question": "apa syarat dan prosedur pengajuan cuti akademik?",
  "debug": {
    "mode": "rag",
    "total_time_s": 4.2,
    "model": "qwen2.5:7b",
    "top_score": 0.87
  }
}
```

**`POST /api/query/stream`** — versi streaming (Server-Sent Events)

```
// Request body sama dengan /api/query
// Response: SSE stream

data: {"type": "token", "delta": "Untuk "}
data: {"type": "token", "delta": "mengajukan "}
...
data: {"type": "meta", "answer": "...", "sources": [...], "debug": {...}}
```

**Field `role`:** `"public"` | `"mahasiswa"` | `"admin"`
- BE bertanggung jawab memverifikasi user dan mengirim role yang sesuai
- `get_info_private` intent hanya terpicu jika user terautentikasi sebagai mahasiswa

---

### Endpoint Lengkap

| Method | Endpoint | Auth | Deskripsi |
|---|---|---|---|
| POST | `/api/query` | — | RAG query untuk integrasi BE |
| POST | `/api/query/stream` | — | Streaming variant `/api/query` |
| POST | `/api/chat` | Cookie/JWT | Chat dengan session management |
| POST | `/api/chat/stream` | Cookie/JWT | Streaming chat |
| GET | `/api/sessions` | Cookie/JWT | Daftar sesi percakapan user |
| POST | `/api/auth/login` | — | Login, dapat JWT cookie |
| POST | `/api/auth/logout` | Cookie | Logout |
| GET | `/api/health` | — | Status semua service |
| POST | `/api/index` | — | Trigger indexing PDF |

---

### Response Mode (field `debug.mode`)

| Mode | Arti |
|---|---|
| `rag` | Dijawab dari dokumen RAG |
| `chitchat` | Dijawab langsung (sapaan/basa-basi) |
| `out_of_scope` | Diluar topik akademik UNHAS |
| `blocked` | Diblokir keyword berbahaya (L1) |
| `blocked_moderation` | Diblokir model moderasi (L2) |
| `clarification_needed` | Intent tidak jelas, minta klarifikasi |
| `get_info_private` | Dijawab dari API UNHAS |
| `cache_hit` | Dijawab dari cache Redis |

---

## Environment Variables

Semua config dibaca dari `.env`. Template tersedia di `.env.dev` (dev lokal) dan `.env.poc` (POC/production).

### Variabel Utama

| Variable | Default Dev | Deskripsi |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | Provider LLM: `ollama` / `vllm` / `openai` |
| `LLM_MODEL` | `qwen2.5:7b` | Nama model |
| `LLM_BASE_URL` | `http://localhost:11434` | URL endpoint LLM |
| `EMBED_PROVIDER` | `huggingface` | Provider embedding: `huggingface` / `tei` |
| `EMBED_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Model embedding |
| `RERANKER_PROVIDER` | `sentence_transformers` | Provider reranker |
| `QDRANT_URL` | `http://localhost:6333` | URL Qdrant |
| `QDRANT_COLLECTION` | `unhas_docs` | Nama collection |
| `DATABASE_URL` | `postgresql://...` | Connection string PostgreSQL |
| `REDIS_URL` | `redis://localhost:6379/0` | Connection string Redis |
| `JWT_SECRET` | *(set di .env)* | Secret key JWT |
| `MODERATION_BACKEND` | `passthrough` | `passthrough` / `ollama` |
| `INTENT_MODEL_PATH` | `models/intent_classifier` | Path model IndoBERT |
| `RATE_LIMIT_TEXT_PER_MINUTE` | `20` | Rate limit per user per menit |
| `ALLOWED_ORIGINS` | `http://localhost:*` | CORS origins |
| `UNHAS_API_BASE_URL` | *(kosong)* | Base URL API UNHAS (isi jika sudah tersedia) |

---

## Untuk Tim 1 (BE)

### Cara Memanggil RAG Core

Tim 1 **tidak perlu** mengelola session, history penyimpanan, atau auth di sisi RAG core. Cukup:

1. Verifikasi user di BE sendiri
2. Kirim request ke `/api/query` dengan `role` yang sesuai
3. Simpan history di DB BE sendiri, kirim kembali tiap request

```python
import httpx

RAG_BASE_URL = "http://rag-core:8000"  # sesuaikan host

def ask_rag(question: str, history: list, role: str = "public"):
    resp = httpx.post(
        f"{RAG_BASE_URL}/api/query",
        json={
            "question": question,
            "history": history,  # [{"role": "user"|"assistant", "content": "..."}]
            "role": role,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()
```

### Streaming

```python
def ask_rag_stream(question: str, history: list, role: str = "public"):
    with httpx.stream(
        "POST",
        f"{RAG_BASE_URL}/api/query/stream",
        json={"question": question, "history": history, "role": role},
        timeout=120,
    ) as resp:
        for line in resp.iter_lines():
            if line.startswith("data: "):
                event = json.loads(line[6:])
                yield event  # {"type": "token", "delta": "..."} atau {"type": "meta", ...}
```

### Catatan Arsitektur

- Repo ini adalah **RAG core saja** — session management, auth mahasiswa, dan integrasi sistem UNHAS lainnya ada di BE Tim 1
- `get_info_private` intent saat ini fallback ke RAG karena `UNHAS_API_BASE_URL` belum dikonfigurasi. Akan aktif setelah endpoint UNHAS tersedia dan `UNHAS_API_BASE_URL` di-set
- Rate limiting ada di RAG core (20 req/menit per IP) — BE bisa tambahkan rate limiting tersendiri di atasnya

---

## Progress

- [x] RAG pipeline (retrieval + rerank + generation)
- [x] PDF indexing (PaddleOCR + Unstructured.io)
- [x] JSON narrative pipeline (preprocess-template + index_narratives)
- [x] Multi-turn conversation dengan history
- [x] Streaming response (SSE)
- [x] Auth (JWT + bcrypt, PostgreSQL)
- [x] Session & message persistence (PostgreSQL)
- [x] Redis semantic cache
- [x] Rate limiting (slowapi)
- [x] Layer 1: Harmful keyword filter
- [x] Layer 2: Moderation (Llama Guard 3, passthrough di dev)
- [x] Layer 3: Intent classifier (IndoBERT 4-kelas)
- [x] Layer 4: Private API handler (function calling, fallback ke RAG)
- [x] Layer 6: Output filter (redact AI names, stack, NIM, JWT, URL)
- [x] Structured logging (structlog JSON)
- [x] `/api/query` endpoint untuk integrasi BE eksternal
- [x] Docker compose (dev + POC)
- [x] Fine-tuned IndoBERT intent classifier (4-kelas, hosted di HuggingFace)
- [ ] Integrasi API UNHAS (menunggu endpoint tersedia)
- [ ] Evaluasi pipeline (RAGAS atau sejenisnya)
- [ ] Vision RAG (aktif di POC dengan Qwen3-VL)
