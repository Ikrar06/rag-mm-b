# RAG Core — Chatbot Akademik UNHAS

**RAG Core** adalah komponen inti retrieval-augmented generation untuk chatbot akademik Universitas Hasanuddin. Repo ini hanya mencakup **pipeline RAG + API-nya** — bukan frontend produksi maupun BE utama.

```
Tim 1 (BE + FE)  ──HTTP──▶  /api/query  ──▶  RAG Core (repo ini)  ──▶  Qdrant + LLM
Tim 3 (Evaluasi) ──────────────────────────▶  (uji pipeline via /api/query)
```

---

## Daftar Isi

1. [Arsitektur Pipeline](#arsitektur-pipeline)
2. [Tech Stack](#tech-stack)
3. [Intent Classifier (IndoBERT)](#intent-classifier-indobert)
4. [Logging](#logging)
5. [Struktur Folder](#struktur-folder)
6. [Environments (Dev / POC / Production)](#environments)
7. [Setup Dev (RTX 3060)](#setup-dev-rtx-3060)
8. [Setup POC (L40S 48GB)](#setup-poc-l40s-48gb)
9. [Sumber Data RAG](#sumber-data-rag)
10. [Data Pipeline: JSON → Narasi → Qdrant](#data-pipeline-json--narasi--qdrant)
11. [PDF Indexing (Production-Grade)](#pdf-indexing-production-grade)
12. [API Reference](#api-reference)
13. [Environment Variables](#environment-variables)
14. [Untuk Tim 1 (BE + FE)](#untuk-tim-1-be--fe)
15. [Untuk Tim 3 (Evaluasi)](#untuk-tim-3-evaluasi)
16. [Progress](#progress)

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

### Detail Tiap Layer

#### L1 — Keyword Filter
Pengecekan regex berbasis daftar kata kunci berbahaya sebelum query menyentuh model apapun. Sangat cepat (< 1ms). Mengembalikan mode `blocked`.

Kategori yang diblokir: senjata, bahan peledak, narkoba, serangan siber, prompt injection (`"ignore instruction"`, `"abaikan instruksi"`, dll).

#### L2 — Moderation Model
Llama Guard 3 1B (Meta) via Ollama untuk deteksi konten berbahaya yang lebih nuanced dari regex. Di lingkungan **dev dinonaktifkan** (`MODERATION_BACKEND=passthrough`) karena VRAM tidak cukup di RTX 3060 bersamaan model lain. Aktif di **POC** dengan URL Ollama terpisah dari LLM utama (`MODERATION_BASE_URL=http://ollama:11434`, `LLM_BASE_URL=http://vllm:8001`). Mengembalikan mode `blocked_moderation`.

#### L3 — Intent Classifier
IndoBERT fine-tuned 4-kelas (lihat [seksi Intent Classifier](#intent-classifier-indobert)). Menentukan alur routing query:

| Intent | Alur | Mode response |
|---|---|---|
| `chitchat` | Langsung ke LLM dengan system prompt ringan, tanpa RAG | `chitchat` |
| `out_of_scope` | Tolak dengan pesan sopan, tanpa LLM call | `out_of_scope` |
| `get_info_public` | Lanjut ke L5 RAG | `rag` / `rag_low_relevance` / `cache_hit` |
| `get_info_private` | Lanjut ke L4 Private API | `get_info_private` / fallback ke `rag` |

Jika confidence < `INTENT_CONFIDENCE_THRESHOLD` (default 0.6), pipeline meminta klarifikasi daripada menebak intent. Hasil intent dan confidence-nya selalu dikembalikan di field `debug.intent` dan `debug.intent_confidence`.

#### L4 — Private API Handler
Menangani query `get_info_private` dengan memanggil endpoint API UNHAS secara langsung (function calling). Saat ini **fallback ke RAG** karena `UNHAS_API_BASE_URL` belum dikonfigurasi — akan aktif setelah endpoint UNHAS tersedia.

Data yang bisa di-query via API: IPK, KRS, jadwal personal, nilai, status UKT, dosen wali.

#### L5 — RAG Pipeline
Sub-pipeline dengan 4 tahap:

```
Query
  │
  ▼
[5a] Query Condensation   — ubah follow-up multi-turn jadi pertanyaan standalone
  │                          (via LLM + CONDENSE_PROMPT, skip jika single-turn)
  ▼
[5b] Cache Check          — cari jawaban identik di Redis (semantic cache)
  │                          hit → skip retrieval + generation langsung
  ▼
[5c] Retrieval            — embed query → ANN search di Qdrant (top-K chunks)
  │                          model: Qwen3-Embedding-0.6B
  ▼
[5d] Reranking            — cross-encoder scoring untuk re-order chunks
  │                          model: BGE-Reranker-v2-M3
  │                          jika top score < SCORE_THRESHOLD → mode rag_low_relevance
  ▼
[5e] Generation           — LLM generate jawaban dari chunks + prompt template
```

Parameter yang bisa dikonfigurasi via `.env`: `SIMILARITY_TOP_K`, `SCORE_THRESHOLD`, `RERANKER_TOP_N`, `HISTORY_TURNS`, `CHUNK_SIZE`, `CHUNK_OVERLAP`.

#### L6 — Output Filter
Regex post-processing pada jawaban akhir sebagai backstop. Meredact:
- Nama model AI (`qwen`, `llama`, `mistral`, `gpt`, dll.)
- Nama stack teknologi (`LlamaIndex`, `Qdrant`, `FastAPI`, dll.)
- NIM pola regex (format `D/F/H + digit`)
- JWT token (format `eyJ...`)
- URL internal (localhost, IP private)

---

## Tech Stack

| Komponen | Dev (RTX 3060) | POC (L40S 48GB) |
|---|---|---|
| LLM | Qwen2.5:7b via Ollama | Qwen3-VL-8B via vLLM |
| Embedding | Qwen3-Embedding-0.6B (in-process, GPU) | Qwen3-Embedding-0.6B via TEI |
| Re-ranker | BGE-Reranker-v2-M3 (in-process) | BGE-Reranker-v2-M3 via TEI |
| Moderation | passthrough (skip) | Llama Guard 3 1B via Ollama (lebih cocok untuk model kecil daripada vLLM) |
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

## Intent Classifier (IndoBERT)

Layer 3 pipeline menggunakan model **IndoBERT fine-tuned** untuk klasifikasi intent sebelum query masuk ke RAG.

### Detail Model

| Atribut | Nilai |
|---|---|
| Base model | `indobenchmark/indobert-base-p2` |
| Jumlah kelas | 4 |
| Dataset | Sintetis via Gemini API (`gemini-2.5-flash`) |
| Sampel per intent | 1.000 |
| Total dataset | ~4.000 sampel |
| Split | 80% train / 10% val / 10% test |
| HuggingFace | [`ikrarrr/rag-unhas-intent-classifier`](https://huggingface.co/ikrarrr/rag-unhas-intent-classifier) |

### Kelas Intent

| Label | Deskripsi | Contoh |
|---|---|---|
| `chitchat` | Sapaan, basa-basi, pertanyaan tentang bot | "halo", "terima kasih", "kamu bisa apa?" |
| `out_of_scope` | Topik di luar akademik UNHAS | "harga laptop bagus apa?", "resep rendang" |
| `get_info_public` | Info akademik publik atau data mahasiswa/dosen tertentu (nama disebutkan) | "syarat cuti?", "NIM Budi Santoso berapa?" |
| `get_info_private` | Data akademik milik si penanya sendiri (ada kata saya/gue/aku/-ku) | "IPK saya berapa?", "jadwal gue hari ini?" |

> **Catatan penting:** Query tentang data orang lain yang namanya disebutkan (mis. *"NIM Maria berapa?"*) diklasifikasikan sebagai `get_info_public`, bukan `get_info_private`.

### Melatih Ulang

```powershell
# 1. Generate dataset baru (butuh GEMINI_API_KEY)
$env:GEMINI_API_KEY="your_key"
python generate-dataset-intent-gemini.py --samples 1000

# 2. Training (output langsung ke folder model)
python train_classifier.py --output_dir C:\path\to\rag-prototype\models\intent_classifier

# 3. Salin best_model ke root model path
Copy-Item -Recurse -Force models\intent_classifier\best_model\* models\intent_classifier\

# 4. Upload ke HuggingFace
hf upload ikrarrr/rag-unhas-intent-classifier models/intent_classifier .
```

> Script `generate-dataset-intent-gemini.py` dan `train_classifier.py` disimpan di luar repo (tidak di-commit) karena hanya dipakai saat retraining.

---

## Logging

Pipeline menggunakan **structlog** dengan output JSON untuk kompatibilitas dengan log aggregator (ELK, Loki, dll.).

### Format Log

Setiap request HTTP otomatis mendapat **correlation ID** (`request_id`) yang diteruskan ke semua log dalam satu request:

```json
{
  "event": "http_request method=POST path=/api/query status=200 duration_ms=4231.5 request_id=a3f9b1c2",
  "request_id": "a3f9b1c2",
  "timestamp": "2026-05-14T10:23:45.123456Z",
  "level": "info"
}
```

Client bisa mengirim `X-Request-ID` header sendiri, atau sistem generate otomatis (8 karakter hex). Header dikembalikan di response sebagai `X-Request-ID`.

### Log Penting yang Dihasilkan Pipeline

| Event | Level | Keterangan |
|---|---|---|
| `http_request` | INFO | Setiap request masuk: method, path, status, duration |
| `intent_classifier_loaded` | INFO | Saat model IndoBERT pertama kali di-load ke memori |
| `cache_hit` | DEBUG | Query dijawab dari Redis |
| `rag_low_relevance` | WARNING | Top reranker score di bawah threshold |
| `stream_error` | ERROR | Error saat streaming response |
| `db_init_failed` | ERROR | PostgreSQL tidak terhubung saat startup |
| `chat_error` | ERROR | Error di `/api/chat` endpoint |

### Konfigurasi Log Level

```bash
# Di .env
LOG_LEVEL=INFO    # INFO (default) / DEBUG / WARNING / ERROR
```

Mode `DEBUG` menampilkan detail tiap tahap pipeline (retrieval scores, intent confidence, cache status).

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
│   │   └── query.py             # POST /api/query, /api/query/stream  ← untuk Tim 1 (BE + FE)
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
│   ├── index_documents.py       # Helper untuk indexing dokumen
│   ├── download_models.py       # Download model dari HuggingFace (intent + reranker)
│   └── start_demo.ps1           # Script demo (Windows)
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── docker-compose.yml           # Base compose minimal (Qdrant saja)
├── docker-compose.dev.yml       # Dev: Qdrant + Redis + PostgreSQL (LLM via Ollama di host)
├── docker-compose.poc.yml       # POC: backend container + DB services (vLLM/TEI/Ollama dicomment)
├── Dockerfile                   # Build image backend untuk POC
├── .dockerignore                # Exclude venv, models, data besar dari build context
├── .env.dev                     # Template env untuk dev
├── .env.poc                     # Template env untuk POC
├── requirements.txt
└── CLAUDE.md
```

---

## Environments

Sistem ini didesain untuk 3 level deployment, dari yang paling ringan ke paling besar. Setiap level punya tujuan, infrastruktur, dan tradeoff berbeda.

### Perbandingan 3 Environment

| Aspek | **Dev** | **POC** | **True Production** |
|---|---|---|---|
| **Tujuan** | Development, debugging, demo lokal | Staging di server GPU, load testing, demo ke stakeholder | Layanan publik UNHAS untuk ribuan user |
| **Compose file** | `docker-compose.dev.yml` | `docker-compose.poc.yml` | K8s manifests (belum dibuat) |
| **Hardware** | Laptop RTX 3060 12GB | Server L40S 48GB (1 host) | Multi-host cluster + managed services |
| **Orkestrasi** | docker-compose | docker-compose | Kubernetes / ECS |
| **LLM** | Ollama (`qwen2.5:7b`) di host | vLLM (`Qwen3-VL-8B`) di container | vLLM cluster + autoscale |
| **Embedding/Reranker** | In-process (HuggingFace + sentence-transformers) | TEI service terpisah | TEI cluster atau managed inference |
| **PostgreSQL** | Container, password hardcoded `dev` | Container, password dari env var | **Managed DB** (RDS / Cloud SQL) — HA, backup otomatis, replica |
| **Redis** | Container `redis:alpine` 512MB | Container `redis-stack` 4GB (RedisSearch) | **Managed cache** (ElastiCache / Memorystore) — cluster mode |
| **Object Storage** | Tidak ada (vision feature off) | Container MinIO | **S3 / GCS** — durabilitas 11 nines |
| **Backend** | Jalan di host (`uvicorn --reload`) | Container, single replica | Multiple replica + HPA + service mesh |
| **Moderation (L2)** | `passthrough` (skip, VRAM tidak cukup) | Llama Guard 3 1B via Ollama (instance terpisah dari vLLM) | Managed safety API (OpenAI Moderation, AWS Comprehend) — zero-ops |
| **TLS / HTTPS** | HTTP saja (localhost) | nginx/Traefik reverse proxy manual | Managed Load Balancer + cert otomatis |
| **High Availability** | Tidak ada | Tidak ada (host mati = semua mati) | Multi-zone, auto-failover |
| **Auto-scaling** | Tidak | Manual (`docker compose --scale`) | HPA berdasarkan CPU/QPS |
| **Secrets** | `.env.dev` (hardcoded `dev`) | `.env` di server (file biasa) | Vault / AWS Secrets Manager |
| **Logging** | `stdout` di terminal | `docker logs` di host | Centralized (ELK / Loki / Datadog) |
| **Monitoring** | Tidak ada | Prometheus + Grafana (opsional) | Full observability stack + alerting |
| **Rate limit** | 20 req/menit | 8 req/menit (text), 2/menit (vision) | Per-user quota + WAF rate limit |
| **Disaster Recovery** | Tidak ada | Manual backup script | Snapshot otomatis + cross-region replication |
| **Target user load** | 1 user (developer) | < 500 concurrent | 30.000+ mahasiswa UNHAS |

### Kapan Pakai yang Mana?

**Dev:**
- Coding fitur baru di laptop sendiri
- Reproduce bug yang dilaporkan
- Demo ke teman / dosen secara lokal
- Eksperimen prompt template, hyperparameter

**POC:**
- Demo arsitektur lengkap ke stakeholder UNHAS
- Load testing dengan beban realistis (puluhan-ratusan user)
- Validasi sebelum migrasi ke production sungguhan
- MVP untuk pilot user terbatas (mis. 1 fakultas)
- Testing fitur vision dengan vLLM multimodal

**True Production:**
- Layanan publik untuk seluruh mahasiswa UNHAS
- Requirement SLA uptime > 99.5%
- Audit / compliance (data perlindungan personal)
- Skala dinamis sesuai jam sibuk akademik

> **Strategi migrasi POC → Production:** POC compose sengaja didesain agar **arsitekturnya sudah mendekati production** — 1 container Docker mapping ke 1 Deployment Kubernetes. Saat migrasi, fokus utamanya bukan refactor kode tapi:
> 1. Lepas service stateful (PostgreSQL, Redis, MinIO) → ganti ke managed service
> 2. Convert `docker-compose.poc.yml` → K8s manifests (atau Helm chart)
> 3. Tambah Ingress + cert-manager untuk HTTPS
> 4. Set up HPA, PDB, NetworkPolicy
> 5. Centralized logging + monitoring stack

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

## Setup POC (L40S 48GB)

> **Posisi POC:** Staging deployment di server GPU untuk demo, load testing, dan validasi sebelum production sungguhan. **Bukan production-grade.** Lihat tabel [Environments](#environments) untuk perbedaan POC vs true production.

Semua service di `docker-compose.poc.yml` sudah aktif (tidak ada yang di-comment). Tinggal `cp .env.poc .env`, isi credential, dan `docker compose up`.

### Arsitektur Service POC

```
   Internet
      │
      ▼
  [Reverse Proxy / Load Balancer]   ← nginx/Traefik (di luar compose)
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Docker Network: ragchat                     │
│                                                                 │
│   [Backend :8000] ──────────┬─────────────────────────────────  │
│         │                   │                                   │
│         ├──▶ [vLLM :8001]              (GPU — Qwen3-VL-8B)      │
│         ├──▶ [Ollama :11434]           (GPU — Llama Guard 3 1B) │
│         ├──▶ [TEI Embed :8002]         (CPU — Embedding)        │
│         ├──▶ [TEI Rerank :8003]        (CPU — Reranker)         │
│         ├──▶ [Qdrant :6333]            (Vector DB)              │
│         ├──▶ [PostgreSQL :5432]        (Session DB)             │
│         ├──▶ [Redis Stack :6379]       (Semantic cache)         │
│         └──▶ [MinIO :9000]             (Image storage utk vision RAG) │
│                                                                 │
│   [Prometheus :9090] ◀── scrape /metrics dari backend            │
│   [Grafana :3000]    ◀── dashboard performance                  │
└─────────────────────────────────────────────────────────────────┘
```

### Prasyarat

| Komponen | Minimum | Rekomendasi |
|---|---|---|
| GPU | NVIDIA VRAM 24GB | NVIDIA L40S 48GB |
| RAM | 32 GB | 64 GB |
| Disk | 100 GB SSD | 500 GB NVMe |
| OS | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |
| Docker | 24.0+ | 26.0+ |
| Python | 3.11 | 3.11 |

### Step 1 — Install Docker + NVIDIA Container Toolkit

> **Skip step ini jika server sudah punya Docker + NVIDIA toolkit** (cek dengan: `docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi` — jika berhasil menampilkan GPU info, langsung ke Step 2).

```bash
# Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker

# NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verifikasi GPU dari container
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### Step 2 — Clone & Siapkan Model + Data

```bash
git clone https://github.com/Ikrar06/rag-prototype
cd rag-prototype

# Python venv untuk persiapan model lokal
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download intent classifier (~500MB) + reranker (~2.3GB)
python scripts/download_models.py
```

**Siapkan data JSON UNHAS** sebelum generate narasi:

```bash
# Taruh file JSON dari API UNHAS di data/json/
ls data/json/
# Harus ada: fakultas.json, prodi.json, mahasiswa.json, jadwal.json,
#            mata-kuliah.json, jenjang.json, kurikulum.json, prasyarat.json,
#            rps.json, kelas.json, fasilitas.json, pmb.json, pengumuman.json

# Generate narasi teks (output ke data/narratives/)
python scripts/preprocess-template.py
```

> File JSON didapat dari API UNHAS atau tim akademik. Jika belum tersedia, minimal `fakultas.json` dan `prodi.json` untuk smoke test.

### Step 3 — Konfigurasi `.env`

```bash
cp .env.poc .env
nano .env
```

Wajib ganti:

| Variable | Cara isi |
|---|---|
| `JWT_SECRET` | `python -c "import secrets; print(secrets.token_hex(32))"` |
| `POSTGRES_PASSWORD` | password random (≥ 16 karakter) untuk PostgreSQL container |
| `DATABASE_URL` | ganti `CHANGE_ME` dengan nilai `POSTGRES_PASSWORD` yang sama (harus identik agar backend bisa connect ke PostgreSQL) |
| `MINIO_USER` | username admin MinIO |
| `MINIO_PASSWORD` | password admin MinIO (≥ 8 karakter) |
| `MINIO_ACCESS_KEY` | bisa sama dengan `MINIO_USER` |
| `MINIO_SECRET_KEY` | bisa sama dengan `MINIO_PASSWORD` |
| `GRAFANA_PASSWORD` | password admin Grafana |
| `ALLOWED_ORIGINS` | domain frontend production, dipisah koma |
| `HF_TOKEN` | (opsional) HuggingFace token, lihat Step 4 |
| `UNHAS_API_BASE_URL` | (opsional) kosongkan jika belum ada |

**Contoh konkret:** kalau `POSTGRES_PASSWORD=Rahasia123Banget`, maka:
```
DATABASE_URL=postgresql://ragchat:Rahasia123Banget@postgres:5432/ragchat
```

### Step 4 — HF_TOKEN (jika butuh model gated)

Qwen3-VL-8B-Instruct dan Qwen3-Embedding-0.6B saat ini **public** — tidak butuh token. Tapi jika nanti pakai model gated (mis. Llama 3, Gemma):

```bash
# 1. Buat token di https://huggingface.co/settings/tokens (scope: read)
# 2. Accept license model di halaman HuggingFace
# 3. Set di .env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

vLLM dan TEI otomatis pakai `HF_TOKEN` dari environment.

### Step 5 — Build & Start Services

```bash
# Build image backend (sekali saja, ~5-10 menit)
docker compose -f docker-compose.poc.yml build backend

# Start semua service
docker compose -f docker-compose.poc.yml up -d

# Cek status
docker compose -f docker-compose.poc.yml ps
```

**Saat pertama kali jalan:**
- **vLLM** download `Qwen/Qwen3-VL-8B-Instruct` dari HuggingFace (~16 GB) → 10–30 menit
- **TEI** download `Qwen3-Embedding-0.6B` (~600 MB) dan `bge-reranker-v2-m3` (~2.3 GB)
- **Ollama** masih kosong, perlu pull model di step 6
- **Backend** akan start lebih dulu lalu retry health check sampai dependency ready (grace 60 detik, retry 3×). Wajar jika `docker compose ps` menunjukkan backend `unhealthy` di menit-menit awal — biarkan saja sampai vLLM selesai load.

Pantau progress download:
```bash
docker compose -f docker-compose.poc.yml logs -f vllm
```

Tunggu sampai log vLLM muncul: `Uvicorn running on http://0.0.0.0:8001`.

### Step 6 — Pull Model Llama Guard 3 (untuk Moderation)

Setelah Ollama container jalan, pull model `llama-guard3:1b`:

```bash
docker compose -f docker-compose.poc.yml exec ollama ollama pull llama-guard3:1b
```

Output yang diharapkan:
```
pulling manifest
pulling 8c3c8f4... 100% ▕████████████████▏ 1.6 GB
verifying sha256 digest
writing manifest
success
```

Verifikasi:
```bash
docker compose -f docker-compose.poc.yml exec ollama ollama list
# Harus muncul: llama-guard3:1b
```

### Step 7 — Buat Bucket MinIO

> **Untuk apa MinIO?** Menyimpan gambar yang di-upload user untuk fitur **vision RAG** (Qwen3-VL multimodal). Use case: user upload foto KRS/KTM/kartu ujian/formulir → backend simpan ke MinIO → URL gambar dikirim ke vLLM untuk dianalisis. Kalau pakai S3-compatible API jadi mudah migrate ke S3/GCS saat production.
>
> Skip step ini jika `LLM_SUPPORTS_VISION=false` di `.env` (text-only mode).

Pakai env var dari host (perlu `-e` flag agar diteruskan ke container):

```bash
# Load .env ke shell agar variabel tersedia
set -a && source .env && set +a

# Setup MinIO client di dalam container
docker compose -f docker-compose.poc.yml exec -e MINIO_USER -e MINIO_PASSWORD minio \
  mc alias set local http://localhost:9000 "$MINIO_USER" "$MINIO_PASSWORD"

# Buat bucket
docker compose -f docker-compose.poc.yml exec minio mc mb local/ragchat-images
```

Atau via console UI: buka `http://<server-ip>:9001` → login pakai `MINIO_USER`/`MINIO_PASSWORD` → **Create Bucket** → nama `ragchat-images`.

### Step 8 — Index Data ke Qdrant

```bash
docker compose -f docker-compose.poc.yml exec backend \
  python backend/services/index_narratives.py --force
```

Output:
```
Indexing fakultas... 5 narrative chunks
Indexing prodi... 11 narrative chunks
...
Total: ~150 chunks indexed to Qdrant collection 'unhas_docs'
```

### Step 9 — Verifikasi

```bash
# Health check semua service
curl http://localhost:8000/api/health
```

Response yang diharapkan:
```json
{
  "status": "healthy",
  "ollama": true,
  "qdrant": true,
  "postgres": true,
  "redis": true,
  "intent_model": true
}
```

```bash
# Test query end-to-end
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question":"apa syarat cuti akademik?","history":[],"role":"public"}'
```

Buka di browser (ganti `<server-ip>` dengan IP/hostname server, atau `localhost` jika akses dari mesin yang sama):
- **Chat UI**: `http://<server-ip>:8000`
- **Swagger**: `http://<server-ip>:8000/docs`
- **MinIO Console**: `http://<server-ip>:9001` (login: `MINIO_USER` / `MINIO_PASSWORD`)
- **Grafana**: `http://<server-ip>:3000` (login: `admin` / `GRAFANA_PASSWORD`)

### Step 10 — Reverse Proxy + HTTPS (Wajib untuk Public Access)

Backend port 8000 **jangan diekspos langsung** ke internet. Pakai nginx:

```nginx
# /etc/nginx/sites-available/ragchat
server {
    listen 443 ssl http2;
    server_name chatbot.unhas.ac.id;

    ssl_certificate     /etc/letsencrypt/live/chatbot.unhas.ac.id/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/chatbot.unhas.ac.id/privkey.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WAJIB untuk SSE streaming
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 180s;
    }
}
```

```bash
# Aktifkan + reload nginx
sudo ln -s /etc/nginx/sites-available/ragchat /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# Generate cert Let's Encrypt
sudo certbot --nginx -d chatbot.unhas.ac.id
```

### Maintenance

```bash
# Update kode + rebuild
git pull
docker compose -f docker-compose.poc.yml build backend
docker compose -f docker-compose.poc.yml up -d

# Backup PostgreSQL
docker compose -f docker-compose.poc.yml exec postgres \
  pg_dump -U ragchat ragchat > backup_$(date +%Y%m%d).sql

# Backup Qdrant volume
docker run --rm -v rag-prototype_qdrant_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/qdrant_$(date +%Y%m%d).tar.gz /data

# Scale backend (butuh load balancer di depan)
docker compose -f docker-compose.poc.yml up -d --scale backend=2

# View log filter request_id
docker compose -f docker-compose.poc.yml logs backend | grep "request_id=a3f9b1c2"
```

### Troubleshooting

| Masalah | Penyebab | Solusi |
|---|---|---|
| Build backend gagal: `zlib.h: No such file or directory` | Header zlib tidak terinstall di base image | Pastikan `zlib1g-dev` ada di Dockerfile (sudah include di repo terbaru) |
| `vllm` OOM saat startup | VRAM tidak cukup | Edit `--gpu-memory-utilization 0.85` → `0.75` di compose |
| vLLM stuck "Downloading..." berjam-jam | Network lambat / firewall HF | Set `HF_HUB_ENABLE_HF_TRANSFER=1` (sudah default) |
| `ollama list` kosong | Lupa pull model | Jalankan Step 6 |
| `MODERATION_BACKEND=ollama` tapi tidak block | Llama Guard belum di-pull | Step 6 |
| Health check `intent_model: false` | Volume `./models` belum di-mount | Step 2 (download model) lalu restart backend |
| MinIO bucket not found error | Bucket belum dibuat | Step 7 |
| Query selalu `out_of_scope` | Intent model salah load | Cek log: `docker compose logs backend \| grep intent_classifier` |
| Streaming response stuck | nginx buffering aktif | Pastikan `proxy_buffering off;` di config |
| `degraded` di health check | Ada service `false` | `docker compose ps` lalu `docker compose logs <service>` |
| `connection refused` ke vLLM | vLLM masih loading model | Tunggu sampai log: `Uvicorn running on http://0.0.0.0:8001` |

### Production Checklist

Sebelum go-live, pastikan:

- [ ] Semua `CHANGE_ME` di `.env` sudah diganti
- [ ] `JWT_SECRET` minimal 32 byte hex
- [ ] HTTPS aktif (Let's Encrypt atau cert lain)
- [ ] `ALLOWED_ORIGINS` dibatasi ke domain production
- [ ] Firewall: hanya port 443 terbuka publik (port 9001, 3000 untuk akses internal saja)
- [ ] Backup otomatis PostgreSQL + Qdrant terjadwal (cron)
- [ ] Log rotation aktif (`logrotate`)
- [ ] Grafana dashboard sudah di-import untuk monitoring backend
- [ ] Rate limit production-grade (default `8/min` text, `2/min` vision)
- [ ] Test query end-to-end dengan 3 role: `public`, `mahasiswa`, `admin`
- [ ] Llama Guard 3 sudah di-pull (`ollama list`)
- [ ] Health check return `healthy`, bukan `degraded`

---

## Sumber Data RAG

Sistem mengindex **dua sumber data komplementer** ke dalam **satu Qdrant collection** (`unhas_docs`). Setiap chunk punya field `source_type` di metadata untuk identifikasi asalnya:

| Sumber | `source_type` | Isi | Pipeline | Sifat |
|---|---|---|---|---|
| **API SatuData UNHAS** (JSON) | `narrative` | Data tabular akademik: fakultas, prodi, mahasiswa, jadwal, mata kuliah, dll | JSON → narasi teks via `preprocess-template.py` → index | Data live, sering update |
| **PDF Resmi UNHAS** | `pdf` | Dokumen formal: SOP, Pedoman, Rubrik, Manual aplikasi, dll | PDF → layout extraction (text + table + image description) → index | Data statis, jarang update |

**Kenapa dua sumber dipakai bersamaan:**

- **JSON narrative** bagus untuk pertanyaan **faktual numerik** (jumlah mahasiswa, daftar prodi, kuota kelas) — datanya selalu fresh dari API.
- **PDF resmi** bagus untuk pertanyaan **prosedural** (cara cuti, syarat skripsi, format formulir) — datanya otoritatif tapi statis.

**Saat user query**, retrieval di Qdrant mencari chunk paling relevan secara semantik **tanpa peduli source_type**. Reranker BGE meranking berdasarkan konteks query. Misal:
- Query "berapa jumlah mahasiswa FT?" → kemungkinan match chunk dari `mahasiswa.json` (source_type=narrative)
- Query "bagaimana cara cuti akademik?" → kemungkinan match chunk dari `SOP_Cuti.pdf` (source_type=pdf)
- Query "siapa wali akademik saya?" → match chunk procedural (PDF) + faktual (JSON dosen, jika ada)

**Catatan untuk Tim 3 (Evaluasi):** field `source_type` di response `sources[].source_type` (planned) bisa dipakai sebagai dimensi evaluasi — apakah retrieval menarik dari sumber yang tepat untuk tipe pertanyaan tertentu.

Detail per sumber:
- **Sumber 1** — JSON Narrative: lihat section [Data Pipeline: JSON → Narasi → Qdrant](#data-pipeline-json--narasi--qdrant)
- **Sumber 2** — PDF Resmi: lihat section [PDF Indexing (Production-Grade)](#pdf-indexing-production-grade)

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

## PDF Indexing (Production-Grade)

Selain narasi JSON, RAG core juga mengindex dokumen PDF resmi UNHAS (SOP, Pedoman, Rubrik, dll) di `data/pdfs/`. Pipeline indexing dirancang untuk handle PDF dengan konten campuran: teks, tabel, gambar diagram, dan scan dokumen.

### Strategi Extraction

Setiap PDF di-route ke salah satu strategi berdasarkan karakteristiknya:

| Strategy | Library | Kapan dipakai | Output |
|---|---|---|---|
| **fast** | PyMuPDF + PaddleOCR fallback | PDF text-heavy (SOP, surat resmi) | Text per halaman |
| **hi_res** | Unstructured.io `partition_pdf` | PDF rich (Pedoman, Manual dengan tabel/gambar) | Element terstruktur: Title, Text, Table, Image |
| **auto** | Detektor heuristik | Default — pilih otomatis per file | Pilih fast/hi_res berdasarkan kepadatan gambar |

Konfigurasi via `.env`:
```bash
PDF_EXTRACTION_STRATEGY=auto         # auto | fast | hi_res
PDF_EXTRACT_TABLES=true              # extract tabel ke Markdown
PDF_EXTRACT_IMAGES=true              # extract gambar untuk description
PDF_DESCRIBE_IMAGES=auto             # auto: deskripsikan kalau LLM_SUPPORTS_VISION
PDF_MIN_IMAGE_SIZE_KB=20             # skip gambar < 20 KB (icon/logo)
PDF_TABLE_MAX_CHARS=2000             # tabel < 2000 char → digabung dengan section
```

### Element Types yang Diproses

| Element | Treatment | Element_type di metadata |
|---|---|---|
| Title/Heading | Awal section baru, di-prepend ke chunk | `Title` |
| NarrativeText, ListItem | Digabung sampai mendekati CHUNK_SIZE | `NarrativeText` |
| Table | Convert HTML → Markdown; chunk independen jika besar | `Table` |
| Image (informative) | Describe via vision LLM → text chunk | `ImageDescription` |
| Image (dekoratif) | Skip (logo, tanda tangan, header decoration) | — |
| Header/Footer/PageNumber | Selalu skip | — |

**Klasifikasi image informative vs dekoratif:**
1. Heuristic awal: ukuran file < `PDF_MIN_IMAGE_SIZE_KB` atau dimensi terlalu kecil → skip
2. Aspect ratio ekstrem (mis. line separator > 15:1) → skip
3. Sisanya dikirim ke Qwen-VL untuk deskripsi. Jika hasil deskripsi diawali `DEKORATIF` atau `TIDAK JELAS`, chunk tidak disimpan.

### Kapan Vision Model Dipanggil & Dimana Output-nya?

**Vision model HANYA dipanggil saat indexing**, tidak saat user query:

| Operasi | Vision dipanggil? |
|---|---|
| User chat di `/api/query` atau `/api/chat` | ❌ Tidak |
| Indexing PDF, strategy `fast` | ❌ Tidak (text-only) |
| Indexing PDF, strategy `hi_res`, `LLM_SUPPORTS_VISION=false` (dev) | ❌ Tidak (graceful skip) |
| Indexing PDF, strategy `hi_res`, `LLM_SUPPORTS_VISION=true` (POC) | ✅ Per gambar informative |

**Alur deskripsi gambar → chunk Qdrant:**

```
Image element (Unstructured.io)
   ↓ base64 di memory
describe_image(image_bytes)
   → POST ke vLLM /v1/chat/completions (atau Ollama /api/generate)
   → return string deskripsi: "Diagram alur pengajuan cuti..."
   ↓
Replace element Image → element ImageDescription (text)
   ↓
Smart chunking → chunk independen:
   text: "## {section}\n\n[Deskripsi Gambar] {deskripsi dari vision LLM}"
   metadata: { element_type: "ImageDescription", page, file_name, ... }
   ↓
Embed teks deskripsi via Qwen3-Embedding → vector 1024-dim
   ↓
Upsert ke Qdrant collection `unhas_docs`
```

**Yang disimpan vs tidak:**

| Item | Disimpan? | Lokasi |
|---|---|---|
| Teks deskripsi gambar | ✅ | Qdrant chunk (`element_type=ImageDescription`) |
| Vector embedding teks deskripsi | ✅ | Qdrant (searchable via ANN) |
| Metadata (file, page, section) | ✅ | Qdrant payload |
| Image bytes asli (PNG/JPG) | ❌ | Dibuang setelah dideskripsikan — hemat storage |
| In-memory description cache | ⚠️ Sementara | Hilang saat process restart, tujuannya skip duplicate call di run yang sama |

**Kenapa tidak simpan image asli:** RAG retrieval pakai semantic similarity di vector teks. Image asli tidak punya nilai untuk pencarian ulang. Sources yang dikembalikan ke user cuma `file_name + page`, bukan gambar.

**Saat user query:** sistem hanya semantic search di Qdrant, retrieval chunk teks (termasuk yang `element_type=ImageDescription`), kirim ke LLM utama sebagai context. Vision LLM **tidak dipanggil ulang**.

**Implikasi cost:** Vision LLM = one-time cost saat indexing per file. Re-indexing pun cuma proses file yang content hash-nya berubah (lihat section incremental di bawah). Untuk 28 PDF dengan rata-rata 20 gambar informative per file, total ~560 panggilan vision LLM untuk first-time indexing, lalu hampir nol untuk update rutin.

### Incremental Indexing dengan Content Hash

Setiap PDF di-hash (SHA256) dan disimpan di metadata Qdrant. Saat re-index:

| Kondisi | Aksi |
|---|---|
| File baru (belum ada di Qdrant) | Index full |
| File ada, hash sama | Skip (no-op) |
| File ada, hash berbeda | Delete chunks lama → index ulang |

Ini menghemat waktu re-index drastis. Untuk PDF set 28 file, biasanya cuma 1-2 file yang berubah per update — sisanya skip.

### Cara Indexing

```powershell
# Append (skip yang sudah ada, hanya proses file baru/berubah)
python scripts/index_documents.py

# Force full re-index (hapus semua chunks PDF lama)
python scripts/index_documents.py --force

# Custom directory
python scripts/index_documents.py /path/to/pdfs --force
```

Atau via API:
```bash
curl -X POST http://localhost:8000/api/index \
  -H "Content-Type: application/json" \
  -d '{"directory":"data/pdfs","force":true}'
```

### Performance & Resource

| Operasi | Waktu (per page) | Resource |
|---|---|---|
| Strategy `fast` (text-only) | ~0.1s | CPU |
| Strategy `fast` + OCR fallback | ~2-5s | GPU (PaddleOCR) |
| Strategy `hi_res` (layout detection) | ~5-15s | CPU (heavy) |
| Image description (Qwen-VL) | ~2-5s per image | GPU (vLLM) |
| Embedding chunk | ~50ms per chunk | GPU/CPU |

Untuk 28 PDF UNHAS dengan rata-rata 20 halaman dan 3 gambar informatif per halaman:
- First-time full index: ~30-90 menit (tergantung strategy auto-detection)
- Incremental update 1-2 file: ~1-5 menit

### Filter Berdasarkan Element Type di Retrieval

Metadata `element_type` di setiap chunk bisa dipakai untuk targeting retrieval. Contoh query:
- "Berapa nominal UKT 2025 untuk fakultas teknik?" → cenderung match chunk `element_type=Table`
- "Bagaimana diagram alur pengajuan cuti?" → cenderung match `element_type=ImageDescription`

Sistem reranker BGE otomatis mempertimbangkan konteks teks, tapi untuk advanced filtering (mis. filter HARD pakai Qdrant filter), bisa ditambahkan di pipeline retrieval.

---

## API Reference

Semua endpoint tersedia di `http://localhost:8000/docs` (Swagger UI).

### Untuk Integrasi BE Eksternal (Tim 1 — BE + FE)

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

## Untuk Tim 1 (BE + FE)

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

## Untuk Tim 3 (Evaluasi)

Tim 3 melakukan evaluasi kualitas pipeline RAG secara kuantitatif. Endpoint yang dipakai sama dengan Tim 1: **`POST /api/query`**.

### Metrik yang Bisa Dievaluasi

| Metrik | Cara Ukur |
|---|---|
| **Faithfulness** | Apakah jawaban sesuai dengan source dokumen? |
| **Answer Relevancy** | Apakah jawaban menjawab pertanyaan? |
| **Context Recall** | Apakah dokumen yang diambil relevan? |
| **Context Precision** | Seberapa presisi retrieval? |

### Setup Evaluasi

```python
import httpx, json

RAG_BASE_URL = "http://rag-core:8000"

def evaluate_query(question: str, expected_answer: str = None):
    resp = httpx.post(
        f"{RAG_BASE_URL}/api/query",
        json={"question": question, "history": [], "role": "public"},
        timeout=60,
    )
    result = resp.json()
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "mode": result["debug"]["mode"],
        "top_score": result["debug"]["top_score"],
        "intent": result["debug"].get("intent"),
    }
```

> **Catatan:** Field `debug.mode` berguna untuk evaluasi — query yang masuk sebagai `rag` atau `cache_hit` adalah kandidat evaluasi RAG. Query `chitchat` / `out_of_scope` bukan domain RAG.

---

## Progress

- [x] RAG pipeline (retrieval + rerank + generation)
- [x] PDF indexing production-grade (layout-aware, table→markdown, image→vision description, content hash incremental)
- [x] JSON narrative pipeline (preprocess-template + index_narratives)
- [x] Multi-turn conversation dengan history
- [x] Streaming response (SSE)
- [x] Auth (JWT + bcrypt, PostgreSQL)
- [x] Session & message persistence (PostgreSQL)
- [x] Redis semantic cache
- [x] Rate limiting (slowapi)
- [x] Layer 1: Harmful keyword filter
- [x] Layer 2: Moderation (Llama Guard 3 via Ollama dengan URL terpisah dari LLM utama)
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
