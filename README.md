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
9. [Data Pipeline: JSON → Narasi → Qdrant](#data-pipeline-json--narasi--qdrant)
10. [API Reference](#api-reference)
11. [Environment Variables](#environment-variables)
12. [Untuk Tim 1 (BE + FE)](#untuk-tim-1-be--fe)
13. [Untuk Tim 3 (Evaluasi)](#untuk-tim-3-evaluasi)
14. [Progress](#progress)

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
>
> Service yang membutuhkan GPU (vLLM, TEI, Ollama) dan monitoring stack dicomment di `docker-compose.poc.yml` karena konfigurasinya menyesuaikan infrastruktur server — uncomment dan sesuaikan sebelum deploy.

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
│         ├──▶ [vLLM :8001]              (GPU — LLM serving)      │
│         ├──▶ [Ollama :11434]           (GPU — Llama Guard 3)    │
│         ├──▶ [TEI Embed :8002]         (CPU/GPU — Embedding)    │
│         ├──▶ [TEI Rerank :8003]        (CPU/GPU — Reranker)     │
│         ├──▶ [Qdrant :6333]            (Vector DB)              │
│         ├──▶ [PostgreSQL :5432]        (Session DB)             │
│         ├──▶ [Redis :6379]             (Semantic cache)         │
│         └──▶ [MinIO :9000]             (Image storage)          │
│                                                                 │
│   Monitoring (opsional):                                        │
│         [Prometheus]   ◀── scrape /metrics dari backend          │
│         [Grafana :3000] ◀── dashboard performance + RAG         │
└─────────────────────────────────────────────────────────────────┘
```

Semua service berkomunikasi via **Docker internal network** — tidak ada port yang terbuka ke publik kecuali backend port 8000 (di belakang reverse proxy untuk HTTPS).

### Prasyarat Hardware & Software

| Komponen | Minimum | Direkomendasikan |
|---|---|---|
| GPU | NVIDIA dengan VRAM 24GB | NVIDIA L40S 48GB |
| CUDA | 12.1+ | 12.2+ |
| RAM | 32 GB | 64 GB |
| Disk | 100 GB SSD (model + data + image storage) | 500 GB NVMe |
| OS | Ubuntu 22.04 LTS / Debian 12 | Ubuntu 22.04 LTS |
| Docker | Docker Engine 24.0+ | Docker Engine 26.0+ |
| Docker Compose | v2.20+ | v2.27+ |
| NVIDIA Container Toolkit | latest | latest |
| Python | 3.11 (untuk persiapan model lokal) | 3.11 |

### 1. Persiapan Server

#### 1.1 Install Docker & NVIDIA Container Toolkit

```bash
# Docker Engine
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker

# NVIDIA Container Toolkit (wajib untuk GPU dalam container)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verifikasi GPU dapat diakses dari container
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

#### 1.2 Clone Repository

```bash
git clone https://github.com/Ikrar06/rag-prototype
cd rag-prototype
```

### 2. Persiapan Model & Data

Model dan data perlu disiapkan **sebelum** container backend di-build, karena akan di-mount sebagai volume read-only.

#### 2.1 Install Python Dependencies (Lokal)

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 2.2 Download Model

```bash
python scripts/download_models.py
```

Output: `models/intent_classifier/` (~500MB) + `models/bge-reranker-v2-m3/` (~2.3GB).

> **Catatan:** Model **LLM** (Qwen3-VL-8B) dan **Embedding** (Qwen3-Embedding-0.6B) tidak perlu didownload manual — di-load otomatis oleh vLLM dan TEI saat startup dari HuggingFace cache di volume `hf_cache`.

#### 2.3 Generate Narasi & Index Data

```bash
# 1. Pastikan file JSON ada di data/json/
ls data/json/   # fakultas.json, prodi.json, mahasiswa.json, dst.

# 2. Generate narasi teks
python scripts/preprocess-template.py

# 3. Hasilnya di data/narratives/<endpoint>/*.txt
```

Indexing ke Qdrant dilakukan **setelah** Qdrant container jalan (step 5).

### 3. Konfigurasi Environment

```bash
cp .env.poc .env
nano .env   # atau editor favorit
```

#### 3.1 Variable Wajib Diganti (CHANGE_ME)

| Variable | Wajib | Keterangan |
|---|---|---|
| `JWT_SECRET` | ✅ | String random ≥ 32 byte. Generate: `python -c "import secrets; print(secrets.token_hex(32))"` |
| `POSTGRES_PASSWORD` | ✅ | Password PostgreSQL container |
| `DATABASE_URL` | ✅ | Ganti `CHANGE_ME` di string dengan password yang sama di atas |
| `MINIO_USER` | ✅ | Username admin MinIO |
| `MINIO_PASSWORD` | ✅ | Password admin MinIO (min 8 karakter) |
| `MINIO_ACCESS_KEY` | ✅ | Access key untuk API MinIO (bisa sama dengan `MINIO_USER`) |
| `MINIO_SECRET_KEY` | ✅ | Secret key untuk API MinIO (bisa sama dengan `MINIO_PASSWORD`) |
| `ALLOWED_ORIGINS` | ✅ | Domain frontend production, dipisah koma |
| `UNHAS_API_BASE_URL` | ❌ | Base URL API UNHAS — kosongkan jika belum tersedia |

#### 3.2 Perbedaan Env POC vs Dev

| Variable | Dev | POC |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `vllm` |
| `LLM_MODEL` | `qwen2.5:7b` | `Qwen/Qwen3-VL-8B-Instruct` |
| `LLM_BASE_URL` | `http://localhost:11434` | `http://vllm:8001` |
| `LLM_SUPPORTS_VISION` | `false` | `true` |
| `LLM_MAX_TOKENS` | `1024` | `2048` |
| `EMBED_PROVIDER` | `huggingface` (in-process) | `tei` (service terpisah) |
| `EMBED_BASE_URL` | — | `http://tei-embed:8002` |
| `RERANKER_PROVIDER` | `sentence_transformers` | `tei` |
| `RERANKER_BASE_URL` | — | `http://tei-rerank:8003` |
| `MODERATION_BACKEND` | `passthrough` | `ollama` (Llama Guard 3 via instance Ollama terpisah) |
| `MODERATION_BASE_URL` | — | `http://ollama:11434` (terpisah dari `LLM_BASE_URL`) |
| `QDRANT_URL` | `http://localhost:6333` | `http://qdrant:6333` |
| `REDIS_URL` | `redis://localhost:6379/0` | `redis://redis:6379/0` |
| `DATABASE_URL` | `postgresql://...@localhost:5432/...` | `postgresql+asyncpg://...@postgres:5432/...` |
| `RATE_LIMIT_TEXT_PER_MINUTE` | `20` | `8` |
| `RATE_LIMIT_VISION_PER_MINUTE` | — | `2` |
| `LOG_LEVEL` | `DEBUG` | `INFO` |
| `INTENT_MODEL_PATH` | `models/intent_classifier` | `/app/models/intent_classifier` |
| `STORAGE_BACKEND` | — | `minio` |

### 4. Uncomment & Konfigurasi Service GPU

Edit `docker-compose.poc.yml`, uncomment dan sesuaikan blok berikut:

#### 4.1 vLLM (LLM Server)

```yaml
vllm:
  image: vllm/vllm-openai:latest
  command: >
    --model Qwen/Qwen3-VL-8B-Instruct
    --port 8001
    --max-model-len 8192
    --gpu-memory-utilization 0.92
    --max-num-seqs 64
    --limit-mm-per-prompt image=2
    --enable-prefix-caching
    --served-model-name qwen3-vl-8b
  volumes:
    - hf_cache:/root/.cache/huggingface
  restart: unless-stopped
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

Parameter penting:

| Flag | Fungsi | Rekomendasi |
|---|---|---|
| `--gpu-memory-utilization` | % VRAM untuk model | `0.92` untuk L40S 48GB; turunkan ke `0.85` jika OOM |
| `--max-model-len` | Max context length token | `8192` (cukup untuk RAG + history) |
| `--max-num-seqs` | Max concurrent requests | `64` untuk 50–100 user, `128` untuk lebih |
| `--limit-mm-per-prompt image=2` | Max gambar per request | `2` (sesuai konfigurasi backend) |
| `--enable-prefix-caching` | Cache prefix prompt | Wajib aktif (hemat VRAM + latency) |

#### 4.2 TEI — Text Embeddings Inference

```yaml
tei-embed:
  image: ghcr.io/huggingface/text-embeddings-inference:cpu-latest
  command: --model-id Qwen/Qwen3-Embedding-0.6B --port 8002
  ports:
    - "8002"
  restart: unless-stopped

tei-rerank:
  image: ghcr.io/huggingface/text-embeddings-inference:cpu-latest
  command: --model-id BAAI/bge-reranker-v2-m3 --port 8003
  ports:
    - "8003"
  restart: unless-stopped
```

> **GPU vs CPU:** Ganti `cpu-latest` ke `cuda12.2-latest` untuk akselerasi GPU. Embedding/reranker cukup ringan untuk CPU jika user < 100 concurrent.

#### 4.3 Ollama untuk Moderasi (Llama Guard 3)

```yaml
ollama:
  image: ollama/ollama:latest
  volumes:
    - ollama_data:/root/.ollama
  restart: unless-stopped
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

Setelah container jalan, pull model Llama Guard:
```bash
docker compose -f docker-compose.poc.yml exec ollama ollama pull llama-guard3:1b
```

#### 4.4 Monitoring (Opsional)

```yaml
prometheus:
  image: prom/prometheus:latest
  volumes:
    - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    - prometheus_data:/prometheus
  restart: unless-stopped

grafana:
  image: grafana/grafana:latest
  environment:
    GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}
  volumes:
    - grafana_data:/var/lib/grafana
  ports:
    - "3000:3000"
  restart: unless-stopped
```

Aktifkan volume di bagian bawah file:
```yaml
volumes:
  pg_data:
  redis_data:
  qdrant_data:
  minio_data:
  hf_cache:           # uncomment
  ollama_data:        # uncomment
  prometheus_data:    # uncomment jika monitoring aktif
  grafana_data:       # uncomment jika monitoring aktif
```

Buat `monitoring/prometheus.yml`:
```yaml
global:
  scrape_interval: 15s
scrape_configs:
  - job_name: 'backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: /metrics
```

### 5. Build & Jalankan Services

```bash
# Build image backend (sekitar 5-10 menit, tergantung jaringan)
docker compose -f docker-compose.poc.yml build backend

# Jalankan semua service
docker compose -f docker-compose.poc.yml up -d

# Cek status
docker compose -f docker-compose.poc.yml ps
```

Output yang diharapkan: semua service `running` atau `healthy`.

Ikuti log backend saat startup pertama:
```bash
docker compose -f docker-compose.poc.yml logs -f backend
```

> **Catatan vLLM:** Saat pertama kali jalan, vLLM akan download model dari HuggingFace (~16GB). Tunggu sampai log menunjukkan `Uvicorn running on http://0.0.0.0:8001`. Untuk model gated (perlu HF token), tambahkan environment variable di service vLLM:
> ```yaml
> environment:
>   HF_TOKEN: ${HF_TOKEN}
> ```

### 6. Setup Post-Deploy

#### 6.1 Buat Bucket MinIO

```bash
# Akses console MinIO via browser: http://server-ip:9001
# Login dengan MINIO_USER / MINIO_PASSWORD
# Buat bucket dengan nama sesuai MINIO_BUCKET (default: ragchat-images)

# Atau via CLI:
docker compose -f docker-compose.poc.yml exec minio \
  mc alias set local http://localhost:9000 $MINIO_USER $MINIO_PASSWORD

docker compose -f docker-compose.poc.yml exec minio \
  mc mb local/ragchat-images
```

#### 6.2 Index Data ke Qdrant

```bash
# Index narasi yang sudah di-generate di step 2.3
docker compose -f docker-compose.poc.yml exec backend \
  python backend/services/index_narratives.py --force
```

#### 6.3 Seed User Awal (Opsional)

User otomatis di-seed dari `data/users.json` saat backend startup. Untuk seed manual:
```bash
docker compose -f docker-compose.poc.yml exec backend \
  python -c "from backend.services.auth import seed_users_from_json; seed_users_from_json()"
```

### 7. Verifikasi Deployment

#### 7.1 Health Check

```bash
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

`"status": "degraded"` berarti ada service yang belum terhubung — cek `docker compose logs <service_name>`.

#### 7.2 Test Query

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "apa syarat cuti akademik?",
    "history": [],
    "role": "public"
  }'
```

#### 7.3 Swagger UI

Buka di browser: `http://server-ip:8000/docs`

### 8. Reverse Proxy + HTTPS (Production)

Untuk production, **JANGAN** expose port 8000 langsung. Pakai reverse proxy dengan HTTPS:

Contoh konfigurasi **nginx**:
```nginx
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

        # SSE streaming
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 180s;
    }
}
```

### 9. Maintenance & Operations

#### 9.1 Update Image

```bash
# Pull image latest
docker compose -f docker-compose.poc.yml pull

# Rebuild backend (jika ada perubahan kode)
docker compose -f docker-compose.poc.yml build backend

# Restart dengan image baru
docker compose -f docker-compose.poc.yml up -d
```

#### 9.2 Backup Data

```bash
# Backup volume PostgreSQL
docker compose -f docker-compose.poc.yml exec postgres \
  pg_dump -U ragchat ragchat > backup_$(date +%Y%m%d).sql

# Backup volume Qdrant
docker run --rm -v rag-prototype_qdrant_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/qdrant_$(date +%Y%m%d).tar.gz /data
```

#### 9.3 Scale Backend

```bash
# Scale ke 2 replica (butuh load balancer di depan)
docker compose -f docker-compose.poc.yml up -d --scale backend=2
```

#### 9.4 View Logs

```bash
# Backend log saja
docker compose -f docker-compose.poc.yml logs -f backend

# Semua service, last 100 baris
docker compose -f docker-compose.poc.yml logs --tail=100

# Filter log dengan request_id tertentu
docker compose -f docker-compose.poc.yml logs backend | grep "request_id=a3f9b1c2"
```

### 10. Troubleshooting

| Masalah | Penyebab | Solusi |
|---|---|---|
| `vllm` OOM saat startup | VRAM tidak cukup | Turunkan `--gpu-memory-utilization` ke 0.85 atau `--max-model-len` ke 4096 |
| `backend` tidak connect ke `vllm` | Service belum ready | Tambah `depends_on: vllm: condition: service_healthy` |
| Health check `intent_model: false` | Model belum di-mount | Pastikan `./models` di host sudah ada, restart backend |
| `MinIO` bucket not found | Bucket belum dibuat | Jalankan step 6.1 |
| Query selalu `out_of_scope` | Intent model salah load | Cek log: `intent_classifier_loaded path=...` |
| Streaming response stuck | nginx buffering aktif | Tambah `proxy_buffering off;` di nginx config |
| `degraded` di health check | Ada service `false` | Cek `docker compose logs <service>` untuk yang false |

### 11. Production Checklist

Sebelum go-live, pastikan:

- [ ] Semua `CHANGE_ME` di `.env` sudah diganti dengan nilai random
- [ ] `JWT_SECRET` minimal 32 byte hex
- [ ] HTTPS aktif di reverse proxy (Let's Encrypt atau cert lain)
- [ ] `ALLOWED_ORIGINS` sudah dibatasi ke domain production
- [ ] Firewall: hanya port 443 (HTTPS) yang terbuka publik
- [ ] Backup otomatis PostgreSQL + Qdrant terjadwal (cron)
- [ ] Log rotation aktif (`logrotate` atau setara)
- [ ] Monitoring (Prometheus + Grafana) aktif dengan alert
- [ ] Rate limit production-grade di `.env` (default `8/min` text, `2/min` vision)
- [ ] Test query end-to-end dengan beberapa role (`public`, `mahasiswa`, `admin`)
- [ ] Verifikasi MinIO bucket policy (private read, presigned URL untuk download)

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
- [x] PDF indexing (PaddleOCR + Unstructured.io)
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
