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
12. [Retrieval Strategy (Production-Grade)](#retrieval-strategy-production-grade)
13. [API Reference](#api-reference)
14. [Environment Variables](#environment-variables)
15. [Untuk Tim 1 (BE + FE)](#untuk-tim-1-be--fe)
16. [Untuk Tim 3 (Evaluasi)](#untuk-tim-3-evaluasi)
17. [Progress](#progress)

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
git clone https://github.com/ai-llm-unhas/rag-core-system
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

**Spreadsheet Sycn**

Untuk QA/evaluasi, jalankan QA sheet sync via service `qa-sheet-sync` di compose dev. Cara setup dan command ada di [automation_qa/README.md](automation_qa/README.md).

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

### 6. Index Data ke Qdrant

Dua sumber data masuk ke **collection Qdrant yang sama** (`unhas_docs`). Bisa di-index berurutan atau independen. Pastikan Qdrant container sudah jalan dulu (Step 3).

#### 6a. Index Narrative dari JSON API UNHAS

```powershell
# 1. Taruh file JSON di data/json/
ls data/json/   # fakultas.json, prodi.json, mahasiswa.json, dst.

# 2. Generate narasi teks (output ke data/narratives/)
python scripts/preprocess-template.py

# 3. Index narratives ke Qdrant
python backend/services/index_narratives.py

# Force re-index (jika ada perubahan narasi):
python backend/services/index_narratives.py --force

# Re-index endpoint tertentu saja:
python backend/services/index_narratives.py --force --endpoint fakultas,prodi
```

> **Endpoint JSON yang didukung:** `fakultas`, `prodi`, `jenjang`, `kurikulum`, `mata-kuliah`, `prasyarat`, `rps`, `kelas`, `jadwal`, `fasilitas`, `pmb`, `pengumuman`, `mahasiswa`

#### 6b. Index Dokumen PDF Resmi

```powershell
# 1. Taruh file PDF di data/pdfs/
ls data/pdfs/   # SOP_*.pdf, Pedoman_*.pdf, dll.

# 2. Index PDF ke Qdrant (incremental — skip file dengan hash sama)
python scripts/index_documents.py

# Force full re-index (hapus semua chunks PDF lama):
python scripts/index_documents.py --force

# Custom directory:
python scripts/index_documents.py /path/to/pdfs --force
```

> **Note:** Indexing PDF di dev pakai strategy `fast` (lihat `PDF_EXTRACTION_STRATEGY` di `.env`). Gambar tidak akan dideskripsikan karena `LLM_SUPPORTS_VISION=false` — itu wajar. Untuk image description aktif, lihat Setup POC. Detail lengkap: section [PDF Indexing (Production-Grade)](#pdf-indexing-production-grade).

#### Verifikasi Indexing

```powershell
# Cek total chunks di Qdrant
curl http://localhost:6333/collections/unhas_docs

# Atau via API backend (perlu server jalan dulu)
curl http://localhost:8000/api/health
```

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
   Internet (browser user + QA reviewer)
      │
      ▼ port 80 (HTTP) atau 443 (HTTPS via nginx)
  [Reverse Proxy / Load Balancer]   ← nginx/Traefik (di luar compose, opsional)
      │
      ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       Docker Network: ragchat                          │
│                                                                        │
│   [Backend :8000]                                                      │
│   (GPU — PaddleOCR)                                                    │
│   ↑ port 80:8000 exposed publik                                        │
│   ↑ /api/chat, /api/query, /api/files/{key}, /api/health, dst          │
│         │                                                              │
│         ├──▶ [vLLM :8001]              (GPU — Qwen3-VL-8B multimodal)  │
│         ├──▶ [Ollama :11434]           (GPU — Llama Guard 3 1B)        │
│         ├──▶ [TEI Embed :8002]         (GPU — Qwen3-Embedding-0.6B)    │
│         ├──▶ [TEI Rerank :8003]        (GPU — bge-reranker-v2-m3)      │
│         ├──▶ [Qdrant :6333]            (Vector DB — unhas_docs)        │
│         ├──▶ [PostgreSQL :5432]        (Session + messages + images)   │
│         ├──▶ [Redis Stack :6379]       (Semantic cache, skip vision)   │
│         └──▶ [MinIO :9000]             (Image storage — internal only) │
│                                          loopback bind 127.0.0.1       │
│                                                                        │
│   [QA Sheet Sync] ──▶ PostgreSQL ──▶ Google Sheets (1 jam/cycle)       │
│   (image standalone, image URL = backend /api/files/*)                 │
│                                                                        │
│   [Prometheus :9090]  ◀── scrape /metrics dari backend                 │
│   [Grafana :3000]     ◀── dashboard performance                        │
└────────────────────────────────────────────────────────────────────────┘

Image upload flow (vision):
  Browser → POST /api/chat (with base64 image)
         → backend validate + resize + upload ke MinIO internal
         → presigned key disimpan ke postgres messages.images
         → response answer ke browser

Image download (QA Sheet):
  Reviewer click URL di Sheet (http://<vm>/api/files/<key>)
         → backend stream object dari MinIO internal
         → image tampil di browser reviewer
  (port 9000 MinIO TIDAK perlu di-expose publik — backend yang proxy)
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
| NVIDIA Driver | ≥ 545.x (CUDA 12.4 runtime compat) | ≥ 590.x |

### VRAM Budget (L40S 48GB)

Semua service GPU dalam **1 GPU yang sama**, di-share via NVIDIA Container Toolkit. Allocation by design:

| Service | VRAM | Tipe alloc |
|---|---|---|
| vLLM (Qwen3-VL-8B, `--gpu-memory-utilization 0.80`) | ~38 GB | Statis (reserve di startup) |
| TEI Embed (Qwen3-Embedding-0.6B) | ~1.5 GB | Statis |
| TEI Rerank (bge-reranker-v2-m3) | ~2.3 GB | Statis |
| Ollama (Llama Guard 3 1B, 4-bit) | ~1.5 GB | Statis |
| **Backend (PaddleOCR)** | **~1-2 GB** | **On-demand (saat indexing/OCR)** |
| **Total** | **~44.5 GB** | Free margin: **~3-4 GB** |

**Catatan tuning:**
- vLLM dulu pakai `--gpu-memory-utilization 0.85` (~40.8 GB). **Diturunkan ke 0.80** supaya PaddleOCR di backend punya margin yang aman.
- Kalau vLLM butuh throughput lebih tinggi (max-num-seqs > 64 atau prompt panjang > 8192), naikkan lagi `--gpu-memory-utilization` dan **matikan PaddleOCR GPU** (`OCR_USE_GPU=false`). Trade-off: indexing PDF scanned jadi 5-15× lebih lambat tapi memungkinkan vLLM concurrent lebih tinggi.
- PaddleOCR alokasi 1-2 GB peak hanya saat ada page yang butuh OCR (PDF scanned). Untuk PDF digital (text-based), VRAM tidak ke-touch.

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
git clone https://github.com/ai-llm-unhas/rag-core-system
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

**Siapkan dokumen PDF resmi:**

PDF resmi UNHAS biasanya didistribusikan sebagai **zip archive** (mis. dari Drive, repo internal). Pakai `scripts/fetch_pdfs.py` untuk extract otomatis ke `data/pdfs/`:

```bash
# Opsi A — extract dari folder lokal yang berisi .zip
python scripts/fetch_pdfs.py --source /path/to/folder/with/zips

# Opsi B — clone GitHub repo (private/public) lalu extract semua .zip di dalamnya
python scripts/fetch_pdfs.py --source https://github.com/org/pdf-data-repo

# Dry run dulu kalau ragu (lihat apa yang akan di-extract tanpa eksekusi)
python scripts/fetch_pdfs.py --source /path/to/zips --dry-run

# Force overwrite kalau PDF sudah ada dan ingin replace
python scripts/fetch_pdfs.py --source /path/to/zips --force
```

Script ini:
- Scan rekursif semua `.zip` di source
- Extract **hanya file `.pdf`** (buang struktur folder dalam zip)
- Skip PDF yang sudah ada di `data/pdfs/` kecuali `--force`
- Atomic write (extract ke `.tmp` lalu rename, supaya tidak ada file parsial)

Verifikasi hasil:
```bash
ls data/pdfs/
# Contoh: 1.-SOP_Cuti_Akademik.pdf, 2.-Pedoman_Penulisan_Skripsi.pdf,
#         3.-UKT-TAHUN-2025.pdf, ... (200+ file)
```

> PDF akan di-index di Step 8. Folder `data/pdfs/` ke-mount sebagai volume read-only ke container backend (`./data/pdfs:/app/data/pdfs:ro` di `docker-compose.poc.yml`). Indexing PDF di POC akan otomatis pakai Qwen3-VL untuk deskripsi gambar (lihat Step 8).

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
| `MINIO_PASSWORD` | password admin MinIO (≥ 8 karakter) — generate: `python -c "import secrets; print(secrets.token_urlsafe(24))"` |
| `MINIO_ACCESS_KEY` | sama persis dengan `MINIO_USER` |
| `MINIO_SECRET_KEY` | sama persis dengan `MINIO_PASSWORD` |
| `BACKEND_PUBLIC_URL` | URL publik backend yang reviewer QA pakai untuk akses gambar via `/api/files/*`. Format: `http://<ip-vm>` (tanpa port kalau port 80 mapping; dengan port kalau langsung 8000) |
| `GRAFANA_PASSWORD` | password admin Grafana |
| `ALLOWED_ORIGINS` | domain frontend production, dipisah koma |
| `HF_TOKEN` | (opsional) HuggingFace token, lihat Step 4 |
| `UNHAS_API_BASE_URL` | (opsional) kosongkan jika belum ada |
| `QA_SYNC_GOOGLE_CREDENTIALS_FILE` | path service account JSON di container — default `/run/secrets/google_service_account.json` |
| `QA_SYNC_SPREADSHEET_ID` | ID Google Sheet untuk evaluasi QA (dari URL sheet) |
| `QA_SYNC_WORKSHEET_NAME` | nama tab worksheet (default `Sheet1`) |
| `QA_SYNC_INTERVAL_SECONDS` | interval auto-sync DB → Sheet (default 3600 = 1 jam) |
| `QA_SYNC_START_FROM` | (opsional) ISO timestamp untuk skip message historical. Format: `2026-05-25T14:00:00+08:00` |

**Contoh konkret:** kalau `POSTGRES_PASSWORD=Rahasia123Banget`, maka:
```
DATABASE_URL=postgresql://ragchat:Rahasia123Banget@postgres:5432/ragchat
```

**Catatan `BACKEND_PUBLIC_URL`:** ini krusial untuk QA Sheet integration. URL gambar yang masuk ke kolom `gambar_user` di Sheet adalah `{BACKEND_PUBLIC_URL}/api/files/<key>`. Reviewer click URL → backend stream image dari MinIO internal → tampil di browser. Approach ini menghindari kebutuhan expose port 9000 MinIO publik (firewall provider friendly).

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

**Spreadsheet Sycn**

Untuk QA/evaluasi, `qa-sheet-sync` sudah ikut jalan saat `docker compose -f docker-compose.poc.yml up -d`. Detail setup dan log ada di [automation_qa/README.md](automation_qa/README.md).

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

### Step 7.5 — Seed User Demo

User demo akan otomatis ke-seed dari `users.json` saat backend startup. Verifikasi:

```bash
docker compose -f docker-compose.poc.yml exec postgres \
  psql -U ragchat -d ragchat -c "SELECT username, role FROM users;"
```

Kalau kosong (0 rows), trigger seed manual:

```bash
docker compose -f docker-compose.poc.yml exec backend \
  python -c "from backend.services.auth import seed_users_from_json; seed_users_from_json()"
```

**User demo default** (bisa diedit di `users.json` sebelum build):

| Username | Password | Role |
|---|---|---|
| `mahasiswa1` | `demo123` | mahasiswa |
| `staf1` | `staf456` | staf_akademik |
| `calon1` | `calon789` | calon_mahasiswa |

> Untuk production, **JANGAN** pakai password ini. Buat user via script terpisah atau integrasi SSO UNHAS. Password di-hash dengan bcrypt sebelum disimpan ke `users.password_hash`.

### Step 8 — Index Data ke Qdrant

Dua sumber data masuk ke **collection `unhas_docs` yang sama**. Index berurutan:

**8a. Index Narrative dari JSON API UNHAS**

```bash
# Pastikan narasi sudah ter-generate di Step 2 (data/narratives/*.txt)
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

**8b. Index Dokumen PDF Resmi**

**Verifikasi dulu** PDF sudah ke-mount ke container backend (host `data/pdfs/` → container `/app/data/pdfs/`):

```bash
docker compose -f docker-compose.poc.yml exec backend ls /app/data/pdfs | head
# Harus muncul daftar PDF. Kalau "No such file or directory", mount belum aktif —
# cek docker-compose.poc.yml service backend → volumes harus include:
#   - ./data/pdfs:/app/data/pdfs:ro
# Lalu: docker compose -f docker-compose.poc.yml up -d --force-recreate backend
```

Jalankan indexing:

```bash
docker compose -f docker-compose.poc.yml exec backend \
  python scripts/index_documents.py
```

Output progres yang diharapkan (per-PDF):
```
Loading embedding model...
embed_provider=tei base_url=http://tei-embed:8002
pdf_processing total=214 new=214 changed=0
pdf_extract file=3.-UKT-TAHUN-2025.pdf strategy=hi_res hash=ff1e880c
Reading PDF for file: /app/data/pdfs/3.-UKT-TAHUN-2025.pdf ...
Loading the Table agent ...                           ← table-transformer download (~115 MB, sekali)
Downloading yolox_l0.05.onnx ...                      ← layout model (~217 MB, sekali)
Table model successfully loaded to cpu
pdf_chunked file=3.-UKT-TAHUN-2025.pdf chunks=7
...
embedding_start total_chunks=4400
indexing_complete documents_indexed=4400
```

**First-run akan download model layout sekali (~330 MB total):** `yolox_l0.05.onnx` (layout detection) + `table-transformer-structure-recognition` (table parsing) + `en_core_web_sm` (spaCy). Cached di volume `hf_cache` — re-run tidak download ulang.

> **Catatan POC:** Karena `LLM_SUPPORTS_VISION=true` dan vLLM sudah jalan, indexing PDF akan **otomatis panggil Qwen3-VL** untuk deskripsi gambar informative — terlihat di log sebagai `HTTP Request: POST http://vllm:8001/v1/chat/completions` setelah tahap extract per PDF. Image deskoratif (logo, sampul) atau "TIDAK JELAS" di-skip otomatis. Detail flow: section [PDF Indexing (Production-Grade)](#pdf-indexing-production-grade).

> **Estimasi durasi (214 PDF, L40S 48GB):** first-time full index ~30-90 menit tergantung dominasi tabel/gambar per PDF. Re-indexing incremental cuma ~1-5 menit (lewat hash check).

**Cek total chunks:**
```bash
curl http://localhost:6333/collections/unhas_docs
# expect: points_count ≈ 4000-5000 (narrative + PDF chunks)
```

**Aman di-Ctrl+C kapan saja.** Indexing per-PDF bersifat atomic — chunk hanya commit ke Qdrant setelah satu PDF selesai diproses full. PDF yang sudah masuk Qdrant di-skip otomatis saat re-run via SHA256 content hash. Cukup jalankan command yang sama lagi.

**Reset & re-index dari nol** (kalau ubah `CHUNK_SIZE` / `PDF_EXTRACT_STRATEGY` / `EMBED_MODEL`):
```bash
# Opsi 1 — pakai flag --force (delete collection + re-index)
docker compose -f docker-compose.poc.yml exec backend \
  python scripts/index_documents.py --force

# Opsi 2 — manual delete via Qdrant API lalu re-run
curl -X DELETE http://localhost:6333/collections/unhas_docs
docker compose -f docker-compose.poc.yml exec backend \
  python scripts/index_documents.py
```

#### Tuning TEI — Default GPU, Fallback CPU

Default `docker-compose.poc.yml` pakai **TEI GPU** (`cuda12.2-latest` image) untuk production-grade performance:
- Indexing ~4400 chunks: **2-5 menit** (vs 30-60 menit CPU)
- Query runtime latency: **20-50ms per call** (vs 100-500ms CPU)
- VRAM cost: ~1.5 GB (embed) + ~2.3 GB (rerank) = ~4 GB total

Default `.env.poc`:
```bash
EMBED_BATCH_SIZE=32          # GPU handle batch besar
EMBED_TIMEOUT=120            # GPU cepat, timeout kecil cukup
```

**TEI tag berdasarkan GPU compute capability** (default `89-latest` untuk L40S/RTX 4090):

| GPU | Compute Capability | TEI Tag |
|---|---|---|
| NVIDIA T4 | 7.5 (Turing) | `turing-latest` |
| A100, A30 | 8.0 (Ampere) | `latest` |
| RTX 3090, A10 | 8.6 (Ampere) | `86-latest` |
| **L40S, L40, RTX 4090** | **8.9 (Ada Lovelace)** | **`89-latest`** ← default |
| H100 | 9.0 (Hopper) | `hopper-latest` |

Cek compute capability GPU:
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

**Kalau VRAM ketat** (vLLM butuh full allocation atau GPU < 32 GB), fallback ke TEI CPU:

```yaml
# docker-compose.poc.yml — edit tei-embed dan tei-rerank
tei-embed:
  image: ghcr.io/huggingface/text-embeddings-inference:cpu-latest   # was 89-latest
  command: --model-id Qwen/Qwen3-Embedding-0.6B --port 8002
  environment:
    HF_TOKEN: ${HF_TOKEN:-}
  volumes:
    - hf_cache:/data
  restart: unless-stopped
  # HAPUS blok `deploy:` saat pakai CPU image
```

Lalu turunkan batch + naikkan timeout di `.env`:
```bash
EMBED_BATCH_SIZE=8           # batch kecil supaya single batch ga timeout
EMBED_TIMEOUT=600            # 10 menit per batch — safety margin
```

Restart:
```bash
docker compose -f docker-compose.poc.yml up -d --force-recreate tei-embed tei-rerank
docker compose -f docker-compose.poc.yml restart backend
```

**Cek VRAM available** sebelum migrate:
```bash
nvidia-smi
# Pastikan free ≥ 5 GB setelah vLLM + Ollama jalan
```

Kalau vLLM butuh full VRAM, kurangi `--gpu-memory-utilization` di vLLM command dari `0.85` → `0.75` supaya free ~4-5 GB untuk TEI GPU.

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

#### Safe Git Pull (saat ada modifikasi lokal VM)

**Penting** — di VM POC, file seperti `docker-compose.poc.yml` (port mapping, GPU device IDs) atau `.env` sering di-modify lokal oleh operator/senior. **Jangan** langsung `git pull` — bisa overwrite atau conflict.

Procedure aman:

```bash
# 1. Cek dulu file apa saja yang berubah di lokal vs repo
git status

# 2. Cek file apa saja yang akan masuk dari remote
git fetch origin main
git log HEAD..origin/main --oneline      # commit baru yang akan di-pull
git diff HEAD..origin/main --stat        # file mana saja yang berubah di remote

# 3. Cross-reference: kalau remote mengubah file yang sama dengan local modif → conflict
#    Lihat overlap dengan: git diff HEAD..origin/main --name-only | xargs -I{} sh -c 'git status --porcelain {} 2>/dev/null'

# 4. Strategi A — kalau tidak ada overlap (safe path):
git pull origin main

# 5. Strategi B — kalau ADA overlap (mis. compose berubah di remote DAN dimodif lokal):
#    Stash dulu lokal modif → pull → manual merge
git stash push -m "vm-local-$(date +%Y%m%d)" -- docker-compose.poc.yml .env
git pull origin main
git stash pop                            # akan merge; resolve conflict marker manual
# atau lihat isi stash dulu:
git stash show -p stash@{0}

# 6. Strategi C — paranoid: backup file kritis manual
cp docker-compose.poc.yml docker-compose.poc.yml.vm-backup
cp .env .env.vm-backup
git pull origin main
diff docker-compose.poc.yml.vm-backup docker-compose.poc.yml   # lihat perubahan
# kalau perlu, restore manual: cp docker-compose.poc.yml.vm-backup docker-compose.poc.yml
```

#### Setelah Pull — Rebuild & Restart

**Skenario A — perubahan di backend Python (paling sering):**
```bash
# Rebuild — cache hit di semua layer kecuali COPY backend/
# Estimasi: 1-3 menit
sudo docker compose -f docker-compose.poc.yml build backend

# Recreate
sudo docker compose -f docker-compose.poc.yml up -d --force-recreate backend qa-sheet-sync

# Flush cache supaya tidak return jawaban lama
sudo docker exec rag-prototype-redis-1 redis-cli FLUSHDB
```

**Skenario B — perubahan di frontend (HTML/CSS/JS):**
```bash
# Quick path (tanpa rebuild image, ~10 detik):
sudo docker cp frontend/index.html rag-prototype-backend-1:/app/frontend/
sudo docker cp frontend/style.css rag-prototype-backend-1:/app/frontend/
sudo docker cp frontend/app.js rag-prototype-backend-1:/app/frontend/
# Hard refresh browser (Ctrl+Shift+R) — tidak perlu restart container

# Persistent path (rebuild image):
sudo docker compose -f docker-compose.poc.yml build backend
sudo docker compose -f docker-compose.poc.yml up -d --force-recreate backend
```

**Skenario C — perubahan di Dockerfile/requirements (jarang):**
```bash
# Rebuild full — bisa 10-25 menit kalau perlu re-install CUDA base + paddle/torch
sudo docker compose -f docker-compose.poc.yml build backend
sudo docker compose -f docker-compose.poc.yml up -d --force-recreate backend
```

**Skenario D — perubahan di docker-compose.poc.yml (port, env, dll):**
```bash
# Tidak perlu rebuild, cukup recreate
sudo docker compose -f docker-compose.poc.yml up -d
# Atau force-recreate kalau perubahan env var:
sudo docker compose -f docker-compose.poc.yml up -d --force-recreate backend
```

**Skenario E — perubahan di prompt template (`.py`) saja:**
```bash
# Prompt di-import saat module load — restart tidak cukup,
# harus recreate container karena file di image lama.
sudo docker compose -f docker-compose.poc.yml build backend
sudo docker compose -f docker-compose.poc.yml up -d --force-recreate backend

# ATAU shortcut tanpa rebuild (tidak persistent):
sudo docker cp backend/prompts/templates.py rag-prototype-backend-1:/app/backend/prompts/templates.py
sudo docker compose -f docker-compose.poc.yml restart backend
sudo docker exec rag-prototype-redis-1 redis-cli FLUSHDB
```

**Verifikasi setelah recreate:**
```bash
# Cek startup log — harus muncul keyword_filter_loaded, storage_backend, intent_classifier_loaded
sudo docker compose -f docker-compose.poc.yml logs backend --tail 30

# Cek health
curl http://localhost/api/health | python3 -m json.tool
# Expected: status=healthy, vision_enabled=true, moderation_circuit=closed
```

#### Backup & Operations

```bash
# Backup PostgreSQL (termasuk messages.images — sudah include data attachment URL)
sudo docker compose -f docker-compose.poc.yml exec postgres \
  pg_dump -U ragchat ragchat > backup_$(date +%Y%m%d).sql

# Backup Qdrant volume (chunks dokumen + narasi)
sudo docker run --rm -v rag-prototype_qdrant_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/qdrant_$(date +%Y%m%d).tar.gz /data

# Backup MinIO data (gambar user upload)
sudo docker run --rm -v rag-prototype_minio_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/minio_$(date +%Y%m%d).tar.gz /data

# Force sync QA Sheet manual (di luar interval otomatis)
sudo docker exec rag-prototype-qa-sheet-sync-1 \
  python -m automation_qa.sync_to_sheets --once

# Dry-run sync (read DB & Sheet tanpa append — verify count)
sudo docker exec rag-prototype-qa-sheet-sync-1 \
  python -m automation_qa.sync_to_sheets --once --dry-run

# Flush Redis cache (semantic cache + intent cache)
sudo docker exec rag-prototype-redis-1 redis-cli FLUSHDB

# Scale backend (butuh load balancer di depan)
sudo docker compose -f docker-compose.poc.yml up -d --scale backend=2

# View log filter request_id
sudo docker compose -f docker-compose.poc.yml logs backend | grep "request_id=a3f9b1c2"

# Tail log live multiple service
sudo docker compose -f docker-compose.poc.yml logs -f backend qa-sheet-sync vllm
```

#### SSH Tunnel untuk Admin Tools

MinIO Console dan Qdrant Dashboard bind ke loopback `127.0.0.1` (tidak ke-expose publik). Akses via SSH tunnel dari laptop:

```bash
# Di laptop, terminal baru:
ssh -p 5617 -L 9001:localhost:9001 -L 6333:localhost:6333 ubuntu@<vm-ip>
# Biarkan terbuka

# Di browser laptop:
http://localhost:9001        # MinIO Console (login: MINIO_USER / MINIO_PASSWORD)
http://localhost:6333/dashboard   # Qdrant Dashboard
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
| Indexing PDF: `FileNotFoundError: '/app/data/pdfs'` | Volume `./data/pdfs:/app/data/pdfs:ro` belum di-mount ke backend | Edit `docker-compose.poc.yml` (cek Step 8b verifikasi), lalu `docker compose ... up -d --force-recreate backend` |
| Indexing PDF: Qwen3-VL `/v1/chat/completions` timeout | vLLM masih sibuk load model atau request queue penuh | Tunggu vLLM idle, atau set `LLM_SUPPORTS_VISION=false` sementara untuk skip image description |
| PaddleOCR `Switching to CPU instead` walaupun pakai POC compose | Backend container belum ada `deploy.resources.reservations.devices` | Pastikan `docker-compose.poc.yml` service backend punya blok devices nvidia (lihat VRAM Budget section). Rebuild + `--force-recreate backend` |
| PaddleOCR error `ConvertPirAttribute2RuntimeAttribute not support` | Bug PaddlePaddle PIR + oneDNN saat CPU mode | Pakai GPU mode (default POC). Kalau terpaksa CPU, set env `FLAGS_use_mkldnn=0` di service backend |
| Backend OOM saat OCR + vLLM concurrent | VRAM tidak cukup | Turunkan vLLM `--gpu-memory-utilization` (0.80 → 0.75), atau set `OCR_USE_GPU=false` di `.env` |
| `paddlepaddle-gpu` install gagal di Dockerfile | Network / channel Paddle tidak reachable | Cek koneksi ke `paddlepaddle.org.cn`. Kalau diblok, install dari PyPI mirror: `pip install paddlepaddle-gpu==3.0.0 --index-url https://pypi.tuna.tsinghua.edu.cn/simple` |
| Vision query: `(psycopg2.errors.UndefinedColumn) column messages.images does not exist` | DB existing belum punya kolom `images` (Message model ditambah setelah deploy) | Restart backend — startup akan auto-run `ALTER TABLE messages ADD COLUMN IF NOT EXISTS images JSON` via `init_db()` |
| Image upload sukses tapi URL di Sheet 404 | `BACKEND_PUBLIC_URL` di `.env` salah / pakai `localhost` padahal reviewer dari laptop lain | Set `BACKEND_PUBLIC_URL=http://<ip-publik-vm>` (tanpa port kalau port 80), restart backend |
| Image upload `403 SignatureDoesNotMatch` | Pakai presigned MinIO URL langsung (versi lama). Versi sekarang serve via backend `/api/files/*` | Pull commit terbaru — `MinIOStorage.put_sync` return `BACKEND_PUBLIC_URL/api/files/<key>`, bukan presigned MinIO URL |
| Image upload `bucket_check_failed` | MinIO credential salah / bucket belum ada | Verify `MINIO_ACCESS_KEY=MINIO_USER`, `MINIO_SECRET_KEY=MINIO_PASSWORD`. Backend auto-create bucket `ragchat-images` di startup |
| QA Sheet sync error: kolom shift | Header sheet belum di-update jadi 14 kolom (tambah `gambar_user` di posisi C) | Manual insert kolom di Google Sheet (lihat `automation_qa/README.md`) |
| Pertanyaan medis di-block (`blocked_moderation` S6) | Llama Guard 3 over-restrictive untuk konteks akademik | Sudah di-fix — `S6: Specialized Advice` dihapus dari moderation prompt. Pull commit terbaru |
| Bot tawarkan bantuan untuk topik OOS (mis. "tips Mobile Legend") | LLM 8B interpret "acknowledge dulu" terlalu jauh | Sudah di-fix di OOS prompt — pull commit terbaru, rebuild backend, flush Redis cache |
| Response selalu pakai "Semoga membantu. Mau cek hal lain?" | Cached response lama atau prompt belum re-deploy | `docker exec rag-prototype-backend-1 grep -c "Semoga membantu" /app/backend/prompts/templates.py` — kalau > 3 berarti image lama, rebuild backend |
| Frontend tidak ada tombol attach gambar | File frontend di image lama, perlu re-copy atau rebuild | Pakai docker cp shortcut (Skenario B Maintenance) atau rebuild backend |

### Production Checklist

Sebelum go-live, pastikan:

**Security & Credentials:**
- [ ] Semua `CHANGE_ME` di `.env` sudah diganti
- [ ] `JWT_SECRET` minimal 32 byte hex
- [ ] `MINIO_PASSWORD` random ≥ 16 karakter (generate via `secrets.token_urlsafe(24)`)
- [ ] `POSTGRES_PASSWORD` random ≥ 16 karakter
- [ ] HTTPS aktif (Let's Encrypt atau cert lain)
- [ ] `ALLOWED_ORIGINS` dibatasi ke domain production
- [ ] Firewall: hanya port 443 + 80 terbuka publik
- [ ] Port 9000 (MinIO API), 9001 (MinIO Console), 6333 (Qdrant), 3000 (Grafana) tidak ke-expose publik — bind ke loopback atau buka SSH tunnel
- [ ] Service account Google credential (`automation_qa/secrets/google_service_account.json`) tidak di-commit ke git

**Data & Backup:**
- [ ] Backup otomatis PostgreSQL terjadwal (cron, termasuk `messages.images`)
- [ ] Backup otomatis Qdrant volume terjadwal
- [ ] Backup otomatis MinIO volume terjadwal (kalau image attachment penting)
- [ ] Log rotation aktif (`logrotate`)

**Functional:**
- [ ] PDF index ke Qdrant sudah selesai (cek `points_count` > 0)
- [ ] Test query end-to-end dengan 3 role: `public`, `mahasiswa`, `admin`
- [ ] Test vision upload: kirim gambar via UI → URL muncul di QA Sheet, click-able
- [ ] Test OOS handling: pertanyaan random (game/makanan/dll) → response ramah, BUKAN canned
- [ ] Test medis ringan: "perut sakit" → arahkan ke poliklinik UNHAS, BUKAN blocked
- [ ] Llama Guard 3 sudah di-pull (`ollama list`)
- [ ] Health check return `healthy`, bukan `degraded`
- [ ] Health check `vision_enabled: true` (kalau pakai vision)

**Observability:**
- [ ] Grafana dashboard sudah di-import untuk monitoring backend
- [ ] Prometheus scrape backend `/metrics` aktif
- [ ] Rate limit production-grade (default `8/min` text, `2/min` vision)

**QA Sheet Integration:**
- [ ] Service account email di-share ke Google Sheet dengan role Editor
- [ ] Header sheet 14 kolom (sudah include `gambar_user` di posisi C)
- [ ] `QA_SYNC_START_FROM` di-set ke timestamp launch (skip data historis)
- [ ] Sync test: kirim message dummy → tunggu sync cycle / force sync → row muncul di Sheet
- [ ] URL gambar di Sheet click-able dan tampil di browser reviewer

**Tuning:**
- [ ] `LLM_TEMPERATURE` di-set sesuai use case (0.3 untuk balance accuracy + variasi)
- [ ] `SCORE_THRESHOLD` + `LOW_CONFIDENCE_BUFFER` dikalibrasi berdasarkan QA evaluation
- [ ] `RERANKER_TIMEOUT` sesuai (5s untuk GPU, 30s untuk CPU)

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

## Retrieval Strategy (Production-Grade)

### Masalah yang Diselesaikan

PDF panjang (mis. `DRAFT PANDUAN KKN.pdf` = 100 chunks, `Rubrik 2024.pdf` = 118 chunks) sering memuat satu jawaban yang **tersebar di beberapa chunk berturutan**. Top-K retrieval naif bisa cuma ambil 1-2 chunk awal, info penting di chunk berikutnya hilang.

### Solusi: Multi-Stage Retrieval dengan Neighbor Expansion

```
User Query
    ↓
[Step 1] Vector Search di Qdrant
    → Ambil top-K chunks paling relevan (SIMILARITY_TOP_K=12)
    ↓
[Step 2] Neighbor Expansion
    → Untuk setiap chunk hasil retrieval:
      Fetch chunks tetangga (chunk_index ± NEIGHBOR_EXPANSION_RADIUS)
      yang berada di file_name + section yang sama
    → Dedup by (file_name, chunk_index)
    → Cap total di MAX_EXPANDED_CHUNKS=30 supaya tidak overflow LLM context
    ↓
[Step 3] Cross-Encoder Reranking
    → BGE-Reranker-v2-M3 re-score semua kandidat (retrieved + neighbors)
    → Ambil top RERANKER_TOP_N=6 chunks dengan relevansi tertinggi
    ↓
[Step 4] Generation
    → LLM generate jawaban dari top-N chunks + history + prompt template
```

### Konfigurasi via .env

```bash
# Step 1 — Initial vector search
SIMILARITY_TOP_K=12              # Berapa chunks ambil dari Qdrant (lebih banyak = recall tinggi)
SCORE_THRESHOLD=0.3              # Minimum top score; di bawah ini → mode "rag_low_relevance"

# Step 2 — Neighbor expansion
NEIGHBOR_EXPANSION_ENABLED=true  # set false untuk disable expansion
NEIGHBOR_EXPANSION_RADIUS=2      # ±2 chunks per retrieved → max 5 chunks per section
MAX_EXPANDED_CHUNKS=30           # Cap total kandidat sebelum reranking

# Step 3 — Reranking
RERANKER_TOP_N=6                 # Berapa chunks dikirim ke LLM sebagai context

# Indexing-side
CHUNK_SIZE=512                   # Ukuran chunk saat indexing
CHUNK_OVERLAP=128                # Overlap antar chunk → kurangi info loss di boundary
```

### Kenapa Strategi Ini Bekerja

| Aspek | Implementasi | Manfaat |
|---|---|---|
| **Recall vs Precision** | TOP_K=12 → reranker pilih 6 | Cast wide net, lalu filter ketat |
| **Konteks utuh** | Neighbor expansion via metadata `section` + `chunk_index` | Jawaban yang span multi-chunk tetap utuh |
| **Tidak overflow LLM** | `MAX_EXPANDED_CHUNKS=30` cap kandidat | Reranker handle final selection |
| **Boundary loss** | `CHUNK_OVERLAP=128` (sebelumnya 50) | Info di boundary tidak hilang |
| **Cross-encoder rerank** | BGE-Reranker-v2-M3 (state-of-the-art untuk Bahasa Indonesia) | Re-score akurat berdasarkan query-chunk pair |

### Contoh Kerja

User tanya: "Apa syarat pendaftaran KKN?"

1. **Vector search** → match 12 chunks, ada 3 chunks dari `DRAFT PANDUAN KKN.pdf` di section "Pendaftaran" (chunks #15, #18, #20)

2. **Neighbor expansion** → fetch chunks tetangga di section sama:
   - Dari #15 → fetch #13, #14, #16, #17
   - Dari #18 → fetch #16, #17, #19, #20 (#16, #17 sudah ada, skip)
   - Dari #20 → fetch #18, #19, #21, #22 (#18, #19, #21 sudah ada)
   - Total expanded: 12 (original) + ~8 unique neighbors = ~20 kandidat

3. **Rerank** → BGE re-score 20 kandidat → ambil 6 paling relevan untuk query "syarat pendaftaran KKN"

4. **Generation** → LLM dapat 6 chunks **berurutan** dari section "Pendaftaran", jawaban lengkap dan tidak terpotong.

### Tuning untuk Use Case Spesifik

| Skenario | Setting Rekomendasi |
|---|---|
| Banyak PDF panjang dengan info procedural berurutan | `NEIGHBOR_EXPANSION_RADIUS=3`, `MAX_EXPANDED_CHUNKS=40` |
| Mostly fact lookup (definisi, daftar nominal) | `NEIGHBOR_EXPANSION_RADIUS=1`, `RERANKER_TOP_N=4` (precision tinggi) |
| Latency-sensitive (chatbot real-time) | `NEIGHBOR_EXPANSION_ENABLED=false`, `SIMILARITY_TOP_K=8` |
| Konteks luas (research, analisis) | `RERANKER_TOP_N=8`, `MAX_EXPANDED_CHUNKS=50` |

### Performance Impact

| Tahap | Latency Tambahan | Resource |
|---|---|---|
| Neighbor expansion (Qdrant scroll) | ~50-200ms per query | I/O Qdrant |
| Reranker dengan 30 kandidat (vs 8) | ~200-500ms | GPU/CPU |
| **Total overhead** | **~300-700ms per query** | — |

Trade-off: latency naik sedikit, **kualitas jawaban naik drastis** untuk PDF panjang.

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
| POST | `/api/query` | — | RAG query untuk integrasi BE (support `images[]` untuk vision) |
| POST | `/api/query/stream` | — | Streaming variant `/api/query` |
| POST | `/api/chat` | Cookie/JWT | Chat dengan session management (support `images[]`) |
| POST | `/api/chat/stream` | Cookie/JWT | Streaming chat |
| GET | `/api/sessions` | Cookie/JWT | Daftar sesi percakapan user |
| POST | `/api/auth/login` | — | Login, dapat JWT cookie |
| POST | `/api/auth/logout` | Cookie | Logout |
| GET | `/api/health` | — | Status semua service (termasuk `vision_enabled`, `moderation_circuit`) |
| GET | `/api/files/{key}` | — | Serve image yang user upload (proxy ke MinIO). Dipakai di QA Sheet |
| POST | `/api/index` | — | Trigger indexing PDF |
| GET | `/metrics` | — | Prometheus metrics |

---

### Vision Input (Image Attachment)

Endpoint `/api/query`, `/api/chat`, dan stream variant-nya menerima field `images[]` opsional:

```json
{
  "query": "ini KRS saya, SKS-nya sudah cukup belum?",
  "session_id": "uuid-xxx",
  "images": [
    {
      "mime_type": "image/jpeg",
      "data": "<base64-encoded-image-tanpa-prefix-data:>"
    }
  ]
}
```

**Constraint:**
- Max 2 gambar per request (`MAX_IMAGES_PER_MESSAGE`)
- Max 10 MB per gambar (`MAX_IMAGE_SIZE_MB`)
- MIME yang diterima: `image/jpeg`, `image/png`, `image/webp`
- Auto-resize ke 1280px (`IMAGE_RESIZE_MAX_DIM`) — hemat token VL

**Pipeline saat ada `images[]`:**
1. L1 keyword filter pada query text → block atau lanjut
2. L2 Llama Guard moderation pada query text → block atau lanjut
3. L3 intent classifier — early reject kalau OOS confidence ≥ 0.85
4. Image upload ke MinIO internal → URL disimpan di `messages.images`
5. L5c retrieval RAG berdasarkan query text → konteks tambahan
6. VL generation (Qwen3-VL) dengan multimodal prompt
7. L6 output filter

**Response `debug.mode` saat vision:**
- `vision_rag` — sukses, image dianalisis dengan konteks RAG
- `vision_error` — image invalid atau VL gagal
- `out_of_scope` — text + image dianggap OOS (early reject sebelum VL call)

**URL gambar untuk QA Sheet:**
Setiap gambar yang user upload disimpan di MinIO dengan key `chat-uploads/<session_id>/<uuid>.{jpg|png}`. URL yang masuk ke kolom `gambar_user` di Google Sheet adalah `{BACKEND_PUBLIC_URL}/api/files/<key>` — di-stream lewat backend (port 80), tidak butuh port 9000 MinIO publik.

---

### Response Mode (field `debug.mode`)

| Mode | Arti |
|---|---|
| `rag` | Dijawab dari dokumen RAG |
| `rag_low_relevance` | RAG retrieve tapi top_score di bawah threshold — bot akui tidak tahu |
| `vision_rag` | Dijawab dari Qwen3-VL multimodal + konteks RAG (ada image attachment) |
| `vision_error` | Vision pipeline gagal (image invalid / VL timeout) |
| `chitchat` | Dijawab langsung (sapaan/basa-basi/reaksi follow-up) |
| `out_of_scope` | Diluar topik akademik UNHAS — response LLM-generated ramah |
| `blocked` | Diblokir keyword berbahaya (L1 hard_block) |
| `blocked_moderation` | Diblokir Llama Guard 3 (L2) — kategori serius (kekerasan, weapons, hate, self-harm) |
| `clarification_needed` | Intent confidence rendah, minta klarifikasi |
| `get_info_private` | Dijawab dari API UNHAS (intent classifier route `get_info_private`) |
| `cache_hit` | Dijawab dari Redis semantic cache (skip kalau ada image) |

---

## Environment Variables

Semua config dibaca dari `.env`. Template tersedia di `.env.dev` (dev lokal) dan `.env.poc` (POC/production).

### Variabel Utama

| Variable | Default Dev | Deskripsi |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | Provider LLM: `ollama` / `vllm` / `openai` |
| `LLM_MODEL` | `qwen2.5:7b` | Nama model |
| `LLM_BASE_URL` | `http://localhost:11434` | URL endpoint LLM |
| `LLM_TEMPERATURE` | `0.3` | Temperature LLM utama (RAG generation) |
| `LLM_SUPPORTS_VISION` | `false` | True kalau LLM support multimodal (Qwen3-VL di POC) |
| `EMBED_PROVIDER` | `huggingface` | Provider embedding: `huggingface` / `tei` |
| `EMBED_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Model embedding |
| `RERANKER_PROVIDER` | `sentence_transformers` | Provider reranker |
| `RERANKER_TIMEOUT` | `5` | Timeout TEI rerank service (5s GPU, 30s CPU) |
| `QDRANT_URL` | `http://localhost:6333` | URL Qdrant |
| `QDRANT_COLLECTION` | `unhas_docs` | Nama collection |
| `SCORE_THRESHOLD` | `0.3` | Threshold reranker — di bawah ini → low_relevance fallback |
| `LOW_CONFIDENCE_BUFFER` | `0.15` | Buffer marginal disclaimer di atas threshold |
| `DATABASE_URL` | `postgresql://...` | Connection string PostgreSQL |
| `REDIS_URL` | `redis://localhost:6379/0` | Connection string Redis |
| `JWT_SECRET` | *(set di .env)* | Secret key JWT |
| `MODERATION_BACKEND` | `passthrough` | `passthrough` / `ollama` |
| `MODERATION_TIMEOUT` | `20` | Timeout Llama Guard call (sebelum circuit breaker fail-open) |
| `INTENT_MODEL_PATH` | `models/intent_classifier` | Path model IndoBERT |
| `INTENT_CONFIDENCE_THRESHOLD` | `0.6` | Threshold confidence intent — di bawah ini → clarification_needed |
| `OCR_USE_GPU` | `true` | PaddleOCR pakai GPU (POC). False untuk dev RTX 3060 atau backend tanpa GPU |
| `MAX_IMAGES_PER_MESSAGE` | `2` | Max attachment image per request |
| `MAX_IMAGE_SIZE_MB` | `10` | Max size per image (setelah base64 decode) |
| `IMAGE_RESIZE_MAX_DIM` | `1280` | Auto-resize jika dimensi melebihi (hemat token VL) |
| `RATE_LIMIT_TEXT_PER_MINUTE` | `8` | Rate limit text query per user per menit |
| `RATE_LIMIT_VISION_PER_MINUTE` | `3` | Rate limit vision query per user per menit |
| `ALLOWED_ORIGINS` | `http://localhost:*` | CORS origins |
| `UNHAS_API_BASE_URL` | *(kosong)* | Base URL API UNHAS (isi jika sudah tersedia) |
| **Storage** | | |
| `STORAGE_BACKEND` | `filesystem` | `filesystem` (dev) / `minio` (POC) |
| `BACKEND_PUBLIC_URL` | `http://localhost:8000` | URL backend publik untuk QA Sheet (POC: `http://<ip-vm>`) |
| `MINIO_ENDPOINT` | `minio:9000` | Endpoint internal MinIO (docker DNS) |
| `MINIO_ACCESS_KEY` / `_SECRET_KEY` | *(set di .env)* | MinIO credential |
| `MINIO_BUCKET` | `ragchat-images` | Bucket untuk image upload |
| **QA Sheet Sync** | | |
| `QA_SYNC_GOOGLE_CREDENTIALS_FILE` | `automation_qa/secrets/google_service_account.json` | Path SA JSON |
| `QA_SYNC_SPREADSHEET_ID` | *(set di .env)* | ID Google Sheet target |
| `QA_SYNC_WORKSHEET_NAME` | `Sheet1` | Nama worksheet |
| `QA_SYNC_INTERVAL_SECONDS` | `3600` | Interval auto-sync (1 jam) |
| `QA_SYNC_START_FROM` | *(kosong)* | ISO timestamp untuk skip message historical |

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
- [x] Retrieval production-grade (multi-stage: vector search → neighbor expansion → BGE rerank)
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
