"""Configuration for RAG Chatbot UNHAS — v2 POC-ready."""

import os
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# === Paths ===
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "pdfs")
IMAGES_DIR = os.path.join(_PROJECT_ROOT, "data", "images")

# =============================================================================
# LLM — Switchable provider: "ollama" (dev) atau "vllm" (POC)
# =============================================================================
# Dev (RTX 3060): LLM_PROVIDER=ollama, LLM_MODEL=qwen2.5:7b via Ollama
# POC (L40S):     LLM_PROVIDER=vllm,   LLM_MODEL=Qwen/Qwen3-VL-8B-Instruct via vLLM
# =============================================================================

LLM_PROVIDER: Literal["ollama", "vllm", "openai"] = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
LLM_API_KEY = os.getenv("LLM_API_KEY", "not-needed")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", "120"))
LLM_SUPPORTS_VISION = os.getenv("LLM_SUPPORTS_VISION", "false").lower() == "true"

# =============================================================================
# Vision — model deskripsi gambar, TERPISAH dari model generation
# =============================================================================
# LLM_MODEL dipakai dua peran: generation lewat llm_factory, dan deskripsi
# gambar lewat image_describer. Di server riset keduanya berbeda — LLM_MODEL
# menunjuk model teks sementara deskripsi gambar butuh model multimodal.
#
# Default jatuh ke LLM_MODEL supaya perilaku lama tidak berubah bila tidak diset.
VISION_MODEL = os.getenv("VISION_MODEL", "") or LLM_MODEL

# Parameter generasi deskripsi gambar. Deskripsi BUKAN metadata — ia isi chunk
# yang diindeks, jadi nondeterminisme di sini mengubah teks yang divektorkan DAN
# jumlah chunk (verdict DEKORATIF membuang element, menggeser seluruh penomoran
# sesudahnya). Lihat INSPECTION_REPORT_2.md G10 konsekuensi 2.
VISION_TEMPERATURE = float(os.getenv("VISION_TEMPERATURE", "0"))
VISION_MAX_TOKENS = int(os.getenv("VISION_MAX_TOKENS", "300"))

# Seed generasi. Dikirim ke Ollama lewat `options.seed`; dicatat di manifest dan
# di tiap baris images.jsonl. Kosongkan (nilai negatif) untuk tidak mengirim.
RESEARCH_VISION_SEED = int(os.getenv("RESEARCH_VISION_SEED", "1337"))

# =============================================================================
# Cache deskripsi gambar — ARTEFAK EKSPERIMEN, bukan optimasi
# =============================================================================
# Isi cache adalah SUMBER ISI CHUNK. Dua run yang menghasilkan chunk sama hanya
# dapat dibuktikan berasal dari deskripsi yang sama bila cache-nya sama.
# Diperlakukan setara document_registry.json: berkasnya dibagikan antar fork,
# hash isinya dicatat di manifest, dan TIDAK dihapus di tengah eksperimen.
#
# Lokasi default berada di LUAR clone mana pun supaya dapat dibagikan. Lihat
# CHANGES.md untuk opsi berbagi dan konsekuensi izin berkasnya.
VISION_CACHE_ENABLED = os.getenv("VISION_CACHE_ENABLED", "false").lower() == "true"
VISION_CACHE_PATH = os.getenv(
    "VISION_CACHE_PATH", "~/rag_mm_b_shared/vision_cache.db"
)

# Panjang konteks Ollama. Tanpa ini Ollama MEMOTONG ke 4096 token secara senyap,
# dan satu gambar saja bisa menghabiskannya — deskripsi jadi terpotong tanpa
# jejak. Dikirim eksplisit lewat `options.num_ctx`.
VISION_NUM_CTX = int(os.getenv("VISION_NUM_CTX", "8192"))

# Backward-compat alias
OLLAMA_BASE_URL = LLM_BASE_URL

# =============================================================================
# Embedding — Switchable provider: "huggingface" (dev, in-process) atau "tei" (POC)
# =============================================================================
# Dev:  in-process HuggingFace model
# POC:  Text Embeddings Inference (TEI) HTTP service — dikelola IOH
# =============================================================================

EMBED_PROVIDER: Literal["huggingface", "tei"] = os.getenv("EMBED_PROVIDER", "huggingface")
EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "")   # hanya dipakai kalau EMBED_PROVIDER=tei
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cuda")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "8"))   # kecil untuk TEI CPU, naikin kalau pakai GPU
EMBED_TIMEOUT = int(os.getenv("EMBED_TIMEOUT", "600"))   # detik, untuk TEI HTTP call (CPU mode butuh besar)
EMBED_DIMENSION = 1024

# Backward-compat aliases
EMBED_MODEL_NAME = EMBED_MODEL

# =============================================================================
# Re-ranker — Switchable provider: "sentence_transformers" (dev) atau "tei" (POC)
# =============================================================================

RERANKER_PROVIDER: Literal["sentence_transformers", "tei"] = os.getenv("RERANKER_PROVIDER", "sentence_transformers")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", os.path.join(_PROJECT_ROOT, "models", "bge-reranker-v2-m3"))
RERANKER_BASE_URL = os.getenv("RERANKER_BASE_URL", "")   # hanya dipakai kalau RERANKER_PROVIDER=tei
RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", "6"))
# Timeout TEI rerank service. GPU 5s cukup; CPU naikkan ke 30s.
RERANKER_TIMEOUT = float(os.getenv("RERANKER_TIMEOUT", "5"))
# Buffer di atas SCORE_THRESHOLD untuk marginal confidence. top_score yang
# berada di [SCORE_THRESHOLD, SCORE_THRESHOLD + buffer] → tambah disclaimer
# halus ke akhir jawaban. Default 0.15 → marginal zone score 0.30-0.45.
LOW_CONFIDENCE_BUFFER = float(os.getenv("LOW_CONFIDENCE_BUFFER", "0.15"))
RERANKER_USE_FP16 = True

# Backward-compat alias
RERANKER_MODEL_NAME = RERANKER_MODEL

# =============================================================================
# Qdrant
# =============================================================================

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# Versi server Qdrant yang DIHARAPKAN. Kosong = tidak diperiksa.
#
# Dicatat di run_manifest.json bersama versi yang benar-benar terdeteksi dari
# server, supaya selisihnya terlihat. Port ikut terbawa lewat QDRANT_URL.
# Tanpa ini, run yang dijalankan sebelum dan sesudah Qdrant dipindah port atau
# di-upgrade tidak dapat dibedakan dari artefaknya.
QDRANT_SERVER_VERSION = os.getenv("QDRANT_SERVER_VERSION", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", os.getenv("QDRANT_COLLECTION_NAME", "unhas_docs"))
QDRANT_COLLECTION_NAME = QDRANT_COLLECTION  # backward-compat

# =============================================================================
# PaddleOCR
# =============================================================================

OCR_LANG = "id"
# True kalau container backend punya akses GPU (POC L40S). False untuk dev RTX 3060 (CPU fallback).
OCR_USE_GPU = os.getenv("OCR_USE_GPU", "true").lower() == "true"

# =============================================================================
# Chunking
# =============================================================================

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "128"))

# =============================================================================
# PDF Processing (production-grade indexing)
# =============================================================================

# Strategy: "fast" (PyMuPDF only) atau "hi_res" (Unstructured.io layout-aware)
# - fast: cepat, cocok untuk PDF text-heavy (SOP)
# - hi_res: lambat tapi extract table+image structure, cocok untuk Pedoman/Manual
# - auto: deteksi otomatis per file (rekomendasi)
PDF_EXTRACTION_STRATEGY = os.getenv("PDF_EXTRACTION_STRATEGY", "auto")

# Image extraction & description
PDF_EXTRACT_IMAGES = os.getenv("PDF_EXTRACT_IMAGES", "true").lower() == "true"
PDF_DESCRIBE_IMAGES = os.getenv("PDF_DESCRIBE_IMAGES", "auto").lower()  # auto | true | false
PDF_MIN_IMAGE_SIZE_KB = int(os.getenv("PDF_MIN_IMAGE_SIZE_KB", "20"))  # skip image < 20KB (decorative)
PDF_MAX_IMAGE_DIM = int(os.getenv("PDF_MAX_IMAGE_DIM", "1280"))         # resize besar untuk hemat token VL

# Table handling
PDF_EXTRACT_TABLES = os.getenv("PDF_EXTRACT_TABLES", "true").lower() == "true"
PDF_TABLE_MAX_CHARS = int(os.getenv("PDF_TABLE_MAX_CHARS", "2000"))      # keep table utuh jika < ini

# Kecualikan field metadata non-semantik dari teks yang di-embed.
# Default false: perilaku indexing lama dipertahankan persis.
#
# Saat true, hanya `section` yang ikut divektorkan; file_hash/page/chunk_index/
# element_type/extraction_strategy/source_type/file_name dikecualikan. Field-nya
# TETAP tersimpan penuh di payload Qdrant — pengecualian hanya memengaruhi teks
# yang di-embed dan get_metadata_str(), bukan node.metadata.
#
# Alasan: metadata memotong kuota chunk lewat
# effective_chunk_size = chunk_size - metadata_len (llama-index sentence.py:158).
# Terukur pada korpus contoh: 128 token -> 14 token, kuota 384 -> 498.
#
# CATATAN JALUR QUERY: SentenceTransformerRerank membaca MetadataMode.EMBED
# (sbert_rerank.py:75), jadi mengaktifkan flag ini mengubah teks yang di-rerank
# saat RERANKER_PROVIDER=sentence_transformers (.env.dev, .env.alt.poc).
# Jalur TEI tidak terpengaruh — ia memakai get_content() default MetadataMode.NONE.
INDEX_EXCLUDE_METADATA_FROM_EMBED = os.getenv(
    "INDEX_EXCLUDE_METADATA_FROM_EMBED", "false"
).lower() == "true"

# Field yang dikecualikan dari teks yang di-embed saat
# INDEX_EXCLUDE_METADATA_FROM_EMBED aktif. Daftarnya lengkap: TIDAK ADA metadata
# yang divektorkan, sehingga `text_content` di chunks.jsonl sama persis dengan
# string yang di-embed.
#
# Sebagian besar field di sini memang provenance (id sintetis, hash, koordinat).
# `section` adalah pengecualian dan dikecualikan karena alasan berbeda: nilainya
# SUDAH muncul di dalam text_content untuk chunk tabel dan deskripsi gambar
# (prefix "## {section}"), sehingga menyertakannya lagi lewat metadata memberi
# bobot ganda yang tidak merata antar tipe chunk.
#
# Terukur pada dokumen sintetis berstruktur realistis — porsi token `section`
# terhadap teks yang di-embed:
#     ImageDescription  46% (2x)  ->  27% (1x)
#     Table             43% (2x)  ->  25% (1x)
#     NarrativeText      7%        ->   1%
# Bobot ganda jatuh persis pada dua strata yang diteliti, sehingga artefaknya
# tidak dapat dipisahkan dari efek strategi indexing saat analisis.
#
# BIAYA YANG DITERIMA: chunk teks kedua dan seterusnya dalam satu section
# kehilangan sinyal section di ruang vektor (hanya chunk pertama yang membawa
# "# {title}" di teksnya). Peran NEIGHBOR_EXPANSION karenanya lebih besar
# daripada konfigurasi lama.
#
# Ditaruh di config (bukan indexing.py) agar dapat diimpor tanpa menyeret
# qdrant_client — scripts/probe_rechunk.py mengandalkan itu.
EMBED_EXCLUDED_METADATA_KEYS = (
    "file_name",
    "file_hash",
    "page",
    "chunk_index",
    "element_type",
    "extraction_strategy",
    "source_type",
    "section",
    # Tahap 2 — identitas & metadata struktural.
    "document_id",
    "chunk_id",
    "text_sha",
    "raw_html",
    "table_format",
    "bbox",
    # Tahap 4 — tautan ke images.jsonl.
    "image_id",
)

# Nama lama, dipertahankan agar impor yang ada tidak patah.
NON_SEMANTIC_METADATA_KEYS = EMBED_EXCLUDED_METADATA_KEYS

# Batas token untuk setiap chunk yang dikeluarkan _chunk_elements.
# 0 = mati (perilaku lama dipertahankan persis).
#
# Akar masalah yang ditangani: pemeriksaan flush di preprocessing.py terjadi
# SEBELUM append, sehingga satu element yang lebih besar dari CHUNK_SIZE masuk
# utuh tanpa pernah diperiksa. Di jalur fast, _extract_fast membuat satu element
# per HALAMAN (570-1030 token), yang lalu dipecah ulang oleh node parser
# LlamaIndex menjadi beberapa titik Qdrant dengan chunk_index yang sama.
#
# Satuannya TOKEN, memakai tokenizer yang sama dengan splitter LlamaIndex,
# supaya batas di sini dan effective_chunk_size di sana tidak bisa berselisih.
#
# Nilai yang disarankan: <= effective_chunk_size terkecil yang direncanakan.
# Dengan INDEX_EXCLUDE_METADATA_FROM_EMBED aktif dan seluruh field masa depan
# terpasang, effective_chunk_size = 380. Nilai 350 memberi sisa aman.
#
# Tabel dan deskripsi gambar TIDAK PERNAH dipecah walau melewati batas ini:
# memecah tabel merusak relasi baris-kolom yang diukur RCAA, dan memecah
# deskripsi gambar merusak relasi satu-deskripsi-satu-gambar. Keduanya hanya
# dicatat sebagai warning.
INDEX_MAX_CHUNK_TOKENS = int(os.getenv("INDEX_MAX_CHUNK_TOKENS", "0"))

# Matikan node parser LlamaIndex saat indexing. Default false (perilaku lama).
#
# Tanpa flag ini, from_documents() memakai Settings.node_parser dan memecah ulang
# Document yang melewati Settings.chunk_size. Node anak mewarisi metadata induk
# APA ADANYA — chunk_index, chunk_id, dan text_sha ikut tersalin — jadi beberapa
# titik Qdrant berbagi satu identitas dengan teks berbeda.
#
# Flag terpisah dari INDEX_MAX_CHUNK_TOKENS secara sengaja: keduanya perlu bisa
# dinyalakan sendiri-sendiri agar run_manifest.json mencatat dua keputusan yang
# memang berbeda.
#
# Menutup juga perbedaan dua entry point: POST /api/index tidak pernah memanggil
# _configure_settings sehingga memakai default LlamaIndex (1024/200), sedangkan
# CLI memakai 512/128. Dengan PassthroughNodeParser keduanya tidak lagi bergantung
# pada Settings sama sekali.
#
# CARA mematikannya ada di backend/services/node_passthrough.py, dan itu BUKAN
# transformations=[] — daftar kosong bersifat falsy dan justru memulihkan
# SentenceSplitter default tanpa peringatan apa pun.
INDEX_DISABLE_NODE_PARSER = os.getenv(
    "INDEX_DISABLE_NODE_PARSER", "false"
).lower() == "true"

# Ambang minimum token untuk sebuah chunk diemisikan. 0 = mati (perilaku lama).
#
# Mengaktifkan DUA penyaring sekaligus:
#   1. Chunk dengan token < nilai ini tidak diemisikan.
#   2. Chunk yang teks ternormalisasinya persis sama dengan baris judul section
#      ("# {section}") tidak diemisikan, BERAPA PUN panjangnya.
#
# Penyaring kedua ada karena panjang saja tidak memisahkan judul dari konten:
# terukur pada korpus contoh, judul panjang = 18 token sedangkan kalimat asli
# terpendek = 8 token. Ambang yang cukup tinggi untuk menangkap semua judul akan
# ikut membuang kalimat asli. Judul-saja terbukti tidak membawa konten unik
# karena `section` sudah ada di metadata setiap chunk.
#
# Nilai riset yang dibekukan: 8 (tepat di batas kalimat asli terpendek terukur).
INDEX_MIN_CHUNK_TOKENS = int(os.getenv("INDEX_MIN_CHUNK_TOKENS", "0"))

# =============================================================================
# Identitas & metadata struktural (Tahap 2) — semua opt-in, default mati
# =============================================================================

# Aktifkan field identitas dan metadata struktural di payload chunk:
# document_id, chunk_id, text_sha, raw_html, table_format, bbox.
#
# MEMBUTUHKAN data/document_registry.json terisi. Berkas PDF yang tidak
# terdaftar (atau terdaftar dengan document_id kosong) DILEWATI saat indexing
# dengan error log — tidak diberi id provisional, karena anotasi gold yang
# terlanjur menempel pada id sementara akan patah saat id sebenarnya menyusul.
#
# Semua field ini provenance, bukan konten semantik, jadi masuk
# NON_SEMANTIC_METADATA_KEYS dan tidak ikut divektorkan.
INDEX_STRUCTURAL_METADATA = os.getenv(
    "INDEX_STRUCTURAL_METADATA", "false"
).lower() == "true"

# Jadikan tabel kecil (< PDF_TABLE_MAX_CHARS) chunk mandiri, bukan digabung ke
# buffer teks sekitarnya. Default false (perilaku lama).
#
# Diperlukan agar setiap tabel punya hubungan 1:1 dengan satu chunk beserta
# raw_html-nya — deck riset menuntut structured_summary tabel di images.jsonl
# sama dengan text_as_html chunk terkait, dan itu mustahil bila tabel melebur
# ke dalam prosa.
#
# PERHATIAN: ini MENGGESER BATAS CHUNK TEKS di sekitar tabel, bukan sekadar
# menambah chunk tabel. Prosa yang tadinya satu chunk bersama tabel kini
# terbelah menjadi chunk sebelum dan sesudah. Fork lain wajib memakai nilai
# yang sama.
INDEX_TABLES_AS_OWN_CHUNKS = os.getenv(
    "INDEX_TABLES_AS_OWN_CHUNKS", "false"
).lower() == "true"

# Tulis gambar hasil ekstraksi PDF ke IMAGES_DIR. Default false (perilaku lama:
# gambar hanya hidup di memori sebagai base64 lalu dibuang).
#
# Penulisan terjadi SEBELUM keputusan deskripsi, sehingga gambar yang dinilai
# DEKORATIF atau gagal dideskripsikan TETAP tersimpan — peneliti fork lain
# memvektorkan gambar aslinya, dan riset ini butuh kemampuan menilai ulang
# tanpa mengulang ekstraksi PDF.
#
# TIDAK ADA penyaringan ukuran. Ukuran dicatat di metadata supaya penyaringan
# dapat dilakukan di hilir. (Catatan: is_likely_informative di image_describer
# adalah fungsi mati tanpa pemanggil, jadi PDF_MIN_IMAGE_SIZE_KB selama ini
# tidak berefek pada apa pun — dan sengaja TIDAK dihidupkan di sini.)
#
# MEMBUTUHKAN document_id, jadi hanya berjalan saat INDEX_STRUCTURAL_METADATA
# aktif dan berkas terdaftar di document_registry.json.
#
# Hanya jalur hi_res yang mengekstrak gambar; jalur fast tidak sama sekali.
INDEX_PERSIST_IMAGES = os.getenv("INDEX_PERSIST_IMAGES", "false").lower() == "true"

# =============================================================================
# RESEARCH_MODE — gerbang konfigurasi eksperimen
# =============================================================================

# Bila true, indexing memvalidasi bahwa SELURUH flag riset bernilai sesuai
# daftar beku di CHANGES.md, dan menolak run bila ada yang jatuh ke default.
# Default false: perilaku produksi tidak berubah.
#
# Alasannya sebuah mode kegagalan senyap yang terverifikasi: load_dotenv()
# mencari .env dari lokasi config.py KE ATAS, bukan dari direktori kerja. `.env`
# yang ditaruh di direktori kerja lain DIABAIKAN TANPA PERINGATAN, seluruh flag
# riset jatuh ke default, dan indexing tetap berjalan — menghasilkan chunk yang
# salah setelah berjam-jam. Sekelas dengan PDF_EXTRACTION_STRATEGY=auto yang
# sudah ditolak _check_image_strategy.
RESEARCH_MODE = os.getenv("RESEARCH_MODE", "false").lower() == "true"

# Nilai beku dari CHANGES.md, bagian "DAFTAR FINAL FLAG RISET — BEKU".
# Satu sumber kebenaran; jangan ubah tanpa mengubah CHANGES.md dan re-index.
RESEARCH_EXPECTED_FLAGS: dict[str, object] = {
    "INDEX_EXCLUDE_METADATA_FROM_EMBED": True,
    "INDEX_MAX_CHUNK_TOKENS": 350,
    "INDEX_MIN_CHUNK_TOKENS": 8,
    "INDEX_DISABLE_NODE_PARSER": True,
    "INDEX_STRUCTURAL_METADATA": True,
    "INDEX_TABLES_AS_OWN_CHUNKS": True,
    "INDEX_PERSIST_IMAGES": True,
    "PDF_EXTRACTION_STRATEGY": "hi_res",
    "CHUNK_SIZE": 512,
    "CHUNK_OVERLAP": 128,
    "PDF_TABLE_MAX_CHARS": 2000,
    "NEIGHBOR_EXPANSION_ENABLED": True,
    "NEIGHBOR_EXPANSION_RADIUS": 2,
    "MAX_EXPANDED_CHUNKS": 30,
}

# Escape hatch: terima korpus gambar yang TIDAK LENGKAP. Default false.
#
# Saat INDEX_PERSIST_IMAGES aktif, indexing menolak dua kondisi yang membuat
# sebagian dokumen kehilangan seluruh gambarnya tanpa jejak di berkas hasil:
#   1. PDF_EXTRACTION_STRATEGY bukan "hi_res" (ditolak sebelum run dimulai)
#   2. Ada dokumen yang jatuh dari hi_res ke fast (gagal keras di akhir run,
#      setelah dump ditulis, sebelum apa pun masuk Qdrant)
#
# Setel true HANYA bila kamu memang menerima korpus tidak lengkap dan sudah
# memeriksa `degraded_documents` di run_manifest.json. Nilai true tercatat di
# manifest, sehingga keputusan itu terbawa bersama datanya.
ALLOW_INCOMPLETE_IMAGE_CORPUS = os.getenv(
    "ALLOW_INCOMPLETE_IMAGE_CORPUS", "false"
).lower() == "true"

# Direktori dump chunk untuk ditinjau tim evaluasi. Kosong = tidak ada dump
# (perilaku lama). Diisi = tiap run indexing menulis
# <dir>/<run_id>/{chunks.jsonl, chunks_review.csv, run_manifest.json}.
#
# chunks.jsonl mengikuti skema lapis 2 dataset publikasi riset, bukan dump ad hoc.
#
# Dump hanya SETIA (identik dengan yang masuk Qdrant) bila
# INDEX_DISABLE_NODE_PARSER=true. Tanpa itu node parser LlamaIndex masih memecah
# dan mem-strip whitespace. run_manifest.json mencatat statusnya di
# `dump_faithful`.
CHUNK_DUMP_DIR = os.getenv("CHUNK_DUMP_DIR", "")

# Registry pemetaan nama berkas PDF -> document_id.
#
# ARTEFAK BERSAMA, setara vision_cache.db. document_id menentukan chunk_id,
# image_id, DAN nama direktori gambar sekaligus — jadi konsistensi antar fork
# datang dari BERKAS YANG SAMA, bukan dari dua eksekusi scaffolder yang
# kebetulan sepakat.
#
# Default berada di LUAR clone mana pun supaya cukup satu berkas untuk kedua
# fork, tanpa perlu dikirim-kirim. Yang perlu disepakati bukan cara mengirimnya,
# tapi siapa yang boleh menjalankan generate ulang dan kapan — lihat
# document_registry_notes.md yang ditulis bersama registry-nya.
#
# Generate/perbarui: python scripts/scaffold_document_registry.py
DOCUMENT_REGISTRY_PATH = os.path.expanduser(os.getenv(
    "DOCUMENT_REGISTRY_PATH",
    "~/rag_mm_b_shared/document_registry.json",
))

# =============================================================================
# RAG Pipeline
# =============================================================================

SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "12"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.3"))

# Neighbor expansion — production-grade strategy untuk PDF panjang.
# Setelah retrieval top-K, fetch chunks tetangganya (di file & section yang sama)
# supaya konteks panjang tidak terpotong. Reranker akan re-rank gabungan.
NEIGHBOR_EXPANSION_ENABLED = os.getenv("NEIGHBOR_EXPANSION_ENABLED", "true").lower() == "true"
NEIGHBOR_EXPANSION_RADIUS = int(os.getenv("NEIGHBOR_EXPANSION_RADIUS", "2"))
MAX_EXPANDED_CHUNKS = int(os.getenv("MAX_EXPANDED_CHUNKS", "30"))

# =============================================================================
# Conversation history (multi-turn)
# =============================================================================

HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "5"))
HISTORY_MAX_TOKENS = int(os.getenv("HISTORY_MAX_TOKENS", "1500"))

# =============================================================================
# Vision — aktif saat LLM_SUPPORTS_VISION=true (POC dengan Qwen3-VL)
# =============================================================================

MAX_IMAGES_PER_MESSAGE = int(os.getenv("MAX_IMAGES_PER_MESSAGE", "2"))
MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
IMAGE_RESIZE_MAX_DIM = int(os.getenv("IMAGE_RESIZE_MAX_DIM", "1280"))

# =============================================================================
# PostgreSQL — sessions, messages, audit
# Dev: postgresql://ragchat:dev@localhost:5432/ragchat  (lihat docker-compose.dev.yml)
# POC: postgresql+asyncpg://... (IOH provision)
# =============================================================================

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://ragchat:dev@localhost:5432/ragchat")

# =============================================================================
# Redis — cache + rate limit + circuit breaker
# Dev: redis://localhost:6379/0  (lihat docker-compose.dev.yml)
# POC: redis://redis:6379/0
# =============================================================================

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))

# =============================================================================
# Auth (JWT)
# =============================================================================

JWT_SECRET = os.getenv("JWT_SECRET", "unhas-rag-demo-secret-2026")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "24"))

# =============================================================================
# Intent Classifier (IndoBERT)
# =============================================================================

INTENT_MODEL_PATH = os.getenv(
    "INTENT_MODEL_PATH",
    os.path.join(_PROJECT_ROOT, "models", "intent_classifier"),
)
INTENT_CONFIDENCE_THRESHOLD = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.6"))

# =============================================================================
# Moderation (Layer 2 — Llama Guard 3)
# Dev:  passthrough (skip)
# POC:  ollama      (llama-guard3:1b via Ollama)
#
# MODERATION_BASE_URL: URL Ollama untuk moderasi. Default ke LLM_BASE_URL untuk
# backward-compat (dev), tapi di POC harus diset terpisah karena LLM utama
# pakai vLLM (LLM_BASE_URL=http://vllm:8001), sementara moderasi tetap di Ollama.
# =============================================================================

MODERATION_BACKEND = os.getenv("MODERATION_BACKEND", "passthrough")
MODERATION_MODEL = os.getenv("MODERATION_MODEL", "llama-guard3:1b")
MODERATION_BASE_URL = os.getenv("MODERATION_BASE_URL", LLM_BASE_URL)
MODERATION_TIMEOUT = int(os.getenv("MODERATION_TIMEOUT", "20"))

# =============================================================================
# Rate Limiting
# =============================================================================

RATE_LIMIT_TEXT_PER_MINUTE = int(os.getenv("RATE_LIMIT_TEXT_PER_MINUTE", "10"))
RATE_LIMIT_VISION_PER_MINUTE = int(os.getenv("RATE_LIMIT_VISION_PER_MINUTE", "3"))

# =============================================================================
# CORS
# =============================================================================

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if o.strip()
]

# =============================================================================
# Logging
# =============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# =============================================================================
# Private API (integrasi sistem UNHAS — aktif saat URL dikonfigurasi)
# =============================================================================

UNHAS_API_BASE_URL = os.getenv("UNHAS_API_BASE_URL", "")

# =============================================================================
# POC-only — Storage (MinIO)
# =============================================================================

# STORAGE_BACKEND: Literal["filesystem", "minio"] = os.getenv("STORAGE_BACKEND", "filesystem")
# STORAGE_LOCAL_PATH = os.getenv("STORAGE_LOCAL_PATH", "./storage")
# MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
# MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "")
# MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "")
# MINIO_BUCKET = os.getenv("MINIO_BUCKET", "ragchat-images")

# =============================================================================
# Instrumentasi riset Tahap 5 — jalur QUERY
# =============================================================================
#
# Berbeda dari flag INDEX_* yang membentuk korpus, flag di bawah ini hanya
# memengaruhi jalur query dan TIDAK memicu re-index. Semuanya default mati:
# dengan seluruhnya false, perilaku pipeline identik dengan sebelum Tahap 5.
#
# Tidak satu pun menghapus layer. Yang dilakukan adalah MELEWATI titik keluar
# supaya query sampai ke retrieval penuh, sehingga item test set berlabel
# expected_behavior="abstain_or_flag_conflict" benar-benar diukur di retrieval
# dan bukan ditolak lebih awal oleh L1/L3.

# Lewati lima titik keluar yang tidak punya saklar sendiri (C9 #1, #3, #5, #6, #8):
# L1 keyword hard-block, identity override, L3 chitchat, L3 out_of_scope, dan
# condensation <ACK>. Setiap penyelamatan dicatat ke log dengan penanda titiknya
# supaya jumlah item test set yang terpengaruh dapat dihitung.
#
# TIDAK melewati L2 moderation: itu punya saklar sendiri
# (MODERATION_BACKEND=passthrough) dan melewatinya diam-diam berarti mengubah
# perilaku keamanan tanpa jejak di konfigurasi moderation.
RESEARCH_BYPASS_ROUTING = os.getenv("RESEARCH_BYPASS_ROUTING", "false").lower() == "true"

# Matikan query condensation (C11). Pemicunya `if history:`, bukan flag, jadi
# satu-satunya jalan lewat konfigurasi sebelumnya adalah tidak mengirim history
# sama sekali — yang mustahil untuk /api/chat (history diambil dari PostgreSQL)
# dan menghapus jalur multi-turn dari perbandingan.
RESEARCH_DISABLE_CONDENSATION = os.getenv(
    "RESEARCH_DISABLE_CONDENSATION", "false"
).lower() == "true"

# Matikan output filter (Layer 6). Jawaban yang dinilai metrik lapis 2 (RAGAS)
# harus keluaran model apa adanya: pola di output_filter.py mencocoki kosakata
# akademik yang sah — `transformers`, `bert`, dan `meta` semuanya muncul di
# teks akademik Indonesia.
#
# filter_token() pada jalur streaming ikut dimatikan; kalau tidak, token yang
# di-stream tersaring sedangkan jawaban akhir tidak, dan keduanya jadi berbeda.
RESEARCH_DISABLE_OUTPUT_FILTER = os.getenv(
    "RESEARCH_DISABLE_OUTPUT_FILTER", "false"
).lower() == "true"

# Sertakan hasil ketiga tahap retrieval (dense, ekspansi tetangga, rerank) di
# debug.retrieval_stages pada response. Default mati supaya payload produksi
# tidak membengkak — tiga tahap x SIMILARITY_TOP_K node, masing-masing dengan
# teks pratinjau.
#
# Saat aktif, text_preview pada sources dikembalikan ke teks chunk apa adanya:
# SourceLabelPostprocessor menyisipkan "[nama_file]\n" ke node.text sebelum
# _build_sources dipanggil, sehingga pratinjau produksi bukan teks chunk.
# Konteks yang dilihat LLM TIDAK diubah — labelnya tetap ada di sana.
RESEARCH_VERBOSE_RETRIEVAL = os.getenv(
    "RESEARCH_VERBOSE_RETRIEVAL", "false"
).lower() == "true"


def research_query_flags() -> dict[str, bool]:
    """Snapshot flag riset jalur query, untuk dicatat di keluaran eksperimen.

    Sebagian mengubah angka yang dilaporkan, jadi tiap berkas hasil harus
    membawa nilainya. Dipakai scripts/retrieval_dump.py.

    CATATAN: fungsi ini BELUM dipanggil chunk_dump.write_run() — berkas itu
    dibekukan pada Tahap 5. Lihat CHANGES.md, bagian "Menyatukan ke
    run_manifest.json", untuk baris persis yang perlu ditambahkan nanti.
    """
    return {
        "RESEARCH_BYPASS_ROUTING": RESEARCH_BYPASS_ROUTING,
        "RESEARCH_DISABLE_CONDENSATION": RESEARCH_DISABLE_CONDENSATION,
        "RESEARCH_DISABLE_OUTPUT_FILTER": RESEARCH_DISABLE_OUTPUT_FILTER,
        "RESEARCH_VERBOSE_RETRIEVAL": RESEARCH_VERBOSE_RETRIEVAL,
    }
