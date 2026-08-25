# DEPLOY.md — pasang harness riset di server DGX

Dari repo yang sudah ter-clone sampai siap indexing.

**Kondisi server yang diasumsikan** (jangan pakai panduan ini untuk lingkungan lain):

| | |
|---|---|
| OS / arsitektur | Ubuntu 24.04, aarch64 |
| GPU | NVIDIA GB10, compute capability **sm_121**, CUDA 13.2 |
| Memori | **128 GB terpadu** — bukan VRAM terpisah |
| Python | 3.12.3 |
| sudo | **tidak ada** |
| Docker | **tidak ada** (user bukan anggota grup `docker`) |
| Grup user | `users`, `conda`, `jupyterhub` |
| Disk | ~169 GB sisa dari 916 GB |
| Repo | `~/rag_mm_b` |

---

## ⚠ Mesin dipakai bersama

Memori GB10 **terpadu**: tidak ada VRAM terpisah, jadi 128 GB itu dipakai bersama
oleh sistem operasi, proses indexing kamu, model embedding dan reranker yang
kamu muat, **dan** apa pun yang dijalankan peneliti lain.

Yang perlu diingat:

- **Instance Qdrant kedua** dari peneliti lain ikut memakan kolam yang sama.
  Qdrant memetakan segmen ke memori; korpus besar berarti jejak besar.
- **Ollama** yang dijalankan siapa pun di mesin ini memuat bobot model ke kolam
  yang sama.
- Proses indexing kamu sendiri memuat model embedding (Qwen3-Embedding-0.6B) dan
  reranker (bge-reranker-v2-m3) sekaligus.

Sebelum menjalankan indexing panjang, lihat siapa lagi yang sedang memakai:

```bash
nvidia-smi                       # proses GPU dan pemakaiannya
free -g                          # kolam terpadu; ini angka yang berlaku
ps aux | grep -E "qdrant|ollama|python" | grep -v grep
```

`scripts/verify_env.py` melaporkan memori tersedia dan mendaftar collection
Qdrant yang sudah ada di server — pakai itu untuk melihat apakah orang lain
sudah memakai instance yang sama.

---

## Langkah 1 — binari sistem lewat conda

`unstructured[pdf]` dengan `strategy="hi_res"` butuh dua binari yang biasanya
dipasang lewat `apt`. Tanpa sudo, conda satu-satunya jalur.

```bash
conda --version                  # pastikan tersedia (user ada di grup `conda`)

# Environment terpisah HANYA untuk binari, bukan untuk Python paket.
conda create -y -n ragbin -c conda-forge tesseract poppler
conda activate ragbin

tesseract --version              # harus muncul
pdftoppm -v                      # poppler; harus muncul
which tesseract pdftoppm
```

Bahasa Indonesia untuk tesseract (`ind`) dipakai `partition_pdf(languages=["ind","eng"])`:

```bash
conda install -y -c conda-forge tesseract-data-ind tesseract-data-eng
tesseract --list-langs           # harus memuat `ind` dan `eng`
```

> Kalau paket data bahasa itu tidak tersedia di conda-forge, unduh
> `ind.traineddata` dan `eng.traineddata` dari repo `tesseract-ocr/tessdata`
> ke `$CONDA_PREFIX/share/tessdata/` secara manual, lalu ulangi `--list-langs`.

**Environment `ragbin` harus aktif setiap kali indexing dijalankan**, karena
`tesseract` dan `pdftoppm` dicari lewat `PATH`.

---

## Langkah 2 — venv Python dengan `--system-site-packages`

```bash
cd ~/rag_mm_b
python3 --version                # harus 3.12.x

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

`--system-site-packages` **wajib**: torch bawaan DGX OS adalah satu-satunya build
yang memuat kernel sm_121. Wheel torch dari PyPI dibangun untuk arsitektur lain
dan akan gagal senyap di GPU ini.

Konfirmasi torch sistem terlihat dari venv **sebelum** memasang apa pun:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(), torch.cuda.get_arch_list())"
```

Catat keluarannya. Kalau `sm_121` tidak ada di `get_arch_list()`, hentikan —
torch sistemnya sendiri sudah tidak cocok, dan tidak ada yang bisa diperbaiki
dari sisi Python.

---

## Langkah 3 — periksa lingkungan sebelum memasang paket

```bash
python scripts/verify_env.py
```

Wajar bila banyak yang `GAGAL` di titik ini — paket belum dipasang. Yang harus
sudah lolos: **versi Python**, **torch**, **tesseract**, **poppler**.

---

## Langkah 4 — pasang paket bertahap

Bertahap supaya kegagalan mudah diisolasi, dan supaya pembayangan torch
ketahuan di tahap mana ia terjadi.

**Verifikasi ketersediaan wheel dulu** (tidak memasang apa pun):

```bash
pip download --no-deps --only-binary=:all: \
    --platform manylinux2014_aarch64 --python-version 312 \
    -d /tmp/whlcheck -r requirements-research.txt
```

Paket yang gagal di sini tidak punya wheel aarch64 dan akan mencoba compile dari
sdist — tanpa sudo, itu kemungkinan besar gagal.

Lalu pasang per blok, **jalankan `verify_env.py` setelah tiap blok**:

```bash
pip install --upgrade pip

# 4a — inti LlamaIndex + vector store (semua pure-python, cepat)
pip install \
  llama-index-core==0.14.13 \
  llama-index-vector-stores-qdrant==0.10.1 \
  llama-index-embeddings-huggingface==0.6.1 \
  llama-index-llms-ollama==0.10.1 \
  qdrant-client==1.17.1
python scripts/verify_env.py | grep -E "torch|llama-index-core|qdrant-client"

# 4b — embedding & reranker. Titik risiko torch PERTAMA.
pip install \
  sentence-transformers==3.3.1 transformers==4.47.1 \
  tokenizers==0.21.0 huggingface-hub==0.27.1 tiktoken==0.14.0
python scripts/verify_env.py | grep -A2 "torch / GPU"

# 4c — ekstraksi PDF. Titik risiko torch TERBESAR:
#      unstructured[pdf] menarik effdet + unstructured-inference,
#      yang meminta torchvision dan timm.
pip install "unstructured[pdf]==0.16.11" numpy==1.26.4 \
  pymupdf==1.24.14 pdfplumber==0.11.4 markdownify==0.13.1 pillow==11.0.0
python scripts/verify_env.py | grep -A2 "torch / GPU"

# 4d — utilitas
pip install httpx==0.28.1 python-dotenv==1.0.1 PyYAML==6.0.2 \
  pydantic==2.11.7 prometheus-client==0.23.1

python scripts/verify_env.py
```

### Kalau torch terbayangi

Gejalanya: setelah suatu blok, `get_device_capability()` berubah, atau `sm_121`
hilang dari `get_arch_list()`, atau `torch.__version__` berbeda dari yang kamu
catat di Langkah 2.

```bash
pip list --format=freeze | grep -iE "^(torch|torchvision|timm)="   # apa yang masuk venv
pip uninstall -y torch torchvision                                  # buang yang dari PyPI
python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
pip install --no-deps timm torchvision   # kalau memang dibutuhkan, tanpa dependensi
```

`--no-deps` mencegah torchvision menarik torch versinya sendiri.

---

## Langkah 5 — konfigurasi

```bash
cd ~/rag_mm_b
cp .env.research .env
```

**`.env` harus di root repo** (sejajar `backend/`). Terverifikasi:
`config.py` memanggil `load_dotenv()` yang mencari dari lokasi `config.py`
ke atas, **bukan** dari direktori kerja. `.env` di direktori kerja lain
**diabaikan senyap** dan seluruh flag riset jatuh ke default — indexing tetap
berjalan tapi menghasilkan chunk yang salah.

Verifikasi flag benar-benar termuat:

```bash
python -c "
from backend.config import *
for k in ('INDEX_EXCLUDE_METADATA_FROM_EMBED','INDEX_MAX_CHUNK_TOKENS','INDEX_MIN_CHUNK_TOKENS',
          'INDEX_DISABLE_NODE_PARSER','INDEX_STRUCTURAL_METADATA','INDEX_TABLES_AS_OWN_CHUNKS',
          'INDEX_PERSIST_IMAGES','PDF_EXTRACTION_STRATEGY'):
    print(f'{k:<36}{globals()[k]!r}')"
```

Harus persis: `True 350 8 True True True True 'hi_res'`.

Lebih baik lagi, `RESEARCH_MODE=true` di `.env.research` membuat indexing
**mencetak nilai efektif tiap flag ke stdout saat start** dan **menolak run**
bila ada yang tidak sesuai daftar beku di `CHANGES.md`. Uji tanpa korpus:

```bash
python -c "
import backend.services.indexing as ix
ix.print_effective_flags()
ix._check_research_mode(); ix._check_flag_consistency()
ix._check_image_strategy(); ix._check_vision_reachable()
print('SEMUA GERBANG LOLOS')"
```

Kalau `.env` salah lokasi, blok yang tercetak akan menunjukkan nilai default
dengan penanda `<-- BEDA`, dan `_check_research_mode` menolak sebelum satu PDF
pun disentuh.

### `QDRANT_COLLECTION` tidak ada di `.env` — disengaja

Kirim eksplisit setiap run:

```bash
QDRANT_COLLECTION=rag_mm_b_varian_c python -m scripts.index_documents
```

Alasannya: `_ensure_collection` punya jalur pemulihan yang **menghapus**
collection bila namanya bertabrakan. Dua sesi shell di clone yang **sama**
memuat `.env` yang sama, jadi nama collection yang salah tidak akan tertangkap
oleh apa pun kecuali disiplin. Nilainya tercatat di `run_manifest.json`, jadi
salah kirim tetap terlacak setelahnya.

---

## Langkah 6 — Qdrant tanpa Docker

Binary rilis resmi, aarch64 musl (statis, tidak bergantung versi glibc).

**Terverifikasi 2026-08-25:**

| | |
|---|---|
| Versi | `v1.17.1` — cocok dengan `qdrant-client==1.17.1` yang di-pin |
| Aset | `qdrant-aarch64-unknown-linux-musl.tar.gz` |
| Ukuran | 33.069.416 byte (32 MB) |
| sha256 | `9347a4db839f53fe123cc775bd87e4dd02f6c2750783bea02ea4fcae9c923164` |
| Isi | satu berkas: `qdrant` (tanpa direktori config) |

```bash
mkdir -p ~/opt/qdrant ~/rag_mm_b_data/qdrant/{storage,snapshots} ~/rag_mm_b_data/log
cd ~/opt/qdrant

curl -L -o qdrant.tar.gz \
  https://github.com/qdrant/qdrant/releases/download/v1.17.1/qdrant-aarch64-unknown-linux-musl.tar.gz

echo "9347a4db839f53fe123cc775bd87e4dd02f6c2750783bea02ea4fcae9c923164  qdrant.tar.gz" | sha256sum -c -
tar xzf qdrant.tar.gz && chmod +x qdrant && rm qdrant.tar.gz
./qdrant --version
```

### Konfigurasi lewat env var

Terverifikasi dari source `v1.17.1`: `settings.rs:313` memakai
`Environment::with_prefix("QDRANT").separator("__")`, dan kunci config-nya
`storage.storage_path`, `storage.snapshots_path`, `service.host`,
`service.http_port`, `service.grpc_port`.

**Port 16333/16334, bukan default 6333/6334** — supaya tidak bentrok dengan
instance peneliti lain. Cek dulu:

```bash
ss -ltn | grep -E ':(6333|6334|16333|16334)\b' || echo "keempat port bebas"
```

Kalau 16333 ternyata terpakai, pilih pasangan lain dan **perbarui `QDRANT_URL`
di `.env`** — bukan hanya di perintah start. Port ikut tercatat di
`run_manifest.json` lewat `provenance.qdrant_url`, sehingga run sebelum dan
sesudah pindah port dapat dibedakan dari artefaknya saja.

### Skrip start/stop

Tanpa systemd (butuh sudo), `setsid` melepas proses dari sesi SSH sehingga ia
tetap hidup setelah koneksi ditutup.

`~/opt/qdrant/start.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
BASE="$HOME/rag_mm_b_data"
PIDFILE="$BASE/qdrant.pid"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "Qdrant sudah jalan (PID $(cat "$PIDFILE"))"; exit 0
fi

export QDRANT__SERVICE__HOST=127.0.0.1
export QDRANT__SERVICE__HTTP_PORT=16333
export QDRANT__SERVICE__GRPC_PORT=16334
export QDRANT__STORAGE__STORAGE_PATH="$BASE/qdrant/storage"
export QDRANT__STORAGE__SNAPSHOTS_PATH="$BASE/qdrant/snapshots"
export QDRANT__TELEMETRY_DISABLED=true

setsid nohup "$HOME/opt/qdrant/qdrant" \
  >> "$BASE/log/qdrant.log" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
sleep 3
curl -sf http://127.0.0.1:16333/ >/dev/null \
  && echo "Qdrant hidup, PID $(cat "$PIDFILE")" \
  || { echo "GAGAL start — lihat $BASE/log/qdrant.log"; exit 1; }
```

`~/opt/qdrant/stop.sh`:

```bash
#!/usr/bin/env bash
PIDFILE="$HOME/rag_mm_b_data/qdrant.pid"
[ -f "$PIDFILE" ] || { echo "PID file tidak ada"; exit 0; }
PID=$(cat "$PIDFILE")
kill "$PID" 2>/dev/null && echo "SIGTERM ke $PID"
for _ in $(seq 20); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
kill -0 "$PID" 2>/dev/null && { echo "masih hidup, SIGKILL"; kill -9 "$PID"; }
rm -f "$PIDFILE"
```

```bash
chmod +x ~/opt/qdrant/{start,stop}.sh
~/opt/qdrant/start.sh
curl -s http://127.0.0.1:16333/ | head -c 200; echo
```

`bind 127.0.0.1` berarti Qdrant **tidak** terjangkau dari luar mesin. Itu
disengaja: tidak ada autentikasi yang dikonfigurasi.

> **`--config-path` sebagai alternatif.** `main.rs:117` menerima argumen
> `config_path`. Kalau env var ternyata tidak terbaca sesuai harapan, tulis
> `config.yaml` dengan kunci di atas dan jalankan
> `./qdrant --config-path ~/rag_mm_b_data/qdrant/config.yaml`.

---

## Langkah 7 — verifikasi lingkungan penuh

```bash
cd ~/rag_mm_b && source .venv/bin/activate && conda activate ragbin
python scripts/verify_env.py
```

Semua pemeriksaan wajib harus lolos. Perhatikan juga baris
`collection lain di server` — kalau ada nama yang bukan milikmu, peneliti lain
memakai instance yang sama dan pemisahan collection jadi kritis.

---

## Langkah 8 — jalankan ulang `probe_rechunk.py` dan bandingkan

**Ini prasyarat, bukan formalitas.** Seluruh angka Tahap 1–4 diukur di Python
3.14 karena venv proyek belum ada. Perilaku `SentenceSplitter` bergantung pada
versi `llama-index-core` dan `tiktoken`.

```bash
cd ~/rag_mm_b
python scripts/probe_rechunk.py            # dengan .env terpasang, flag riset aktif
```

Banner `LINGKUNGAN EKSEKUSI` harus menunjukkan `python : 3.12.x` **tanpa**
peringatan target.

### Angka acuan, diukur di Python 3.14 / llama-index-core 0.14.13 / tiktoken 0.14.0

**Rasio karakter per token (teks akademik Indonesia):**

| sampel | chars | token | char/tok |
|---|---:|---:|---:|
| paragraf SOP (naratif) | 1.403 | 397 | **3,53** |
| tabel Markdown | 1.979 | 811 | **2,44** |
| deskripsi flowchart | 602 | 176 | 3,42 |
| judul section pendek | 38 | 12 | 3,17 |

**Sembilan kasus uji — semuanya harus `unik` / `unik`:**

| kasus | token | chunk | node | chunk_index | chunk_id |
|---|---:|---:|---:|---|---|
| teks pendek | 81 | 1 | 1 | unik | unik |
| teks ~CHUNK_SIZE char | 160 | 1 | 1 | unik | unik |
| teks 1 element panjang | 476 | 2 | 2 | unik | unik |
| teks 1 element sangat panjang | 1.187 | 4 | 4 | unik | unik |
| tabel kecil | 224 | 1 | 1 | unik | unik |
| tabel besar | 811 | 1 | 1 | unik | unik |
| deskripsi gambar | 196 | 1 | 1 | unik | unik |
| fast path 1 halaman A4 | 713 | 3 | 3 | unik | unik |
| fast path 1 halaman padat | 1.029 | 4 | 4 | unik | unik |

**Anggaran metadata** — nol, karena tidak ada metadata yang divektorkan:

| skenario | field | meta tok | efektif |
|---|---:|---:|---:|
| sekarang (8 field) | 8 | **0** | 512 |
| + chunk_id, document_id | 10 | 0 | 512 |
| + image_id, visual_type | 12 | 9 | 503 |
| + manifest lengkap | 16 | 70 | 442 |

**Verdict yang diharapkan:**

```
VERDICT: LULUS — 9/9 kasus, nol tabrakan chunk_index, 18 chunk_id unik.
```

### Kalau angkanya berbeda

Kolom `chunk` dan `node` yang berbeda berarti pembagian chunk berubah — dan
seluruh nilai beku di `CHANGES.md` perlu dihitung ulang sebelum indexing.
Rasio char/token yang berbeda menandakan versi `tiktoken` tidak sesuai pin.
**Jangan lanjut indexing sampai selisihnya dijelaskan.**

Bandingkan juga dengan baseline (semua flag mati) untuk memastikan perilaku
lama tetap utuh:

```bash
env -u INDEX_EXCLUDE_METADATA_FROM_EMBED -u INDEX_MAX_CHUNK_TOKENS \
    -u INDEX_MIN_CHUNK_TOKENS -u INDEX_DISABLE_NODE_PARSER \
    -u INDEX_STRUCTURAL_METADATA -u INDEX_TABLES_AS_OWN_CHUNKS \
    python scripts/probe_rechunk.py | grep VERDICT
# Diharapkan: GAGAL — chunk_index bertabrakan di 5 kasus
```

> Perintah di atas hanya bekerja bila flag berasal dari environment. Karena
> `.env` sudah terpasang, cara yang andal adalah memindahkan `.env` sementara:
> `mv .env .env.off && python scripts/probe_rechunk.py | grep VERDICT && mv .env.off .env`

---

## Langkah 8b — model vision di Ollama

Deskripsi gambar memakai Ollama (vLLM butuh Docker). Kondisi server yang
terverifikasi: `ollama` di `/usr/local/bin/ollama`, server hidup di
`localhost:11434`, model `qwen3-vl:8b` sudah ditarik.

```bash
curl -s http://127.0.0.1:11434/api/tags | python -m json.tool | grep -E '"(name|digest)"'
```

Harus memuat `qwen3-vl:8b` beserta digest `sha256:...` penuh.

> **`qwen2.5:7b` milik pengguna lain.** Jangan `ollama rm`, jangan `ollama pull`
> ulang. `ollama pull` pada tag yang sama dapat mengganti bobot tanpa mengubah
> nama tag — itu akan mengubah deskripsi gambar di tengah eksperimen.

Deskripsi gambar adalah **isi chunk**, jadi seluruh parameternya wajib identik
antara run varian (b) dan (c). `.env.research` sudah menyetelnya:
`VISION_MODEL=qwen3-vl:8b`, `VISION_TEMPERATURE=0`, `VISION_MAX_TOKENS=300`,
`RESEARCH_VISION_SEED=1337`, `VISION_NUM_CTX=8192`.

`VISION_NUM_CTX` wajib eksplisit: tanpa itu Ollama memotong konteks ke 4096
token secara senyap, dan satu gambar saja bisa menghabiskannya.

Setelah indexing, periksa `models.vision.digest_resolved` di
`run_manifest.json`. `false` berarti digest tidak terbaca dari `/api/tags` saat
run itu — bobot yang menghasilkan deskripsi tidak dapat dibuktikan setelahnya.

---

## Langkah 9 — unduh model

```bash
python scripts/download_models.py --model reranker
```

`bge-reranker-v2-m3` (~2,3 GB) diunduh ke `models/`. Model embedding
(`Qwen3-Embedding-0.6B`) diunduh otomatis oleh `HuggingFaceEmbedding` saat
pertama dipakai, ke cache HuggingFace default.

`--model intent` **tidak diperlukan**: intent classifier hanya dipakai jalur
query, dan jalur itu tidak dipakai riset. Tanpa modelnya, `classify_intent`
mengembalikan fallback tanpa menggagalkan apa pun.

Awasi kuota: cache HuggingFace ada di `~/.cache/huggingface` dan **dipakai
bersama** semua proyekmu di mesin ini.

```bash
du -sh ~/.cache/huggingface ~/rag_mm_b_data 2>/dev/null
df -h ~
```

---

## ⛔ Langkah yang BELUM BISA dijalankan

Dua prasyarat riset belum ada. Semua di atas dapat diselesaikan tanpa keduanya.

### A. Korpus PDF — `data/pdfs/` masih kosong

Tanpa PDF, ini semua belum bisa:

- `python -m scripts.index_documents` — tidak ada yang di-index
- Verifikasi bahwa `partition_pdf` benar-benar mengisi `coordinates`, sehingga
  `bbox` tidak selalu null
- Verifikasi bahwa element `Image` benar-benar punya `image_base64` terisi
- Proporsi dokumen yang jatuh ke fallback `hi_res` → `fast`
- V9 dan V13 di `INSPECTION_REPORT_2.md`
- Pemeriksaan kelayakan strata (deck menuntut ≥15–20 dokumen flowchart, ≥15–20
  tabel, ≥10–15 formulir, ≥10 figur deskriptif)

### B. `data/document_registry.json` — belum ada

Butuh korpus dulu untuk kerangkanya, lalu kurasi manual:

```bash
python scripts/scaffold_document_registry.py     # setelah PDF ada
# lalu isi document_id tiap entri secara manual
```

**Sepakati berkas registry ini dengan peneliti fork lain SEBELUM salah satu
mulai meng-index.** `document_id` menentukan `chunk_id`, `image_id`, **dan** nama
direktori gambar di disk sekaligus. Registry berbeda berarti `gold_chunk_ids`
dan `gold_image_ids` tidak akan cocok lintas fork.

Berkas PDF yang tidak terdaftar (atau `document_id`-nya kosong) akan
**dilewati** saat indexing, dengan error log dan hitungan ringkasan.

---

## Urutan menjalankan indexing (setelah A dan B ada)

```bash
cd ~/rag_mm_b
source .venv/bin/activate
conda activate ragbin
~/opt/qdrant/start.sh

python scripts/verify_env.py || exit 1

QDRANT_COLLECTION=rag_mm_b_varian_c python -m scripts.index_documents --force
```

Setelah selesai, periksa `run_manifest.json` di `data/dumps/<run_id>/`:

| Field | Harus |
|---|---|
| `dump_faithful` | `true` |
| `degraded_documents` | `[]` — kalau tidak kosong, run sudah digagalkan |
| `research_flags.ALLOW_INCOMPLETE_IMAGE_CORPUS` | `false` |
| `provenance.qdrant_url` | port yang benar |
| `provenance.qdrant_server_version_detected` | sama dengan `_declared` |
| `provenance.document_registry_sha256` | sama dengan peneliti fork lain |
| `research_flags.RESEARCH_MODE` | `true` |
| `models.vision.digest_resolved` | `true` |
| `models.vision.vision_model_digest` | sama dengan peneliti fork lain |

Bandingkan blok `chunking`, `research_flags`, `hardcoded_constants`, dan
`models` dengan manifest peneliti fork lain — semuanya harus identik. Daftar
lengkap yang boleh dan tidak boleh berbeda ada di bagian akhir `CHANGES.md`.

---

## Ringkasan berkas yang ditambahkan untuk deployment ini

| Berkas | Isi |
|---|---|
| `requirements-research.txt` | Paket jalur indexing + retrieval saja, tanpa torch/paddle/fastapi |
| `.env.research` | Flag riset beku + penyesuaian server |
| `scripts/verify_env.py` | Pemeriksaan lingkungan, tidak mengubah apa pun |
| `DEPLOY.md` | Berkas ini |

`requirements.txt` asli **tidak diubah** — ia tetap berlaku untuk deployment POC
x86 dengan Docker.
