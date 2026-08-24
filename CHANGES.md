# CHANGES — Tahap 1: identitas chunk & pemecahan ganda

Branch `chunk-identity-and-dump`, bercabang dari tag `baseline-riset`.
Salinan riset di `github.com/Ikrar06/rag-mm-b`. Push hanya ke `origin`.

**Seluruh perubahan opt-in dan default mati.** Tanpa satu pun env var diset,
jalur indexing berperilaku persis seperti `baseline-riset` — diverifikasi lewat
`scripts/probe_rechunk.py` (baseline: 5/9 kasus bertabrakan, sama seperti
sebelum perubahan).

Jalur query tidak disentuh sama sekali: tidak ada perubahan di `rag_pipeline.py`,
`moderation.py`, `intent_classifier.py`, `output_filter.py`, `auth.py`, maupun
`routers/`.

---

## Masalah yang diperbaiki

`indexing.py` memanggil `VectorStoreIndex.from_documents()` tanpa argumen
`transformations=`, sehingga LlamaIndex menerapkan node parser default dan
**memecah ulang** Document yang melewati `Settings.chunk_size`. Node anak
mewarisi metadata induk, termasuk `chunk_index` — jadi beberapa titik Qdrant
berbagi satu `chunk_index`, dan `chunk_index` bukan kunci unik.

Terukur pada bentuk data yang benar-benar diproduksi repo ini (Tahap 0):

| Bentuk chunk | Token | Node sebelum |
|---|---:|---:|
| 1 halaman A4 jalur `fast` | 713 | 3 |
| 1 halaman padat jalur `fast` | 1.029 | 4 |
| Tabel besar (`PDF_TABLE_MAX_CHARS`) | 811 | 3 |
| Element teks panjang | 476 | 2 |

Ambang pecah sebenarnya **bukan** `CHUNK_SIZE` (512) melainkan
`effective_chunk_size = 512 - metadata_len` = **384**, karena metadata memakan
kuota (`llama_index/core/node_parser/text/sentence.py:158`).

---

## Perubahan per berkas

### `backend/config.py`

| Baris | Tambahan |
|---|---|
| `113-131` | `INDEX_EXCLUDE_METADATA_FROM_EMBED` (bool, default `false`) |
| `133-146` | `NON_SEMANTIC_METADATA_KEYS` (tuple, 7 field) |
| `148-168` | `INDEX_MAX_CHUNK_TOKENS` (int, default `0` = mati) |
| `170-188` | `INDEX_DISABLE_NODE_PARSER` (bool, default `false`) |

### `backend/services/indexing.py`

| Baris | Perubahan |
|---|---|
| `20-29` | Impor tiga konstanta baru dari `config` |
| `163-186` | `_embed_and_store` bercabang: `transformations=[]` saat `INDEX_DISABLE_NODE_PARSER` |
| `271-275` | `excluded_keys` dihitung sekali per berkas |
| `277-292` | `Document(...)` menerima `excluded_embed_metadata_keys` + `excluded_llm_metadata_keys` |

### `backend/services/preprocessing.py`

| Baris | Perubahan |
|---|---|
| `27` | Impor `INDEX_MAX_CHUNK_TOKENS` |
| `39` | Singleton `_tokenizer` |
| `308-430` | Blok baru: `_SPLIT_SEPARATORS`, `_get_tokenizer`, `_ntok`, `_pack`, `_hard_cut`, `_merge_fitting`, `_apply_overlap`, `_split_for_budget` |
| `471-487` | Helper `emit()` baru di `_chunk_elements` |
| `489-500` | `flush()` memakai `emit(splittable=True)` |
| `528-533` | Cabang `Table` memakai `emit(splittable=False)` |
| `549-554` | Cabang `ImageDescription` memakai `emit(splittable=False)` |

### `scripts/probe_rechunk.py`

Berkas baru. Verifikasi tanpa Qdrant dan tanpa model embedding.

---

## Env var baru

| Nama | Tipe | Default | Efek |
|---|---|---|---|
| `INDEX_EXCLUDE_METADATA_FROM_EMBED` | bool | `false` | Hanya `section` yang divektorkan; 7 field lain dikecualikan |
| `INDEX_MAX_CHUNK_TOKENS` | int | `0` (mati) | Batas token per chunk keluaran `_chunk_elements` |
| `INDEX_DISABLE_NODE_PARSER` | bool | `false` | `transformations=[]` → 1 Document = 1 node |

Nilai yang dipakai saat verifikasi: `true` / `350` / `true`.

---

## Hasil terukur

`INDEX_EXCLUDE_METADATA_FROM_EMBED`:

| | metadata_len | effective_chunk_size |
|---|---:|---:|
| Mati | 128 tok | 384 |
| Nyala | **14 tok** | **498** |

`file_hash` sendirian menyumbang 68 dari 128 token (53%) untuk nol nilai semantik.

Probe 9 kasus:

| Kasus | Token | Baseline | Final (1A+1B+1C) |
|---|---:|---|---|
| teks pendek | 81 | 1 chunk → 1 node | 1 → 1 |
| teks ~CHUNK_SIZE char | 160 | 1 → 1 | 1 → 1 |
| teks 1 element panjang | 476 | 1 → **2 tabrakan** | 2 → 2 unik |
| teks 1 element sangat panjang | 1.187 | 1 → **5 tabrakan** | 4 → 4 unik |
| tabel kecil | 224 | 1 → 1 | 1 → 1 |
| tabel besar | 811 | 1 → **3 tabrakan** | 1 → 1 unik |
| deskripsi gambar | 196 | 1 → 1 | 1 → 1 |
| fast path 1 halaman A4 | 713 | 1 → **3 tabrakan** | 3 → 3 unik |
| fast path 1 halaman padat | 1.029 | 1 → **4 tabrakan** | 4 → 4 unik |

**Baseline: 5/9 bertabrakan. Final: 0/9. Kriteria lulus terpenuhi.**

---

## Keputusan sadar: tabel TIDAK dipecah

Ini keputusan desain, bukan keterbatasan implementasi.

**Alasan.** Memecah Markdown tabel memisahkan baris header dari baris data.
Metrik **RCAA (Row-Column Alignment Accuracy)** di lapis 3 mengukur persis apakah
sistem membaca sel dari baris dan kolom yang benar. Fragmen tabel tanpa header
membuat pertanyaan seperti *"IPK minimum untuk S2"* tidak terjawab dari chunk
mana pun — kegagalan yang disebabkan pipeline, bukan oleh strategi indexing yang
sedang dibandingkan. Alasan yang sama berlaku untuk `structured_summary` tabel di
`images.jsonl`, yang menurut deck sama dengan `text_as_html` chunk terkait.

**Konsekuensi yang harus diterima.** Chunk tabel **jauh lebih panjang** daripada
chunk teks. Chunk teks dibatasi `INDEX_MAX_CHUNK_TOKENS` (350 token pada
verifikasi); chunk tabel dibatasi `PDF_TABLE_MAX_CHARS` (2.000 karakter ≈ 811
token pada rasio terukur 2,44 char/token). Jadi:

- Distribusi panjang chunk menjadi bimodal. Statistik "rata-rata panjang chunk"
  akan menyesatkan bila dilaporkan tanpa dipisah per `element_type`.
- Chunk tabel mengonsumsi porsi konteks LLM yang jauh lebih besar per chunk.
  Dengan `RERANKER_TOP_N=6`, enam chunk tabel ≈ 4.900 token, mendekati
  `--max-model-len 8192` di `docker-compose.poc.yml:66`.
- Chunk tabel hanya aman selama `INDEX_DISABLE_NODE_PARSER=true`. Menyalakan 1B
  tanpa 1C menyisakan tabel besar tetap dipecah — inilah alasan 1C wajib, bukan
  sekadar jaring pengaman.

**`PDF_TABLE_MAX_CHARS` kini menentukan batas atas ukuran chunk.** Sebelumnya ia
hanya memutuskan tabel jadi chunk sendiri atau digabung ke buffer. Sekarang, karena
tabel tidak pernah dipecah lebih lanjut, nilainya adalah satu-satunya yang membatasi
seberapa besar sebuah chunk bisa jadi. **Wajib masuk `run_manifest.json`.**

Deskripsi gambar juga tidak dipecah, dengan alasan sejalan: satu deskripsi = satu
gambar, dan memecahnya merusak relasi `narrative_summary` ↔ `image_id`. Praktisnya
tidak pernah terpicu karena `image_describer.py:150` membatasi `max_tokens=300`.

---

## Yang harus diterapkan IDENTIK oleh peneliti lain di fork yang sama

Perbandingan dua strategi indexing hanya valid bila ekstraksi identik. Berikut yang
wajib sama persis.

### 1. Ketiga env var, dengan nilai yang sama

```
INDEX_EXCLUDE_METADATA_FROM_EMBED=true
INDEX_MAX_CHUNK_TOKENS=350
INDEX_DISABLE_NODE_PARSER=true
```

Ketiganya mengubah **isi dan batas chunk**. Fork yang menyalakan sebagian saja
menghasilkan pembagian chunk yang berbeda, dan seluruh perbandingan runtuh.

### 2. Wajib masuk `run_manifest.json`

Di luar daftar "Kelompok 1" di `INSPECTION_REPORT.md` bagian E17, tambahkan:

| Var | Alasan |
|---|---|
| `INDEX_EXCLUDE_METADATA_FROM_EMBED` | Menentukan teks yang divektorkan |
| `INDEX_MAX_CHUNK_TOKENS` | Batas atas chunk teks |
| `INDEX_DISABLE_NODE_PARSER` | Menentukan ada tidaknya pemecahan kedua |
| **`PDF_TABLE_MAX_CHARS`** | **Kini batas atas ukuran chunk tabel** — lihat keputusan di atas |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | Sudah wajib sebelumnya; `CHUNK_OVERLAP` kini juga dipakai `_apply_overlap` |

### 3. Re-index penuh diperlukan

`chunk_index` **berubah nilainya** saat 1B aktif: element yang sebelumnya jadi satu
chunk kini jadi beberapa, dan penomoran berikutnya bergeser. Index lama dan index
baru tidak sebanding. Jalankan `--force`, dan jangan mencampur chunk dari dua
konfigurasi dalam satu collection.

### 4. Perhatian: reranker `sentence_transformers` terpengaruh

`SentenceTransformerRerank` membaca `MetadataMode.EMBED`
(`llama_index/core/postprocessor/sbert_rerank.py:75`), jadi
`INDEX_EXCLUDE_METADATA_FROM_EMBED=true` **mengubah teks yang di-rerank**.

| Environment | `RERANKER_PROVIDER` | Terpengaruh |
|---|---|---|
| `.env.poc` | `tei` | Tidak |
| `.env.dev` | `sentence_transformers` | **Ya** |
| `.env.alt.poc` | `sentence_transformers` | **Ya** |

Jalur TEI memakai `get_content()` dengan default `MetadataMode.NONE`, jadi ia tidak
pernah melihat metadata. Konsekuensi yang berdiri sendiri: **kedua reranker sudah
melihat teks berbeda sejak sebelum perubahan ini** — TEI tanpa metadata, ST dengan
metadata. Ini memperkuat temuan F4 di `INSPECTION_REPORT_2.md`.

Untuk perbandingan yang adil, kedua fork harus memakai `RERANKER_PROVIDER` yang sama.

### 5. Dua entry point kini seragam

Sebelumnya `POST /api/index` memakai `Settings` default LlamaIndex (1024/200) karena
tidak pernah memanggil `_configure_settings`, sedangkan CLI memakai 512/128. Dengan
`INDEX_DISABLE_NODE_PARSER=true` keduanya tidak lagi bergantung `Settings`, jadi
perbedaan itu hilang. **Tanpa flag itu, perbedaannya masih ada** — jangan mencampur
dokumen yang di-index lewat dua jalur berbeda.

---

## Verifikasi

```bash
# Baseline — harus 5/9 bertabrakan
python scripts/probe_rechunk.py

# Final — harus 0/9
INDEX_EXCLUDE_METADATA_FROM_EMBED=true \
INDEX_MAX_CHUNK_TOKENS=350 \
INDEX_DISABLE_NODE_PARSER=true \
python scripts/probe_rechunk.py
```

Probe butuh `llama-index-core==0.14.13`, `tiktoken`, `markdownify==0.13.1`, dan
`pymupdf` (yang terakhir hanya karena `preprocessing.py:21` mengimpor `fitz` di
level modul). Tidak butuh Qdrant, tidak memuat model embedding.

---

## Residu yang diketahui

**Chunk judul yatim.** Saat sebuah element teks panjang dipecah, baris judul
`# {section}` yang ditambahkan `_chunk_elements` bisa terpisah menjadi chunk sendiri
berukuran ~6 token bila potongan pertama sesudahnya sudah memenuhi anggaran.
`_merge_fitting` menggabungkan serpihan bertetangga yang muat, tetapi judul + potongan
350-token tidak muat sehingga tetap terpisah. Dampaknya kecil (`section` tetap ada di
metadata setiap chunk), tetapi chunk pendek itu ikut divektorkan.

**Belum terjawab tanpa korpus.** Probe memakai teks susunan tangan yang meniru bentuk
dokumen di deck. Yang belum diketahui: berapa **proporsi** chunk nyata yang melewati
ambang, dan berapa banyak tabel yang melewati `PDF_TABLE_MAX_CHARS`. Setelah
`data/pdfs` terisi, jalankan V9 di `INSPECTION_REPORT_2.md`.

---

## Temuan sampingan (tidak diperbaiki, sesuai batasan tugas)

1. **`is_likely_informative` adalah fungsi mati.** `image_describer.py:71-92` tidak
   punya pemanggil di `backend/` maupun `scripts/`, sehingga ambang
   `PDF_MIN_IMAGE_SIZE_KB` (`:76`) tidak berefek pada apa pun.

2. **`RERANKER_USE_FP16` dan `RATE_LIMIT_VISION_PER_MINUTE` tidak pernah dibaca.**
   Keduanya didefinisikan di `config.py` tetapi tidak punya pembaca.

3. **`JWT_ALGORITHM` di `config.py` tidak dipakai.** `auth.py:23` meng-hardcode
   `"HS256"` dan membaca `JWT_SECRET`/`JWT_EXPIRE_HOURS` lewat `os.getenv` sendiri,
   bukan lewat `config`.

4. **`_html_table_to_markdown` diam-diam mengembalikan HTML mentah saat gagal**
   (`preprocessing.py:186`), sehingga chunk `element_type="Table"` kadang berisi
   Markdown kadang HTML tanpa penanda mana yang mana.

---

## Catatan git

`.gitignore:43` memuat pola `*.md`, jadi berkas ini **tidak ter-track**:

```bash
git add -f CHANGES.md
```
