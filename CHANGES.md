# CHANGES — Tahap 1–3: identitas chunk, metadata struktural, dump evaluasi

> **PRASYARAT SEBELUM INDEXING PERTAMA.** Seluruh angka di dokumen ini diukur di
> venv probe **Python 3.14**, sedangkan indexing riil berjalan di **Python 3.12**
> (`requirements.txt:3`). Perilaku `SentenceSplitter` bergantung pada versi
> `llama-index-core` dan `tiktoken`, jadi angka-angka ini **wajib diverifikasi
> ulang di venv target** sebelum korpus di-index. `scripts/probe_rechunk.py`
> kini mencetak versi Python + paket kunci dan memperingatkan bila tidak cocok.
> Yang paling perlu dicek ulang: rasio char/token, `metadata_len`,
> `effective_chunk_size`, dan ambang pecah empiris.

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

**Belum terjawab tanpa korpus.** Probe memakai teks susunan tangan yang meniru bentuk
dokumen di deck. Yang belum diketahui: berapa **proporsi** chunk nyata yang melewati
ambang, berapa banyak tabel yang melewati `PDF_TABLE_MAX_CHARS`, dan apakah
`el.metadata.coordinates` benar-benar terisi pada `strategy="hi_res"` untuk korpus ini.
Setelah `data/pdfs` terisi, jalankan V9 dan V13 di `INSPECTION_REPORT_2.md`.

**Chunk judul yatim — SELESAI di Tahap 2.** Lihat bagian Tahap 2 di bawah.

---
---

# TAHAP 2 — identitas berlapis tiga & metadata struktural

## Residu Tahap 1 yang ditutup

**Chunk judul yatim.** Diselesaikan dengan `INDEX_MIN_CHUNK_TOKENS`, yang
mengaktifkan **dua** penyaring sekaligus. Pengukuran menunjukkan ambang panjang saja
tidak cukup:

| Token | Isi |
|---:|---|
| 6 | `# Persyaratan Sidang` — judul |
| 8 | `Pengajuan ditolak.` — kalimat asli |
| 10 | `Berkas diverifikasi oleh admin prodi.` — kalimat asli |
| 18 | `# Persyaratan Pengajuan Izin Ujian Akhir Online Bagi Mahasiswa` — judul |

Ambang 18+ untuk menangkap semua judul akan ikut membuang kalimat asli 8–10 token.
Karena itu penyaring kedua bersifat struktural: chunk yang teks ternormalisasinya
persis sama dengan `# {section}` dibuang berapa pun panjangnya. Nol false positive —
`section` sudah ada di metadata setiap chunk, jadi chunk itu terbukti tidak membawa
konten unik. Nilai riset: **8**.

**`INDEX_MAX_CHUNK_TOKENS=350` dibekukan** dan didokumentasikan di `.env.example`
beserta turunan angkanya.

## Perubahan per berkas (Tahap 2)

### Berkas baru

| Path | Isi |
|---|---|
| `backend/services/document_registry.py` | Pemetaan nama berkas → `document_id`, dengan validasi slug dan deteksi id ganda |
| `scripts/scaffold_document_registry.py` | Membuat/memperbarui kerangka registry; aman dijalankan berulang |
| `.env.example` | Flag riset Tahap 1 & 2 beserta alasan tiap angka |

### `backend/config.py`

| Baris | Tambahan |
|---|---|
| `190-212` | `INDEX_MIN_CHUNK_TOKENS` (int, default `0`) |
| `214-241` | `INDEX_STRUCTURAL_METADATA` (bool, default `false`) |
| `243-248` | `INDEX_TABLES_AS_OWN_CHUNKS` (bool, default `false`) |
| `250-253` | `DOCUMENT_REGISTRY_PATH` |
| `146-153` | 6 field Tahap 2 ditambahkan ke `NON_SEMANTIC_METADATA_KEYS` |

### `backend/services/preprocessing.py`

| Baris | Perubahan |
|---|---|
| `107-130` | `normalize_for_hash`, `text_sha` |
| `211-224` | `_html_table_to_markdown` kini mengembalikan `(text, format)` |
| `227-267` | `_element_bbox` — normalisasi koordinat ke [0,1] |
| `548-559` | `_page_segment` — penanganan `page=0` sebagai `pNA` |
| `601-676` | `emit()` dengan `chunk_id`, `text_sha`, `extra`, `inherited_prefix`, dua penyaring |
| `680-695` | `_union_bbox` |
| `538-543` | `_split_for_budget` tidak lagi menempel overlap |

### `backend/services/indexing.py`

| Baris | Perubahan |
|---|---|
| `41-49` | `_STRUCTURAL_METADATA_KEYS` |
| `51-73` | `_check_flag_consistency` |
| `75-86` | `_resolve_document_id` |
| `316-380` | Resolusi registry, cek sha256, passthrough field struktural, ringkasan error |

## Hasil terukur (Tahap 2)

Probe 9 kasus, seluruh flag aktif:

| Kasus | Token | Baseline | Final |
|---|---:|---|---|
| teks pendek | 81 | 1 chunk → 1 node | 1 → 1, id unik |
| teks ~CHUNK_SIZE char | 160 | 1 → 1 | 1 → 1, id unik |
| teks 1 element panjang | 476 | 1 → **2 tabrakan** | 2 → 2, id unik |
| teks 1 element sangat panjang | 1.187 | 1 → **5 tabrakan** | 4 → 4, id unik |
| tabel kecil | 224 | 1 → 1 | 1 → 1, id unik |
| tabel besar | 811 | 1 → **3 tabrakan** | 1 → 1, id unik |
| deskripsi gambar | 196 | 1 → 1 | 1 → 1, id unik |
| fast path 1 halaman A4 | 713 | 1 → **3 tabrakan** | 3 → 3, id unik |
| fast path 1 halaman padat | 1.029 | 1 → **4 tabrakan** | 4 → 4, id unik |

**Baseline 5/9 bertabrakan → final 0/9, dengan 18 `chunk_id` unik.**

### Anggaran metadata final

| | field | metadata_len | effective_chunk_size |
|---|---:|---:|---:|
| Tanpa exclusion | 14 | **987 tok** | **−475 → ValueError** |
| Dengan exclusion | 14 | **14 tok** | **498** |

`raw_html` sebuah tabel 30 baris sendirian mencapai ~772–987 token. Anggaran
`INDEX_MAX_CHUNK_TOKENS=350` menyisakan **148 token headroom** terhadap 498.

### Independensi `text_sha`

Uji: satu paragraf diubah, dua paragraf lain identik.

| | sha bertahan |
|---|---|
| Sebelum perbaikan `inherited_prefix` | 1 dari 3 |
| Sesudah | **2 dari 3** |

Yang tersisa berbeda adalah chunk yang isinya memang ikut berubah. Ada **dua**
mekanisme overlap di `_chunk_elements` dan keduanya harus dikeluarkan dari dasar
hash: overlap antar-potongan hasil pemecahan, dan overlap antar-chunk dari buffer.
Memindahkan `_apply_overlap` saja hanya menutup yang pertama.

## Kombinasi flag yang ditolak

`INDEX_STRUCTURAL_METADATA=true` **tanpa** `INDEX_EXCLUDE_METADATA_FROM_EMBED`
maupun `INDEX_DISABLE_NODE_PARSER` akan membuat `SentenceSplitter` melempar
`ValueError("Metadata length (772) is longer than chunk size (512)")` di tengah
indexing. `_check_flag_consistency` menolaknya di awal dengan penjelasan.

| Kombinasi | Hasil |
|---|---|
| semua mati | lolos |
| `STRUCTURAL` saja | **ditolak** |
| `STRUCTURAL` + `EXCLUDE` | lolos |
| `STRUCTURAL` + `DISABLE_NODE_PARSER` | lolos |

## Tambahan untuk peneliti fork lain

Di luar daftar Tahap 1, ini juga wajib identik:

```
INDEX_MIN_CHUNK_TOKENS=8
INDEX_STRUCTURAL_METADATA=true
INDEX_TABLES_AS_OWN_CHUNKS=true
```

**`INDEX_TABLES_AS_OWN_CHUNKS` menggeser batas chunk teks, bukan sekadar menambah
chunk tabel.** Prosa yang tadinya satu chunk bersama tabel kecil kini terbelah
menjadi chunk sebelum dan sesudah tabel — sehingga `chunk_index`, `chunk_id`, dan
`text_sha` seluruh dokumen berubah, bukan hanya di sekitar tabel. Fork yang memakai
nilai berbeda menghasilkan pembagian chunk yang tidak sebanding.

**`data/document_registry.json` harus sama persis di kedua fork.** `document_id`
yang berbeda menghasilkan `chunk_id` yang berbeda untuk konten yang sama, dan
`gold_chunk_ids` tidak akan cocok lintas fork. Registry ini sebaiknya dibagikan,
bukan dibuat ulang masing-masing.

**Re-index penuh diperlukan lagi.** `chunk_index` bergeser karena penyaring
`INDEX_MIN_CHUNK_TOKENS` membuang chunk, dan `INDEX_TABLES_AS_OWN_CHUNKS` mengubah
batas. Index dari Tahap 1 dan Tahap 2 tidak sebanding.

## Verifikasi (Tahap 2)

```bash
# Kerangka registry — jalan tanpa korpus
python scripts/scaffold_document_registry.py

# Baseline — harus 5/9 bertabrakan
python scripts/probe_rechunk.py

# Final — harus 0/9 dan chunk_id unik
INDEX_EXCLUDE_METADATA_FROM_EMBED=true \
INDEX_MAX_CHUNK_TOKENS=350 \
INDEX_MIN_CHUNK_TOKENS=8 \
INDEX_DISABLE_NODE_PARSER=true \
INDEX_STRUCTURAL_METADATA=true \
INDEX_TABLES_AS_OWN_CHUNKS=true \
python scripts/probe_rechunk.py
```

## Catatan lingkungan verifikasi

Probe dijalankan di venv terisolasi dengan **Python 3.14**, sedangkan repo menarget
**Python 3.12** (`requirements.txt:3`). Konsekuensinya dua paket pinned tidak dapat
dipasang di sana: `unstructured==0.16.11` (butuh `<3.13`) dan
`llama-index-vector-stores-qdrant==0.10.1` (butuh `<3.14`).

- API koordinat `unstructured` diverifikasi dengan memeriksa **wheel resmi 0.16.11**
  (`elements.py:164`, `CoordinatesMetadata.points`/`.system`), bukan dari ingatan.
- `_check_flag_consistency` diuji dengan men-stub modul qdrant — fungsinya murni
  boolean, jadi stub tidak mengurangi validitas.
- Yang **belum** diverifikasi dengan menjalankan `partition_pdf` sungguhan: apakah
  `coordinates` benar-benar terisi pada korpus ini. Menunggu PDF.

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

---
---

# TAHAP 3 — dump chunk untuk tim evaluasi

## Kesetiaan dump — jawaban atas pertanyaan gerbang

Titik hook ada di `indexing.py` antara konstruksi `Document` dan
`_embed_and_store`. Pertanyaannya: apakah chunk di titik itu identik dengan yang
masuk Qdrant? **Jawabannya bersyarat**, dan diverifikasi empiris:

| Kondisi | Node | `text` identik? |
|---|---|---|
| `INDEX_DISABLE_NODE_PARSER=true` | 1 | **Ya** |
| Parser aktif, chunk muat anggaran | 1 | **Tidak** — trailing whitespace di-strip |
| Parser aktif, chunk kebesaran | >1 | **Tidak** — dipecah |

Bahkan pada `Document` yang tidak dipecah, `SentenceSplitter` mem-strip trailing
whitespace (terukur: 390 → 389 karakter). Karena itu dump **tidak dipaksakan**:
`run_manifest.json` mencatat `dump_faithful`, dan bila `false` modul menulis
baris ERROR. Dump dengan `dump_faithful: false` tidak boleh dipakai sebagai
dataset.

**Yang di-embed bukan `text_content`.** Vektor dihitung dari
`get_content(MetadataMode.EMBED)` = `"section: <nilai>\n\n<text_content>"` saat
`INDEX_EXCLUDE_METADATA_FROM_EMBED` aktif. Manifest merekamnya di
`embedded_text_shape`.

## Berkas baru

| Path | Isi |
|---|---|
| `backend/services/chunk_dump.py` | Pemetaan `Document` → skema lapis 2, penulisan tiga berkas, pembangunan manifest |

## Perubahan per berkas

### `backend/config.py`
| Baris | Tambahan |
|---|---|
| `250-260` | `CHUNK_DUMP_DIR` (str, default `""` = mati) |

### `backend/services/preprocessing.py`
| Baris | Perubahan |
|---|---|
| `134-160` | `ExtractionReport` — dataclass pencatat degradasi |
| `165-168` | `STRATEGY_IMAGE_THRESHOLD`, `STRATEGY_SAMPLE_PAGES` (literal diberi nama) |
| `199-202` | `OCR_TEXT_THRESHOLD_CHARS`, `OCR_RENDER_DPI` |
| `205-246` | `_extract_text_from_page_fast` → `(text, ocr_error)`, **`except` per halaman** |
| `249-273` | `_extract_fast(..., report=)` mengumpulkan halaman gagal |
| `~300` | `_extract_hi_res(..., report=)` mencatat fallback diam |
| `~880` | `extract_from_pdf` mengembalikan `report` di hasil |

### `backend/services/image_describer.py`
| Baris | Perubahan |
|---|---|
| `37-42` | `DESCRIPTION_MAX_TOKENS`, `DESCRIPTION_TEMPERATURE` diberi nama |

### `backend/services/indexing.py`
| Baris | Perubahan |
|---|---|
| `~318` | `extraction_reports` dikumpulkan per berkas |
| `~394` | Hook `chunk_dump.write_run(...)` sebelum `_embed_and_store` |

## Perbaikan: kegagalan OCR tidak lagi membuang seluruh PDF

Blok `try` di `_extract_text_from_page_fast` hanya punya `finally`, tanpa
`except`. `ImportError` paddleocr atau kegagalan `predict()` merambat naik lewat
`_extract_fast` dan `extract_from_pdf` sampai ditangkap `indexing.py` — yang lalu
**melewati seluruh PDF**. Satu halaman scan di halaman 40 membuang 99 halaman lain.

Terverifikasi dengan PDF 2 halaman dan `paddleocr` absen (kondisi aarch64 tanpa
wheel Paddle):

```
INFO   page_ocr_fallback page=2
ERROR  page_ocr_failed page=2 error=ModuleNotFoundError: No module named 'paddleocr'
       — halaman didegradasi ke teks PyMuPDF apa adanya (5 karakter)
elements dihasilkan : 2 -> [1, 2]        # sebelumnya: 0, seluruh PDF dibuang
halaman gagal OCR   : [{'page': 2, 'error': '...', 'fallback_chars': 5}]
```

Tidak diubah jadi gagal keras, sesuai permintaan — hanya dicatat.

## Contoh isi berkas

### `chunks.jsonl` (satu baris, dirapikan)

```json
{
  "chunk_id": "sop-izin-ujian-online-v2_p2_c01",
  "document_id": "sop-izin-ujian-online-v2",
  "page_number": 2,
  "chunk_type": "table",
  "text_content": "## Persyaratan Sidang Skripsi\n\n| Persyaratan | S1 | S2 |\n| --- | ...",
  "text_as_html": "<table><thead><tr><th>Persyaratan</th>...",
  "bbox": [0.1, 0.62, 0.9, 0.75],
  "text_sha": "ea11b8e7a4559b0b",
  "chunk_index": 1,
  "element_type": "Table",
  "table_format": "markdown",
  "section": "Persyaratan Sidang Skripsi",
  "file_name": "sop_izin_ujian_2.pdf",
  "file_hash": "9f2b1c4e...",
  "extraction_strategy": "hi_res"
}
```

Tujuh field pertama adalah skema lapis 2; sisanya kolom tambahan.

### `chunks_review.csv` — 19 kolom

```
chunk_id, document_id, page_number, chunk_type, element_type, chunk_index,
text_sha, n_chars, has_table_html, table_html_chars, table_format, bbox,
section, file_name, text_preview, text_content,
visual_type, verdict, catatan_reviewer
```

Tiga kolom terakhir kosong untuk reviewer. `visual_type` enum:
`flowchart` / `tabel_sebagai_gambar` / `formulir` / `figur_deskriptif`.

**`raw_html` sengaja TIDAK ada di CSV.** HTML tabel bisa ribuan karakter berisi
newline dan tanda kutip; walau modul `csv` mengutipnya dengan benar, satu sel
sebesar itu membuat spreadsheet tak terbaca. Yang dibawa hanya `has_table_html`
(ya/kosong) dan `table_html_chars`. HTML utuhnya ada di `chunks.jsonl`.

`text_content` **tidak dipotong**, dan `text_preview` (180 karakter, newline
dirapatkan) disediakan sebagai kolom terpisah untuk pembacaan cepat.

### `run_manifest.json` — ringkasan struktur

```json
{
  "run_id": "20260825T025423Z-cb1bb53d",
  "n_chunks": 3,
  "dump_faithful": true,
  "embedded_text_shape": "section: <nilai>\\n\\n<text_content>",
  "chunking":            { 12 variabel Kelompok 1 },
  "research_flags":      { 6 flag Tahap 1 & 2 },
  "hardcoded_constants": { ambang OCR 50, DPI 300, ambang strategi 1.0,
                           sampel 5 halaman, max_tokens 300, temperature 0.1,
                           image_description_prompt_sha256 },
  "models":              { embedding, vision, reranker + penanda `authoritative` },
  "environment":         { python, platform, versi 4 paket kunci },
  "provenance":          { git_commit, qdrant_collection,
                           document_registry_sha256 },
  "extraction_reports":  { per dokumen },
  "degraded_documents":  [ ... ],
  "n_degraded_documents": 2
}
```

Penanda `authoritative: false` dipasang pada `embedding` dan `reranker` saat
provider-nya `tei` — pada jalur itu bobot sebenarnya ditentukan `--model-id` di
`docker-compose.poc.yml`, bukan nilai di config.

`INDEX_TABLES_AS_OWN_CHUNKS` ikut dicatat meski tidak diminta eksplisit: ia
menggeser batas chunk teks di seluruh dokumen.

## Field skema yang bernilai null, dan alasannya

| Field | Kapan null |
|---|---|
| `chunk_id`, `document_id` | `INDEX_STRUCTURAL_METADATA` mati, atau berkas tidak ada di registry |
| `page_number` | `page == 0`, penanda "tidak diketahui" dari jalur hi_res — dinormalkan jadi `null`, bukan dilaporkan sebagai halaman nol |
| `text_as_html` | Chunk bukan tabel; **atau** tabel kecil saat `INDEX_TABLES_AS_OWN_CHUNKS` mati (melebur ke prosa sehingga tidak ada hubungan 1:1) |
| `bbox` | Jalur `fast` (tidak menyediakan koordinat sama sekali); atau `el.metadata.coordinates` kosong di jalur hi_res |
| `text_sha` | `INDEX_STRUCTURAL_METADATA` mati |
| `table_format` | Chunk bukan tabel |

Satu jebakan pemetaan yang perlu diketahui reviewer: **`chunk_type` bisa salah di
jalur `fast`.** Jalur itu menandai setiap element `"NarrativeText"`, termasuk
tabel yang tergilas jadi teks datar — chunk semacam itu terpetakan `"text"`
walau isinya tabel. Kolom `extraction_strategy` di `chunks.jsonl` memungkinkan
reviewer menyaringnya.

## Tambahan untuk peneliti fork lain

`CHUNK_DUMP_DIR` **tidak** memengaruhi isi chunk — murni observasi. Tidak perlu
sama antar fork. Tetapi `run_manifest.json` dari kedua fork **wajib dibandingkan
sebelum eksperimen dimulai**: seluruh blok `chunking`, `research_flags`,
`hardcoded_constants`, dan `models` harus identik. Yang boleh berbeda hanya
`run_id`, `created_at`, `qdrant_collection`, dan `git_commit`.

## Verifikasi (Tahap 3)

Korpus belum ada, jadi diuji dengan element sintetis yang melewati
`_chunk_elements` yang sama:

```bash
CHUNK_DUMP_DIR=/tmp/dump \
INDEX_EXCLUDE_METADATA_FROM_EMBED=true INDEX_MAX_CHUNK_TOKENS=350 \
INDEX_MIN_CHUNK_TOKENS=8 INDEX_DISABLE_NODE_PARSER=true \
INDEX_STRUCTURAL_METADATA=true INDEX_TABLES_AS_OWN_CHUNKS=true \
python -m scripts.index_documents
```

Hasil uji sintetis: 3 chunk, `dump_faithful: true`, 2 dokumen tercatat
terdegradasi (satu fallback hi_res→fast, satu halaman gagal OCR).

**Belum diverifikasi tanpa korpus:** apakah `partition_pdf` benar-benar mengisi
`coordinates` sehingga `bbox` tidak selalu null, dan berapa proporsi dokumen
nyata yang jatuh ke fallback.
