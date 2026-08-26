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
verifikasi); **chunk tabel tidak dibatasi apa pun.** Jadi:

- Distribusi panjang chunk menjadi bimodal. Statistik "rata-rata panjang chunk"
  akan menyesatkan bila dilaporkan tanpa dipisah per `element_type`.
- Chunk tabel mengonsumsi porsi konteks LLM yang jauh lebih besar per chunk.
  Dengan `RERANKER_TOP_N=6`, enam chunk tabel ≈ 4.900 token, mendekati
  `--max-model-len 8192` di `docker-compose.poc.yml:66`.
- Chunk tabel hanya aman selama `INDEX_DISABLE_NODE_PARSER=true`. Menyalakan 1B
  tanpa 1C menyisakan tabel besar tetap dipecah — inilah alasan 1C wajib, bukan
  sekadar jaring pengaman.

> **KOREKSI (lihat "Perbaikan wajib" di akhir berkas).** Versi sebelumnya dokumen
> ini menyatakan *"`PDF_TABLE_MAX_CHARS` kini menentukan batas atas ukuran chunk"*.
> **Itu salah.** `PDF_TABLE_MAX_CHARS` adalah **AMBANG**, bukan **BATAS ATAS**:
> `preprocessing.py` memakainya sebagai `len(text) > PDF_TABLE_MAX_CHARS` untuk
> memutuskan apakah tabel jadi chunk sendiri atau digabung ke buffer. Tabel yang
> melewati ambang menjadi chunk sendiri **berapa pun besarnya** — tidak ada yang
> memangkasnya. Di korpus nyata 214 dokumen, chunk tabel terbesar mencapai **4.091
> token**, lebih dari lima kali "811 token" yang dikira jadi plafon.
>
> Nilainya **tetap wajib masuk `run_manifest.json`** — bukan karena membatasi
> ukuran chunk, tapi karena menentukan tabel mana yang jadi chunk sendiri dan mana
> yang menyatu ke buffer teks, dan itu menggeser batas chunk di seluruh dokumen.

Deskripsi gambar juga tidak dipecah, dengan alasan sejalan: satu deskripsi = satu
gambar, dan memecahnya merusak relasi `narrative_summary` ↔ `image_id`. Praktisnya
tidak pernah terpicu karena `image_describer.DESCRIPTION_MAX_TOKENS` membatasinya di 300.

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
| **`PDF_TABLE_MAX_CHARS`** | **Ambang** "tabel jadi chunk sendiri" — menggeser batas chunk di seluruh dokumen. BUKAN batas atas ukuran chunk; lihat koreksi di atas |
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

---
---

# TAHAP 4 — simpan gambar dan images.jsonl

## Keputusan sadar: `section` ikut dikecualikan dari embedding

Pengukuran pada dokumen sintetis berstruktur realistis — berapa persen chunk yang
nilai `section`-nya sudah muncul di dalam `text_content`:

| element_type | chunk | section ada di text | share token | ganda? |
|---|---:|---:|---:|---|
| ImageDescription | 4 | **100%** | 46% | ya, 2× |
| Table | 3 | **100%** | 43% | ya, 2× |
| NarrativeText | 19 | 21% | 7% | sebagian |

100% pada tabel dan gambar bersifat **struktural**, bukan kebetulan: prefix
`## {section}` dipasang tanpa syarat pada keduanya. Chunk teks hanya membawanya
pada chunk pertama tiap section (lewat `# {title}`).

Sesudah `section` dikecualikan:

| element_type | share sebelum | share sesudah |
|---|---:|---:|
| ImageDescription | 46% (2×) | **27% (1×)** |
| Table | 43% (2×) | **25% (1×)** |
| NarrativeText | 7% | **1%** |

**Alasan utama bukan angka di atas.** Di bawah konfigurasi lama, `text_content`
di `chunks.jsonl` **bukan** string yang divektorkan — yang divektorkan adalah
`"section: <nilai>\n\n<text>"`. Dataset lapis 2 yang dipublikasikan karenanya
tidak cukup untuk mereproduksi index, dan itu bertentangan langsung dengan fungsi
lapis 2 menurut deck: *"menghindarkan pihak lain dari harus mereplikasi pipeline
ekstraksi"*. Sekarang `embedded_text_shape` = `<text_content>`.

Alasan kedua: bobot ganda jatuh persis pada `Table` dan `ImageDescription` — dua
strata yang justru diteliti — sehingga artefaknya tidak dapat dipisahkan dari
efek strategi indexing saat analisis.

**BIAYA YANG DITERIMA.** Chunk teks kedua dan seterusnya dalam satu section
kehilangan sinyal `section` di ruang vektor. **Peran `NEIGHBOR_EXPANSION` menjadi
lebih besar daripada konfigurasi lama** — `NEIGHBOR_EXPANSION_ENABLED`,
`NEIGHBOR_EXPANSION_RADIUS`, dan `MAX_EXPANDED_CHUNKS` naik kepentingannya dan
wajib identik antar fork.

Konstanta `NON_SEMANTIC_METADATA_KEYS` diganti nama menjadi
`EMBED_EXCLUDED_METADATA_KEYS`: nama lama menjadi kontradiktif karena `section`
justru semantik dan dikecualikan atas alasan berbeda. Alias lama dipertahankan.

## Env var baru

| Nama | Default | Efek |
|---|---|---|
| `INDEX_PERSIST_IMAGES` | `false` | Tulis gambar hasil ekstraksi ke `IMAGES_DIR` |

## Penulisan gambar

`IMAGES_DIR` (`config.py:12`) sudah ada sejak awal dan diimpor `preprocessing`
tapi tidak pernah dipakai. Sekarang dipakai.

**Urutan yang diwajibkan:** `_persist_image_elements` berjalan **sebelum**
`_describe_image_elements`. Gambar yang nanti dinilai `DEKORATIF` atau gagal
dideskripsikan **tetap tersimpan** — strategi pembanding memvektorkan gambar
aslinya, dan riset ini harus bisa menilai ulang tanpa mengulang ekstraksi PDF.

**Tidak ada penyaringan ukuran.** Ukuran, dimensi, dan MIME dicatat supaya
penyaringan dilakukan di hilir. `is_likely_informative` tetap fungsi mati tanpa
pemanggil, dan `PDF_MIN_IMAGE_SIZE_KB` tetap tidak berefek — sengaja tidak
dihidupkan.

`sha256` dihitung atas bytes **asli, sebelum resize apa pun**: resize adalah
re-encode lossy yang keluarannya bergantung versi Pillow, sehingga hash setelah
resize tidak stabil antar lingkungan.

### Tiga bug yang diperbaiki sekalian

| Bug | Perbaikan |
|---|---|
| MIME di-hardcode `image/png` di jalur vLLM | `_detect_mime` membaca format sebenarnya; ekstensi berkas mengikutinya |
| Cabang JPEG di `_resize_image_if_needed` tidak terjangkau (`img.resize()` tidak membawa `.format`) | Format dibaca **sebelum** resize; ditambah konversi RGB karena JPEG tidak mendukung alpha |
| Respons kosong tidak menulis cache | Kini ditulis, sehingga gambar itu tidak dipanggilkan model berulang kali dalam satu proses |

## Contoh hasil

### Struktur direktori

```
data/images/sop-izin-ujian-online-v2/p2_img00.png    2787 bytes   800x600
data/images/sop-izin-ujian-online-v2/p2_img01.jpg     757 bytes   120x60   <- logo kecil, TIDAK disaring
data/images/sop-izin-ujian-online-v2/pNA_img00.png   1388 bytes   400x400  <- halaman tidak diketahui
```

Ekstensi mengikuti format asli — bukti perbaikan bug MIME.

### `images.jsonl`

```json
{"image_id": "sop-izin-ujian-online-v2_p2_img00",
 "document_id": "sop-izin-ujian-online-v2",
 "page_number": 2,
 "file_path": "images/sop-izin-ujian-online-v2/p2_img00.png",
 "visual_type": null,
 "structured_summary": null,
 "narrative_summary": "Diagram alir pengajuan izin dengan simpul keputusan TOEFL…",
 "sha256": "f34bd453947b77be…", "mime_type": "image/png",
 "size_bytes": 2787, "width": 800, "height": 600,
 "source_file": "sop_izin_ujian_2.pdf"}
```

Tujuh field pertama adalah skema lapis 2. Gambar yang tidak dideskripsikan tetap
punya baris, dengan `narrative_summary: null`.

### `images_review.csv` — 15 kolom

```
image_id, document_id, page_number, file_path, mime_type, size_kb, width, height,
sha256, has_narrative, narrative_preview, source_file,
visual_type, verdict, catatan_reviewer
```

`file_path` **relatif terhadap `images_root`** yang dicatat di
`run_manifest.json`. `visual_type` diisi dengan salah satu dari empat enum:
`flowchart` / `tabel_sebagai_gambar` / `formulir` / `figur_deskriptif`.

`has_narrative` kosong menandai gambar yang **tidak** dideskripsikan — bisa
karena `DEKORATIF`, vision mati, atau panggilan gagal. Gambarnya tetap ada dan
tetap layak dianotasi.

### Tautan chunk → gambar

```
sop-izin-ujian-online-v2_p2_c01 -> sop-izin-ujian-online-v2_p2_img00
                                -> images/sop-izin-ujian-online-v2/p2_img00.png
```

`image_id` kini ada di `chunks.jsonl` dan `chunks_review.csv`, sehingga
`chunks.jsonl` dapat di-join ke `images.jsonl` untuk stratifikasi lapis 1.

## Implikasi jalur `fast` — tidak dikerjakan

`_extract_fast` **tidak mengekstrak gambar sama sekali**: ia hanya memanggil
`page.get_text()` per halaman dan menghasilkan satu element `NarrativeText`.
Konsekuensinya, dokumen yang diproses lewat jalur `fast` **tidak akan punya satu
baris pun di `images.jsonl`**, bahkan bila halamannya penuh diagram.

Ini bertemu dengan dua hal yang sudah tercatat sebelumnya:
- `detect_strategy` merutekan PDF hasil scan ke `fast` (ambang `> 1.0` strict,
  halaman scan = tepat 1 gambar/halaman).
- Fallback diam `hi_res` → `fast` membuat dokumen kehilangan gambar tanpa
  peringatan — kini tercatat di `run_manifest.json`.

**Keputusan riset sudah memaksa `hi_res`**, jadi ini tidak dikerjakan. Tetapi
`PDF_EXTRACTION_STRATEGY=hi_res` menjadi wajib, bukan opsional: dengan `auto`,
sebagian dokumen akan senyap kehilangan seluruh gambarnya. Periksa
`degraded_documents` di manifest sebelum memakai hasil run mana pun.

## Penegakan: korpus gambar tidak boleh diam-diam tidak lengkap

Dua lapis, karena menolak `auto` saja tidak cukup.

**Lapis 1 — `_check_image_strategy`, sebelum run dimulai.** Menolak
`PDF_EXTRACTION_STRATEGY` selain `hi_res` saat `INDEX_PERSIST_IMAGES` aktif.

`fast` ikut ditolak: mode kegagalannya identik dan lebih parah (seluruh korpus,
bukan sebagian), jadi membiarkannya lolos sementara `auto` ditolak akan tidak
koheren.

**Lapis 2 — `_check_no_silent_fallback`, di akhir run.** Fallback `hi_res` →
`fast` terjadi saat runtime, setelah pemeriksaan konfigurasi lewat. Run
digagalkan dengan daftar dokumennya, bukan sekadar dicatat di
`degraded_documents` — catatan yang harus diperiksa manual pasti suatu saat
tidak diperiksa.

**Urutan gerbang disengaja:**

```
konstruksi Document -> tulis dump -> GERBANG -> _embed_and_store
                        (bukti          (gagal)    (tidak tercapai)
                       tersimpan)
```

Dump ditulis lebih dulu supaya `run_manifest.json` dan `images.jsonl` ada untuk
diperiksa, dan gerbang berjalan sebelum embedding supaya tidak ada yang masuk
Qdrant.

**Tabel kebenaran, terverifikasi:**

| Konfigurasi | Lapis 1 | Lapis 2 |
|---|---|---|
| `PERSIST_IMAGES=false`, strategy apa pun | lolos | lolos |
| `PERSIST_IMAGES=true`, `strategy=auto` | **ditolak** | — |
| `PERSIST_IMAGES=true`, `strategy=fast` | **ditolak** | — |
| `PERSIST_IMAGES=true`, `strategy=hi_res`, tanpa fallback | lolos | lolos |
| `PERSIST_IMAGES=true`, `strategy=hi_res`, ada fallback | lolos | **ditolak** |
| ditambah `ALLOW_INCOMPLETE_IMAGE_CORPUS=true` | lolos | lolos |

Escape hatch `ALLOW_INCOMPLETE_IMAGE_CORPUS` default `false` dan harus disetel
sadar. Nilainya tercatat di `run_manifest.json`, sehingga keputusan menerima
korpus tidak lengkap terbawa bersama datanya.

## Yang TIDAK dikerjakan di tahap ini

Kontrak `describe_image` bervarian untuk `structured_summary` — butuh
perancangan prompt tersendiri. Sampai itu ada, `structured_summary` tetap `null`
di seluruh baris `images.jsonl`, dan hanya varian naratif (b) yang dapat
direplikasi dari dataset.

## Tambahan untuk peneliti fork lain

```
INDEX_PERSIST_IMAGES=true
PDF_EXTRACTION_STRATEGY=hi_res
```

Ditambah konsekuensi keputusan `section`:

```
NEIGHBOR_EXPANSION_ENABLED=true
NEIGHBOR_EXPANSION_RADIUS=<nilai yang sama>
MAX_EXPANDED_CHUNKS=<nilai yang sama>
```

Ketiganya naik kepentingannya karena chunk teks tengah-section kini tanpa sinyal
`section` di ruang vektor.

**`document_registry.json` yang sama menjadi lebih kritis lagi**: `document_id`
kini menentukan bukan hanya `chunk_id` tetapi juga `image_id` dan nama direktori
gambar di disk. Fork dengan registry berbeda menghasilkan struktur direktori
berbeda, dan `gold_image_ids` tidak akan cocok.

## Verifikasi (Tahap 4)

Diuji dengan gambar sintetis nyata (PNG 800×600, JPEG 120×60, satu berhalaman
tidak diketahui) yang melewati `_persist_image_elements` dan `_chunk_elements`
yang sama:

```bash
CHUNK_DUMP_DIR=/tmp/dump INDEX_PERSIST_IMAGES=true \
INDEX_EXCLUDE_METADATA_FROM_EMBED=true INDEX_MAX_CHUNK_TOKENS=350 \
INDEX_MIN_CHUNK_TOKENS=8 INDEX_DISABLE_NODE_PARSER=true \
INDEX_STRUCTURAL_METADATA=true INDEX_TABLES_AS_OWN_CHUNKS=true \
PDF_EXTRACTION_STRATEGY=hi_res \
python -m scripts.index_documents
```

Hasil: 3 gambar tersimpan (termasuk logo 757 byte yang tidak disaring), ekstensi
mengikuti format asli, 1 dari 3 punya `narrative_summary`, tautan chunk→gambar
dapat di-join. Regresi probe tetap 5/9 (default mati) dan 0/9 (flag riset penuh).

**Belum diverifikasi tanpa korpus:** apakah `partition_pdf` benar-benar
menghasilkan element `Image` dengan `image_base64` terisi pada korpus ini, dan
berapa proporsi gambar nyata yang dinilai `DEKORATIF`.

---
---

# DAFTAR FINAL FLAG RISET — BEKU

> Daftar ini yang dikirim ke peneliti fork lain. **Tidak diubah lagi setelah
> eksperimen dimulai.** Mengubah salah satunya menuntut re-index penuh dan
> membatalkan seluruh anotasi gold yang menunjuk `chunk_id` atau `image_id`.

## Salin apa adanya ke `.env`

```bash
# ── Gerbang konfigurasi ──
# Memvalidasi seluruh flag di bawah terhadap daftar ini sebelum run dimulai,
# dan mencetak nilai efektifnya ke stdout.
RESEARCH_MODE=true

# ── Tahap 1: identitas chunk & pemecahan ganda ──
INDEX_EXCLUDE_METADATA_FROM_EMBED=true
INDEX_MAX_CHUNK_TOKENS=350
INDEX_MIN_CHUNK_TOKENS=8
INDEX_DISABLE_NODE_PARSER=true

# ── Tahap 2: identitas berlapis tiga & metadata struktural ──
INDEX_STRUCTURAL_METADATA=true
INDEX_TABLES_AS_OWN_CHUNKS=true

# ── Tahap 4: gambar ──
INDEX_PERSIST_IMAGES=true
PDF_EXTRACTION_STRATEGY=hi_res

# ── Cache deskripsi gambar: ARTEFAK, bagikan berkasnya, jangan dihapus ──
VISION_CACHE_ENABLED=true
VISION_CACHE_PATH=~/rag_mm_b_shared/vision_cache.db

# ── Tahap 4B: deskripsi gambar — WAJIB IDENTIK antara varian (b) dan (c) ──
LLM_SUPPORTS_VISION=true
VISION_MODEL=qwen3-vl:8b
VISION_TEMPERATURE=0
VISION_MAX_TOKENS=300
RESEARCH_VISION_SEED=1337
VISION_NUM_CTX=8192

# ── Nilai lama yang naik kepentingannya, jangan diubah ──
CHUNK_SIZE=512
CHUNK_OVERLAP=128
PDF_TABLE_MAX_CHARS=2000
NEIGHBOR_EXPANSION_ENABLED=true
NEIGHBOR_EXPANSION_RADIUS=2
MAX_EXPANDED_CHUNKS=30

# ── JANGAN disetel kecuali sadar menerima korpus tidak lengkap ──
# ALLOW_INCOMPLETE_IMAGE_CORPUS=false

# ── Observasi saja, boleh berbeda antar fork ──
# CHUNK_DUMP_DIR=data/dumps
```

## Alasan tiap nilai

| Flag | Nilai | Kenapa nilai itu | Kalau berbeda antar fork |
|---|---|---|---|
| `INDEX_EXCLUDE_METADATA_FROM_EMBED` | `true` | Nol metadata divektorkan; `text_content` = string yang di-embed, sehingga dataset lapis 2 cukup untuk mereproduksi index | Teks yang divektorkan berbeda → seluruh skor tidak sebanding |
| `INDEX_MAX_CHUNK_TOKENS` | `350` | `512 − 132` (metadata terburuk terencana) `= 380`; 350 memberi sisa aman 30 token | Pembagian chunk berbeda |
| `INDEX_MIN_CHUNK_TOKENS` | `8` | Kalimat asli terpendek yang terukur = 8 token; judul berkisar 6–18 | Jumlah chunk berbeda → `chunk_index` bergeser |
| `INDEX_DISABLE_NODE_PARSER` | `true` | `1 Document = 1 node`; tanpa ini `chunk_id` bukan kunci unik, dan dump tidak setia | `chunk_id` bertabrakan |
| `INDEX_STRUCTURAL_METADATA` | `true` | Menghidupkan `chunk_id`, `document_id`, `text_sha`, `raw_html`, `bbox`, `image_id` | Tidak ada identitas untuk `gold_chunk_ids` |
| `INDEX_TABLES_AS_OWN_CHUNKS` | `true` | Tiap tabel 1:1 dengan satu chunk beserta `raw_html`-nya | **Menggeser batas chunk teks di seluruh dokumen**, bukan hanya di sekitar tabel |
| `INDEX_PERSIST_IMAGES` | `true` | `images.jsonl` dan `gold_image_ids` mustahil tanpanya | Tidak ada gambar untuk dianotasi |
| `PDF_EXTRACTION_STRATEGY` | `hi_res` | Jalur `fast` tidak mengekstrak gambar sama sekali; **ditegakkan di kode** | Ditolak sebelum run dimulai |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `128` | Ambang flush dan panjang overlap di `_chunk_elements` | Pembagian chunk berbeda |
| `PDF_TABLE_MAX_CHARS` | `2000` | **Ambang** tabel-jadi-chunk-sendiri. Tabel di atas ambang menjadi chunk sendiri **tanpa batas ukuran** — terukur sampai 4.091 token | Komposisi dan ukuran chunk tabel berbeda |
| `NEIGHBOR_EXPANSION_*`, `MAX_EXPANDED_CHUNKS` | `true` / `2` / `30` | Naik kepentingannya sejak `section` dikecualikan: chunk teks tengah-section kini tanpa sinyal `section` di ruang vektor | Konteks yang sampai ke LLM berbeda |

## Yang WAJIB sama tapi bukan env var

| Item | Kenapa |
|---|---|
| **`document_registry.json`** | `document_id` menentukan `chunk_id`, `image_id`, **dan** nama direktori gambar di disk. `document_id` kini di-generate otomatis dari nama berkas, tapi konsistensi tetap datang dari **berkas yang sama** — bukan dari dua eksekusi scaffolder. Satu berkas di `~/rag_mm_b_shared/`, tidak dikirim-kirim. **Jangan generate ulang dari nol setelah anotasi dimulai** |
| **Cache deskripsi gambar** (`vision_cache.db`) | Menyusul `document_registry.json` sebagai artefak yang **harus dibagikan berkasnya**. Isinya adalah sumber `narrative_summary`, yang merupakan isi chunk. Cache berbeda → deskripsi berbeda → membandingkan model, bukan strategi. **Tidak boleh dihapus di tengah eksperimen** — lihat konsekuensinya di bawah |
| `RERANKER_PROVIDER` | `SentenceTransformerRerank` membaca `MetadataMode.EMBED`, jalur TEI tidak — keduanya melihat teks berbeda |
| Versi paket | `llama-index-core`, `tiktoken`, `unstructured`, `pymupdf` — perilaku splitter dan ekstraksi bergantung padanya |
| Prompt deskripsi gambar | `image_description_prompt_sha256` di manifest harus sama |

## Cara memverifikasi kesesuaian antar fork

Bandingkan `run_manifest.json` dari kedua fork. **Harus identik** pada blok
`chunking`, `research_flags`, `hardcoded_constants`, `models`, dan
`provenance.document_registry_sha256`.

**Boleh berbeda:** `run_id`, `created_at`, `n_chunks`, `n_images`,
`provenance.git_commit`, `provenance.qdrant_collection` (harus berbeda),
`environment.platform`.

**Harus `false`:** `research_flags.ALLOW_INCOMPLETE_IMAGE_CORPUS`.
**Harus `true`:** `dump_faithful`.
**Harus kosong:** `degraded_documents`.

## Prasyarat sebelum indexing pertama

1. Jalankan ulang `scripts/probe_rechunk.py` di venv target **Python 3.12** dan
   bandingkan angkanya (lihat peringatan di awal dokumen ini).
2. Isi `data/document_registry.json` — `python scripts/scaffold_document_registry.py`
   lalu isi `document_id` tiap berkas.
3. Sepakati berkas registry itu dengan peneliti fork lain sebelum salah satu
   mulai meng-index.

---
---

# TAHAP 4B — determinisme dan provenance deskripsi gambar

## Kenapa ini bukan soal metadata

Deskripsi gambar adalah **isi chunk yang diindeks**, bukan metadata.
Nondeterminisme di sini mengubah teks yang divektorkan, dan lebih buruk mengubah
**jumlah chunk**: verdict `DEKORATIF` membuat element dibuang di
`_describe_image_elements`, sehingga seluruh penomoran sesudahnya bergeser —
`chunk_index`, `chunk_id`, dan `text_sha` ikut berubah. Mekanismenya tercatat di
`INSPECTION_REPORT_2.md` G10 konsekuensi 2.

## Gerbang baru: `RESEARCH_MODE`

`load_dotenv()` mencari `.env` dari lokasi `backend/config.py` **ke atas**, bukan
dari direktori kerja — terverifikasi dengan tiga skenario. `.env` yang ditaruh di
direktori kerja lain **diabaikan tanpa peringatan**, seluruh flag riset jatuh ke
default, dan indexing tetap berjalan menghasilkan chunk yang salah.

`RESEARCH_MODE=true` memvalidasi seluruh flag terhadap `RESEARCH_EXPECTED_FLAGS`
di `config.py` — satu sumber kebenaran yang mencerminkan daftar beku di atas —
dan **mencetak nilai efektif tiap flag ke stdout** saat start. `print()`, bukan
`logger`: harus terlihat apa pun konfigurasi logging, dan harus muncul sebelum
indexing berjalan berjam-jam.

## Gerbang baru: `_check_vision_reachable`

`LLM_SUPPORTS_VISION` default `false`. Dikombinasikan dengan
`PDF_DESCRIBE_IMAGES=auto` (juga default), `should_describe` bernilai false dan
**setiap** element Image dibuang tanpa satu pun error — `narrative_summary` null
di seluruh `images.jsonl`, dan varian (b) tidak dapat direplikasi.

Ditolak saat `INDEX_PERSIST_IMAGES` aktif. Escape hatch sama:
`ALLOW_INCOMPLETE_IMAGE_CORPUS`.

## Env var baru

| Nama | Default | Efek |
|---|---|---|
| `RESEARCH_MODE` | `false` | Validasi flag terhadap daftar beku + cetak nilai efektif |
| `VISION_MODEL` | jatuh ke `LLM_MODEL` | Model deskripsi gambar, terpisah dari model generation |
| `VISION_TEMPERATURE` | `0` | Dikirim ke **kedua** provider |
| `VISION_MAX_TOKENS` | `300` | Dikirim ke kedua provider |
| `RESEARCH_VISION_SEED` | `1337` | `options.seed` (Ollama) / `seed` (vLLM). Negatif = tidak dikirim |
| `VISION_NUM_CTX` | `8192` | `options.num_ctx` Ollama |

## Determinisme — sejauh mana dijamin

**Yang dijamin suhu 0 + seed tetap:** sampling greedy menghilangkan variasi dari
pengambilan acak token. Dengan model, bobot, prompt, dan gambar yang sama pada
mesin dan build runtime yang sama, keluarannya konsisten antar-panggilan.

**Yang TETAP bisa bervariasi:**

- **Non-determinisme numerik GPU.** Reduksi floating-point pada kernel batch
  tidak asosiatif; urutan penjumlahan bisa berbeda antar-panggilan tergantung
  ukuran batch dan penjadwalan. Selisih sekecil itu jarang mengubah token
  terpilih, tapi **bisa** saat dua kandidat teratas nyaris seri — dan pada
  keputusan `DEKORATIF` vs bukan, satu token yang berubah mengubah jumlah chunk.
- **Perubahan bobot di balik tag yang sama.** `ollama pull qwen3-vl:8b` dapat
  mengganti bobot tanpa mengubah nama tag. Karena itu `vision_model_digest`
  dicatat per gambar.
- **Versi runtime.** Upgrade Ollama dapat mengubah kernel, kuantisasi, atau
  penanganan `num_ctx`.
- **Ukuran batch dan panjang konteks.** Isi `num_ctx` yang berbeda mengubah jalur
  komputasi walau prompt sama.

Ringkasnya: suhu 0 menghilangkan sumber variasi **terbesar** dan **satu-satunya
yang dapat dikendalikan dari sisi klien**. Ia tidak menjadikan pipeline
deterministik bit-per-bit. Karena itu `text_sha` dan `sha256` gambar tetap
diperlukan sebagai jaring pengaman, dan cache persisten (di bawah) menjadi satu-
satunya cara menghilangkan variasi lintas-run sepenuhnya.

## Paritas jalur vLLM dan Ollama

| Aspek | Sebelum | Sesudah |
|---|---|---|
| `temperature` | vLLM 0.1; Ollama **tidak dikirim** | Keduanya `VISION_TEMPERATURE` |
| batas token | vLLM 300; Ollama **tidak dikirim** | Keduanya `VISION_MAX_TOKENS` |
| `seed` | tidak ada | Keduanya, bila ≥ 0 |
| `num_ctx` | tidak ada → Ollama potong ke 4096 senyap | `VISION_NUM_CTX` eksplisit |
| respons tak terduga | vLLM melempar (tertangkap, tercatat); Ollama `.get()` → **None senyap** | Keduanya mencatat `ERROR image_describer_bad_response` |
| respons kosong | senyap | `WARNING image_describer_empty` |
| provider lain | `logger.debug` lalu buang semua | `ValueError` keras |
| MIME | di-hardcode `image/png` | `PIL_FORMAT_MAP` — satu peta untuk ekstensi disk dan MIME payload |

Terverifikasi dengan stub untuk sembilan bentuk respons: **nol kegagalan tanpa
baris log**.

## Provenance per gambar

Tiap baris `images.jsonl` kini membawa:

```json
{"vision_provider": "ollama", "vision_model": "qwen3-vl:8b",
 "vision_model_digest": "sha256:901cae7321...",
 "vision_temperature": 0.0, "vision_max_tokens": 300,
 "vision_seed": 1337, "vision_num_ctx": 8192,
 "prompt_sha256": "a2502e28e1739f63..."}
```

`vision_model_digest` diambil dari `GET /api/tags` — **digest sha256 penuh**,
bukan ID pendek yang tampil di `ollama list`. Tag bergerak setelah `ollama pull`;
digest tidak. Bila endpoint tidak terjangkau, digest `null` dan
`models.vision.digest_resolved` di manifest bernilai `false` — bukan diam.

## Cache persisten — DIKERJAKAN

SQLite di lokasi yang dapat dibagikan. Kunci:

```
sha256( sha256(bytes gambar ASLI) | variant | prompt_sha256 | vision_model_digest )
```

Hash gambar dihitung **sebelum resize**, jadi tidak bergantung versi Pillow.

**Varian (b) dan (c) berbagi cache.** `narrative_summary` yang sama wajib dipakai
keduanya supaya perbedaan skor berasal dari strategi indexing, bukan dari dua
panggilan model yang kebetulan berbeda. Komponen `variant` membedakan **jenis
ringkasan**, bukan run.

### Yang di-cache dan yang tidak

| Verdict | Di-cache? | Alasan |
|---|---|---|
| `described` | ya | Keputusan model |
| `decorative` | ya | Keputusan model — `DEKORATIF` |
| `unclear` | ya | Keputusan model — `TIDAK JELAS` |
| timeout / HTTP error | **tidak** | Transient |
| respons kosong | **tidak** | Transient |
| bentuk respons tak terduga | **tidak** | Transient |

`image_describer` sebelumnya menulis string kosong untuk `DEKORATIF` sementara
respons kosong tidak menulis sama sekali — dua hal berbeda dengan representasi
yang sama. `_classify` memisahkannya eksplisit; cache persisten **tidak
mewarisi** ambiguitas itu.

Kalau kegagalan transient ikut masuk, satu timeout menjadi permanen: gambar itu
hilang dari seluruh eksperimen dan hanya bisa dipulihkan dengan menghapus cache
— yang sendirinya membatalkan anotasi gold.

Terverifikasi dengan stub enam kasus:

| | run 1 | run 2 |
|---|---|---|
| 3 verdict model | disimpan | **HIT**, model tidak dipanggil |
| 3 kegagalan transient | ditolak | **MISS**, model dicoba ulang |

### Provenance di manifest

```json
"vision_cache": {
  "enabled": true, "path": "~/rag_mm_b_shared/vision_cache.db",
  "readonly": false, "entries_total": 1284,
  "content_sha256": "eec704e85deca0ac...",
  "verdicts": {"described": 1100, "decorative": 150, "unclear": 34},
  "hit": 1200, "miss": 84, "stored": 84, "skipped_failure": 3
}
```

`content_sha256` meng-hash **baris terurut**, bukan byte berkas — bebas dari
halaman bebas SQLite dan jurnal WAL, dan tidak berubah saat proses lain menulis
entri yang tidak relevan. Dua cache dengan hash sama memuat deskripsi yang sama
persis.

### Gerbang

`RESEARCH_MODE=true` menolak run bila cache memuat konfigurasi vision yang
berbeda dari yang sedang dipakai. Kunci sudah memuat digest dan hash prompt, jadi
entri asing tidak akan pernah dikembalikan sebagai hit — yang ditolak masalah
lain: cache dengan lebih dari satu konfigurasi berarti sebagian chunk berasal
dari model A dan sebagian dari model B.

| Kondisi | Hasil |
|---|---|
| `RESEARCH_MODE=false` | lolos |
| `VISION_CACHE_ENABLED=false` | lolos |
| Cache satu konfigurasi, cocok | lolos |
| Cache memuat konfigurasi lain | **ditolak** |

### Lokasi dan akses bersama

Default `~/rag_mm_b_shared/vision_cache.db` — di **luar clone mana pun** supaya
dapat dibagikan. Tiga opsi berbagi, dengan konsekuensi izinnya:

| Opsi | Cara | Konsekuensi |
|---|---|---|
| **A. Satu pemilik menulis, pihak kedua read-only** | Pemilik menjalankan indexing gambar lebih dulu; pihak kedua menunjuk `VISION_CACHE_PATH` ke berkas yang sama, yang bagi dia tidak dapat ditulis | **Paling sederhana tanpa sudo.** Sudah didukung: modul mendeteksi berkas tak-dapat-ditulis, membuka read-only, dan mencatat WARNING. Hit tetap dipakai; gambar yang belum ada di cache dipanggilkan model tiap run oleh pihak kedua |
| **B. Direktori group-writable** | `chmod 2775` pada direktori bersama, kedua user di grup yang sama, `umask 002` | Butuh grup bersama yang sudah ada (`users` tersedia) dan disiplin `umask`. SQLite WAL membuat berkas `-wal` dan `-shm` yang juga perlu izin tulis. Tanpa sudo, membuat grup baru tidak mungkin |
| **C. Salin berkas** | Pemilik menjalankan indexing, lalu `cp` cache ke home pihak kedua | Tidak butuh izin bersama sama sekali. Tapi kedua salinan bisa melenceng; `content_sha256` di manifest yang membuktikan keduanya sama |

**Rekomendasi: A**, dengan **C sebagai cadangan** bila izin read pun bermasalah.
B menuntut koordinasi `umask` yang mudah terlewat, dan kegagalannya senyap —
berkas tertulis dengan izin salah baru ketahuan saat pihak kedua gagal membaca.

### Cara memeriksa isinya

```bash
python scripts/inspect_vision_cache.py                    # ringkasan
python scripts/inspect_vision_cache.py --sample 5         # contoh deskripsi
python scripts/inspect_vision_cache.py --image-sha <sha>  # telusuri satu gambar
```

Read-only, tidak memanggil model. Menampilkan jumlah entri, sebaran verdict,
kombinasi (model, digest, prompt) beserta rentang waktu penulisannya, dan hash
isi untuk dicocokkan dengan manifest.

## Tambahan untuk peneliti fork lain

Seluruh blok VISION di `.env.research` **wajib identik**:

```
LLM_SUPPORTS_VISION=true
VISION_MODEL=qwen3-vl:8b
VISION_TEMPERATURE=0
VISION_MAX_TOKENS=300
RESEARCH_VISION_SEED=1337
VISION_NUM_CTX=8192
```

Plus `RESEARCH_MODE=true` supaya penyimpangan tertangkap sebelum indexing, bukan
sesudah.

Bandingkan `models.vision` di `run_manifest.json` kedua fork — termasuk
`vision_model_digest`. Digest yang berbeda berarti bobot berbeda, walau nama
tag-nya sama.

---
---

# CATATAN METODOLOGIS UNTUK PAPER

Dua hal yang perlu dinyatakan eksplisit di bagian metode, karena keduanya
membatasi klaim reproduksibilitas yang dapat dibuat.

## 1. Batas determinisme deskripsi gambar

Deskripsi gambar dihasilkan LLM multimodal dan menjadi **isi chunk yang
diindeks** — bukan metadata. Nondeterminisme di sana mengubah teks yang
divektorkan, dan lebih jauh mengubah **jumlah chunk**: verdict `DEKORATIF`
membuat element dibuang, sehingga `chunk_index`, `chunk_id`, dan `text_sha`
seluruh dokumen sesudahnya bergeser.

**Yang dikendalikan.** `temperature=0` dengan `seed` tetap menghilangkan variasi
dari **sampling** — pengambilan token menjadi greedy, bukan acak. Dengan model,
bobot, prompt, gambar, mesin, dan build runtime yang sama, keluarannya konsisten
antar-panggilan.

**Yang tetap bisa bervariasi:**

- **Non-determinisme numerik GPU.** Reduksi floating-point pada kernel batch
  tidak asosiatif; urutan penjumlahan dapat berbeda antar-panggilan tergantung
  ukuran batch dan penjadwalan. Selisih sekecil itu jarang mengubah token
  terpilih — tapi **bisa** saat dua kandidat teratas nyaris seri. Pada keputusan
  `DEKORATIF` vs bukan, satu token yang berubah mengubah jumlah chunk.
- **Bobot berganti di balik tag yang sama.** `ollama pull` pada tag yang sama
  dapat mengganti bobot tanpa mengubah namanya. Karena itu `vision_model_digest`
  (sha256 penuh dari `/api/tags`, bukan ID pendek) dicatat per gambar.
- **Versi runtime.** Upgrade Ollama dapat mengubah kernel, kuantisasi, atau
  penanganan `num_ctx`.
- **Panjang konteks efektif.** `num_ctx` yang berbeda mengubah jalur komputasi
  walau prompt sama. Karena itu ia disetel eksplisit, bukan dibiarkan default.

**Klaim yang dapat dipertahankan:** suhu 0 menghilangkan sumber variasi terbesar
dan satu-satunya yang dapat dikendalikan dari sisi klien. Pipeline **tidak**
deterministik bit-per-bit.

**Yang menutup sisanya:** cache persisten. Setelah sebuah gambar dideskripsikan
sekali, seluruh run berikutnya memakai teks yang sama tanpa memanggil model —
menghilangkan variasi lintas-run sepenuhnya untuk gambar yang sudah ada di
cache. Itulah sebabnya cache berstatus artefak eksperimen, bukan optimasi.

## 2. Konsekuensi menghapus cache

Cache **tidak boleh dihapus di tengah eksperimen**. Rantai akibatnya:

1. Gambar yang entrinya hilang **dipanggilkan model ulang**.
2. Verdict dapat berbeda karena non-determinisme di atas — terutama pada gambar
   yang ambigu antara `DEKORATIF` dan deskripsi nyata.
3. Verdict yang berubah mengubah **jumlah element**, karena element `Image`
   tanpa deskripsi dibuang.
4. Jumlah element yang berubah **menggeser `chunk_index`** seluruh dokumen itu,
   dan karenanya `chunk_id`.
5. **Seluruh anotasi gold yang menempel pada `chunk_id` menjadi tidak valid**
   untuk dokumen tersebut — `gold_chunk_ids` menunjuk chunk yang berbeda isinya.

`text_sha` menyediakan jalur pemulihan parsial: anotasi dapat dipetakan ulang ke
chunk dengan hash isi yang sama. Tapi chunk yang isinya memang berubah — yaitu
chunk deskripsi gambar itu sendiri, dan chunk sesudahnya yang bergeser
overlap-nya — tidak dapat dipulihkan otomatis.

**Karena itu:** perlakukan `vision_cache.db` seperti korpus PDF-nya sendiri.
Cadangkan sebelum perubahan konfigurasi apa pun, dan cocokkan
`vision_cache.content_sha256` di `run_manifest.json` antar fork sebelum
membandingkan skor.

---
---

# REGISTRY DOKUMEN — dari isian manual ke generate otomatis

`document_id` sekarang diisi otomatis dari nama berkas. Yang berubah bukan
sekadar cara mengisinya: **registry naik status menjadi artefak bersama**,
sejajar `vision_cache.db`. Konsistensi antar fork datang dari berkas yang sama,
bukan dari dua eksekusi scaffolder yang kebetulan menghasilkan keluaran sama.

Satu berkas di `~/rag_mm_b_shared/document_registry.json` — di luar clone mana
pun, tidak dikirim-kirim.

## Aturan slug

| # | Langkah | Contoh |
|---|---|---|
| 1 | Buang ekstensi | `Panduan KKN.pdf` → `Panduan KKN` |
| 2 | Buang prefix penomoran `^\(?\d{1,2}\)?\s*[.)_-]+\s*` | `1.-SOP Izin` → `SOP Izin` |
| 3 | NFKD + buang non-ASCII | `Sürat Edaran Dékan` → `Surat Edaran Dekan` |
| 4 | Lowercase | → `surat edaran dekan` |
| 5 | Non-alfanumerik → `-` | `sop.izin.ujian` → `sop-izin-ujian` |
| 6 | Rapatkan `-`, strip ujung | `--a--b--` → `a-b` |

Hasil terukur:

| Nama berkas | `document_id` |
|---|---|
| `sop_pengurusan_izin_ujian_akhir_online_2.pdf` | `sop-pengurusan-izin-ujian-akhir-online-2` |
| `1.-SOP Pengurusan Izin Ujian.pdf` | `sop-pengurusan-izin-ujian` |
| `01_Pedoman Akademik 2025.pdf` | `pedoman-akademik-2025` |
| `2024_Kalender_Akademik.pdf` | `2024-kalender-akademik` |
| `Sürat Edaran Dékan.pdf` | `surat-edaran-dekan` |

### Tiga keputusan dan alasannya

**Prefix dibatasi dua digit, bukan tiga.** Asimetri kerugiannya: prefix bermakna
yang terpangkas menghasilkan slug yang **terlihat normal** — kesalahannya
senyap. Nomor urut tiga digit yang gagal terpangkas menghasilkan slug jelek tapi
**jelas** dan bisa diedit. Untuk artefak yang menentukan `chunk_id` dan
`image_id` sekaligus, kesalahan yang terlihat lebih baik daripada yang tidak.

Batas ini juga melindungi tahun empat digit: `2024_Kalender.pdf` tetap
`2024-kalender`, tidak tergerus jadi `kalender` yang akan bertabrakan dengan
berkas tahun lain.

**Penanda versi TIDAK ditebak.** `..._online_2.pdf` menjadi `...-online-2`,
bukan `-v2`. Angka di akhir nama ambigu antara urutan dan versi; menerjemahkannya
berarti mengklaim dua dokumen adalah revisi satu sama lain. Versi sebenarnya
menyusul bersama metadata temporal di lapis 4, yang memang sudah ditunda.

**Slug tidak dipotong panjangnya.** Dokumen akademik sering berbeda hanya di
ujung nama — "Program Sarjana" versus "Program Magister". Memotong di batas
panjang akan menabrakkan keduanya, dan tabrakan adalah hal yang justru harus
dihindari.

## Jaminan: entri lama tidak pernah ditimpa

Slug **hanya dihitung** untuk nama berkas yang belum punya `document_id`. Entri
yang sudah terisi disalin apa adanya — termasuk saat aturan slug berubah, dan
termasuk hasil suntingan manual untuk menyelesaikan tabrakan.

Terverifikasi dengan mensimulasikan aturan slug v2 yang jauh berbeda
(`\d{1,9}` alih-alih `\d{1,2}`) pada registry berisi 7 entri v1:

```
entri baru (document_id di-generate)   : 0
entri lama (TIDAK ditimpa)             : 7
beda dari slug aturan sekarang         : 2
[WARN ] Registry sebelumnya dibuat dengan aturan slug versi 1, sekarang 2.
        Entri lama tetap dipertahankan apa adanya.

entri berubah: TIDAK ADA — jaminan terpenuhi
```

Yang membuat jaminan itu **terlihat**, bukan sekadar dijanjikan: tiap entri auto
membawa `slug_rule_version` yang berlaku saat ia dibuat, dan
`document_registry_notes.md` melaporkan entri yang `document_id`-nya tidak lagi
sama dengan slug aturan sekarang — lengkap dengan kedua nilainya berdampingan.

## Tabrakan gagal keras

Dua berkas dengan slug sama menghentikan proses dan **registry tidak ditulis
sama sekali** (kode keluar 1):

```
[ERROR] 1 document_id dipakai lebih dari satu berkas:

          'sop-pengurusan-izin-ujian'
            - 1.-SOP Pengurusan Izin Ujian.pdf
            - SOP Pengurusan Izin Ujian.pdf

        TIDAK ada sufiks otomatis: sufiks berbasis urutan tidak stabil,
        dan document_id yang bergeser membatalkan seluruh anotasi gold.
        Registry TIDAK ditulis.
```

Penyelesaiannya manual — edit `document_id` salah satu berkas, jalankan ulang.
**Itu satu-satunya bagian yang manual.** Slug kosong (nama berkas tanpa karakter
alfanumerik) juga gagal keras.

## Format registry

```json
{
  "_meta": {
    "generated_at": "2026-08-25T09:00:00+00:00",
    "slug_rule_version": 1,
    "tool": "scripts/scaffold_document_registry.py",
    "n_documents": 42
  },
  "documents": {
    "sop_pengurusan_izin_ujian_akhir_online_2.pdf": {
      "document_id": "sop-pengurusan-izin-ujian-akhir-online-2",
      "sha256": "9f2b1c...",
      "slug_rule_version": 1,
      "source": "auto"
    }
  }
}
```

Bentuk datar lama (tanpa `_meta`/`documents`) tetap dibaca — terverifikasi.

## `document_registry_notes.md`

Ditulis bersama registry, di direktori yang sama. **Berkas inilah yang dibaca
peneliti lain sebelum mulai menganotasi**, jadi peringatan tidak cukup di stdout.

Isinya: header generate (waktu, versi aturan, jumlah), `document_id` yang hanya
berisi angka, entri yang tidak lagi sama dengan slug aturan sekarang, berkas yang
`sha256`-nya berubah, entri tanpa berkas di folder, dan protokol regenerate.

## Siapa yang boleh menjalankan generate ulang

Karena registry satu berkas di lokasi bersama, yang perlu disepakati bukan cara
mengirimnya — tapi siapa yang menjalankan ulang dan kapan.

**Aman kapan saja.** Menambah entri untuk PDF baru tanpa menyentuh yang lama.
Dua kali berturut-turut tanpa PDF baru tidak mengubah apa pun kecuali stempel
waktu.

**Butuh kesepakatan lebih dulu:**

- Setelah anotasi gold dimulai. Entri lama memang tidak ditimpa, tapi PDF baru
  menambah dokumen ke korpus dan mengubah komposisi strata.
- Bila ada tabrakan slug. Suntingan itu menentukan `document_id` permanen.
- Bila `sha256` sebuah berkas berubah. Perlu diputuskan apakah itu revisi yang
  butuh `document_id` baru.

**Jangan pernah:** menghapus registry lalu generate ulang dari nol setelah
indexing berjalan. Slug memang deterministik terhadap nama berkas, tapi entri
hasil suntingan manual akan hilang — `document_id` berubah, dan seluruh
`gold_chunk_ids` serta `gold_image_ids` yang menunjuknya jadi tidak valid.

---

# PERBAIKAN WAJIB — `chunk_id` duplikat di Qdrant

**Peneliti fork lain wajib menerapkan perbaikan ini.** Ia bukan penyempurnaan:
tanpanya, `chunk_id` tidak unik dan anotasi gold kehilangan arti tunggalnya.

## Gejala di korpus nyata

Indexing 214 dokumen menghasilkan:

| | |
|---|---|
| Titik di Qdrant | 26.087 |
| `chunk_id` unik | 25.475 |
| Titik berlebih | **388**, tersebar di 30 dokumen |
| Peringatan `chunk_over_budget_kept_whole` | 486, terbesar 4.091 token |

Empat titik yang bertabrakan memiliki `chunk_id`, `chunk_index`, **dan** `text_sha`
identik, tetapi panjang teks berbeda: 739, 1041, 523, 1055 karakter.

## Penyebab — bukan yang diduga pertama

Dugaan awal: `_split_for_budget` (Tahap 1B) memecah chunk tapi tidak menaikkan
pencacah per-halaman. **Dugaan itu salah.** `emit()` di `preprocessing.py` sudah
benar — pencacah dinaikkan di dalam loop per pecahan, dan `text_sha` dihitung dari
`hash_basis = core`, yaitu isi pecahan itu sendiri:

```python
for position, (core, final_text) in enumerate(zip(pieces, final_texts)):
    ordinal = page_counters.get(segment, 0)
    page_counters[segment] = ordinal + 1          # <- naik per pecahan
    chunk["chunk_id"] = f"{document_id}_{segment}_c{ordinal:02d}"
    chunk["text_sha"] = text_sha(hash_basis)      # <- per pecahan
```

Yang menentukan justru **`text_sha` yang ikut identik**. Kalau `_chunk_elements`
menghasilkan pecahan berbeda dengan id sama, `text_sha`-nya akan berbeda karena
dihitung dari teks masing-masing. Tiga field identik dengan teks berbeda hanya bisa
berarti satu hal: metadata **disalin apa adanya** ke beberapa node — tanda tangan
node parser LlamaIndex.

### Sebab sebenarnya: `transformations=[]` tidak mematikan apa pun

`VectorStoreIndex.from_documents` menulis (llama_index/core/indices/base.py:109):

```python
transformations = transformations or Settings.transformations
```

Daftar kosong bersifat **falsy** di Python. `[] or Settings.transformations`
menghasilkan `Settings.transformations` — `SentenceSplitter` default. Argumen
`transformations=[]` yang dipakai Tahap 1C karenanya **tidak pernah berefek**.
Tidak ada peringatan, tidak ada galat, tidak ada jejak di log.

Terbukti langsung, satu tabel 3.949 token:

```
transformations=[]         -> 11 node, 1 chunk_id unik, 10 duplikat
[PassthroughNodeParser()]  ->  1 node, 1 chunk_id unik,  0 duplikat
```

Chunk tabel adalah korban utamanya karena `PDF_TABLE_MAX_CHARS` **ambang**, bukan
plafon (lihat koreksi di Tahap 1): tabel besar keluar dari `_chunk_elements` utuh
dengan `splittable=False` — sengaja, demi RCAA — lalu dipecah ulang oleh parser
kedua. Itu juga menjelaskan 486 peringatan `chunk_over_budget_kept_whole`: chunk
itu memang melewati anggaran, dan justru chunk itulah yang dipecah diam-diam.

### Kenapa probe Tahap 0 tidak menangkapnya

Probe **menirukan** efek flag alih-alih menjalankannya:

```python
if INDEX_DISABLE_NODE_PARSER:
    total_nodes += 1        # asumsi: 1 Document = 1 node
```

Probe yang menguji asumsinya sendiri akan selalu lulus. Sembilan kasus lama tetap
0 tabrakan bahkan di kode yang rusak.

## Perbaikan

### Berkas baru: `backend/services/node_passthrough.py`

`PassthroughNodeParser` — satu Document menjadi tepat satu node, teks utuh.

Memakai `build_nodes_from_splits` (helper yang sama dengan `SentenceSplitter`),
bukan sekadar mengembalikan `nodes` apa adanya. Alasannya: transformasi no-op
meneruskan objek `Document`, dan Qdrant menyimpan tipe itu di payload
(`_node_type`) — bentuk titik berubah dari yang dipakai jalur query. Dengan
helper tersebut, node yang dihasilkan **identik bentuknya** dengan sebelumnya:
`TextNode`, relasi SOURCE, metadata dan daftar kunci terkecuali terwarisi. Satu-
satunya perbedaan adalah teksnya tidak dipotong. Terverifikasi berdampingan.

`MetadataMode.NONE` wajib: `BaseNode.get_content()` default-nya `MetadataMode.ALL`,
yang akan menyisipkan metadata ke dalam teks node.

Berkas ini juga memuat `build_transformations()` — **satu sumber kebenaran** yang
dipakai `indexing.py` **dan** `probe_rechunk.py`. Ditempatkan di sini, bukan di
`indexing.py`, supaya probe dapat mengujinya tanpa menarik klien Qdrant.

### `backend/services/indexing.py`

`_embed_and_store` kini satu jalur, `transformations=build_transformations()`.
Cabang `if/else` lama dihapus: `None` sudah berarti "pakai `Settings`".

### Berkas baru: `backend/services/index_verify.py`

Memeriksa **hasil**, bukan niat. Gerbang lain memeriksa flag dan laporan ekstraksi
sebelum apa pun dikirim; ini membaca kembali titik yang benar-benar tersimpan.
Perlu, karena `chunk_id` duplikat lahir **di dalam** LlamaIndex — setelah gerbang
terakhir. Tidak ada pemeriksaan sebelum-kirim yang bisa melihatnya.

- Dipanggil di akhir `index_documents` saat `INDEX_STRUCTURAL_METADATA` aktif.
- **Menolak** (`ValueError`) saat `RESEARCH_MODE=true`; **memperingatkan** di luar itu.
- Gagal membaca Qdrant **tidak** menjatuhkan run — data sudah tersimpan, dan
  pemeriksaan yang tidak bisa berjalan bukan alasan membuang pekerjaan berjam-jam.
  Ia melaporkan dirinya gagal (`status="gagal_dibaca"`), dan itu terlihat di log.
- Titik **tanpa** `chunk_id` diperingatkan terpisah: tidak dapat dirujuk
  `gold_chunk_ids` sama sekali.
- Membaca `chunk_id` dari akar payload, dengan cadangan `_node_content` — supaya
  tidak diam-diam melaporkan "0 duplikat" hanya karena tidak menemukan field-nya.

### `backend/services/chunk_dump.py` — field manifest baru

`chunking.node_transformations` mencatat **nama kelas transformasi yang
benar-benar dijalankan**, diselesaikan dengan aturan yang sama seperti
`from_documents`:

```json
"node_transformations": ["PassthroughNodeParser"]
```

Dicatat terpisah dari `research_flags.INDEX_DISABLE_NODE_PARSER` karena bug ini
persis kasus **flag benar, perilaku salah**. Manifest run yang bermasalah akan
menunjukkan `INDEX_DISABLE_NODE_PARSER: true` berdampingan dengan
`["SentenceSplitter"]` — satu baris yang cukup untuk menyatakan run itu tidak
sahih. Manifest yang hanya mencatat flag tidak menunjukkan apa pun.

### Berkas baru: `scripts/verify_chunk_ids.py`

Pemeriksaan yang sama, read-only, untuk koleksi yang **sudah** ada — tanpa perlu
mengindeks ulang lebih dulu. Keluar dengan kode 1 bila ada duplikat.

```bash
python scripts/verify_chunk_ids.py --collection rag_mm_b_varian_c
python scripts/verify_chunk_ids.py --all      # semua contoh, tanpa dipotong
```

### `scripts/probe_rechunk.py`

Dua perubahan; yang pertama lebih penting daripada yang kedua.

**1. Bagian B menjalankan transformasi produksi, bukan simulasinya.** Chunk
dibungkus jadi Document persis seperti `indexing.py:608-613`, lalu dilewatkan
`run_transformations(docs, build_transformations() or Settings.transformations)`.
`chunk_index` dan `chunk_id` kini dibaca dari **node**, bukan dari chunk — titik
Qdrant dibuat per node, dan di situlah duplikat lahir.

**2. Kasus regresi baru: tabel 4.024 token.** Kasus tabel lama (811 token) tidak
pernah memicu apa pun — lolos cabang `kept_whole` DAN muat di satu node. Kasus
baru meniru tabel 4.091 token dari korpus nyata.

Bagian C kini juga mendemonstrasikan jebakannya secara langsung:

```
JEBAKAN — kenapa transformations=[] tidak mematikan apa pun:
  base.py:109  transformations = transformations or Settings.transformations
  diminta []  ->  efektif ['SentenceSplitter']   (daftar kosong DIABAIKAN)
```

## Verifikasi

```bash
INDEX_DISABLE_NODE_PARSER=true INDEX_STRUCTURAL_METADATA=true \
INDEX_MAX_CHUNK_TOKENS=350 INDEX_MIN_CHUNK_TOKENS=8 \
INDEX_TABLES_AS_OWN_CHUNKS=true INDEX_EXCLUDE_METADATA_FROM_EMBED=true \
python scripts/probe_rechunk.py
```

Sepuluh kasus, termasuk yang baru — semuanya `chunk` = `node`, `chunk_id` unik:

```
tabel sangat besar (jauh > anggaran, REGRESI)    4024     1     1  unik   unik
```

Bukti kasus baru benar-benar menangkap bug lama — `build_transformations()`
dikembalikan sementara ke `[]`:

```
tabel sangat besar (jauh > anggaran, REGRESI)    4024     1     5  TABRAKAN  TABRAKAN  <-- MASIH DIPECAH
```

Sembilan kasus lain tetap lolos di kode rusak itu. Hanya kasus baru yang menyala.

Dengan flag riset **mati**, perilaku produksi semula tidak berubah:
`build_transformations()` mengembalikan `None`, `SentenceSplitter` tetap berjalan.

`index_verify` diuji dengan klien palsu — tujuh kasus, semua lolos: paginasi lewat
`offset`, `strict=True` menolak, `strict=False` memperingatkan, cadangan
`_node_content`, `_node_content` rusak, payload tanpa `chunk_id`, dan `scroll` yang
melempar galat (tidak menjatuhkan run).

## Yang harus dilakukan peneliti fork lain

1. Ambil `node_passthrough.py`, `index_verify.py`, dan perubahan `indexing.py`.
2. **Periksa koleksi yang sudah ada** dengan `scripts/verify_chunk_ids.py`.
3. **Re-index penuh.** `chunk_id` yang sudah tersimpan tidak dapat diperbaiki di
   tempat — tabel yang terlanjur pecah tersimpan sebagai beberapa titik dengan
   teks yang tidak lagi utuh.
4. **Jangan memulai anotasi gold sebelum langkah 2 lolos.** `gold_chunk_ids` yang
   menunjuk `chunk_id` duplikat tidak dapat diselamatkan setelah anotasi berjalan.

## Pelajaran yang berlaku di luar bug ini

**Daftar kosong bukan cara mematikan sesuatu di Python.** `[]`, `{}`, `0`, dan `""`
semuanya falsy; API yang menulis `x = x or default` akan mengabaikannya. Pola
`or default` ada di banyak tempat di LlamaIndex.

**Probe harus memanggil kode produksi, bukan menirukan perilakunya.** Simulasi
menguji apa yang kita kira benar. Itu sebabnya `build_transformations()` kini hidup
di modul yang dapat diimpor probe.

**Verifikasi niat tidak menggantikan verifikasi hasil.** Seluruh gerbang sebelum
ini memeriksa konfigurasi dan data sebelum dikirim, dan semuanya lolos. Yang salah
terjadi di dalam pustaka pihak ketiga.

---

# TAHAP 5 — instrumentasi jalur retrieval untuk evaluasi

Berbeda dari Tahap 1–4 yang membentuk korpus, tahap ini hanya menyentuh jalur
**query**. Tidak ada perubahan yang memicu re-index: `preprocessing.py`,
`indexing.py`, `chunk_dump.py`, `image_describer.py`, `node_passthrough.py`, dan
seluruh flag `INDEX_*` tidak disentuh.

**Prinsip: tidak ada layer yang dihapus.** Yang dilakukan hanya MELEWATI titik
keluar dan MEREKAM apa yang sudah dihitung pipeline. Dengan seluruh flag
`RESEARCH_*` mati, jalur eksekusi identik dengan sebelum tahap ini.

## Empat flag baru — semuanya default mati

| Flag | Efek |
|---|---|
| `RESEARCH_BYPASS_ROUTING` | Lewati lima titik keluar tanpa saklar (C9 #1, #3, #5, #6, #8) |
| `RESEARCH_DISABLE_CONDENSATION` | Matikan query condensation (C11) |
| `RESEARCH_DISABLE_OUTPUT_FILTER` | Matikan Layer 6, termasuk `filter_token` pada streaming |
| `RESEARCH_VERBOSE_RETRIEVAL` | Sertakan hasil tiga tahap retrieval di `debug.retrieval_stages` |

### 5A — titik keluar tanpa saklar

Lima titik yang sebelumnya tidak punya env var kini dapat dilewati:
`l1_hard_block`, `identity_override`, `l3_chitchat`, `l3_out_of_scope`,
`condense_ack`, plus `l1_hard_block_condensed` (re-check L1 atas pertanyaan
terkondensasi, C9 #9 — turunan dari #1).

Setiap penyelamatan dicatat dua kali: ke log
(`research_bypass_routing point=<titik>`) dan ke `debug.routing_bypassed`,
sehingga jumlah item test set yang terpengaruh dapat dihitung **per titik**.

**L2 moderation sengaja TIDAK dilewati.** Ia sudah punya saklar sendiri
(`MODERATION_BACKEND=passthrough`); melewatinya diam-diam berarti mengubah
perilaku keamanan tanpa jejak di konfigurasi moderation. Terverifikasi: dengan
`RESEARCH_BYPASS_ROUTING=true`, query yang ditolak L2 tetap `blocked_moderation`.

### 5B — condensation

Pemicunya `if history:`, bukan flag. Kini `if history and not
RESEARCH_DISABLE_CONDENSATION:`.

### 5C — output filter

`filter_output` dan `filter_token` dibungkus `_filter_output`/`_filter_token`.
Keduanya dimatikan bersama — kalau hanya salah satu, token yang di-stream
tersaring sementara jawaban akhir tidak, dan keduanya jadi berbeda.

### 5D — instrumentasi retrieval

`_retrieve_and_rerank` mendapat parameter **keluaran** `stages`, pola yang sama
dengan `timings` milik F-2. Diisi tiga tahap: `dense`, `expansion`, `rerank`.

Tiap node membawa `chunk_id`, `document_id`, `image_id`, `element_type`, `page`,
`file_name`, `chunk_index`, `text_sha`, `rank`, `score`, dan `is_neighbor`.

**Tetangga terbedakan.** `_expand_with_neighbors` memberi `score=0.0`; pembedaan
itu terbawa sebagai `is_neighbor` sehingga tidak tertukar dengan node dense yang
kebetulan berskor rendah.

**Dua kontaminasi ditangani:**

*`text_preview` bukan teks chunk.* `SourceLabelPostprocessor` menulis ulang
`node.node.text` menjadi `"[nama_file]\n" + teks` **sebelum** `_build_sources`
dipanggil. Tahap direkam SEBELUM labeler berjalan, dan `_build_sources` menerima
`clean_texts` saat `RESEARCH_VERBOSE_RETRIEVAL` aktif. **Konteks yang dilihat LLM
tidak diubah** — labelnya tetap ada di node, karena menghapusnya akan mengubah
generasi, bukan hanya pelaporan.

*Skor berubah makna diam-diam saat rerank gagal.* Lihat bagian bug produksi.

**Cache.** `retrieval_stages` dan `routing_bypassed` dibuang sebelum `cache_set`.
Tanpa itu, cache hit akan menyajikan hasil retrieval **pertanyaan lain** sebagai
milik pertanyaan ini — mencemari data evaluasi secara senyap.

### 5E — `scripts/retrieval_dump.py`

Retrieval tanpa generation, tanpa endpoint HTTP. Memakai `_retrieve_and_rerank`
yang SAMA dengan produksi, bukan salinannya — menulis ulang logikanya akan
mengukur pipeline yang berbeda dari yang dilayani ke pengguna.

Masukan: teks polos (satu pertanyaan per baris) **atau** JSONL dengan field
`question`. Deteksi berdasarkan isi, bukan ekstensi. Field lain (`gold_chunk_ids`,
`expected_behavior`, ...) dibawa apa adanya ke `retrieval_runs.jsonl`.
`question_id` duplikat ditolak keras — tanpa id unik, baris keluaran tidak dapat
dipetakan balik ke test set.

Keluaran:

| Berkas | Isi |
|---|---|
| `retrieval_nodes.jsonl` | satu baris per (pertanyaan, tahap, peringkat) — tabel datar untuk metrik |
| `retrieval_runs.jsonl` | satu baris per pertanyaan — ringkasan, timing, galat |
| `retrieval_manifest.json` | konfigurasi retrieval, koleksi, model, flag `RESEARCH_*` |

Pemisahan lapis 1 per modalitas, seperti diminta deck:

```
teks     : image_id is null AND element_type != "ImageDescription"
gambar   : image_id is not null
gabungan : seluruh baris
```

```bash
python scripts/retrieval_dump.py --questions qa_pairs.jsonl \
    --collection rag_mm_b_varian_b_v2 --out hasil_varian_b/
```

## Tiga perbaikan BUG PRODUKSI — di luar flag, laporkan ke tim UniAI

Ketiganya berlaku terlepas dari riset dan aktif tanpa env var apa pun.

### 1. `\bmeta\b` mencocoki kata berimbuhan tanda hubung

`output_filter.py:17` memuat `meta` sebagai alternatif ber-`\b`. Tanda hubung
memenuhi batas kata, sehingga **"meta-analisis" menjadi "[AI vendor]-analisis"**.
Kata seperti `meta-analisis`, `meta-data`, dan `meta-kognitif` lazim di teks
akademik Indonesia — justru korpus yang dilayani sistem ini.

Diganti penjaga eksplisit `(?<![\w-])meta(?![\w-])`. Terverifikasi 9/9: lima
bentuk berimbuhan dibiarkan utuh, tiga penyebutan vendor tetap ter-redact, dan
`metadata` (tanpa hubung) tidak terpengaruh.

Pola `transformers` dan `bert` di `:14` dan `:20` **sengaja tidak diubah** —
keduanya memang nama model/pustaka, dan memperbaikinya butuh keputusan produk,
bukan perbaikan regex. Untuk riset, matikan filternya.

### 2. `HISTORY_TURNS=0` memakai riwayat PENUH

Lima tempat menulis `history[-(HISTORY_TURNS * 2):]`. Di Python `-0 == 0`,
sehingga `HISTORY_TURNS=0` menghasilkan `history[0:]` — **seluruh riwayat**,
kebalikan dari yang dimaksud.

Diganti `_recent_history()` yang mengembalikan `[]` saat `HISTORY_TURNS <= 0`.
Terverifikasi: `HISTORY_TURNS=0` → 0 pesan; `=2` → 4 pesan.

### 3. Fallback rerank tidak punya penanda

Saat rerank gagal (`:684`, `:690`, `:696`), `_TEIRerankPostprocessor`
mengembalikan `nodes[:top_n]` **tanpa** menulis ulang `node.score`. Skor yang
tersisa adalah skor kemiripan dense, bukan skor reranker — lalu dibandingkan
dengan `SCORE_THRESHOLD` yang dikalibrasi untuk skala reranker. Dua skala
berbeda, satu ambang. Sebelumnya hanya terlihat di log dan counter Prometheus.

Kini `debug.rerank_fallback` **selalu** ada: `None` = rerank tidak dijalankan,
`False` = normal, `True` = jatuh ke urutan dense. Disertai log warning yang
menyebut konsekuensinya.

Penandanya `contextvars.ContextVar`, bukan variabel modul: endpoint streaming
mengembalikan generator **sinkron** yang dijalankan Starlette di threadpool, jadi
beberapa query bisa berjalan bersamaan dan penanda satu query tidak boleh terbaca
oleh query lain.

## Perbaikan kecil

- `scripts/verify_env.py` membaca `document_registry.json` lewat
  `split_document_block`. Sebelumnya akar JSON dibaca langsung, sehingga `_meta`
  dan `documents` terhitung sebagai dua entri dokumen — **"0/2 entri punya
  document_id" padahal isinya 214**. Terverifikasi: bentuk baru 214/214, bentuk
  datar lama tetap terbaca. Versi aturan slug ikut dilaporkan.
- `verify_env.py` menurunkan ketidakcocokan arsitektur GPU dari **GAGAL** menjadi
  **PERINGATAN** bila GPU lebih baru daripada arsitektur tertinggi di build
  torch. Keterangan lama ("torch kemungkinan terbayangi wheel PyPI") salah: wheel
  resmi cu130 memang hanya dikompilasi sampai `sm_120`, dan GB10 (`sm_121`)
  berjalan lewat **PTX forward compatibility** — driver meng-JIT ulang PTX
  arsitektur tertinggi. Terbukti bekerja: matmul 4096×4096 fp16 selesai 75 ms.
  GPU yang lebih LAMA daripada build tetap mendapat keterangan berbeda, karena
  PTX forward compatibility tidak menolong ke arah itu.
  Perbandingannya memakai `_parse_sm()`: digit terakhir minor, sisanya mayor —
  `sm_120` adalah `(12, 0)`, bukan `(1, 2, 0)` yang akan terurut di bawah `sm_86`.
- `scripts/verify_chunk_ids.py` memisahkan `chunk_id` dari nama berkas dengan
  lebar kolom dinamis.

## Verifikasi

Matriks 24 skenario, tiap flag **dengan dan tanpa** diaktifkan, dijalankan
terhadap `query()` yang sebenarnya dengan stub `sys.modules` untuk torch,
prometheus, Qdrant, dan LLM. Yang di-stub hanya batas luar; logika routing
dijalankan apa adanya.

Hasil kunci:

```
BASELINE semua mati                    mode=rag, kunci_riset=[]      <- tidak ada kebocoran
BYPASS mati  | L1 hard-block           mode=blocked
BYPASS hidup | L1 hard-block           mode=rag, bypass=[l1_hard_block]
BYPASS hidup | L2 moderation           mode=blocked_moderation       <- TETAP diblokir
CONDENSATION dimatikan                 condense_dipanggil=false
OUTPUT_FILTER aktif                    "...meta-analisis [internal system]."
OUTPUT_FILTER dimatikan                "...meta-analisis transformers."
VERBOSE mati                           stages=null, preview="[sop]\nIsi chunk..."
VERBOSE hidup                          dense=4 expansion=5 rerank=3, preview bersih
HISTORY_TURNS=0                        n_ke_format=0                 <- bug -0 diperbaiki
rerank GAGAL                           rerank_fallback=true
```

**Kriteria lulus terpenuhi:** dengan seluruh `RESEARCH_*` mati, `debug` tidak
memuat satu pun kunci riset, `text_preview` tetap berlabel seperti produksi, dan
mode yang dihasilkan tiap titik keluar sama persis dengan sebelum tahap ini.

## Temuan sampingan — TIDAK diperbaiki

`_IDENTITY_KEYWORDS` (`rag_pipeline.py:159`) memuat `"meta"` dan dicocokkan
sebagai **substring polos**. Akibatnya pertanyaan akademik yang sah dibelokkan ke
chitchat tanpa pernah menyentuh retrieval:

```
"Bagaimana metodologi meta-analisis dipakai dalam skripsi?"
    tanpa bypass : mode=chitchat, 0 sumber
    dengan bypass: mode=rag,      3 sumber
```

Jalur identity juga **tidak memanggil output filter sama sekali**, berbeda dari
jalur chitchat biasa.

Tidak diperbaiki karena di luar cakupan tugas ini dan menyangkut keputusan produk
(daftar itu melindungi dari kebocoran nama model). Untuk riset, tertutup oleh
`RESEARCH_BYPASS_ROUTING`. Layak dilaporkan ke tim UniAI bersama tiga bug di atas.

## Menyatukan ke `run_manifest.json`

Flag `RESEARCH_*` **belum** masuk `run_manifest.json`, karena berkas penulisnya
(`chunk_dump.py`) dibekukan pada tahap ini. Dua catatan:

1. Manifest itu merekam **indexing run**. Flag Tahap 5 semuanya query-time dan
   belum ada nilainya saat indexing berjalan — `retrieval_manifest.json` adalah
   tempat yang tepat secara semantik, dan di sanalah snapshot-nya sekarang.
2. Kalau tetap diinginkan di manifest indexing, tambahkan satu baris ke
   `chunk_dump.write_run()` di dalam blok `"research_flags"`:

   ```python
   **config.research_query_flags(),
   ```

   `research_query_flags()` sudah ada di `backend/config.py` dan belum dipanggil
   siapa pun selain `retrieval_dump.py`.

## Cara mendaratkan perubahan yang BELUM di-commit

`backend/services/rag_pipeline.py` dan `backend/models/schemas.py` **sengaja
tidak di-commit**. Keduanya sudah memuat modifikasi pra-sesi milik instrumentasi
F-2 (`backend/metrics.py`), dan hunk Tahap 5 bercampur dengan hunk itu di fungsi
yang sama: `_TEIRerankPostprocessor`, `_expand_with_neighbors`,
`_retrieve_and_rerank`, `_vision_query`, `query`, `query_stream`.

**Ketergantungan yang harus diketahui:** kode Tahap 5 bersandar pada parameter
`timings=` yang ditambahkan F-2 ke tanda tangan `_retrieve_and_rerank`, dan pada
helper `_timed`/`_safe_inc` milik F-2. Commit yang hanya memuat hunk Tahap 5
**tidak dapat dikompilasi berdiri sendiri**.

**Urutan yang benar:**

1. Pemilik F-2 commit lebih dulu — `backend/metrics.py` beserta perubahannya di
   `rag_pipeline.py`, `schemas.py`, `main.py`, dan `moderation.py`.
2. Setelah itu, `git diff` pada `rag_pipeline.py` dan `schemas.py` hanya akan
   menyisakan hunk Tahap 5, yang dapat di-commit terpisah.

Jangan membalik urutannya: mendaratkan Tahap 5 lebih dulu akan ikut membawa WIP
F-2 ke dalam commit yang bukan miliknya.
