# INSPECTION REPORT — rag-core-system

Inspeksi read-only untuk adaptasi menjadi harness eksperimen retrieval multimodal.

**Target migrasi:** NVIDIA DGX Spark GB10, aarch64, sm_121, CUDA 13.2, 128 GB unified memory, Ubuntu 24.04, mesin bersama.
**Basis kode:** branch `main`, commit `022c669`, dengan working-tree modification pada `backend/main.py`, `backend/models/schemas.py`, `backend/services/moderation.py`, `backend/services/rag_pipeline.py`, dan file untracked `backend/metrics.py`.

**Batasan inspeksi:** semua klaim berasal dari pembacaan source. Ketersediaan wheel aarch64 di PyPI dan ketersediaan tag image multi-arch **tidak** diverifikasi lewat jaringan — bagian yang bergantung pada itu ditandai eksplisit sebagai perlu verifikasi.

---

## A. PIPELINE INDEXING

### A1. Alur lengkap PDF → Qdrant

Dua entry point, keduanya bermuara ke fungsi yang sama:

| Entry | Lokasi |
|---|---|
| CLI | `scripts/index_documents.py:28` `main()` |
| HTTP | `backend/main.py:265` `index_documents()` (endpoint `POST /api/index`) |

Jalur CLI memanggil `_configure_settings()` dulu (`scripts/index_documents.py:54` → `backend/services/rag_pipeline.py:603`) untuk men-set `Settings.embed_model` dan `Settings.llm`. Jalur HTTP **tidak** memanggilnya — bergantung pada `Settings` global yang mungkin sudah di-set oleh `_get_retriever()` bila sebuah query pernah dijalankan sebelumnya (`backend/services/rag_pipeline.py:634`). Kalau belum pernah ada query, `POST /api/index` akan memakai embedding default LlamaIndex, bukan `EMBED_MODEL`.

Rantai fungsi:

```
index_documents()                       backend/services/indexing.py:167
├─ get_collection_count()               backend/services/indexing.py:56       [bila force=True, dipanggil di :187]
├─ clear_collection()                   backend/services/indexing.py:37       [bila force=True, dipanggil di :189]
├─ get_indexed_file_hashes()            backend/services/indexing.py:66       [dipanggil di :197] → Qdrant scroll :77
├─ file_sha256()                        backend/services/preprocessing.py:91  [dipanggil di :204]
├─ delete_file_chunks()                 backend/services/indexing.py:102      [dipanggil di :221, untuk file berubah]
│
├─ extract_from_pdf()                   backend/services/preprocessing.py:412 [dipanggil di indexing.py:235]
│  ├─ file_sha256()                     backend/services/preprocessing.py:91  [dipanggil lagi di :424]
│  ├─ detect_strategy()                 backend/services/preprocessing.py:102 [dipanggil di :428, hanya bila strategy="auto"]
│  │
│  ├─ [hi_res] _extract_hi_res()        backend/services/preprocessing.py:189 [dipanggil di :433]
│  │  ├─ unstructured.partition_pdf()   backend/services/preprocessing.py:201
│  │  ├─ _html_table_to_markdown()      backend/services/preprocessing.py:178 [dipanggil di :224]
│  │  └─ (on exception) _extract_fast() backend/services/preprocessing.py:211
│  │
│  ├─ [hi_res] _describe_image_elements() backend/services/preprocessing.py:261 [dipanggil di :434]
│  │  └─ describe_image()               backend/services/image_describer.py:95 [dipanggil di preprocessing.py:292]
│  │     ├─ _resize_image_if_needed()   backend/services/image_describer.py:44  [dipanggil di :109]
│  │     ├─ _describe_via_vllm()        backend/services/image_describer.py:136 [dipanggil di :113]
│  │     └─ _describe_via_ollama()      backend/services/image_describer.py:162 [dipanggil di :115]
│  │
│  └─ [fast] _extract_fast()            backend/services/preprocessing.py:156 [dipanggil di :436]
│     └─ _extract_text_from_page_fast() backend/services/preprocessing.py:130 [dipanggil di :162]
│        ├─ _get_ocr_engine()           backend/services/preprocessing.py:40   [dipanggil di :145]
│        └─ _extract_texts_from_ocr_result() backend/services/preprocessing.py:55 [dipanggil di :150]
│
├─ chunk_documents()                    backend/services/preprocessing.py:446 [dipanggil di indexing.py:240]
│  └─ _chunk_elements()                 backend/services/preprocessing.py:308 [dipanggil di :460]
│
├─ Document(...)                        backend/services/indexing.py:249-261  [konstruksi metadata]
│
└─ _embed_and_store()                   backend/services/indexing.py:149      [dipanggil di :274]
   ├─ _ensure_collection()              backend/services/indexing.py:119      [dipanggil di :152]
   ├─ QdrantVectorStore(...)            backend/services/indexing.py:154
   └─ VectorStoreIndex.from_documents() backend/services/indexing.py:160      ← TITIK MASUK QDRANT
```

**Titik masuk Qdrant yang sebenarnya** adalah `backend/services/indexing.py:160`. Embedding dan upsert dilakukan di dalam `VectorStoreIndex.from_documents()` milik LlamaIndex — tidak ada kode proyek yang memanggil `client.upsert()` secara langsung untuk jalur PDF.

**Catatan:** ada jalur indexing kedua yang menulis ke collection yang **sama** dengan skema metadata berbeda: `backend/services/index_narratives.py:303-315`. Lihat A3.

### A2. Di mana image bytes dibuang

Image bytes tidak pernah menyentuh disk sama sekali. Ada **empat** titik pembuangan berurutan:

| # | Lokasi | Yang terjadi |
|---|---|---|
| 1 | `backend/services/preprocessing.py:206` | `extract_image_block_to_payload=True` — Unstructured menaruh gambar sebagai base64 di metadata, **bukan** menulis file. Alternatifnya (`extract_image_block_output_dir`) tidak dipakai. |
| 2 | `backend/services/preprocessing.py:272` | Bila `should_describe` False, **semua** element `Image` dibuang total (`[e for e in elements if e["category"] != "Image"]`). |
| 3 | `backend/services/preprocessing.py:294` | Bila `describe_image()` mengembalikan `None` (dekoratif, error, vision off), element di-`continue` — gambar hilang tanpa jejak. |
| 4 | `backend/services/preprocessing.py:296-301` | **Titik pembuangan utama.** Element baru dibangun dengan `"metadata": {}` (baris 300). Field `image_base64` dari baris 241 tidak diteruskan. Setelah baris ini tidak ada referensi ke bytes gambar di mana pun. |

Pembuangan kedua terjadi lagi di chunking: `backend/services/preprocessing.py:381-388` membangun chunk `ImageDescription` tanpa key `metadata` sama sekali, dan `backend/services/indexing.py:251-260` hanya membaca key yang sudah di-whitelist.

**Yang perlu diubah agar gambar tersimpan ke disk dengan path tercatat di metadata** (deskripsi, bukan kode):

1. `backend/config.py:12` — `IMAGES_DIR` sudah didefinisikan tapi **tidak pernah dipakai**. Diimpor di `backend/services/preprocessing.py:26` lalu dibiarkan menganggur (dead import). Ini titik sambung yang sudah tersedia.
2. `backend/services/preprocessing.py:232-242` — cabang `Image`/`Figure` perlu menulis base64 ke file di bawah `IMAGES_DIR` dan menyimpan path relatif di element metadata. Perlu skema penamaan deterministik (mis. `{file_hash}_p{page}_{idx}.png`) supaya re-index tidak menghasilkan path baru.
3. `backend/services/preprocessing.py:296-301` — dict output harus meneruskan `image_path`, bukan `{}`.
4. `backend/services/preprocessing.py:381-388` — cabang `ImageDescription` di `_chunk_elements` perlu meneruskan metadata element ke chunk.
5. `backend/services/indexing.py:251-260` — whitelist metadata `Document` perlu ditambah field baru.
6. Tiga cabang `continue` yang membuang gambar (`:272`, `:285`, `:290`, `:294`) perlu dipisah: penulisan file harus terjadi **sebelum** keputusan deskripsi, karena riset ini butuh gambar asli tersimpan terlepas dari apakah deskripsinya berhasil.
7. `backend/services/preprocessing.py:156-173` (`_extract_fast`) tidak melakukan ekstraksi gambar sama sekali. PDF yang dirutekan ke `fast` tidak akan pernah menghasilkan file gambar. Perlu diperhatikan karena `detect_strategy` merutekan PDF hasil scan ke `fast` (lihat A2 catatan di bawah).

**Catatan tambahan tentang routing:** `detect_strategy()` (`backend/services/preprocessing.py:102-125`) hanya menghitung `page.get_images()` pada `min(5, total_pages)` halaman **pertama** (baris 112-116) dan memakai ambang strict `> 1.0` (baris 122). Halaman hasil scan = tepat 1 gambar full-page → rata-rata 1.0 → dirutekan ke `fast`. Docstring baris 105 menyebut "Banyak gambar/tabel → hi_res" tapi tabel tidak pernah dihitung. `PDF_MIN_IMAGE_SIZE_KB` tidak dipakai di sini (hanya di `backend/services/image_describer.py:76`), jadi logo header/footer berulang bisa memicu `hi_res` palsu.

### A3. Field metadata yang ditulis ke payload Qdrant

**Jalur PDF** — `backend/services/indexing.py:251-260`:

| Field | Tipe | Asal nilai |
|---|---|---|
| `file_name` | `str` | `chunk["file_name"]` ← `pdf_path.name` (`backend/services/preprocessing.py:440`) → diteruskan ke `_chunk_elements(elements, file_name)` (`:460`) → ditulis di `:332`, `:365`, `:384` |
| `file_hash` | `str` | `result["file_hash"]` ← `file_sha256(pdf_path)` (`backend/services/preprocessing.py:424`, fungsi di `:91-97`) |
| `page` | `int` | `chunk["page"]`. Jalur fast: `page.number + 1` (`backend/services/preprocessing.py:168`). Jalur hi_res: `el.metadata.page_number` atau `0` (`:216`) |
| `chunk_index` | `int` | `chunk["chunk_index"]` ← `len(chunks)` saat append (`backend/services/preprocessing.py:333`, `:366`, `:385`) — indeks berjalan per file |
| `element_type` | `str` | `chunk["element_type"]`. Chunk teks: `"+".join(sorted(current_categories))` atau `"text"` (`:334`). Tabel besar: literal `"Table"` (`:367`). Gambar: literal `"ImageDescription"` (`:386`) |
| `section` | `str` | `chunk["section"]` ← teks element `Title` terakhir yang ditemui (`backend/services/preprocessing.py:348`). String kosong bila belum ada Title |
| `extraction_strategy` | `str` | `result["strategy"]` ← `"fast"` atau `"hi_res"` (`backend/services/preprocessing.py:442`) |
| `source_type` | `str` | Literal `"pdf"` (`backend/services/indexing.py:259`) |

**Jalur narasi (.txt)** — `backend/services/index_narratives.py:305-314`, menulis ke **collection yang sama** (`QDRANT_COLLECTION_NAME`, `backend/services/index_narratives.py:337`) dengan skema **berbeda**:

| Field | Tipe | Asal nilai |
|---|---|---|
| `source_file` | `str` | `txt_path.name` (`:306`) |
| `file_name` | `str` | `txt_path.name` (`:307`) — sama dengan `source_file` |
| `endpoint` | `str` | nama subfolder (`:308`) |
| `item_id` | `str` | `txt_path.stem` (`:309`) |
| `page` | `None` | selalu `None` (`:310`) |
| `chunk_index` | `int` | enumerate index (`:311`) |
| `total_chunks` | `int` | `len(chunks)` (`:312`) |
| `element_type` | `str` | literal `"narrative"` (`:313`) |

Jalur narasi **tidak** menulis `file_hash`, `section`, `extraction_strategy`, maupun `source_type`. Konsekuensinya untuk eksperimen: `_expand_with_neighbors` memfilter dengan `section` (`backend/services/rag_pipeline.py:783-784`) yang tidak ada di chunk narasi, dan `get_indexed_file_hashes()` (`backend/services/indexing.py:85-88`) hanya mencatat point yang punya kedua field `file_name` dan `file_hash` — chunk narasi tidak pernah masuk peta hash sehingga deteksi incremental untuk PDF tidak terganggu, tapi `delete_file_chunks(file_name)` (`:102`) bisa menghapus chunk narasi bila namanya bertabrakan.

**Payload tambahan dari LlamaIndex.** Selain field di atas, `VectorStoreIndex.from_documents()` menulis key internalnya sendiri. Kode di repo ini membuktikannya secara tidak langsung: `backend/services/rag_pipeline.py:810-819` membaca `payload.get("text")` dan, bila kosong, mem-parse `payload["_node_content"]` sebagai JSON untuk mengambil `.text`. Jadi payload nyata berisi minimal `text` dan/atau `_node_content` di luar delapan field di atas. Struktur persisnya tidak diverifikasi terhadap instance Qdrant yang hidup.

### A4. Perlakuan tabel

**HTML tidak dipertahankan sampai Qdrant. Hanya Markdown yang bertahan.**

Rantainya:

1. `backend/services/preprocessing.py:204` — `infer_table_structure=PDF_EXTRACT_TABLES` diteruskan ke `partition_pdf`, sehingga Unstructured menghasilkan `el.metadata.text_as_html`.
2. `backend/services/preprocessing.py:223` — HTML diambil: `html = getattr(el.metadata, "text_as_html", None)`.
3. **`backend/services/preprocessing.py:224` — titik konversi.** `md = _html_table_to_markdown(html) if html else el.text`. Fungsinya di `:178-186`, memakai `markdownify(html, heading_style="ATX")` (`:182`). Bila import atau konversi gagal, fungsi mengembalikan HTML mentah sebagai fallback (`:186`).
4. `backend/services/preprocessing.py:229` — HTML asli **disimpan** di element metadata: `"metadata": {"raw_html": html or ""}`.
5. **`backend/services/preprocessing.py:355-374` — titik pembuangan.** Cabang `Table` di `_chunk_elements` membaca `el["text"]` (Markdown) saja. Chunk yang dihasilkan (`:361-368`) tidak punya key `metadata`, jadi `raw_html` mati di sini. Cabang tabel kecil (`:371-372`) hanya meng-append string Markdown ke buffer.
6. `backend/services/indexing.py:251-260` — whitelist metadata tidak menyertakan `raw_html`.

Perilaku tambahan yang relevan untuk eksperimen:

- Tabel dengan `len(text) > PDF_TABLE_MAX_CHARS` (default 2000, `backend/config.py:111`) menjadi chunk independen dengan prefix `## {section}` (`backend/services/preprocessing.py:357-368`).
- Tabel yang lebih kecil **digabung** ke buffer teks dengan prefix `**Tabel:**` (`:371`), jadi kehilangan batas element dan bercampur dengan paragraf sekitarnya.
- Jalur `fast` (`_extract_fast`, `:156-173`) tidak punya penanganan tabel sama sekali. Tabel native PDF akan keluar sebagai teks datar dari `page.get_text()` (`:132`), dan `detect_strategy` tidak pernah merutekan berdasarkan tabel (lihat A2). PDF yang isinya tabel-tabel teks akan selalu masuk `fast` dan kehilangan struktur.

### A5. Prompt deskripsi gambar

**File:** `backend/services/image_describer.py`
**Definisi:** konstanta modul `_DESCRIPTION_PROMPT` di `backend/services/image_describer.py:24-34`

Bukan template ber-parameter — string statis tanpa `.format()`, tanpa placeholder. Isinya (baris 25-33): instruksi Bahasa Indonesia, batas 150 kata, empat aturan per jenis gambar (diagram/flowchart, tabel, screenshot UI, foto/dekorasi), dan larangan menafsir.

Cara dikirim:

| Provider | Bentuk payload | Lokasi |
|---|---|---|
| vLLM | `messages[0].content[0] = {"type":"text","text": _DESCRIPTION_PROMPT}`, gambar menyusul sebagai `image_url` data-URI | `backend/services/image_describer.py:143-149` |
| Ollama | field `prompt` + `images: [b64]` | `backend/services/image_describer.py:167-171` |

Parameter generasi jalur vLLM: `max_tokens=300`, `temperature=0.1` (`backend/services/image_describer.py:150-151`). Jalur Ollama tidak menyetel keduanya, jadi memakai default model — dua provider menghasilkan distribusi output berbeda dari prompt yang sama.

**Coupling yang harus diperhatikan saat mengubah prompt.** Dua sentinel di prompt di-parse oleh kode: `backend/services/image_describer.py:128` memeriksa `cleaned.upper().startswith("DEKORATIF")` dan `startswith("TIDAK JELAS")`. Bila cocok, gambar dibuang (`:129-130`). Mengubah kata di prompt tanpa mengubah baris 128 akan mematikan filter. Sebaliknya, karena memakai `startswith`, jawaban seperti `"Gambar ini DEKORATIF"` atau `"**DEKORATIF**"` lolos filter dan masuk index sebagai chunk.

Cache deskripsi: `_description_cache` (`backend/services/image_describer.py:37`), dict proses-lokal dengan key SHA-256 16 hex pertama (`:40-41`). Gambar identik hanya dipanggilkan LLM sekali per proses. Cache hilang saat proses restart.

Prompt lain yang **bukan** untuk indexing, agar tidak tertukar: `VISION_RAG_PROMPT` di `backend/prompts/templates.py` dipakai untuk image attachment saat chat runtime (`backend/services/rag_pipeline.py:1016-1020`), dikirim lewat `generate_with_vision()` di `backend/services/vision.py:232-288`. Pemisahan ini didokumentasikan di `backend/services/vision.py:17-18`.

---

## B. KETERGANTUNGAN ANTAR MODUL

### B6. Dependency graph

Diverifikasi dengan grep menyeluruh atas semua `from backend.` di `backend/` dan `scripts/`.

**Klaster indexing:**

```
backend/config.py                     ← tanpa dependensi internal
backend/services/image_describer.py   → config                        [:17]
backend/services/preprocessing.py     → config                        [:23]
                                      → image_describer  (lazy, :274)
backend/services/indexing.py          → config                        [:20]
                                      → preprocessing                 [:26]
backend/services/index_narratives.py  → config, indexing              [import di :40-60 area]
```

**Klaster retrieval/generation:**

```
backend/services/llm_factory.py       → config                        [:10]
backend/services/cache.py             → config                        [:19]
backend/services/moderation.py        → config                        [:28]
                                      → metrics (defensif, :38-41)
backend/services/intent_classifier.py → config                        [:21]
backend/services/output_filter.py     → (stdlib saja)
backend/services/keyword_filter.py    → (stdlib + yaml, baca blocked_keywords.yaml di :31)
backend/services/vision.py            → config                        [:32]
                                      → storage (lazy, :205)
backend/services/rag_pipeline.py      → config                        [:18]
                                      → llm_factory                   [:42]
                                      → metrics                       [:43]
                                      → prompts.templates             [:44]
                                      → indexing (get_qdrant_client)  [:51]
                                      → cache                         [:52]
                                      → moderation                    [:53]
                                      → intent_classifier             [:54]
                                      → output_filter                 [:55]
                                      → keyword_filter                [:56]
                                      → vision                        [:57-58]
                                      → private_api (lazy, :1166/:1422)
```

**Klaster layanan chatbot:**

```
backend/main.py       → config, logging_config, limiter, models.schemas, routers.{chat,auth,query}  [:18-27]
backend/routers/chat.py  → db.database, limiter, config, models.schemas, rag_pipeline, auth, output_filter, session, vision  [:11-25]
backend/routers/query.py → limiter, config, models.schemas, rag_pipeline, output_filter, vision      [:9-14]
backend/routers/auth.py  → services.auth                                                             [:6]
backend/services/session.py → db.models                                                              [:8]
backend/services/auth.py    → db.database, db.models (lazy, :56-57, :100-101)
backend/db/database.py      → config                                                                 [:11]
backend/db/models.py        → db.database                                                            [:13]
```

**Yang bisa dipakai berdiri sendiri tanpa FastAPI / auth / PostgreSQL / Redis / intent classifier:**

| Modul | Berdiri sendiri? | Catatan |
|---|---|---|
| `backend/config.py` | Ya | Hanya `os` + `dotenv` |
| `backend/services/preprocessing.py` | Ya | Import berat (`paddleocr` `:43`, `unstructured` `:194`, `image_describer` `:274`) semuanya lazy. Hanya `fitz` yang hard import (`:21`) |
| `backend/services/image_describer.py` | Ya | Hanya `config` + `httpx` (lazy, `:138`/`:164`) + PIL (lazy, `:48`) |
| `backend/services/indexing.py` | Ya | `config` + `preprocessing` + LlamaIndex/Qdrant |
| `backend/services/llm_factory.py` | Ya | — |
| `backend/services/output_filter.py` | Ya | Stdlib |
| `backend/services/keyword_filter.py` | Ya | Butuh `backend/services/blocked_keywords.yaml` ada; gagal load pertama kali = raise (`:141-143`) |
| `backend/services/cache.py` | Ya | `import redis` lazy di `:34`; kalau Redis tak terjangkau → `cache_get` mengembalikan `None` (`:57-58`), pipeline normal |
| `backend/services/moderation.py` | Ya | Dengan `MODERATION_BACKEND=passthrough` tidak ada HTTP call (`:190-191`) |
| `backend/services/vision.py` | Ya | `storage` lazy (`:205`); Postgres/auth tidak disentuh |
| `backend/services/rag_pipeline.py` | **Ya, dengan syarat** | Lihat di bawah |

`rag_pipeline.py` **tidak** mengimpor FastAPI, tidak mengimpor `backend.services.auth` (referensinya hanya di komentar `:856`), dan tidak mengimpor apa pun dari `backend.db`. Jadi bisa di-`import` tanpa FastAPI/auth/PostgreSQL. Tetapi tiga dependensi bersifat **hard import** (gagal saat import bila paket tidak ada), bukan hard runtime:

- `backend/services/intent_classifier.py:18-19` — `torch` dan `transformers` di top level. Modelnya sendiri opsional: kalau `INTENT_MODEL_PATH` tidak valid, `classify_intent` mengembalikan fallback `get_info_public` dengan `low_confidence=False` (`backend/services/intent_classifier.py:82-88`), sehingga query tetap masuk jalur RAG.
- `backend/metrics.py` — `prometheus_client` di top level; `rag_pipeline.py:43` mengimpornya tanpa guard (berbeda dari `moderation.py:38-41` yang defensif).
- `backend/services/indexing.py:26` → `preprocessing.py:21` → `fitz`. Artinya **mengimpor modul retrieval menarik seluruh stack indexing**, termasuk PyMuPDF.

Ketergantungan terbalik yang lebih mengganggu untuk harness riset: `scripts/index_documents.py:20` mengimpor `_configure_settings` dari `rag_pipeline`. Konsekuensinya menjalankan indexing menarik `torch`, `transformers`, `prometheus_client`, `keyword_filter` (+ YAML-nya), `moderation`, `intent_classifier`, dan `vision` — padahal indexing tidak butuh satu pun dari itu. `_configure_settings` sendiri (`backend/services/rag_pipeline.py:603-625`) hanya memakai `EMBED_*`, `get_llm()`, dan `CHUNK_*`; tidak ada alasan struktural ia berada di `rag_pipeline`.

### B7. File yang bisa dihapus total (indexing + retrieval + generation saja)

**Aman dihapus tanpa mengubah kode lain:**

| Path | Alasan |
|---|---|
| `backend/main.py` | Entry point FastAPI |
| `backend/limiter.py` | Hanya dipakai `main.py:23`, `routers/chat.py:12`, `routers/query.py:9` |
| `backend/routers/` (seluruh direktori: `__init__.py`, `auth.py`, `chat.py`, `query.py`) | Hanya diimpor `main.py:25-27` |
| `backend/db/` (seluruh direktori: `database.py`, `models.py`, `schema.sql`) | Diimpor `main.py:213,291`, `routers/chat.py:11`, `services/session.py:8`, `services/auth.py:56,100` — semuanya di daftar hapus |
| `backend/models/schemas.py` | Hanya diimpor `main.py:24`, `routers/chat.py:14`, `routers/query.py:11` |
| `backend/services/auth.py` | Diimpor `routers/auth.py:6`, `routers/chat.py:19`, `main.py:298` |
| `backend/services/session.py` | Diimpor `routers/chat.py:21` |
| `backend/services/storage.py` | Diimpor lazy `main.py:147`, `services/vision.py:205` |
| `frontend/` | Di-mount `main.py:124` |
| `automation_qa/` | Service terpisah, hanya dirujuk `docker-compose.poc.yml:189-201` |
| `users.json` | Dibaca `services/auth.py:21` |
| `backend/logging_config.py` | Hanya `main.py:22` |

**Butuh edit satu baris sebelum dihapus:**

| Path | Yang menahan |
|---|---|
| `backend/services/vision.py` | `rag_pipeline.py:57-58` mengimpornya di **module level**. Menghapus file tanpa mengedit dua baris ini = `ImportError` saat import. `ProcessedImage` juga dipakai sebagai type hint di `:1062`, `:1300` dan cabang `if images:` di `:1074`, `:1316` |
| `backend/services/private_api.py` | Import lazy (`:1166`, `:1422`), jadi tidak error saat import — tapi akan `ImportError` saat runtime bila intent classifier mengembalikan `get_info_private`. Aman bila `UNHAS_API_BASE_URL` kosong **dan** cabang `intent == "get_info_private"` dihapus, atau bila intent classifier dimatikan |

**Jangan dihapus meski terlihat opsional:**

- `backend/metrics.py` — `rag_pipeline.py:43` mengimpornya tanpa guard.
- `backend/services/keyword_filter.py` + `backend/services/blocked_keywords.yaml` — `rag_pipeline.py:56` module-level, dan `_get_bundle()` **raise** bila load pertama gagal (`backend/services/keyword_filter.py:141-143`).
- `backend/services/moderation.py`, `backend/services/intent_classifier.py`, `backend/services/cache.py`, `backend/services/output_filter.py` — semuanya module-level import di `rag_pipeline.py:52-55`.
- `backend/prompts/templates.py` — `rag_pipeline.py:44-50`.

### B8. Import melingkar dan state global

**Import melingkar: tidak ada.** Grep menyeluruh atas semua `from backend.` menghasilkan graf asiklik. Import lazy yang ada (`preprocessing.py:274`, `vision.py:205`, `rag_pipeline.py:1166`/`:1422`, `auth.py:56`) tampaknya untuk menunda biaya import, bukan untuk memutus siklus.

**State global yang menyulitkan pemakaian sebagai library:**

| Variabel | Lokasi | Masalah |
|---|---|---|
| `_retriever` | `backend/services/rag_pipeline.py:63`, ditulis `:629-643` | Singleton tanpa lock. Mengikat `SIMILARITY_TOP_K` (`:642`) dan nama collection (`:639`) saat pertama dipanggil. **Tidak ada fungsi reset** — mengganti collection atau `top_k` antar kondisi eksperimen dalam satu proses tidak mungkin tanpa memanipulasi global |
| `_reranker` | `backend/services/rag_pipeline.py:64`, ditulis `:710-733` | Sama: mengikat `RERANKER_TOP_N` dan provider saat pertama dipanggil, tanpa reset |
| `Settings.embed_model` / `Settings.llm` | `backend/services/rag_pipeline.py:607`/`:616`/`:623` | Singleton **global proses** milik LlamaIndex. Dipanggil ulang tanpa penjagaan di `:634`, `:999` — memanggil `_configure_settings()` dari dua kondisi eksperimen berbeda saling menimpa secara diam-diam |
| `_ocr_engine` | `backend/services/preprocessing.py:37`, ditulis `:41-51` | Mengikat `OCR_LANG` dan `OCR_USE_GPU` saat pertama; tanpa lock, tanpa reset |
| `_description_cache` | `backend/services/image_describer.py:37` | Tumbuh tanpa batas selama proses hidup. Gambar identik dapat deskripsi identik lintas kondisi eksperimen — bagus untuk determinisme, buruk bila ingin mengukur variansi deskripsi |
| `_redis_client` | `backend/services/cache.py:26` | Tanpa lock (`:30-42`) |
| `_storage_instance` | `backend/services/storage.py:137` | Tanpa lock (`:142-144`) |
| `_bundle_cache` | `backend/services/keyword_filter.py:90` | **Auto-reload tiap 300 detik** (`:31`, `:127`). Mengedit `blocked_keywords.yaml` di tengah eksperimen mengubah perilaku tanpa restart |
| `_circuit` | `backend/services/moderation.py:172` | Nilai `failure_threshold=5, recovery_sec=30.0` di-hardcode di konstruktor. State bertahan lintas request — sekali circuit OPEN, request berikutnya di-bypass |
| `_health` | `backend/services/intent_classifier.py:65` | Circuit 60 detik (`:27`). Sekali unhealthy, 60 detik berikutnya semua query dapat fallback `get_info_public` |
| `_BACKEND` | `backend/services/moderation.py:71` | Dievaluasi **saat import**. Mengubah `MODERATION_BACKEND` lewat `os.environ` setelah import tidak berefek |
| `_MAX_IMAGE_BYTES` | `backend/services/vision.py:48` | Sama: dievaluasi saat import |
| `_JWT_SECRET`, `_JWT_EXPIRE_HOURS` | `backend/services/auth.py:22`, `:24` | Membaca `os.getenv` sendiri, tidak lewat `config.py` — dua sumber kebenaran |

Masalah struktural yang mendasari semuanya: **seluruh `backend/config.py` dievaluasi saat import** (`load_dotenv()` di `:7`, lalu semua `os.getenv` sebagai konstanta modul). Tidak ada fungsi `get_config()` atau objek settings yang bisa dibuat ulang. Untuk harness yang menyapu parameter, ini berarti setiap kondisi eksperimen butuh **proses terpisah** — bukan sekadar mengganti variabel dalam satu proses.

---

## C. HAMBATAN VALIDITAS EKSPERIMEN

### C9. Titik di mana jawaban keluar tanpa retrieval penuh

Ada **13** titik. Untuk `query()` (`backend/services/rag_pipeline.py:1057`); `query_stream()` (`:1295`) menduplikasi struktur yang sama dengan nomor baris berbeda.

| # | Titik | `query()` | `query_stream()` | Cara mematikan |
|---|---|---|---|---|
| 1 | L1 keyword hard-block | `:1101-1102` | `:1368-1370` | **Tidak ada env var.** Harus mengosongkan list `hard_block` di `backend/services/blocked_keywords.yaml` |
| 2 | L2 moderation block | `:1112-1121` | `:1377-1381` | `MODERATION_BACKEND=passthrough` → return dini di `moderation.py:190-191` |
| 3 | Identity override → chitchat | `:1126-1131` | `:1385-1391` | **Tidak ada env var.** Dipicu list hardcoded `_IDENTITY_KEYWORDS` (`:159-166`) via substring match (`:262`) |
| 4 | L3 low confidence → clarification | `:1143-1148` | `:1400-1404` | `INTENT_CONFIDENCE_THRESHOLD=0.0` → `low_confidence` selalu False (`intent_classifier.py:139`) |
| 5 | L3 chitchat | `:1149-1154` | `:1405-1412` | **Tidak ada env var.** Bergantung output classifier |
| 6 | L3 out_of_scope | `:1155-1164` | `:1413-1420` | **Tidak ada env var.** Bergantung output classifier |
| 7 | L3 get_info_private → API eksternal | `:1165-1181` | `:1421-1429` | `UNHAS_API_BASE_URL=""` → `handle_private_query` return `None` (`private_api.py:68-70`) → fall-through ke RAG |
| 8 | Condensation `<ACK>` → chitchat | `:1192-1196` | `:1440-1445` | **Tidak ada env var.** Dipicu bila LLM mengembalikan `<ACK>` atau output < 3 char (`:593-594`) |
| 9 | Re-check L1 pada condensed | `:1199-1200` | `:1446-1449` | Sama dengan #1 |
| 10 | Cache hit | `:1207-1217` | `:1456-1470` | `CACHE_ENABLED=false` → `cache_get` return `None` (`cache.py:54-55`) |
| 11 | Score threshold → low-relevance fallback | `:1229-1241` | `:1479-1491` | `SCORE_THRESHOLD=0.0` mematikan cabang skor, **tapi** kondisi `or not reranked_nodes` (`:1229`) tetap memicu bila retrieval kosong |
| 12 | TEI rerank gagal → urutan dense | `:684`, `:690`, `:696` | sama | **Tidak ada off-switch.** Degradasi diam. Terhitung di counter `RERANK_FALLBACK` (`metrics.py`) dan log warning (`:679`, `:686`, `:692`) |
| 13 | Vision path melompati cache + sebagian L3 | `:1074-1075` → `_vision_query` `:907` | `:1316-1333` | Tidak mengirim `images` |

Dua hal yang perlu ditekankan untuk validitas:

**Titik #12 mencemari perbandingan skor.** Saat TEI gagal, `_TEIRerankPostprocessor` mengembalikan `nodes[:top_n]` **tanpa** menulis ulang `node.score` (`:684`, `:690`, `:696`). Skor yang tersisa adalah skor kemiripan dense dari Qdrant, bukan skor reranker. Baris `:887` lalu menghitung `top_score` dari nilai itu, dan `:1229` membandingkannya dengan `SCORE_THRESHOLD` yang dikalibrasi untuk skala reranker. Dua skala berbeda dibandingkan dengan satu ambang. Kegagalan ini tidak muncul di response — hanya di log dan counter Prometheus.

**Titik #3 memakai substring polos.** `_is_identity_question` (`:255-262`) mencocokkan substring pada teks yang sudah di-lowercase. `_IDENTITY_KEYWORDS` memuat `"meta"`, `"gemini"`, `"bert"`-adjacent, dan `"pa "`-style entri pendek. Query akademik yang kebetulan memuat substring itu akan dibelokkan ke chitchat tanpa pernah menyentuh retrieval.

Cara paling bersih mematikan L1–L3 sekaligus untuk eksperimen: pakai `MODERATION_BACKEND=passthrough`, kosongkan `hard_block` di YAML, dan pastikan classifier mengembalikan `get_info_public`. Jalur terakhir bisa dipaksa dengan membuat `INTENT_MODEL_PATH` menunjuk ke direktori tak valid — `_load_model` gagal, `_fallback_result` mengembalikan `get_info_public` dengan `low_confidence=False` (`intent_classifier.py:82-88`), dan #4/#5/#6 semuanya terlewati. Ini efektif tapi bergantung pada jalur kegagalan, bukan konfigurasi eksplisit; layak dicatat sebagai kelemahan permukaan konfigurasi.

### C10. Nondeterminisme di luar temperature LLM

**Tidak ada seeding sama sekali.** Grep atas seluruh `backend/` dan `scripts/` tidak menemukan `random.seed`, `torch.manual_seed`, `np.random.seed`, maupun parameter `seed` yang dikirim ke vLLM/Ollama. `llm_factory.py:34-53` tidak meneruskan `seed` ke `Ollama` maupun `OpenAILike`.

| Sumber | Lokasi | Dampak |
|---|---|---|
| `random.choice` tanpa seed | `backend/services/rag_pipeline.py:202`, dipanggil `:1102`, `:1200`, `:1369`, `:1447`, `:960` | Memilih dari 4 varian `_HARMFUL_RESPONSES` (`:193-198`). Hanya jalur blocked |
| Temperature hardcoded per jalur | `:347` (chitchat 0.8), `:403` (OOS 0.6), `:486` (low-relevance 0.2) | Tidak dikendalikan `LLM_TEMPERATURE`. Jalur RAG utama (`:1257`) dan condensation (`:574`) memakai default config |
| Urutan `client.scroll` Qdrant | `:787-793` | `_expand_with_neighbors` menambahkan node sesuai urutan kembalian scroll. **Digabung dengan cap keras** `MAX_EXPANDED_CHUNKS` (`:765`, `:828-829`) yang memutus loop di tengah — himpunan chunk yang lolos ekspansi bergantung pada urutan, bukan hanya isi |
| Tie-breaking reranker | `:698-707` | TEI: `results[:top_n]` mengikuti urutan yang dikembalikan service, tanpa tie-break eksplisit. Skor sama → urutan ditentukan implementasi TEI |
| Fallback rerank diam | `:684`, `:690`, `:696` | Timeout `RERANKER_TIMEOUT` (default 5 detik, `config.py:60`) adalah kondisi balapan waktu. Beban mesin bersama → sebagian query di-rerank, sebagian tidak, dalam eksperimen yang sama |
| Mutasi in-place node | `:705` (`node.score = ...`), `:284` (`node.node.text = f"[{clean_name}]\n{...}"`) | `SourceLabelPostprocessor` menyisipkan label nama file **ke dalam teks node**. Teks itu lalu masuk `context_str` (`:1244`) dan `text_preview` sumber (`:899`). Konteks yang dilihat LLM bukan teks chunk apa adanya |
| `_description_cache` | `backend/services/image_describer.py:37`, `:106-107`, `:132` | Deterministik dalam satu proses, hilang antar proses. Indexing yang di-restart di tengah jalan menghasilkan deskripsi berbeda untuk gambar yang sama |
| Auto-reload keyword YAML | `backend/services/keyword_filter.py:31`, `:127` | Perilaku L1 dapat berubah 300 detik setelah file disentuh, tanpa restart |
| Circuit breaker stateful | `backend/services/moderation.py:172`, `backend/services/intent_classifier.py:65` | Query ke-N bisa mendapat perlakuan berbeda dari query ke-1 karena state kegagalan sebelumnya |

**Yang sudah deterministik** (tidak perlu dikhawatirkan): iterasi `set` di `_chunk_elements` di-`sorted()` sebelum join (`preprocessing.py:334`); `seen` set (`:754`, `rag_pipeline.py:754`) hanya dipakai untuk uji keanggotaan; `_RAG_KEYWORDS` set hanya menghasilkan boolean (`:267`); dict Python 3.7+ mempertahankan urutan sisip.

**Concurrency.** `Dockerfile:63` menjalankan `uvicorn --workers 1`, jadi satu proses. Endpoint non-stream `chat`/`query_endpoint` adalah `async def` yang memanggil `query()` sinkron (`routers/chat.py:105`, `routers/query.py:75`) — ini memblokir event loop sehingga request non-stream efektif terserialisasi. Tetapi endpoint stream mengembalikan `StreamingResponse(event_generator())` dengan generator **sinkron** (`routers/chat.py:199`, `routers/query.py:130`), yang dijalankan Starlette di threadpool. Jadi `query_stream` bisa berjalan konkuren, dan semua global lazy di B8 (`_retriever`, `_reranker`, `_ocr_engine`, `_redis_client`) diinisialisasi tanpa lock. Jendela balapannya sempit (hanya saat inisialisasi pertama) tapi nyata: dua request stream bersamaan pada proses dingin dapat membangun dua retriever.

### C11. Query condensation — bisa dimatikan lewat env var?

**Tidak. Harus ubah kode.**

Pemicunya adalah `if history:` di `backend/services/rag_pipeline.py:1188` dan `:1436` — sebuah pemeriksaan kebenaran atas list, bukan flag konfigurasi. Tidak ada `CONDENSE_ENABLED` atau sejenisnya di `backend/config.py`. Grep atas seluruh config tidak menemukan variabel apa pun yang menjangkau blok ini.

Tiga hal yang perlu diketahui sebelum mencoba mengakalinya:

1. **`HISTORY_TURNS=0` tidak mematikannya — malah memperburuk.** Baris `:1189` menghitung `history[-(HISTORY_TURNS * 2):]`. Dengan `HISTORY_TURNS=0` ekspresi menjadi `history[-0:]`, dan `-0 == 0` di Python, sehingga slice-nya `history[0:]` — **seluruh** history, bukan nol. Kondisi `if history:` tetap benar, condensation tetap jalan, dan sekarang memakai riwayat penuh. Pola yang sama ada di `:1245`, `:1437`, `:1495`, dan `:1014`.

2. **Satu-satunya jalan lewat konfigurasi adalah tidak mengirim history.** `POST /api/query` menerima history dari body request (`routers/query.py:66`, `:121`), jadi harness bisa mengirim `history: []` dan condensation terlewati sepenuhnya. `POST /api/chat` mengambil history dari PostgreSQL (`routers/chat.py:95`) sehingga tidak bisa dikosongkan dari sisi klien. Untuk eksperimen single-turn, `/api/query` dengan history kosong sudah cukup.

3. **Condensation punya jalur fallback diam sendiri.** `_condense_question` (`:562-598`) memanggil LLM (`:575`) lalu memvalidasi hasilnya lewat `_validate_condensed` (`:541-559`), yang mengembalikan pertanyaan asli bila output terlalu pendek (`:547`), lebih dari 3× panjang asli (`:550`), atau memuat token kebocoran prompt (`:555`). Ada pula pemangkasan prefiks (`:581-584`) dan pengambilan baris non-kosong pertama (`:587-591`). Jadi query yang sampai ke retrieval kadang hasil condensation, kadang query asli — perbedaannya hanya terlihat di log (`:548`, `:551`, `:557`) dan field `condensed_question` pada response.

Untuk harness, mematikan condensation secara eksplisit lebih baik daripada mengandalkan history kosong, karena jalur multi-turn adalah bagian dari yang mungkin ingin dibandingkan.

---

## D. ASUMSI HARDWARE

### D12. Pin versi yang terikat CUDA / x86 / compute capability

**Peringatan metodologis:** ketersediaan wheel di PyPI tidak dapat diverifikasi dari inspeksi source. Kolom "Risiko aarch64" adalah penilaian berdasarkan sifat paket (ekstensi native vs pure-Python) dan sumber distribusinya, bukan hasil pengecekan indeks paket. Semuanya perlu diverifikasi dengan `pip download --platform manylinux2014_aarch64` sebelum migrasi.

**Terikat CUDA secara eksplisit — dipasang di Dockerfile, bukan requirements.txt:**

| Pin | Lokasi | Masalah pada GB10 |
|---|---|---|
| `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` | `Dockerfile:1` | Base image. CUDA 12.6 vs target CUDA 13.2. Image ini multi-arch (ada varian arm64), tapi versi CUDA-nya tidak cocok |
| `paddlepaddle-gpu==3.3.0` dari `paddlepaddle.org.cn/packages/stable/cu126/` | `Dockerfile:35-36` | **Risiko tertinggi.** Indeks pihak ketiga ini mendistribusikan wheel Linux x86_64. Wheel GPU aarch64 untuk cu126 tidak lazim tersedia di sana. Bahkan jika ada, wheel cu126 tidak memuat kernel sm_121 |
| `torch==2.7.0+cu126` | `Dockerfile:44` | PyTorch memublikasikan wheel aarch64 untuk sebagian rilis CUDA, tapi **build cu126 tidak memuat kernel sm_121**. Blackwell GB10 membutuhkan build CUDA 12.8+ dengan `TORCH_CUDA_ARCH_LIST` yang menyertakan `12.1` |
| `torchvision==0.22.0+cu126` | `Dockerfile:45` | Terikat ke versi torch di atas |

**Di requirements.txt — terikat CUDA/GPU secara transitif:**

| Pin | Baris | Sifat | Risiko aarch64 |
|---|---|---|---|
| `paddleocr==3.0.3` | `requirements.txt:81` | Pure-Python, tapi hard-depend `paddlepaddle` | **Tinggi** — mewarisi masalah paddlepaddle-gpu |
| `paddlex==3.0.3` | `:82` | Menarik pohon dependensi berat | **Tinggi** — sama |
| `sentence-transformers==3.3.1` | `:38` | Pure-Python, butuh torch | Sedang — bergantung torch |
| `FlagEmbedding==1.4.0` | `:39` | Pure-Python, butuh torch | Sedang — bergantung torch |
| `transformers==4.47.1` | `:44` | Pure-Python, butuh torch | Sedang — bergantung torch |

**Ekstensi native tanpa keterikatan CUDA — perlu wheel aarch64:**

| Pin | Baris | Jenis ekstensi | Risiko aarch64 |
|---|---|---|---|
| `numpy==1.26.4` | `:50` | C | Rendah — wheel aarch64 lazim. **Tapi ini jangkar constraint**: komentar `:48-49` menjelaskan pin ini menahan Paddle dan `unstructured` |
| `tokenizers==0.21.0` | `:45` | Rust | Rendah–sedang |
| `pymupdf==1.24.14` | `:83` | C (MuPDF) | Sedang |
| `unstructured[pdf]==0.16.11` | `:90` | Menarik `onnxruntime`, `opencv-python`, `pikepdf`, model layout | **Sedang–tinggi** — pohon dependensi paling lebar di file ini; satu wheel yang hilang mematikan seluruh jalur hi_res |
| `pillow==11.0.0` | `:93` | C | Rendah |
| `psycopg2-binary==2.9.10` | `:122` | C (libpq) | Rendah — hanya relevan bila Postgres dipertahankan |
| `bcrypt==4.2.1` | `:99` | Rust | Rendah — hanya untuk auth |
| `python-jose[cryptography]==3.3.0` | `:98` | Menarik `cryptography` (Rust) | Rendah — hanya untuk auth |
| `pydantic==2.11.7` | `:131` | `pydantic-core` (Rust) | Rendah |
| `qdrant-client==1.17.1` | `:56` | Menarik `grpcio` (C++) | Rendah–sedang |
| `pdf2image==1.17.0` | `:84` | Pure-Python, butuh binary `poppler` | Rendah — `poppler-utils` di-apt-install (`Dockerfile:23`), tersedia arm64 |
| `pdfplumber==0.11.4` | `:91` | Pure-Python + Pillow | Rendah |

**Terikat compute capability, di luar requirements.txt:** lihat D13 untuk tag image `text-embeddings-inference:89-latest`, yang merupakan keterikatan sm_89 paling eksplisit di seluruh repo ini.

### D13. Image di docker-compose.poc.yml

| Service | Image + tag | Baris | Arsitektur-spesifik? |
|---|---|---|---|
| `postgres` | `postgres:16` | `:15` | Tidak — official image multi-arch (amd64 + arm64) |
| `redis` | `redis/redis-stack:latest` | `:30` | **Perlu verifikasi.** `redis-stack` (varian dengan RedisInsight) secara historis lebih terbatas dukungan arm64-nya dibanding `redis-stack-server`. Perintahnya di `:31` justru menjalankan `redis-stack-server` |
| `qdrant` | `qdrant/qdrant:latest` | `:37` | Tidak — publikasi multi-arch |
| `minio` | `minio/minio:latest` | `:45` | Tidak — publikasi multi-arch |
| `vllm` | `vllm/vllm-openai:latest` | `:62` | **Ya, tinggi.** Tag default vLLM dibangun untuk x86_64. Dukungan aarch64 (GH200/GB200) memerlukan image varian terpisah atau build dari source, dan build itu juga harus menyertakan sm_121 |
| `tei-embed` | `ghcr.io/huggingface/text-embeddings-inference:89-latest` | `:94` | **Ya, eksplisit.** Angka `89` pada tag **adalah** compute capability 8.9 (Ada Lovelace / L40S). GB10 adalah sm_121. Tag ini secara definisi tidak cocok |
| `tei-rerank` | `ghcr.io/huggingface/text-embeddings-inference:89-latest` | `:110` | **Ya, eksplisit.** Sama dengan di atas |
| `ollama` | `ollama/ollama:latest` | `:128` | Tidak — publikasi multi-arch |
| `backend` | build dari `Dockerfile` | `:143-145` | **Ya** — mewarisi seluruh masalah D12 |
| `qa-sheet-sync` | build dari `automation_qa/Dockerfile` | `:191-192` | Tidak diinspeksi; di luar cakupan indexing/retrieval |

Konsekuensi paling langsung: `tei-embed` dan `tei-rerank` memakai tag yang mengunci compute capability 8.9. Kedua service ini menyediakan `EMBED_PROVIDER=tei` dan `RERANKER_PROVIDER=tei`, artinya **embedding dan reranking dua-duanya berhenti** bila tag tidak diganti — dan bersamanya seluruh jalur retrieval. Repo ini punya jalur alternatif in-process (`EMBED_PROVIDER=huggingface` di `rag_pipeline.py:616`, `RERANKER_PROVIDER=sentence_transformers` di `:727`) yang memindahkan beban ke torch di dalam container backend — tapi jalur itu bergantung pada torch yang memuat kernel sm_121, yang menurut D12 juga belum terpenuhi.

### D14. Tempat yang mengasumsikan VRAM terpisah dari RAM sistem

**Tidak ada pengecekan memori manual di kode Python.** Grep atas `nvidia-smi`, `torch.cuda.mem_get_info`, `memory_allocated`, dan pemeriksaan VRAM eksplisit tidak menemukan apa pun. Semua asumsi berada di konfigurasi dan komentar.

| Lokasi | Isi | Mengapa bermasalah di unified memory |
|---|---|---|
| `docker-compose.poc.yml:67` | `--gpu-memory-utilization 0.80` | Paling kritis. Pada GPU diskret 0.80 berarti 80% dari 48 GB VRAM khusus. Pada GB10 kolam 128 GB **dibagi dengan sistem operasi dan proses lain**. vLLM mem-preallocate KV cache berdasarkan fraksi ini — 0.80 dari 128 GB ≈ 102 GB dicaplok di muka, menyisakan ~26 GB untuk OS, container lain, dan peneliti lain di mesin yang sama |
| `docker-compose.poc.yml:80-86`, `:101-107`, `:117-123`, `:132-138`, `:178-185` | Lima service masing-masing me-reserve `count: 1` GPU | Pada mesin satu-GPU, semuanya menunjuk perangkat yang sama. `count: 1` adalah pernyataan penempatan, **bukan** partisi memori — Docker tidak menegakkan batas memori GPU. Anggaran gabungan (vLLM 0.80 + TEI embed + TEI rerank + Ollama llama-guard + backend PaddleOCR/torch/IndoBERT) tidak dihitung di mana pun |
| `docker-compose.poc.yml:89` | Komentar `VRAM ~1.5 GB (embed) + ~2.3 GB (rerank) = ~4 GB total` | Angka kalibrasi L40S. Tidak ada penegakan; hanya dokumentasi |
| `docker-compose.poc.yml:90-91` | Komentar tentang beralih ke image `cpu-latest` bila VRAM ketat | Menganggap CPU dan GPU punya kolam memori terpisah — pada unified memory pemisahan itu tidak bermakna |
| `.env.poc:139` | Komentar `POC: true (backend container punya akses GPU, ~1-2 GB VRAM peak)` | Anggaran VRAM PaddleOCR, tidak ditegakkan |
| `scripts/index_documents.py:9-10` | `Jika VRAM penuh, matikan Ollama dulu saat indexing: ollama stop qwen2.5:7b` | Prosedur manual yang mengasumsikan kompetisi VRAM. Pada unified memory tekanannya berpindah ke seluruh sistem, bukan hanya ke GPU |
| `backend/config.py:43` | `EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cuda")` | Default `cuda`; dipakai `rag_pipeline.py:618` dan `index_narratives.py`. Tidak ada auto-fallback ke CPU |
| `backend/config.py:84` | `OCR_USE_GPU` default `true` | Diteruskan sebagai `device="gpu"` (`preprocessing.py:50`). Tidak ada auto-fallback |
| `backend/config.py:65` | `RERANKER_USE_FP16 = True` | **Dead code** — tidak dibaca di mana pun. Kalau dimaksudkan sebagai penghematan memori, ia tidak berefek |
| `backend/services/intent_classifier.py:71` | `torch.device("cuda" if torch.cuda.is_available() else "cpu")` | Satu-satunya tempat yang auto-fallback dengan benar |

Untuk mesin bersama, tidak adanya batas memori GPU per-container di seluruh compose adalah masalah operasional yang berdiri sendiri, terlepas dari migrasi arsitektur.

### D15. Apakah PaddleOCR wajib?

**Tidak wajib per halaman, tetapi tidak ada saklar konfigurasi untuk mematikannya, dan kegagalannya menjatuhkan seluruh file.**

Jalur kode yang melewati PaddleOCR:

1. **Import bersifat lazy.** `from paddleocr import PaddleOCR` ada di dalam `_get_ocr_engine()` (`backend/services/preprocessing.py:43`), bukan di top-level modul. Mengimpor `preprocessing.py` tidak menyentuh Paddle.
2. **Dipanggil hanya bersyarat.** `_extract_text_from_page_fast` (`:130-153`) memanggilnya di `:145` **hanya** bila `len(page.get_text().strip()) <= 50` (`:132-134`). Halaman dengan lapisan teks langsung `return text` di `:134` dan tidak pernah menyentuh OCR.
3. **Jalur hi_res tidak memakai Paddle sama sekali.** `_extract_hi_res` (`:189-256`) memanggil `partition_pdf` dari Unstructured, yang memakai **tesseract-ocr** (dipasang di `Dockerfile:24`) — engine yang sama sekali berbeda.

Tiga hal yang membuatnya tetap menjadi hambatan:

**Tidak ada env var untuk mematikannya.** `backend/config.py` hanya menyediakan `OCR_LANG` (`:82`, hardcoded `"id"`, tidak dibaca dari environment) dan `OCR_USE_GPU` (`:84`). Tidak ada `OCR_ENABLED`. Satu-satunya kendali adalah `OCR_USE_GPU=false` yang memindahkan Paddle ke CPU — tetap membutuhkan paket terpasang.

**Kegagalan OCR menjatuhkan seluruh PDF, bukan satu halaman.** Blok `try` di `:144-153` hanya punya `finally` (`:152`) untuk membersihkan file temporer — **tidak ada `except`**. Exception apa pun dari `_get_ocr_engine()` atau `ocr.predict()` merambat naik melalui `_extract_fast` (`:156-173`, tanpa try) ke `extract_from_pdf` (`:412-443`, tanpa try), dan baru ditangkap di `backend/services/indexing.py:236-238` — yang me-log error lalu `continue`, **melewati file itu seluruhnya**. Satu halaman scan di halaman 40 dari PDF 100 halaman membuang 99 halaman lain.

**Jalur hi_res bisa jatuh balik ke Paddle.** `:209-211` menangkap kegagalan `partition_pdf` dan memanggil `_extract_fast(pdf_path)`. Jadi bahkan `PDF_EXTRACTION_STRATEGY=hi_res` tidak menjamin Paddle terhindar.

Konsekuensi praktis untuk migrasi: bila wheel `paddlepaddle-gpu` aarch64/sm_121 tidak tersedia (D12), pilihannya adalah (a) memastikan setiap halaman di korpus punya lapisan teks >50 karakter, (b) menambahkan `except` di `:144` sehingga kegagalan OCR mendegradasi ke teks kosong per halaman alih-alih membuang file, atau (c) mengganti engine. Ketiganya butuh perubahan kode; tidak ada yang bisa dicapai lewat konfigurasi.

---

## E. PERMUKAAN KONFIGURASI

### E16. Semua env var di config.py

`backend/config.py:7` memanggil `load_dotenv()`; seluruh file dievaluasi saat import (lihat B8).

| Nama | Baris | Default | Tipe | Dibaca oleh |
|---|---|---|---|---|
| — (`DATA_DIR`) | `:11` | `<root>/data/pdfs` | `str` | `services/indexing.py`, `scripts/index_documents.py` |
| — (`IMAGES_DIR`) | `:12` | `<root>/data/images` | `str` | `services/preprocessing.py:26` — **diimpor tapi tidak pernah dipakai** |
| `LLM_PROVIDER` | `:21` | `"ollama"` | `str` | `models/schemas.py`, `main.py`, `services/vision.py`, `services/image_describer.py`, `services/llm_factory.py` |
| `LLM_MODEL` | `:22` | `"qwen2.5:7b"` | `str` | `main.py`, `services/vision.py`, `services/rag_pipeline.py`, `services/llm_factory.py`, `services/image_describer.py` |
| `LLM_BASE_URL` / `OLLAMA_BASE_URL` | `:23` | `"http://localhost:11434"` | `str` | `main.py`, `services/vision.py`, `services/llm_factory.py`, `services/image_describer.py` |
| `LLM_API_KEY` | `:24` | `"not-needed"` | `str` | `services/llm_factory.py` |
| `LLM_TEMPERATURE` | `:25` | `0.1` | `float` | `services/llm_factory.py`, `services/vision.py` |
| `LLM_MAX_TOKENS` | `:26` | `1024` | `int` | `services/vision.py`, `services/llm_factory.py` |
| `LLM_REQUEST_TIMEOUT` | `:27` | `120` | `int` | `services/vision.py`, `services/llm_factory.py` |
| `LLM_SUPPORTS_VISION` | `:28` | `false` | `bool` | `routers/query.py`, `models/schemas.py`, `services/vision.py`, `prompts/templates.py`, `services/image_describer.py`, `services/preprocessing.py` |
| `EMBED_PROVIDER` | `:40` | `"huggingface"` | `str` | `services/rag_pipeline.py`, `services/index_narratives.py` |
| `EMBED_MODEL` | `:41` | `"Qwen/Qwen3-Embedding-0.6B"` | `str` | `services/rag_pipeline.py`, `services/index_narratives.py` |
| `EMBED_BASE_URL` | `:42` | `""` | `str` | `services/rag_pipeline.py`, `services/index_narratives.py` |
| `EMBED_DEVICE` | `:43` | `"cuda"` | `str` | `services/rag_pipeline.py`, `services/index_narratives.py` |
| `EMBED_BATCH_SIZE` | `:44` | `8` | `int` | `services/rag_pipeline.py`, `services/index_narratives.py` |
| `EMBED_TIMEOUT` | `:45` | `600` | `int` | `services/rag_pipeline.py`, `services/index_narratives.py` |
| — (`EMBED_DIMENSION`) | `:46` | `1024` | `int` | `services/indexing.py`, `services/index_narratives.py` — **konstanta, bukan env var** |
| `RERANKER_PROVIDER` | `:55` | `"sentence_transformers"` | `str` | `services/rag_pipeline.py` |
| `RERANKER_MODEL` | `:56` | `<root>/models/bge-reranker-v2-m3` | `str` | `services/rag_pipeline.py` |
| `RERANKER_BASE_URL` | `:57` | `""` | `str` | `services/rag_pipeline.py` |
| `RERANKER_TOP_N` | `:58` | `6` | `int` | `services/rag_pipeline.py` |
| `RERANKER_TIMEOUT` | `:60` | `5` | `float` | `services/rag_pipeline.py` |
| `LOW_CONFIDENCE_BUFFER` | `:64` | `0.15` | `float` | `models/schemas.py`, `services/rag_pipeline.py` |
| — (`RERANKER_USE_FP16`) | `:65` | `True` | `bool` | **Tidak dibaca di mana pun** |
| `QDRANT_URL` | `:74` | `"http://localhost:6333"` | `str` | `main.py`, `services/indexing.py`, `services/index_narratives.py` |
| `QDRANT_COLLECTION` / `QDRANT_COLLECTION_NAME` | `:75-76` | `"unhas_docs"` | `str` | `services/indexing.py`, `services/rag_pipeline.py`, `services/index_narratives.py` (lewat alias `QDRANT_COLLECTION_NAME`) |
| — (`OCR_LANG`) | `:82` | `"id"` | `str` | `services/preprocessing.py` — **hardcoded, bukan env var** |
| `OCR_USE_GPU` | `:84` | `true` | `bool` | `services/preprocessing.py` |
| `CHUNK_SIZE` | `:90` | `512` | `int` | `services/rag_pipeline.py`, `services/preprocessing.py` |
| `CHUNK_OVERLAP` | `:91` | `128` | `int` | `services/preprocessing.py`, `services/rag_pipeline.py` |
| `PDF_EXTRACTION_STRATEGY` | `:101` | `"auto"` | `str` | `services/preprocessing.py` |
| `PDF_EXTRACT_IMAGES` | `:104` | `true` | `bool` | `services/preprocessing.py` |
| `PDF_DESCRIBE_IMAGES` | `:105` | `"auto"` | `str` | `services/preprocessing.py` |
| `PDF_MIN_IMAGE_SIZE_KB` | `:106` | `20` | `int` | `services/image_describer.py` |
| `PDF_MAX_IMAGE_DIM` | `:107` | `1280` | `int` | `services/image_describer.py` |
| `PDF_EXTRACT_TABLES` | `:110` | `true` | `bool` | `services/preprocessing.py` |
| `PDF_TABLE_MAX_CHARS` | `:111` | `2000` | `int` | `services/preprocessing.py` |
| `SIMILARITY_TOP_K` | `:117` | `12` | `int` | `services/rag_pipeline.py` |
| `SCORE_THRESHOLD` | `:118` | `0.3` | `float` | `services/rag_pipeline.py` |
| `NEIGHBOR_EXPANSION_ENABLED` | `:123` | `true` | `bool` | `services/rag_pipeline.py` |
| `NEIGHBOR_EXPANSION_RADIUS` | `:124` | `2` | `int` | `services/rag_pipeline.py` |
| `MAX_EXPANDED_CHUNKS` | `:125` | `30` | `int` | `services/rag_pipeline.py` |
| `HISTORY_TURNS` | `:131` | `5` | `int` | `services/rag_pipeline.py` |
| `HISTORY_MAX_TOKENS` | `:132` | `1500` | `int` | `services/rag_pipeline.py` |
| `MAX_IMAGES_PER_MESSAGE` | `:138` | `2` | `int` | `models/schemas.py`, `services/vision.py` |
| `MAX_IMAGE_SIZE_MB` | `:139` | `10` | `int` | `models/schemas.py`, `services/vision.py` |
| — (`ALLOWED_IMAGE_TYPES`) | `:140` | `{jpeg, png, webp}` | `set` | `models/schemas.py`, `services/vision.py` — **konstanta** |
| `IMAGE_RESIZE_MAX_DIM` | `:141` | `1280` | `int` | `services/vision.py` |
| `DATABASE_URL` | `:149` | `postgresql://ragchat:dev@localhost:5432/ragchat` | `str` | `db/database.py` |
| `REDIS_URL` | `:157` | `"redis://localhost:6379/0"` | `str` | `services/cache.py` |
| `CACHE_ENABLED` | `:158` | `true` | `bool` | `services/cache.py` |
| `CACHE_TTL_SECONDS` | `:159` | `86400` | `int` | `services/cache.py` |
| `JWT_SECRET` | `:165` | `"unhas-rag-demo-secret-2026"` | `str` | `services/auth.py` — **tapi lewat `os.getenv` sendiri di `auth.py:22`, bukan lewat config** |
| `JWT_ALGORITHM` | `:166` | `"HS256"` | `str` | **Tidak dibaca.** `auth.py:23` meng-hardcode `"HS256"` |
| `JWT_EXPIRE_HOURS` | `:167` | `24` | `int` | `services/auth.py` — juga lewat `os.getenv` sendiri di `auth.py:24` |
| `INTENT_MODEL_PATH` | `:173-176` | `<root>/models/intent_classifier` | `str` | `main.py`, `services/intent_classifier.py` |
| `INTENT_CONFIDENCE_THRESHOLD` | `:177` | `0.6` | `float` | `services/intent_classifier.py` (di `rag_pipeline.py:988` hanya muncul dalam komentar) |
| `MODERATION_BACKEND` | `:189` | `"passthrough"` | `str` | `services/moderation.py` |
| `MODERATION_MODEL` | `:190` | `"llama-guard3:1b"` | `str` | `services/moderation.py` |
| `MODERATION_BASE_URL` | `:191` | `LLM_BASE_URL` | `str` | `services/moderation.py` |
| `MODERATION_TIMEOUT` | `:192` | `20` | `int` | `services/moderation.py` |
| `RATE_LIMIT_TEXT_PER_MINUTE` | `:198` | `10` | `int` | `routers/chat.py`, `routers/query.py` |
| `RATE_LIMIT_VISION_PER_MINUTE` | `:199` | `3` | `int` | **Tidak dibaca di mana pun** |
| `ALLOWED_ORIGINS` | `:205-212` | 3 origin localhost | `list[str]` | `main.py` |
| `LOG_LEVEL` | `:218` | `"INFO"` | `str` | `main.py` |
| `UNHAS_API_BASE_URL` | `:224` | `""` | `str` | `routers/query.py`, `services/private_api.py` |

**Env var yang dibaca di luar `config.py`** (tidak muncul di tabel di atas — sumber kebenaran terpisah):

| Nama | Lokasi | Default |
|---|---|---|
| `STORAGE_BACKEND` | `services/storage.py:146` | `"filesystem"` |
| `STORAGE_LOCAL_PATH` | `services/storage.py:148` | `"./storage"` |
| `MINIO_ENDPOINT` | `services/storage.py:152` | `"minio:9000"` |
| `MINIO_ACCESS_KEY` | `services/storage.py:153` | `""` |
| `MINIO_SECRET_KEY` | `services/storage.py:154` | `""` |
| `MINIO_BUCKET` | `services/storage.py:155` | `"ragchat-images"` |
| `MINIO_SECURE` | `services/storage.py:156` | `"false"` |
| `BACKEND_PUBLIC_URL` | `services/storage.py:53`, `:157` | `"http://localhost:8000"` |
| `JWT_SECRET`, `JWT_EXPIRE_HOURS` | `services/auth.py:22`, `:24` | duplikat dari config |
| `QA_SYNC_*` | `automation_qa/config.py:36-48` | service terpisah |

Semua `STORAGE_*`/`MINIO_*` di atas dikomentari-mati di `backend/config.py:230-235` — didefinisikan di sana dulu, lalu dipindahkan ke `storage.py` tanpa menghapus komentarnya.

**Env var di `.env.poc` yang tidak dibaca kode mana pun** (dead config — menyesatkan bila dipakai sebagai dokumentasi eksperimen):

| Nama | Lokasi di `.env.poc` | Realitas |
|---|---|---|
| `CACHE_SIMILARITY_THRESHOLD` | — | Tier-2 semantic cache masih TODO (`services/cache.py:4`) |
| `CACHE_SKIP_IF_HAS_IMAGE` | `:75` | Perilaku ini di-hardcode di `rag_pipeline.py:1074-1075`; komentar `vision.py:16` keliru mengklaim ini berasal dari config |
| `CIRCUIT_OPEN_THRESHOLD` | `:83` | Di-hardcode `failure_threshold=5` di `moderation.py:172` |
| `CIRCUIT_RECOVERY_THRESHOLD` | `:84` | Di-hardcode `recovery_sec=30.0` di `moderation.py:172` |
| `RATE_LIMIT_BURST_PER_SECOND` | `:80` | Tidak dibaca |

### E17. Variabel yang memengaruhi hasil indexing — wajib dilaporkan di paper

Dipisah menjadi yang mengubah isi index (wajib), yang mengubah cara query dijawab (wajib bila retrieval dievaluasi end-to-end), dan yang tidak relevan.

**Kelompok 1 — mengubah isi dan struktur chunk yang tersimpan. Semua wajib.**

| Var | Default | Mengapa penting |
|---|---|---|
| `PDF_EXTRACTION_STRATEGY` | `auto` | Menentukan `fast` vs `hi_res` per file (`preprocessing.py:426-436`). `auto` membuat pemilihan bergantung isi PDF — nilainya harus dilaporkan **beserta** distribusi strategi aktual per dokumen, karena `auto` tidak dapat direproduksi hanya dari nilai variabelnya |
| `CHUNK_SIZE` | `512` | Ambang flush buffer (`preprocessing.py:393`) |
| `CHUNK_OVERLAP` | `128` | Panjang tail yang dibawa antar chunk (`preprocessing.py:397`) |
| `PDF_EXTRACT_TABLES` | `true` | Diteruskan sebagai `infer_table_structure` (`preprocessing.py:204`). False = tidak ada tabel terstruktur |
| `PDF_TABLE_MAX_CHARS` | `2000` | Menentukan tabel jadi chunk sendiri atau digabung ke teks (`preprocessing.py:357`) |
| `PDF_EXTRACT_IMAGES` | `true` | Menentukan `extract_image_block_types` (`preprocessing.py:197-198`) |
| `PDF_DESCRIBE_IMAGES` | `auto` | Menentukan gambar jadi chunk atau dibuang (`preprocessing.py:266-272`) |
| `LLM_SUPPORTS_VISION` | `false` | Bersama `PDF_DESCRIBE_IMAGES=auto`, menentukan `should_describe` (`preprocessing.py:268`). **Default `false`** — bila tidak diubah, semua gambar dibuang diam-diam |
| `PDF_MIN_IMAGE_SIZE_KB` | `20` | Ambang buang gambar (`image_describer.py:76`) |
| `PDF_MAX_IMAGE_DIM` | `1280` | Resize sebelum dikirim ke VL (`image_describer.py:44`) — mengubah resolusi yang dilihat model, jadi mengubah isi deskripsi |
| `OCR_LANG` | `"id"` | **Hardcoded di `config.py:82`, bukan env var.** Tetap wajib dilaporkan sebagai konstanta |
| `OCR_USE_GPU` | `true` | Menentukan `device` PaddleOCR (`preprocessing.py:50`). CPU vs GPU dapat menghasilkan output numerik berbeda |
| `EMBED_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Model embedding |
| `EMBED_PROVIDER` | `huggingface` | **Jalur `tei` dan `huggingface` bukan setara.** `tei` mengirim HTTP ke service (`rag_pipeline.py:607-612`), `huggingface` memuat in-process dengan `trust_remote_code=True` (`:616-621`). Pooling dan normalisasi bisa berbeda; vektor tidak dijamin identik |
| `EMBED_BATCH_SIZE` | `8` | Ukuran batch dapat mengubah hasil pada level presisi floating-point |
| `EMBED_DEVICE` | `cuda` | Hanya untuk jalur `huggingface`. CPU vs GPU = presisi berbeda |
| `EMBED_DIMENSION` | `1024` | Konstanta (`config.py:46`). Menentukan `VectorParams` collection (`indexing.py:131`) |
| `QDRANT_COLLECTION` | `unhas_docs` | Identitas collection. **Wajib** karena jalur narasi menulis ke collection yang sama dengan skema berbeda (A3) |
| `LLM_MODEL` + `LLM_PROVIDER` + `LLM_BASE_URL` | — | Model vision yang menulis deskripsi gambar (`image_describer.py:112-118`). Model berbeda = teks chunk berbeda |

Konstanta yang tidak dapat dikonfigurasi tapi tetap wajib dilaporkan karena menentukan isi index: ambang OCR fallback 50 karakter (`preprocessing.py:133`), DPI render OCR 300 (`preprocessing.py:137`), ambang `detect_strategy` 1.0 gambar/halaman dan sampel 5 halaman pertama (`preprocessing.py:112`, `:122`), `max_tokens=300` dan `temperature=0.1` pada deskripsi gambar jalur vLLM (`image_describer.py:150-151`), serta prompt deskripsi gambar itu sendiri (`image_describer.py:24-34`).

**Kelompok 2 — mengubah jawaban tanpa mengubah index. Wajib bila metrik diukur end-to-end.**

`SIMILARITY_TOP_K` (`:117`), `RERANKER_PROVIDER` / `RERANKER_MODEL` / `RERANKER_TOP_N` / `RERANKER_TIMEOUT` (`:55-60`), `SCORE_THRESHOLD` (`:118`), `LOW_CONFIDENCE_BUFFER` (`:64`), `NEIGHBOR_EXPANSION_ENABLED` / `NEIGHBOR_EXPANSION_RADIUS` / `MAX_EXPANDED_CHUNKS` (`:123-125`), `CACHE_ENABLED` (`:158`), `HISTORY_TURNS` (`:131`), `INTENT_CONFIDENCE_THRESHOLD` (`:177`), `MODERATION_BACKEND` (`:189`), `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` (`:25-26`).

Tiga di antaranya butuh catatan khusus di paper: `RERANKER_TIMEOUT` karena timeout mengubah skala skor tanpa memberi sinyal ke pemanggil (C9 #12); `NEIGHBOR_EXPANSION_*` karena berinteraksi dengan urutan scroll Qdrant (C10); `CACHE_ENABLED` karena `true` membuat query berulang tidak menyentuh retrieval sama sekali (C9 #10).

**Tidak relevan untuk reproduksibilitas indexing:** `DATABASE_URL`, `REDIS_URL`, `JWT_*`, `RATE_LIMIT_*`, `ALLOWED_ORIGINS`, `LOG_LEVEL`, `UNHAS_API_BASE_URL`, `STORAGE_*`, `MINIO_*`, `MAX_IMAGES_PER_MESSAGE`, `MAX_IMAGE_SIZE_MB`, `IMAGE_RESIZE_MAX_DIM` (jalur chat runtime, bukan indexing), `QA_SYNC_*`.

---

## RISIKO TERTINGGI

Diurutkan dari yang paling mungkin menggagalkan adaptasi ini.

**1. Tag TEI mengunci compute capability 8.9 — mematikan embedding dan reranking sekaligus.**
`ghcr.io/huggingface/text-embeddings-inference:89-latest` di `docker-compose.poc.yml:94` dan `:110`. Angka `89` pada tag adalah sm_89 (Ada/L40S); GB10 adalah sm_121. Dua service ini menyediakan `EMBED_PROVIDER=tei` dan `RERANKER_PROVIDER=tei` — keduanya mati berarti seluruh jalur retrieval mati. Jalur alternatif in-process (`huggingface` + `sentence_transformers`) memindahkan beban ke torch, yang menurut risiko #3 juga belum siap. Ini yang paling pasti dan paling mudah dikonfirmasi lebih dulu.

**2. `paddlepaddle-gpu` 3.3.0 dari indeks cu126 x86, tanpa jalan konfigurasi untuk melewatinya.**
`Dockerfile:35-36` memasangnya dari `paddlepaddle.org.cn/packages/stable/cu126/`, indeks yang mendistribusikan wheel x86_64. Tidak ada `OCR_ENABLED` di `backend/config.py`, dan kegagalan OCR tidak ditangkap per halaman: blok `try` di `preprocessing.py:144-153` hanya punya `finally`, sehingga exception merambat sampai `indexing.py:236` yang **melewati seluruh PDF**. Satu halaman scan membuang seluruh dokumen. Melewatinya butuh perubahan kode, bukan konfigurasi.

**3. torch 2.7.0+cu126 tidak memuat kernel sm_121.**
`Dockerfile:44-45`. Blackwell GB10 butuh build CUDA 12.8+ dengan arsitektur 12.1 disertakan. Yang bergantung padanya: intent classifier (`intent_classifier.py:18`, `:71`), embedding in-process (`rag_pipeline.py:616-621`), reranker in-process (`rag_pipeline.py:727-731`), `sentence-transformers`, `FlagEmbedding`, `transformers`. Menaikkan torch akan menabrak pin `numpy==1.26.4` (`requirements.txt:50`) yang komentarnya sendiri (`:48-49`) menjelaskan ada demi Paddle dan `unstructured` — satu perubahan menarik tiga pin lain.

**4. Gambar asli tidak pernah menyentuh disk — persis kebutuhan riset yang belum ada.**
Base64 hidup di memori dari `preprocessing.py:206` sampai dibuang di `:300` (`"metadata": {}`). `IMAGES_DIR` sudah didefinisikan (`config.py:12`) dan diimpor (`preprocessing.py:26`) tapi tidak pernah dipakai. Menambahkannya menyentuh enam titik (A2) dan berinteraksi dengan tiga cabang `continue` yang membuang gambar sebelum sempat ditulis (`:272`, `:290`, `:294`). Ditambah: jalur `fast` tidak mengekstrak gambar sama sekali, dan `detect_strategy` merutekan PDF hasil scan ke `fast` (ambang `> 1.0` strict, `preprocessing.py:122`).

**5. HTML tabel dibuang di chunking, dan `detect_strategy` tidak pernah melihat tabel.**
`raw_html` disimpan di element metadata (`preprocessing.py:229`) lalu dibuang karena `_chunk_elements` tidak meneruskan metadata untuk cabang Table (`:361-368`) dan `indexing.py:251-260` mem-whitelist field. Diperparah: docstring `detect_strategy` menjanjikan routing berdasarkan tabel (`:105`) tapi kodenya hanya menghitung `get_images()` (`:116`), sehingga PDF penuh tabel teks selalu masuk `fast` — yang tidak punya penanganan tabel sama sekali. Kebutuhan riset akan HTML relasional membutuhkan perbaikan di kedua tempat.

**6. Konfigurasi dievaluasi saat import, dan singleton retrieval tidak punya reset — satu kondisi eksperimen per proses.**
`backend/config.py` mengevaluasi seluruh `os.getenv` sebagai konstanta modul saat import (`:7` dan seterusnya). `_retriever` (`rag_pipeline.py:63`) mengikat `SIMILARITY_TOP_K` dan nama collection saat pertama dipanggil (`:639`, `:642`); `_reranker` (`:64`) mengikat `RERANKER_TOP_N`; `Settings.embed_model` adalah singleton global LlamaIndex (`:607`, `:616`). Tidak ada fungsi reset untuk satu pun. Menyapu parameter retrieval memerlukan proses terpisah per kondisi, atau refactor konfigurasi. Ditambah beban import yang tidak perlu: `scripts/index_documents.py:20` menarik seluruh stack retrieval (torch, transformers, prometheus, moderation, keyword filter) hanya untuk `_configure_settings`.

**7. Jalur query punya 13 titik keluar sebelum retrieval penuh, dan lima di antaranya tanpa saklar konfigurasi.**
Rinciannya di C9. Yang tanpa env var: L1 hard-block, identity override (substring hardcoded, `rag_pipeline.py:159-166`), chitchat, out_of_scope, dan `<ACK>` condensation. Condensation sendiri tidak dapat dimatikan lewat env var (C11), dan `HISTORY_TURNS=0` justru membuat riwayat **penuh** dipakai karena `history[-0:] == history[0:]` (`:1189`). Paling berbahaya untuk validitas adalah fallback rerank diam (`:684`, `:690`, `:696`): saat TEI timeout, skor yang dibandingkan dengan `SCORE_THRESHOLD` di `:1229` adalah skor dense, bukan skor reranker — dua skala berbeda diuji dengan satu ambang, dan satu-satunya jejaknya ada di log serta counter Prometheus. Pada mesin bersama, timeout 5 detik (`config.py:60`) adalah kondisi balapan yang akan terpicu tidak merata sepanjang eksperimen.
