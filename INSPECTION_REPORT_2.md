# INSPECTION REPORT 2 — rag-core-system

Putaran kedua: menguji repo terhadap desain riset dan stack model, bukan terhadap strukturnya sendiri.
Lanjutan dari `INSPECTION_REPORT.md`. Basis kode sama: branch `main`, commit `022c669`.

---

## Sumber skema

Skema dataset diambil dari `Research Meeting 220826.pdf` (deck "Optimasi Strategi Retrieval Multimodal pada Sistem Tanya Jawab Berbasis Dokumen"), bagian **Publikasi Dataset**, halaman 17–22. Teks diekstrak dengan `pdftotext -f 19 -l 21` untuk memastikan tidak ada baris tabel yang terpotong oleh tepi slide. Keempat berkas skema terbaca lengkap.

Konteks desain riset yang relevan, dari deck yang sama:

| Hal | Sumber di deck |
|---|---|
| Tiga strategi yang dibandingkan: (a) retrieval teks sebagai baseline, (b) retrieval visual dengan **ringkasan naratif**, (c) **structure-preserving visual indexing** — graf berserialisasi untuk flowchart, HTML relasional untuk tabel | Hal. 4 |
| Empat strata tipe visual: flowchart prosedural, tabel persyaratan, formulir/template, figur deskriptif (kelompok kontrol) | Hal. 5–6 |
| Empat lapis metrik: retrieval (IR klasik), generasi (RAGAS), konsistensi struktural (kontribusi baru), reliabilitas (kontribusi baru) | Hal. 7 |
| Lapis 1 dijalankan **terpisah untuk retrieval teks, retrieval gambar, lalu gabungan** | Hal. 8 |
| Lapis 3: SOF, CBI, RCAA, Actor Attribution Accuracy, PHR | Hal. 10 |
| Lapis 4 dihitung hanya pada item dengan `conflict_scenario` bukan null | Hal. 11 |

**Batasan inspeksi (sama seperti putaran pertama):** klaim tentang perilaku internal LlamaIndex, TEI, dan sentence-transformers tidak dapat diverifikasi dari pembacaan source repo ini. Semuanya ditandai eksplisit dan diberi perintah verifikasinya.

---

## F. STACK MODEL

### F1. Setiap pemanggilan model di repo

Dua belas titik pemanggilan. Kolom "Parameter dikirim" hanya mencantumkan yang **eksplisit ada di kode**; sisanya default library atau server.

#### Generation LLM (lewat `llm_factory.get_llm()`)

| # | Pemanggilan | Lokasi | Provider | Parameter dikirim |
|---|---|---|---|---|
| 1 | RAG generation (non-stream) | `rag_pipeline.py:1259` `llm.complete(prompt)` | `LLM_PROVIDER` | `temperature=LLM_TEMPERATURE` (0.1), `max_tokens=LLM_MAX_TOKENS` (1024), `timeout=LLM_REQUEST_TIMEOUT` — semua lewat `llm_factory.py:45-53` |
| 2 | RAG generation (stream) | `rag_pipeline.py:1506` `llm.stream_complete(prompt)` | sama | sama |
| 3 | Chitchat | `rag_pipeline.py:390` `llm.complete(prompt)` | sama | `temperature=0.8` **hardcoded** di `:347` |
| 4 | Out-of-scope | `rag_pipeline.py:481` | sama | `temperature=0.6` **hardcoded** di `:403` |
| 5 | Low-relevance fallback | `rag_pipeline.py:524` | sama | `temperature=0.2` **hardcoded** di `:486` |
| 6 | Condensation | `rag_pipeline.py:575` `llm.complete(prompt)` | sama | `get_llm()` tanpa argumen (`:574`) → `LLM_TEMPERATURE` |
| 7 | Private-API tool selection | `private_api.py:130` `llm.complete(prompt)` | sama | `temperature=0.0` di `:129` |

Konstruksi klien ada di `llm_factory.py:31-53`: cabang `ollama` (`:34-40`) mengirim `num_predict` via `additional_kwargs`, cabang `vllm`/`openai` (`:45-53`) mengirim `max_tokens` + `is_chat_model=True`. Instans di-cache `@lru_cache(maxsize=16)` (`:21`) dengan kunci tujuh parameter — jadi tiap nilai temperature membuat instans terpisah.

**Tidak ada `seed` yang dikirim ke provider mana pun** (`llm_factory.py:34-53`).

#### Vision description (indexing)

| # | Pemanggilan | Lokasi | Provider | Parameter dikirim |
|---|---|---|---|---|
| 8 | Deskripsi gambar via vLLM | `image_describer.py:154-159` | dipilih di `:112-118` berdasar `LLM_PROVIDER` | `model=LLM_MODEL`, `max_tokens=300`, `temperature=0.1` (`:150-151`) |
| 9 | Deskripsi gambar via Ollama | `image_describer.py:174-179` | sama | `model=LLM_MODEL`, `stream=False` (`:169-171`). **Tidak ada** `max_tokens`, **tidak ada** `temperature` |

Tidak lewat `llm_factory` — dua klien `httpx` terpisah. Rinci di F5.

#### Vision RAG (runtime, chat attachment)

| # | Pemanggilan | Lokasi | Provider | Parameter dikirim |
|---|---|---|---|---|
| 10 | `generate_with_vision` | `vision.py:270-273` | Hanya vLLM — dijaga `vision_supported()` (`:75-77`, `:252-253`) | `max_tokens=max_tokens or LLM_MAX_TOKENS`, `temperature=temperature if not None else LLM_TEMPERATURE` (`:265-266`). Pemanggil di `rag_pipeline.py:1024` tidak mengirim override, jadi memakai nilai config |

#### Embedding

| # | Pemanggilan | Lokasi | Provider | Parameter dikirim |
|---|---|---|---|---|
| 11a | TEI HTTP | `rag_pipeline.py:607-612` | `EMBED_PROVIDER=tei` | `model_name`, `base_url`, `embed_batch_size`, `timeout`. **Tidak ada** pooling, normalisasi, instruksi, atau max-length |
| 11b | HuggingFace in-process | `rag_pipeline.py:616-621` | `EMBED_PROVIDER=huggingface` | `model_name`, `device`, `trust_remote_code=True`, `embed_batch_size`. **Tidak ada** pooling, normalisasi, instruksi, atau max-length |

Pemanggilan embedding sebenarnya terjadi di dalam LlamaIndex: `indexing.py:160` (`VectorStoreIndex.from_documents`) untuk indexing dan `rag_pipeline.py:871` (`retriever.retrieve`) untuk query. Kode repo tidak pernah memanggil `.get_text_embedding()` langsung.

**Duplikat:** blok yang sama ada dua kali — `rag_pipeline.py:603-625` (`_configure_settings`) dan `index_narratives.py:183-201` (`_configure_embed`). Perbedaannya: `rag_pipeline.py:623` menyetel `Settings.llm = get_llm()`, `index_narratives.py:201` menyetel `Settings.llm = None`.

#### Reranker

| # | Pemanggilan | Lokasi | Provider | Parameter dikirim |
|---|---|---|---|---|
| 12a | TEI `/rerank` HTTP | `rag_pipeline.py:666-675` | `RERANKER_PROVIDER=tei` | Body: `query`, `texts` (`n.node.get_content()`, `:670`), `raw_scores: False`, `return_text: False`. `timeout=RERANKER_TIMEOUT`. **Model tidak dikirim** — ditentukan server |
| 12b | SentenceTransformerRerank | `rag_pipeline.py:727-731` | `RERANKER_PROVIDER=sentence_transformers` | `model=RERANKER_MODEL`, `top_n=RERANKER_TOP_N`, `keep_retrieval_score=False`. **Tidak ada** `device`, **tidak ada** aktivasi/normalisasi skor |

#### Intent classifier

| # | Pemanggilan | Lokasi | Provider | Parameter dikirim |
|---|---|---|---|---|
| 13 | IndoBERT forward pass | `intent_classifier.py:124-125` | Lokal, `transformers` | Tokenizer (`:115-121`): `truncation=True`, `padding=True`, `max_length=128` (`_MAX_LENGTH`, `:25`). Model: `torch.no_grad()`, `softmax(logits, dim=-1)`. Device auto: `cuda` bila tersedia (`:71`) |

#### Moderation

| # | Pemanggilan | Lokasi | Provider | Parameter dikirim |
|---|---|---|---|---|
| 14 | Llama Guard via Ollama `/api/generate` | `moderation.py:204-208` | `MODERATION_BACKEND=ollama`; `passthrough` melewati seluruhnya (`:190-191`) | `model=MODERATION_MODEL`, `prompt`, `stream: False`, `timeout=MODERATION_TIMEOUT`. **Tidak ada** temperature. Input **dipotong ke 500 karakter** (`:202`) |

### F2. Provenance model — apa yang terekam, apa yang hilang

| Model | Terekam di artefak? | Di mana | Versi/digest? |
|---|---|---|---|
| **LLM generation** | **Ya** | `debug["model"] = LLM_MODEL` di `rag_pipeline.py:1084`, `:1342`, `:941`; terekspos lewat `DebugInfo.model` (`schemas.py:71`); ikut tersimpan ke Postgres lewat `save_assistant_message(debug=...)` (`routers/chat.py:120`) | Tidak. Hanya string dari env |
| **Vision description (indexing)** | **Tidak** | Memakai `LLM_MODEL` yang sama (`image_describer.py:142`, `:168`) tapi **tidak ditulis ke metadata chunk mana pun**. `indexing.py:251-260` tidak punya field untuk itu | Tidak |
| **Embedding** | **Tidak** | Hanya di log `rag_pipeline.py:613`/`:622`. Tidak di payload Qdrant, tidak di response | Tidak |
| **Reranker** | **Tidak** | Jalur ST: log `:732` menyebut `RERANKER_MODEL`. Jalur TEI: log `:721-724` hanya menyebut `base_url` — **nama model tidak diketahui aplikasi**, ditentukan `--model-id` di `docker-compose.poc.yml:111` | Tidak |
| **Intent classifier** | **Tidak** | Path di-log saat load (`intent_classifier.py:75`). `INTENT_MODEL_PATH` adalah direktori lokal — isinya bisa diganti tanpa jejak | Tidak. `download_models.py:47` memanggil `snapshot_download(repo_id=..., local_dir=...)` **tanpa `revision=`** → selalu menarik `main` terbaru |
| **Reranker (unduhan)** | **Tidak** | Sama: `download_models.py:47` tanpa `revision=` untuk `BAAI/bge-reranker-v2-m3` (`:35`) | Tidak |
| **Moderation** | **Tidak** | `MODERATION_MODEL` tidak di-log per panggilan dan tidak masuk response | Tidak |
| **Image container** | **Tidak** | `docker-compose.poc.yml` memakai tag bergerak: `vllm/vllm-openai:latest` (`:62`), `qdrant/qdrant:latest` (`:37`), `ollama/ollama:latest` (`:128`), `redis/redis-stack:latest` (`:30`), `minio/minio:latest` (`:45`), `text-embeddings-inference:89-latest` (`:94`, `:110`) | **Tidak ada satu pun digest** |

Ringkasnya: **hanya nama LLM generation yang bisa dilaporkan langsung dari artefak eksperimen.** Enam model lain hanya hidup di env var dan flag compose, keduanya bisa berubah tanpa meninggalkan jejak di data yang dihasilkan. Untuk paper, ini berarti korpus yang sudah di-index tidak dapat dibuktikan berasal dari model embedding tertentu.

Yang membuatnya lebih rapuh: chunk yang tersimpan **tidak memuat** nama model embedding, model vision, maupun versi prompt deskripsi. Dua run indexing dengan model berbeda menghasilkan payload Qdrant yang tidak dapat dibedakan.

### F3. `EMBED_PROVIDER=tei` vs `=huggingface`

**Temuan utama: repo tidak mengonfigurasi pooling, normalisasi, prefix instruksi, maupun panjang maksimum — pada kedua jalur.** Divergensi muncul lewat *kelalaian*, bukan lewat konfigurasi yang berbeda.

| Aspek | Jalur `tei` | Jalur `huggingface` | Dijamin sama? |
|---|---|---|---|
| Titik konstruksi | `rag_pipeline.py:607-612` | `rag_pipeline.py:616-621` | — |
| Nama model | `model_name=EMBED_MODEL` (`:608`) | `model_name=EMBED_MODEL` (`:617`) | Ya secara string. **Tapi** jalur TEI meneruskannya hanya sebagai label — bobot sebenarnya ditentukan `--model-id` di `docker-compose.poc.yml:95`. Dua nilai bisa berbeda tanpa error |
| **Pooling** | Tidak dikirim | Tidak dikirim | **Tidak.** TEI menentukan server-side dari konfigurasi model; `HuggingFaceEmbedding` menentukan client-side dari default library. Dua implementasi berbeda |
| **Normalisasi** | Tidak dikirim | Tidak dikirim | **Tidak.** Sama alasannya |
| **Prefix instruksi** | Tidak dikirim | Tidak dikirim | Konsisten dalam arti keduanya kosong. **Tapi** `EMBED_MODEL` default `Qwen/Qwen3-Embedding-0.6B` (`config.py:41`) berasal dari keluarga yang memakai prefix instruksi untuk query. Repo tidak pernah menyetel `query_instruction`/`text_instruction` (diverifikasi: grep atas kedua nama tidak menghasilkan apa pun) |
| **Panjang maksimum** | Tidak dikirim | Tidak dikirim | **Tidak.** TEI memotong server-side; `HuggingFaceEmbedding` memotong client-side pada `max_length` default-nya |
| Batch size | `embed_batch_size=EMBED_BATCH_SIZE` (`:610`) | `embed_batch_size=EMBED_BATCH_SIZE` (`:620`) | Ya |
| Timeout | `timeout=float(EMBED_TIMEOUT)` (`:611`) | Tidak berlaku | — |
| Device | Tidak berlaku (server) | `device=EMBED_DEVICE` (`:618`) | — |
| `trust_remote_code` | Tidak berlaku | `True` (`:619`) | — |

**Kesimpulan: vektor kedua jalur TIDAK dijamin identik.** Titik divergensinya adalah **absennya argumen di `rag_pipeline.py:607-612` versus `:616-621`**, ditambah `docker-compose.poc.yml:95` yang menjalankan TEI hanya dengan `--model-id` dan `--port` — tanpa `--pooling`. Tidak ada satu baris pun di repo yang menyamakan kedua konfigurasi.

Dua masalah turunan:

**Dimensi tidak pernah diverifikasi.** `EMBED_DIMENSION = 1024` adalah konstanta di `config.py:46` (bukan env var) dan ditegaskan ke Qdrant di `indexing.py:131` (`VectorParams(size=EMBED_DIMENSION, ...)`). Tidak ada kode yang memeriksa apakah provider benar-benar mengembalikan 1024 dimensi. Ketidakcocokan baru muncul sebagai kegagalan upsert dari Qdrant, bukan sebagai pesan kesalahan konfigurasi.

**Ada dua salinan konfigurasi yang bisa melenceng.** `rag_pipeline.py:603-625` dan `index_narratives.py:183-201` menduplikasi blok yang sama. Keduanya harus diubah bersamaan; tidak ada yang menegakkan itu.

### F4. `RERANKER_PROVIDER=tei` vs `=sentence_transformers`

| Aspek | Jalur `tei` | Jalur `sentence_transformers` | Dijamin sama? |
|---|---|---|---|
| Implementasi | `_TEIRerankPostprocessor`, kelas kustom di repo (`rag_pipeline.py:647-707`) | `SentenceTransformerRerank` dari LlamaIndex (`:727-731`) | — |
| **Model yang dipakai** | **`RERANKER_MODEL` diabaikan sepenuhnya.** Konstruktor hanya menerima `base_url`, `top_n`, `timeout` (`:716-720`). Model ditentukan `docker-compose.poc.yml:111` | `model=RERANKER_MODEL` (`:728`) — default path lokal `<root>/models/bge-reranker-v2-m3` (`config.py:56`) | **Tidak.** Mengganti provider mengubah sumber bobot secara diam-diam |
| **Skala skor** | `raw_scores: False` dikirim eksplisit (`:671`) → TEI menerapkan aktivasi | Tidak ada argumen aktivasi/normalisasi (`:727-731`) → default library | **Tidak.** Keduanya lalu dibandingkan dengan `SCORE_THRESHOLD` yang sama (`config.py:118` → `rag_pipeline.py:1229`) dan `LOW_CONFIDENCE_BUFFER` yang sama (`:64` → `:235`). Bila skalanya berbeda, satu ambang berarti dua hal |
| Input teks | `n.node.get_content()` (`:670`) | Ditentukan LlamaIndex | Kemungkinan sama; tidak diverifikasi |
| **Truncation** | Server-side, tidak dikonfigurasi di `docker-compose.poc.yml:111` | Client-side, `max_length` tidak dikirim (`:727-731`) | **Tidak dijamin** |
| Device | Server (GPU per `docker-compose.poc.yml:117-123`) | **Tidak dikirim** (`:727-731`) — library memilih sendiri | Tidak |
| Penulisan skor | `node.score = float(r["score"])` in-place (`:705`) | Ditangani library | — |
| Jumlah keluaran | `results[:self.top_n]` (`:700`), dengan `idx >= len(nodes)` di-skip (`:702-703`) | `top_n=RERANKER_TOP_N` (`:729`) | Ya secara nominal |
| **Perilaku gagal** | Tiga cabang fallback (`:678-696`) mengembalikan `nodes[:top_n]` **tanpa menulis ulang skor** | **Tidak ada fallback** — exception merambat ke pemanggil | **Tidak.** Semantik kegagalan berlawanan |
| Tie-breaking | Urutan kembalian TEI, tanpa tie-break eksplisit (`:699-707`) | Ditentukan library | Tidak |

Fallback jalur TEI adalah masalah validitas yang sudah dicatat di laporan pertama, dan di sini bertambah jelas: karena `keep_retrieval_score=False` di jalur ST (`:730`) tetapi jalur TEI mempertahankan skor dense saat gagal, dua provider bahkan tidak sepakat tentang skor apa yang tersisa pada node ketika reranking tidak berjalan.

### F5. Vision description — vLLM vs Ollama

| Aspek | vLLM (`image_describer.py:136-159`) | Ollama (`:162-179`) |
|---|---|---|
| Endpoint | `{LLM_BASE_URL}/v1/chat/completions` (`:155`) | `{LLM_BASE_URL}/api/generate` (`:176`) |
| **Format pesan** | `messages: [{role:"user", content:[{type:"text",...},{type:"image_url",...}]}]` (`:143-149`) — array multimodal OpenAI | `prompt: <str>` + `images: [b64]` di top level (`:169-170`) — dua field terpisah |
| **Penanganan gambar** | Data URI `f"data:image/png;base64,{b64}"` (`:147`) | Base64 telanjang tanpa prefix (`:170`) |
| **MIME** | **Selalu di-hardcode `image/png`** (`:147`), tanpa memeriksa format bytes sebenarnya | Tidak ada deklarasi MIME |
| `max_tokens` | `300` (`:150`) | **Tidak dikirim** → default model |
| `temperature` | `0.1` (`:151`) | **Tidak dikirim** → default model |
| `stream` | Tidak dikirim | `False` (`:171`) |
| Timeout | `httpx.Client(timeout=60.0)` (`:154`) | `httpx.Client(timeout=60.0)` (`:174`) |
| Cek status HTTP | `resp.raise_for_status()` (`:156`) | `resp.raise_for_status()` (`:177`) |
| **Ekstraksi hasil** | `data["choices"][0]["message"]["content"]` (`:159`) — **subscript tanpa guard**, melempar `KeyError`/`IndexError` bila bentuk respons tak terduga | `data.get("response")` (`:179`) — **mengembalikan `None` diam-diam** bila key hilang |
| **Tipe kembalian saat gagal** | Praktisnya tidak pernah `None`: entah `str` entah melempar | Bisa `None` secara langsung |

**Apakah keduanya menghasilkan tipe kembalian sama untuk kasus gagal?** Di permukaan luar, ya — keduanya dipanggil di dalam `try` pada `describe_image` (`:111-121`) yang menangkap `Exception`, mencatat warning (`:120`), dan mengembalikan `None` (`:121`). Jadi pemanggil selalu menerima `Optional[str]`.

Di dalam, tidak. Jalur vLLM mencapai `None` lewat *exception yang tertangkap*; jalur Ollama bisa mencapainya lewat *return normal*. Perbedaannya kelihatan di log: kegagalan vLLM meninggalkan baris `image_describer_error error=...` (`:120`), kegagalan Ollama yang berupa key hilang **tidak meninggalkan baris apa pun** — hasil `None` mengalir ke `:123-124` dan keluar tanpa jejak. Untuk eksperimen ini berarti hitungan gambar yang gagal dideskripsikan tidak dapat direkonstruksi dari log pada jalur Ollama.

Tiga temuan tambahan pada jalur ini:

**Provider ketiga membuang semuanya diam-diam.** `llm_factory.py:42` menerima `LLM_PROVIDER="openai"`, tetapi `image_describer.py:112-118` hanya menangani `"vllm"` dan `"ollama"`; selain itu mencatat `logger.debug` (`:117`) dan mengembalikan `None`. Dengan `LLM_PROVIDER=openai`, **seluruh deskripsi gambar hilang** dan hanya menyisakan jejak di level DEBUG.

**Cabang JPEG di resize tidak pernah tereksekusi.** `_resize_image_if_needed` menugaskan ulang `img = img.resize(...)` (`:57`), lalu membaca `fmt = img.format or "PNG"` (`:61`). Objek hasil `.resize()` tidak membawa `.format`, sehingga `fmt` selalu `"PNG"` dan cabang `if fmt == "JPEG"` (`:61-63`) tidak terjangkau. Semua gambar yang di-resize dikirim sebagai PNG — konsisten dengan MIME hardcode di `:147`, tapi berarti gambar yang **tidak** di-resize (`:52-53` mengembalikan bytes asli) bisa berupa JPEG yang tetap dilabeli `image/png`.

**Caching tidak konsisten.** Hasil `DEKORATIF`/`TIDAK JELAS` ditulis ke cache sebagai `""` (`:129`), tetapi respons kosong dari model keluar di `:123-124` **tanpa menulis cache**. Gambar yang menghasilkan respons kosong akan memanggil model berulang kali dalam satu proses.

### F6. Asumsi hardcoded — dimensi, panjang konteks, nama model

**Dimensi embedding:**

| Nilai | Lokasi | Dampak |
|---|---|---|
| `EMBED_DIMENSION = 1024` | `config.py:46` — konstanta, **bukan** env var | Ditegaskan ke Qdrant di `indexing.py:131`. Ganti model embedding berdimensi lain tanpa mengubah baris ini = kegagalan upsert |

**Panjang konteks / token:**

| Nilai | Lokasi | Dampak |
|---|---|---|
| `_MAX_LENGTH = 128` | `intent_classifier.py:25`, dipakai `:120` | Pertanyaan >128 token dipotong sebelum klasifikasi |
| `max_tokens=300` | `image_describer.py:150` | Batas panjang deskripsi gambar. Prompt meminta "maks 150 kata" (`:25`) — dua batas yang tidak saling terkait |
| `text[:500]` | `moderation.py:202` | Input moderasi dipotong 500 karakter |
| `limit * 4` sebagai proksi token | `rag_pipeline.py:532` | Heuristik 4 karakter/token untuk trim history |
| `MAX_CHUNK_TOKENS = 400`, `CHARS_PER_TOKEN = 3.5` | `index_narratives.py:208-209` | **Konstanta karakter-per-token berbeda** dari `rag_pipeline.py:532`. Dua estimator token di satu repo |
| `--max-model-len 8192` | `docker-compose.poc.yml:66` | Batas konteks vLLM. Tidak ada kode yang memeriksa panjang prompt terhadap batas ini sebelum mengirim |
| `n.text[:300]` | `rag_pipeline.py:899` | `text_preview` dipotong 300 karakter |
| `question[:60]`, `[:120]`, `[:50]`, `[:100]`, `[:150]` | `rag_pipeline.py:597`, `:1127`, `:1209`, `:355`, `:410` | Pemotongan untuk log |

**Nama model hardcoded:**

| Lokasi | Isi | Dampak untuk riset |
|---|---|---|
| `output_filter.py:14` | Regex me-redact `claude\|gpt\|gemini\|llama\|qwen\|mistral\|indobert\|grok\|kimi\|deepseek\|ernie\|palm\|bert` → `[AI system]` | **Menyentuh teks jawaban.** Diterapkan di `rag_pipeline.py:1263` dan lagi di `routers/query.py:82` / `routers/chat.py:113` |
| `output_filter.py:17` | Regex vendor termasuk `\bmeta\b` sebagai alternatif berdiri sendiri | **Risiko nyata pada teks akademik Indonesia.** `\bmeta\b` cocok dengan "meta" di "meta-analisis" (batas kata terpenuhi oleh tanda hubung) → jawaban menjadi "[AI vendor]-analisis" |
| `output_filter.py:20` | Regex stack termasuk `transformers`, `redis`, `qdrant`, `langchain` | Sama; `transformers` adalah kata yang bisa muncul di konteks lain |
| `rag_pipeline.py:159-166` | `_IDENTITY_KEYWORDS` memuat nama model dan frasa pendek | Memicu pengalihan ke chitchat lewat pencocokan substring (`:262`) |
| `moderation.py:73-99` | `_PROMPT` memakai chat template Llama Guard literal (`<\|begin_of_text\|>`, `<\|start_header_id\|>`) | Terkunci ke keluarga model itu. Mengganti model moderasi tanpa mengganti template = keluaran tidak terdefinisi |
| `moderation.py:224` | `output.startswith("safe")` | Kontrak keluaran spesifik Llama Guard |
| `image_describer.py:128` | `startswith("DEKORATIF")` / `("TIDAK JELAS")` | Kontrak keluaran yang dipasangkan dengan prompt di `:24-34` |
| `rag_pipeline.py:1149`, `:1155`, `:1165` | Label intent `"chitchat"`, `"out_of_scope"`, `"get_info_private"` dibandingkan sebagai string literal | Label sebenarnya berasal dari `model.config.id2label` (`intent_classifier.py:137`). Model dengan label berbeda **jatuh diam-diam** ke jalur RAG tanpa error |
| `config.py:56` | `RERANKER_MODEL` default path lokal `models/bge-reranker-v2-m3` | — |
| `config.py:82` | `OCR_LANG = "id"` — konstanta, bukan env var | — |
| `docker-compose.poc.yml:64`, `:95`, `:111` | `Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen3-Embedding-0.6B`, `BAAI/bge-reranker-v2-m3` | Satu-satunya tempat model TEI dideklarasikan |

**Yang paling perlu diperhatikan untuk riset:** `output_filter.py` mengubah teks jawaban sebelum jawaban itu dievaluasi. Kalau metrik lapis 2 mengukur kualitas jawaban, yang diukur adalah teks pasca-redaksi, bukan keluaran model. Dan `\bmeta\b` di `:17` adalah pola yang wajar muncul dalam dokumen akademik.

---

## G. KESENJANGAN TERHADAP SKEMA DATASET RISET

### G7. Status per field

Tiga status: **Ada** (sudah tertulis ke payload Qdrant hari ini), **Turunan** (dapat dihitung dari yang ada tanpa ekstraksi baru), **Nol** (harus dibangun).

Rekapitulasi: dari 46 field di keempat berkas, **2 Ada** (`sha256`, `page_number`), **5 Turunan**, **39 Nol**.

#### 3.1 `corpus_metadata.jsonl` — satu baris per dokumen

| Field | Tipe | Status | Bukti / catatan |
|---|---|---|---|
| `document_id` | string | **Nol** | Kode hanya punya `file_name` (`indexing.py:252` ← `preprocessing.py:440`). Contoh deck `"sop-izin-ujian-online-v2"` memuat penanda versi di dalam id — tidak ada mekanisme apa pun yang memproduksi slug semacam itu |
| `title` | string | **Nol** | Tidak pernah diekstrak. `fitz.open()` dipanggil di `preprocessing.py:109` dan `:158`, tetapi hanya untuk iterasi halaman; `doc.metadata` tidak pernah dibaca |
| `source_unit` | string | **Nol** | Tidak ada konsep unit penerbit di kode |
| `source_url` | string atau null | **Nol** | `scripts/fetch_pdfs.py` menerima URL repo GitHub (`:222`, `:233`) atau folder zip, lalu mengekstrak PDF (`:129-152`). **URL per dokumen tidak pernah disimpan** — tidak ada manifest yang ditulis |
| `access_status` | enum | **Nol** | Tidak ada |
| `license` | string | **Nol** | Tidak ada |
| `document_type` | enum | **Nol** | Tidak ada |
| `primary_visual_type` | enum | **Nol** | Tidak ada. Bersinggungan dengan H14 |
| `target_population` | object | **Nol** | Tidak ada. Dipakai lapis 3 (RCAA menunjuk `target_population` untuk memilih sel gold) |
| `effective_start` | date | **Nol** | Tidak ada metadata temporal apa pun di repo |
| `effective_end` | date atau null | **Nol** | Sama |
| `superseded_by` | string atau null | **Nol** | Sama. Dipakai lapis 4 (Temporal Grounding Accuracy) |
| `regulation_number` | string atau null | **Nol** | Tidak ada |
| `sha256` | string | **Ada** | Satu-satunya kecocokan langsung di seluruh skema. `file_sha256` (`preprocessing.py:91-97`) → `result["file_hash"]` (`:441`) → payload `file_hash` (`indexing.py:253`). Perlu ganti nama saja |

Dua field yang sudah ditulis kode tetapi **tidak ada di skema**: `extraction_strategy` (`indexing.py:258`) dan `source_type` (`:259`). Keduanya layak dipertahankan untuk reproduksibilitas (lihat F2), meski di luar skema.

#### 3.2 `chunks.jsonl` — satu baris per unit teks/tabel yang diindeks

| Field | Tipe | Status | Bukti / catatan |
|---|---|---|---|
| `chunk_id` | string | **Nol** | Format deck `{document_id}_p{page}_c{NN}` adalah **ordinal dalam halaman**. Kode punya `chunk_index` yang merupakan pencacah berjalan **per berkas** (`preprocessing.py:333`, `:366`, `:385`) — granularitas berbeda, dan tidak unik per titik Qdrant. Rinci di G10 |
| `document_id` | string | **Nol** | Bergantung pada `document_id` di lapis 1 |
| `page_number` | int | **Ada** | Payload `page` (`indexing.py:254`). **Peringatan:** jalur hi_res memakai `el.metadata.page_number` dengan fallback ke `0` bila absen (`preprocessing.py:216`), sehingga nilai `0` berarti "tidak diketahui", bukan halaman nol |
| `chunk_type` | enum `"text"`/`"table"` | **Turunan** | Dari `element_type` (`indexing.py:256`): `"Table"` → `"table"`, sisanya → `"text"`. **Tetapi pemetaan ini rusak di jalur `fast`**, yang menandai setiap element `"NarrativeText"` (`preprocessing.py:167`) — tabel di dokumen jalur `fast` akan salah terlabel `"text"` |
| `text_content` | string | **Turunan (terkontaminasi)** | Teks ada di payload sebagai `text` dan/atau `_node_content` (dibuktikan pembaca di `rag_pipeline.py:810-819`). **Tiga kontaminasi:** chunk tabel besar diberi prefix `## {section}` (`preprocessing.py:362`), chunk gambar diberi prefix `[Deskripsi Gambar] ` (`:382`), dan chunk teks bisa memuat header `# {title}` yang disisipkan (`:351`) |
| `text_as_html` | string atau null | **Nol** | Sempat ada di `preprocessing.py:229`, dibuang 130 baris kemudian di `:361-368`. Deck menyebut `infer_table_structure=True` — yang memang aktif di `:204` — jadi bahan mentahnya benar-benar diproduksi lalu dibuang. Rinci di G8 |
| `bbox` | array[float] atau null | **Nol** | Unstructured menyediakan koordinat di metadata element; `_extract_hi_res` (`:213-254`) hanya membaca `.category`, `.page_number`, `.text_as_html`, dan `.image_base64` — koordinat tidak pernah disentuh. Jalur `fast` juga tidak, meski PyMuPDF menyediakannya |

#### 3.3 `images.jsonl` — satu baris per gambar/visual yang diekstrak

| Field | Tipe | Status | Bukti / catatan |
|---|---|---|---|
| `image_id` | string | **Nol** | Format deck `{document_id}_p{page}_img{NN}` — ordinal dalam halaman. Tidak ada identitas gambar apa pun di kode. Rinci di G11 |
| `document_id` | string | **Nol** | Bergantung lapis 1 |
| `page_number` | int | **Turunan** | Element `Image` membawa `page` di `preprocessing.py:240`, tetapi dibuang bersama seluruh metadata di `:300` |
| `file_path` | string | **Nol** | Gambar **tidak pernah menyentuh disk**. `extract_image_block_to_payload=True` (`:206`) menaruh base64 di memori; alternatif `extract_image_block_output_dir` tidak dipakai. Laporan 1, A2 |
| `visual_type` | enum | **Nol** | Deck menyatakan ini "label hasil anotasi manusia" — jadi **tidak perlu diturunkan dari model**, dan H14 sebagian terjawab. Yang tetap dibutuhkan: field untuk membawanya, dan penautan ke chunk agar metrik lapis 1 bisa distratifikasi |
| `structured_summary` | object atau null | **Nol** | Dua kasus berbeda. **Tabel:** deck bilang "sama dengan `text_as_html` di chunk terkait" — jadi bergantung sepenuhnya pada penyelamatan HTML (perubahan #4). **Flowchart:** graf `{"nodes":[…],"edges":[{"from","to","condition"}]}` tidak dapat dipasok oleh apa pun di pipeline sekarang. Rinci di G8 |
| `narrative_summary` | string atau null | **Turunan (terkontaminasi)** | Teksnya ada, tetapi hanya sebagai bagian dari chunk `ImageDescription` yang sudah dilebur dengan prefix `## {section}\n\n[Deskripsi Gambar] ` (`preprocessing.py:382`). Tidak ada penyimpanan terpisah, dan tidak ada id gambar untuk menautkannya kembali. Deck menegaskan field ini harus tetap ada agar varian (b) dapat direplikasi — persis kebutuhan yang dibahas di G9 |

#### 3.4 `qa_pairs.jsonl` — satu baris per pertanyaan

Berkas ini adalah hasil anotasi manusia; kode tidak memasok isinya. Yang relevan adalah apakah **target rujukannya** dapat dinyatakan dan diverifikasi terhadap index.

| Field | Tipe | Status | Bukti / catatan |
|---|---|---|---|
| `qa_id` | string | **Nol** (anotasi) | — |
| `question` | string | **Nol** (anotasi) | — |
| `question_category` | string | **Nol** (anotasi) | — |
| `visual_type_required` | enum | **Nol** (anotasi) | Harus konsisten dengan `visual_type` di lapis 2 agar stratifikasi lapis 1 bisa dihitung |
| `reference_answer` | string | **Nol** (anotasi) | — |
| `reference_answer_short` | string | **Nol** (anotasi) | Dipakai RCAA — nilai diskret dari sel tabel |
| `gold_document_ids` | array[string] | **Nol** | Menunjuk `document_id` yang belum ada |
| `gold_chunk_ids` | array[string] | **Nol** | **Field paling rapuh di seluruh skema.** Menunjuk `chunk_id` yang belum ada, dan skema penomoran yang tersedia hari ini tidak stabil lintas re-index. Rinci di G10 |
| `gold_image_ids` | array[string] | **Nol** | Menunjuk `image_id` yang belum ada. Dipakai CBI untuk menemukan anotasi `structured_summary` berisi edge berlabel kondisi |
| `gold_page_numbers` | array[int] | **Turunan** | Satu-satunya field gold yang dapat diverifikasi terhadap index hari ini, lewat payload `page` (`indexing.py:254`) — dengan peringatan fallback `0` di `preprocessing.py:216` |
| `target_population` | object | **Nol** | Dipakai RCAA untuk memilih sel gold. Tidak ada di kode |
| `validity_context` | object | **Nol** | `as_of_date` + `applicable_regulation_id`. Tidak ada metadata temporal di repo, dan **tidak ada mekanisme filter berbasis tanggal di jalur retrieval** — `_expand_with_neighbors` memfilter atas `file_name`, `chunk_index`, `section` (`rag_pipeline.py:779-784`), dan RBAC pun masih komentar (`:854-859`) |
| `conflict_scenario` | string atau null | **Nol** (anotasi) | Menentukan subset lapis 4 |
| `expected_behavior` | enum | **Nol** | Nilai `"abstain_or_flag_conflict"` menuntut perilaku yang tidak ada. Yang paling mendekati adalah `_low_relevance_response` (`rag_pipeline.py:484-524`), tetapi itu dipicu oleh skor rendah (`:1229`), bukan oleh terdeteksinya konflik antar-sumber |
| `difficulty` | object | **Nol** (anotasi) | `scope: "single_document"` vs multi — dapat diverifikasi terhadap `gold_document_ids` |
| `annotators` | array[string] | **Nol** (anotasi) | — |
| `adjudication_note` | string | **Nol** (anotasi) | — |
| `split` | string | **Nol** (anotasi) | — |

#### Tiga kesenjangan lintas-berkas yang baru terlihat dari skema

**Identitas berlapis tiga tidak ada satu pun.** Skema mengandaikan `document_id` → `chunk_id` → `image_id` yang saling merujuk. Kode hanya punya `file_name` dan `chunk_index`, keduanya tidak memenuhi syarat. Semua field `gold_*` bergantung pada rantai ini; sampai ia dibangun, `qa_pairs.jsonl` tidak dapat ditulis dengan cara yang dapat diverifikasi.

**Dimensi temporal sama sekali absen.** `effective_start`, `effective_end`, `superseded_by`, dan `validity_context` mendukung seluruh lapis 4. Tidak ada padanan apa pun di repo — bukan hanya field metadata yang hilang, tetapi juga kemampuan retrieval untuk memfilter berdasarkan masa berlaku. Contoh Q&A di deck (hal. 13, SOP 2022 vs SOP 2025) mengandalkan sistem mengenali dokumen mana yang berlaku; pipeline sekarang akan menarik keduanya dan menyerahkan pilihan ke LLM.

**Retrieval gambar terpisah tidak ada.** Deck (hal. 8) meminta lapis 1 dijalankan terpisah untuk retrieval teks, retrieval gambar, lalu gabungan. Pipeline sekarang punya **satu** jalur retrieval atas **satu** collection (`rag_pipeline.py:636-642`), dan deskripsi gambar dilebur menjadi chunk teks biasa (`preprocessing.py:381-388`) yang bersaing di ruang vektor yang sama. Memisahkan Recall@k gambar dari Recall@k teks tidak mungkin tanpa memberi tanda modalitas pada chunk — `element_type="ImageDescription"` sebenarnya sudah bisa dipakai sebagai penanda, tetapi tidak terekspos di `_build_sources` dengan cara yang memungkinkan pemisahan metrik (`rag_pipeline.py:891-902` memang mengembalikan `element_type`, jadi ini yang paling dekat dari semua kesenjangan).

### G8. `structured_summary` — di mana informasinya hilang

**Untuk tabel — HTML relasional memang sempat ada, lalu dibuang.**

Jejaknya:

| Tahap | Lokasi | Status HTML |
|---|---|---|
| Unstructured menghasilkan | `preprocessing.py:204` (`infer_table_structure=PDF_EXTRACT_TABLES`) | Ada, sebagai `el.metadata.text_as_html` |
| Diambil kode | `preprocessing.py:223` | Ada, di variabel lokal `html` |
| Dikonversi ke Markdown | `preprocessing.py:224` → `_html_table_to_markdown` (`:178-186`) | Markdown dibuat; HTML masih hidup |
| Disimpan di element metadata | `preprocessing.py:229` (`{"raw_html": html or ""}`) | **Masih ada** |
| **Chunking** | `preprocessing.py:355-374` | **HILANG.** Cabang `Table` membaca `el["text"]` saja; dict chunk yang dibangun (`:361-368`) tidak punya key `metadata` |
| Konstruksi Document | `indexing.py:251-260` | Tidak ada field untuk itu |

Jadi titik kehilangannya persis di **`preprocessing.py:361-368`** (tabel besar) dan **`preprocessing.py:371-372`** (tabel kecil, yang bahkan kehilangan batas element karena di-append ke buffer teks). Jaraknya hanya 130 baris dari tempat HTML tersedia — perbaikannya kecil.

Dua batasan yang tetap ada meski HTML diselamatkan:

- Jalur `fast` (`preprocessing.py:156-173`) tidak memanggil `partition_pdf` sama sekali, jadi tidak pernah menghasilkan `text_as_html`. Dan `detect_strategy` (`:102-125`) tidak pernah menghitung tabel — hanya `get_images()` (`:116`). PDF penuh tabel teks selalu masuk `fast`.
- Bila `markdownify` gagal, `_html_table_to_markdown` mengembalikan HTML mentah (`:186`), sehingga `element_type="Table"` kadang berisi Markdown kadang HTML — tanpa penanda mana yang mana.

**Untuk flowchart — tidak ada apa pun, dan tidak ada tempat untuk menaruhnya.**

Yang dibutuhkan riset (nodes, edges berlabel kondisi, aktor) tidak dapat dipasok pipeline mana pun saat ini:

| Kebutuhan | Realitas kode |
|---|---|
| Graf berserialisasi | `describe_image` mengembalikan `Optional[str]` (`image_describer.py:95`) — prosa bebas |
| Node & edge terpisah | Prompt (`:28`) meminta "sebutkan semua langkah, label, dan arah panah" dalam kalimat. Tidak ada permintaan format terstruktur, tidak ada JSON schema, tidak ada parsing |
| Aktor | Tidak disebut di prompt sama sekali (`:24-34`) |
| Penanda "ini flowchart" | Prompt menyuruh model **membedakan** diagram/tabel/screenshot/dekoratif (`:28-31`), tetapi kode hanya mengekstrak dua sentinel: `DEKORATIF` dan `TIDAK JELAS` (`:128`). Klasifikasi jenis visual lainnya larut dalam prosa dan tidak pernah diambil |
| Tempat menyimpan | Chunk `ImageDescription` hanya punya enam key (`preprocessing.py:381-388`), semuanya skalar. Tidak ada field bebas |

Sejauh mana pipeline **bisa** memasoknya: gambarnya sendiri sampai ke model vision sebagai base64 (`image_describer.py:147`), dan modelnya multimodal. Jadi bahan mentahnya ada — yang tidak ada adalah (a) prompt kedua yang meminta struktur, (b) tipe kembalian yang bukan `str`, (c) field metadata untuk menampungnya, dan (d) gambar aslinya yang tersimpan supaya bisa diproses ulang tanpa mengulang ekstraksi PDF.

Titik kehilangan tunggal yang paling menentukan: **`image_describer.py:95`** — tanda tangan fungsi `-> Optional[str]`. Selama tipe kembaliannya string, tidak ada tempat bagi output terstruktur untuk lewat.

### G9. `narrative_summary` berdampingan dengan `structured_summary`

**Ada tiga asumsi satu-deskripsi-per-gambar yang harus dibongkar.** Semuanya struktural, bukan sekadar penamaan.

| # | Asumsi | Lokasi | Mengapa memblokir |
|---|---|---|---|
| 1 | Satu gambar → satu string | `image_describer.py:95` (`-> Optional[str]`), `:113`, `:115`, `:132-133` | Tidak ada jalur bagi dua varian untuk dikembalikan bersamaan |
| 2 | Cache berkunci hash gambar saja | `image_describer.py:37` (`dict[str, str]`), kunci `_hash_image` (`:40-41`), dibaca `:106-107`, ditulis `:129`/`:132` | Kunci tidak memuat varian. Meminta deskripsi naratif lalu terstruktur untuk gambar yang sama akan **mengembalikan hasil varian pertama** dari cache |
| 3 | Satu element Image → satu element ImageDescription | `preprocessing.py:296-301` membangun **satu** dict per gambar; `:381-388` membangun **satu** chunk per element | Relasi 1:1 dipaksakan di dua tempat berurutan |

Asumsi keempat yang lebih halus: **tidak ada id gambar untuk menautkan dua varian.** Chunk `ImageDescription` hanya membawa `page` dan `section` (`preprocessing.py:383`, `:387`). Dua gambar di halaman yang sama, di bawah judul yang sama, tidak dapat dibedakan. Tanpa G11 terselesaikan lebih dulu, "varian A dan varian B dari gambar yang sama" tidak dapat dinyatakan sama sekali.

**Apakah struktur sekarang memungkinkannya?** Tidak, tetapi penghalangnya dangkal. `_describe_image_elements` (`preprocessing.py:261-303`) sudah berbentuk transformasi list→list, jadi menghasilkan dua element dari satu input tidak melawan bentuk fungsinya. Yang berlawanan hanyalah tipe kembalian di `:95` dan kunci cache di `:37`. `element_type` juga sudah bebas-nilai (`indexing.py:256`), jadi `"ImageNarrative"` dan `"ImageStructured"` bisa hidup berdampingan tanpa mengubah skema Qdrant.

Satu peringatan untuk desain eksperimen: bila kedua varian di-index ke collection yang sama, keduanya akan bersaing di retrieval yang sama, dan perbandingan strateginya tercemar. Dua varian untuk gambar yang sama menuntut **dua collection terpisah** — yang berarti `QDRANT_COLLECTION` (`config.py:75`) menjadi parameter eksperimen, dan konsekuensinya lihat I19.

### G10. Apakah `chunk_index` deterministik dan stabil lintas re-index?

**Tidak. Dan lebih buruk dari yang kamu duga: `chunk_index` bahkan tidak unik di dalam satu collection.**

Ini pertanyaan terpenting di bagian ini, jadi saya telusuri penuh.

#### Bagaimana `chunk_index` dihasilkan

Satu-satunya sumber nilainya adalah `len(chunks)` pada saat append, di tiga tempat dalam `_chunk_elements` (`preprocessing.py:308-407`):

| Lokasi | Cabang |
|---|---|
| `preprocessing.py:333` | `flush()` — chunk teks |
| `preprocessing.py:366` | Tabel besar (`len(text) > PDF_TABLE_MAX_CHARS`) |
| `preprocessing.py:385` | Deskripsi gambar |

Jadi ia adalah **penghitung berjalan per berkas**, ditetapkan sesuai urutan element yang masuk. Tidak ada komponen isi, tidak ada hash, tidak ada offset halaman.

#### Konsekuensi 1 — pergeseran kaskade

Ya, persis seperti dugaanmu. Bila satu halaman berubah sehingga jumlah chunk pada halaman itu berubah, **setiap chunk sesudahnya bergeser**. Ini mengikuti langsung dari `len(chunks)` sebagai satu-satunya sumber nilai.

Diperparah oleh siklus re-index: perubahan isi berkas mengubah `file_sha256` (`preprocessing.py:91-97`), yang terdeteksi di `indexing.py:209`, memicu `delete_file_chunks(f.name)` (`:221`) untuk **seluruh berkas**, lalu seluruh ruang indeks berkas itu ditulis ulang dari nol. Tidak ada mekanisme yang mempertahankan indeks lama.

#### Konsekuensi 2 — ketidakstabilan tanpa perubahan berkas sama sekali

Ini yang tidak terlihat dari pembacaan sepintas. `chunk_index` dapat bergeser meski PDF-nya **identik bit-per-bit**:

| Pemicu | Mekanisme |
|---|---|
| Deskripsi gambar berubah verdict | `describe_image` mengembalikan `None` untuk `DEKORATIF` (`image_describer.py:128-130`). Panggilan dijalankan pada `temperature=0.1` (`:151`) — bukan 0. Bila satu gambar dinilai dekoratif di run A dan tidak di run B, satu element hilang/muncul di `preprocessing.py:294`, dan **semua chunk sesudahnya bergeser** |
| Cache deskripsi hilang antar proses | `_description_cache` (`image_describer.py:37`) adalah dict proses-lokal. Re-index di proses baru = query ulang model vision = kesempatan baru untuk verdict berbeda |
| Kegagalan VL transient | `describe_image` mengembalikan `None` pada exception apa pun (`:119-121`), termasuk timeout. Satu timeout = satu chunk hilang = pergeseran |
| Perubahan `CHUNK_SIZE`/`CHUNK_OVERLAP` | Ambang flush di `preprocessing.py:393` dan tail overlap di `:397` |
| `hi_res` jatuh ke `fast` | `preprocessing.py:209-211` menangkap kegagalan `partition_pdf` dan memanggil `_extract_fast`. Hasilnya himpunan element yang sama sekali berbeda |

#### Konsekuensi 3 — `chunk_index` tidak unik per titik Qdrant

Ini temuan yang paling merusak untuk `gold_chunk_ids`.

`indexing.py:160` memanggil `VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)` — **tanpa argumen `transformations=` maupun `node_parser=`**. LlamaIndex karenanya menerapkan node parser default-nya, yang dikonfigurasi dari `Settings.chunk_size` dan `Settings.chunk_overlap` — dan kedua nilai itu disetel di `rag_pipeline.py:624-625` menjadi `CHUNK_SIZE` (512) dan `CHUNK_OVERLAP` (128).

Artinya dokumen yang sudah di-chunk oleh `_chunk_elements` **di-chunk ulang** oleh LlamaIndex. Dan `_chunk_elements` memang menghasilkan chunk yang melebihi 512 karakter:

- Tabel besar sampai `PDF_TABLE_MAX_CHARS` = 2000 karakter (`config.py:111`, `preprocessing.py:357`)
- Pemeriksaan flush di `:393` terjadi **sebelum** append, sehingga satu element tunggal yang lebih panjang dari `CHUNK_SIZE` masuk utuh
- Deskripsi gambar di-append utuh (`:381-388`)

Setiap node hasil pemecahan mewarisi metadata Document-nya — termasuk **`chunk_index` yang sama**. Sebuah tabel 2000 karakter menjadi beberapa titik Qdrant yang semuanya mengklaim `chunk_index` identik.

Yang ikut rusak karenanya:

| Yang rusak | Lokasi |
|---|---|
| Dedup ekspansi tetangga | `rag_pipeline.py:754-759` dan `:803-806` memakai kunci `(file_name, chunk_index)`. Beberapa titik berbagi kunci → hanya satu yang lolos |
| Filter rentang tetangga | `rag_pipeline.py:781` (`Range(gte=..., lte=...)` atas `chunk_index`) mengambil semua duplikat |
| Identitas untuk `gold_chunk_ids` | `(file_name, chunk_index)` bukan kunci |

**Peringatan kejujuran:** klaim tentang node parser default LlamaIndex berasal dari perilaku library, bukan dari kode repo ini. Yang **dapat** saya buktikan dari repo: `indexing.py:160` tidak mengirim override, dan `rag_pipeline.py:624-625` menyetel `Settings.chunk_size`/`chunk_overlap`. Verifikasinya tidak butuh korpus — perintahnya ada di bagian "Menunggu Korpus", butir V1.

#### Konsekuensi 4 — entry point berbeda menghasilkan chunking berbeda

`POST /api/index` (`main.py:265-283`) memanggil `index_documents` **tanpa** memanggil `_configure_settings()` lebih dulu. Jadi `Settings.chunk_size` masih bernilai default LlamaIndex, bukan `CHUNK_SIZE`. Jalur CLI memanggilnya di `scripts/index_documents.py:54`. Dua entry point, dua ukuran chunk sekunder. (Laporan pertama sudah mencatat pola yang sama untuk model embedding.)

#### Jawaban ringkas

`chunk_index` tidak layak dipakai sebagai dasar `gold_chunk_ids`. Ia tidak unik, tidak stabil terhadap perubahan berkas, dan tidak stabil bahkan terhadap re-index berkas yang tidak berubah. **Anotasi gold yang dibangun di atasnya akan batal setiap re-index** — persis kekhawatiranmu, ditambah satu masalah keunikan yang tidak diantisipasi.

Skema sendiri sudah memilih bentuk yang lebih baik: `chunk_id` = `{document_id}_p{page}_c{NN}`, yaitu **ordinal dalam halaman**. Itu strictly lebih tahan daripada pencacah per berkas yang ada sekarang — perubahan di halaman 1 tidak menggeser penomoran halaman 3. Tetapi ia tidak menyelesaikan dua dari empat konsekuensi di atas: pergeseran **di dalam** halaman yang berubah tetap terjadi, dan pemecahan ulang oleh LlamaIndex (konsekuensi 3) tetap menghasilkan beberapa titik Qdrant untuk satu `chunk_id`.

Jadi ada dua pekerjaan terpisah, dan hanya yang pertama yang dituntut skema:

1. **Ganti granularitas penomoran dari per-berkas ke per-halaman**, agar sesuai `chunk_id` di skema. Menyentuh `preprocessing.py:308-407` — pencacah `len(chunks)` di `:333`, `:366`, `:385` perlu direset per halaman dan `document_id` perlu tersedia.
2. **Selesaikan pemecahan ulang LlamaIndex** di `indexing.py:160`, atau `chunk_id` tidak akan pernah menjadi kunci unik apa pun yang kamu simpan. Ini independen dari skema penamaan dan harus dikerjakan lebih dulu.

Saran tambahan yang tidak melawan skema: simpan hash teks chunk ternormalisasi sebagai kolom pendamping. Bahannya sudah ada — `file_sha256` (`preprocessing.py:91-97`) membuktikan pola hashing sudah dipakai di repo. Dengan itu, re-index yang menggeser penomoran dapat memetakan ulang anotasi gold secara otomatis alih-alih membatalkannya. Rincian pertimbangannya sama dengan yang saya uraikan untuk `image_id` di G11.

Terakhir, `chunk_index` yang ada sekarang **harus tetap dipertahankan** sebagai penanda urutan, apa pun keputusan penamaan: `_expand_with_neighbors` memfilter rentangnya di `rag_pipeline.py:781`. Id stabil adalah tambahan, bukan pengganti.

### G11. Identitas gambar yang stabil terhadap isi

Prasyaratnya: gambar harus disimpan lebih dulu (laporan 1, A2 — saat ini tidak pernah menyentuh disk).

**Bahan yang sudah ada di repo.** `_hash_image` (`image_describer.py:40-41`) sudah menghitung `hashlib.sha256(image_bytes).hexdigest()[:16]`. Yang penting: ia dipanggil di `describe_image:105` **sebelum** `_resize_image_if_needed` di `:109`. Jadi hash-nya atas **bytes asli** hasil ekstraksi, bukan atas hasil resize.

Urutan itu menentukan. Hash atas bytes pasca-resize tidak akan stabil, karena `_resize_image_if_needed` melakukan re-encode lossy — `optimize=True` untuk PNG (`:64`) dan kualitas 85 untuk JPEG (`:62`) — yang keluarannya bergantung pada versi Pillow (dipin `11.0.0` di `requirements.txt:93`, tetapi pin bisa berubah). Menaikkan Pillow akan mengubah setiap id. Hash pra-resize tidak punya masalah itu.

**Apa yang stabil dan apa yang tidak:**

| Kandidat penamaan | Stabil terhadap isi? | Catatan |
|---|---|---|
| SHA-256 atas bytes gambar asli | **Ya** | Sudah dihitung di `image_describer.py:41`, tinggal dinaikkan ke pemanggil |
| Truncation 16 hex (64 bit) | Ya, dengan catatan | Cukup untuk skala korpus akademik, tapi 64 hex penuh tidak menambah biaya dan menghilangkan pertanyaan tabrakan dari review |
| Urutan pemrosesan (`enumerate`) | **Tidak** | Bergeser persis seperti `chunk_index` (G10) |
| `(file_name, page, idx-dalam-halaman)` | **Tidak** | `page` sendiri tidak stabil: jalur `hi_res` memakai `el.metadata.page_number` dengan fallback ke `0` (`preprocessing.py:216`) |
| Nama berkas hasil Unstructured | Tidak berlaku | Repo memakai `extract_image_block_to_payload=True` (`:206`), jadi tidak ada berkas yang dihasilkan |

**Dua identitas yang perlu dibedakan.** Hash isi menyatukan gambar identik yang muncul berkali-kali — logo kop surat yang sama di lima dokumen menjadi satu id. Untuk `images.jsonl` itu benar (satu baris per gambar unik). Untuk menautkan gambar ke chunk tempat ia muncul, dibutuhkan id kemunculan yang membawa konteks berkas dan halaman. Keduanya diturunkan dari isi, tidak ada yang diturunkan dari urutan.

#### Rekonsiliasi dengan skema

Skema memilih pendekatan lain: `image_id` berformat `{document_id}_p{page}_img{NN}` dan `file_path` berformat `images/{document_id}/p{page}_img{NN}.png` — **ordinal dalam halaman**, bukan turunan isi. Perlu dinilai jujur, karena ini keputusan desain yang sudah diambil, bukan kelalaian:

**Yang membuatnya lebih baik dari kondisi kode sekarang.** Ordinal *dalam halaman* jauh lebih tahan daripada pencacah global. Gambar baru di halaman 3 tidak menggeser penomoran di halaman 7. Bandingkan dengan `chunk_index` sekarang yang berjalan per berkas (`preprocessing.py:333`) sehingga satu penyisipan di awal menggeser semuanya. Untuk anotasi manusia, id yang terbaca (`p3_img01`) juga jauh lebih praktis daripada hash 64 hex.

**Yang tetap rapuh.** Id ini bergeser bila jumlah gambar pada halaman yang sama berubah — penambahan gambar, atau perubahan apa yang dianggap gambar oleh ekstraktor. Ia juga terikat pada `page_number`, yang di jalur hi_res bisa jatuh ke `0` (`preprocessing.py:216`). Dan yang paling relevan: ia bergantung pada **urutan enumerasi Unstructured di dalam satu halaman** — properti library, bukan properti dokumen, sehingga bisa berubah saat versi `unstructured` dinaikkan (dipin `0.16.11` di `requirements.txt:90`).

**Yang saya sarankan, tanpa melawan skema.** Simpan sha256 bytes asli sebagai field tambahan di `images.jsonl` di luar skema minimum — `_hash_image` (`image_describer.py:40-41`) sudah menghitungnya di titik yang tepat, sebelum resize. Id ordinal tetap menjadi kunci utama seperti di skema; hash menjadi jaring pengaman yang memungkinkan pemetaan ulang otomatis bila re-index menggeser penomoran, alih-alih menganotasi ulang secara manual. Biayanya satu kolom; imbalannya anotasi gold yang bisa dipulihkan.

Alasan yang sama berlaku untuk `chunk_id`. Skema memakai `{document_id}_p{page}_c{NN}` — juga ordinal dalam halaman, juga lebih tahan daripada `chunk_index` per berkas yang ada sekarang. Menyimpan hash teks chunk ternormalisasi sebagai kolom pendamping memberi jalur pemulihan yang sama.

**Interaksi dengan `PDF_MIN_IMAGE_SIZE_KB`.** Filter ukuran (`image_describer.py:76`) berjalan di `is_likely_informative` — yang, perlu dicatat, **tidak pernah dipanggil dari jalur indexing**: grep menunjukkan fungsi itu hanya didefinisikan (`:71`) dan tidak punya pemanggil di `backend/` maupun `scripts/`. Jadi ambang 20 KB saat ini tidak berefek pada apa pun. Bila gambar mulai disimpan, keputusan apakah menulis gambar kecil ke disk perlu dibuat eksplisit — bukan diwarisi dari fungsi mati ini.

---

## H. KESIAPAN EVALUASI

### H12. Apakah response memaparkan hasil per tahap retrieval?

**Tidak. Hanya tahap terakhir, dan itu pun sudah termodifikasi.**

Yang dikembalikan `/api/query`: `answer`, `sources`, `condensed_question`, `debug` (`schemas.py:161-165`). `sources` dibangun `_build_sources` (`rag_pipeline.py:891-902`) dengan enam field: `file_name`, `page`, `chunk_index`, `element_type`, `score`, `text_preview`.

Tidak ada `node_id`, tidak ada id titik Qdrant, tidak ada penanda tahap. Diverifikasi: grep atas `node_id`/`doc_id`/`ref_doc_id` di `rag_pipeline.py` dan `indexing.py` tidak menghasilkan apa pun.

#### Di mana informasi tiap tahap masih ada sebelum dibuang

Semuanya di dalam `_retrieve_and_rerank` (`rag_pipeline.py:840-888`):

| Tahap | Variabel | Baris tersedia | Baris hilang | Isi |
|---|---|---|---|---|
| **Dense** | `nodes` | `:871` (`retriever.retrieve(question)`) | Keluar dari cakupan di `:888` | `list[NodeWithScore]` — peringkat dan skor kemiripan lengkap untuk `SIMILARITY_TOP_K` |
| **Ekspansi tetangga** | `expanded_nodes` | `:878` (`_expand_with_neighbors(nodes)`) | Keluar dari cakupan di `:888` | Nodes dense + tetangga. Tetangga diberi `score=0.0` (`:826`), sehingga **dapat dibedakan** dari nodes dense |
| **Rerank** | `reranked` | `:884` | Diteruskan | Hanya `RERANKER_TOP_N` teratas. Nodes yang tidak lolos **hilang seluruhnya** |

`return reranked, top_score` di `:888` adalah titik pembuangannya. Dense dan expanded tidak pernah keluar dari fungsi.

Jejak parsial yang tersisa:

- `rag_pipeline.py:832-835` mencatat `retrieved={len(nodes)} expanded={len(expanded_nodes)}` — tetapi **hanya bila `len(expanded_nodes) > len(nodes)`** (`:831`). Bila ekspansi tidak menambah apa pun, tidak ada baris log sama sekali.
- `debug["timings_ms"]` (`schemas.py:88-95`) memuat kunci `retrieve`, `neighbor_expansion`, `rerank` — durasi saja, bukan isi.
- `debug["sources_returned"]` (`:1239`, `:1277`) — jumlah pasca-rerank saja.

#### Dua kontaminasi pada data yang memang terekspos

**`text_preview` bukan teks chunk.** `SourceLabelPostprocessor` berjalan di `:885`, **setelah** rerank, dan menulis ulang `node.node.text` menjadi `f"[{clean_name}]\n{node.node.text}"` (`:284`). `_build_sources` baru dipanggil di `:891`, jadi `n.text[:300]` (`:899`) sudah memuat label yang disisipkan. Teks yang sama juga masuk `context_str` (`:1244`).

**`score` berubah makna diam-diam.** Bila TEI rerank gagal (`:684`, `:690`, `:696`), skor yang tersisa adalah skor dense, bukan skor reranker — dan tidak ada field di `DebugInfo` yang menandakannya. Counter `RERANK_FALLBACK` ada di `metrics.py` tetapi tidak muncul di response.

**Kesimpulan untuk lapis 1:** Precision@k, Recall@k, MRR@k, dan nDCG@k **tidak dapat dihitung per tahap** dari response saat ini. Untuk tahap rerank pun peringkatnya ada tetapi identitas chunk-nya tidak stabil (G10). Titik perubahan tunggal yang paling sempit adalah tanda tangan kembalian `_retrieve_and_rerank` di `rag_pipeline.py:888`.

### H13. Menjalankan retrieval tanpa generation

**Lewat API: tidak ada.** Rute yang terdaftar (diverifikasi lewat grep atas semua dekorator `@app`/`@router`):

| Rute | Lokasi |
|---|---|
| `GET /` | `main.py:127` |
| `GET /api/files/{key}` | `main.py:132` |
| `GET /api/health` | `main.py:173` |
| `POST /api/index` | `main.py:258` |
| `POST /api/chat`, `POST /api/chat/stream`, `GET /api/sessions` | `routers/chat.py:54`, `:141`, `:246` |
| `POST /api/query`, `POST /api/query/stream` | `routers/query.py:20`, `:95` |
| `POST /api/login`, `POST /api/logout`, `GET /api/me` | `routers/auth.py:43`, `:82`, `:89` |

Tidak ada endpoint retrieval-only. Setiap jalur query berakhir di generasi LLM.

**Lewat pemanggilan fungsi langsung: bisa, dengan tiga biaya.**

`_retrieve_and_rerank(question, role="public", timings=None)` (`rag_pipeline.py:840`) dapat dipanggil langsung dan mengembalikan `(reranked_nodes, top_score)` tanpa menyentuh LLM. Tetapi:

1. **Mengimpor modulnya mahal dan rapuh.** `rag_pipeline.py:42-58` mengimpor di level modul: `metrics` (butuh `prometheus_client`), `intent_classifier` (butuh `torch` + `transformers` — `intent_classifier.py:18-19`), `keyword_filter` (yang **melempar** bila `blocked_keywords.yaml` gagal dimuat pertama kali — `keyword_filter.py:141-143`), `vision`, `moderation`, dan `indexing` → `preprocessing` → `fitz`.
2. **`_get_retriever()` mengonstruksi klien LLM.** `:634` memanggil `_configure_settings()`, yang menyetel `Settings.llm = get_llm()` di `:623`. Ini membangun klien; tidak ada panggilan generasi yang dilakukan. Apakah konstruksi `OpenAILike` melakukan handshake jaringan adalah perilaku SDK yang tidak dapat saya pastikan dari repo ini — perintah verifikasinya ada di "Menunggu Korpus", butir V4.
3. **Hasilnya tetap hanya tahap rerank** (H12).

**Jalur alternatif yang lebih bersih**, seluruhnya di luar `rag_pipeline`: `get_qdrant_client()` dari `indexing.py:33` (yang hanya bergantung pada `config` dan `preprocessing`) + `QdrantVectorStore` + `VectorStoreIndex.from_vector_store`, meniru `rag_pipeline.py:636-642`. Menghindari torch, prometheus, dan YAML keyword. Ongkosnya: ekspansi tetangga (`:736-837`) dan reranking (`:647-733`) harus dipanggil/ditulis ulang sendiri.

### H14. Stratifikasi per tipe visual

**Tidak ada field yang memungkinkan pengelompokan flowchart / tabel / formulir / figur deskriptif.**

Nilai `element_type` yang mungkin ada di Qdrant, lengkap:

| Nilai | Sumber |
|---|---|
| Gabungan kategori Unstructured, mis. `"NarrativeText"`, `"Title+NarrativeText"` | `preprocessing.py:334` (`"+".join(sorted(current_categories))`) |
| `"text"` | `preprocessing.py:334`, fallback bila himpunan kategori kosong |
| `"NarrativeText"` | `preprocessing.py:167`, literal untuk seluruh jalur `fast` |
| `"Table"` | `preprocessing.py:367` |
| `"ImageDescription"` | `preprocessing.py:386` |
| `"narrative"` | `index_narratives.py:313` |

Jadi tabel **bisa** distratifikasi (`element_type="Table"`), tetapi hanya untuk berkas yang lewat jalur `hi_res` — jalur `fast` menandai segalanya `"NarrativeText"` (`:167`) termasuk tabel yang tergilas menjadi teks datar.

Flowchart, formulir, dan figur deskriptif **semuanya jatuh ke satu ember `"ImageDescription"`**. Tidak ada yang membedakannya.

Yang membuat ini terasa lebih disayangkan: prompt deskripsi **sudah meminta model membedakannya**. `image_describer.py:28-31` memuat empat cabang eksplisit (diagram/flowchart, tabel, screenshot UI, foto/dekorasi). Tetapi `describe_image` hanya mengekstrak dua sentinel di `:128` — `DEKORATIF` dan `TIDAK JELAS`. Klasifikasi jenis visual yang model hasilkan larut dalam prosa dan tidak pernah diambil. Sinyalnya diproduksi lalu dibuang di baris yang sama.

**Skema sudah memutuskan cara mengisinya, dan itu kabar baik.** `images.jsonl.visual_type` ditandai "label hasil anotasi manusia, dipakai untuk stratifikasi", dengan empat nilai `"flowchart"` / `"tabel_sebagai_gambar"` / `"formulir"` / `"figur_deskriptif"`. Jadi label tidak perlu diturunkan dari kepatuhan model terhadap format prompt — pilihan yang jauh lebih dapat dipertahankan di paper, dan yang menghilangkan ketergantungan pada `image_describer.py:128`.

Yang tetap dibutuhkan adalah **propagasi label itu ke chunk**. Metrik lapis 1 diukur atas hasil retrieval, dan hasil retrieval berupa chunk — bukan baris `images.jsonl`. Agar Recall@k dapat dipecah per strata, `visual_type` harus ikut di payload chunk (`indexing.py:251-260`) atau chunk harus membawa `image_id` yang bisa di-join ke `images.jsonl`. Yang kedua lebih rapi dan sejalan dengan struktur skema.

Satu hal yang masih perlu keputusan: dokumen dengan beberapa jenis visual. `corpus_metadata.jsonl` punya `primary_visual_type` per dokumen, `images.jsonl` punya `visual_type` per gambar, dan `qa_pairs.jsonl` punya `visual_type_required` per pertanyaan. Ketiganya bisa berbeda untuk satu pertanyaan. Stratifikasi lapis 1 sebaiknya mengikuti `visual_type_required` (dari sisi pertanyaan), bukan `primary_visual_type` (dari sisi dokumen) — tetapi ini keputusan desain eksperimen, dan kode tidak memaksakan salah satunya.

### H15. Logging structlog di jalur query

**Temuan utama: structlog dikonfigurasi tetapi tidak pernah dipakai untuk emisi log. Docstring-nya salah.**

`logging_config.py:3-4` menyatakan: *"Setelah configure_logging() dipanggil, semua stdlib logging.getLogger(...) akan menghasilkan JSON log."* Itu tidak terjadi.

Alasannya, dari `logging_config.py:14-33`:

- `structlog.configure(...)` (`:14-27`) menyetel pipeline dengan `JSONRenderer` (`:22`) dan `PrintLoggerFactory` (`:25`). Ini hanya berlaku untuk logger yang diperoleh lewat `structlog.get_logger()`.
- `logging.basicConfig(format="%(message)s", level=..., force=True)` (`:29-33`) mengonfigurasi stdlib secara terpisah, mencetak pesan mentah.
- **Tidak ada `structlog.stdlib.ProcessorFormatter`** yang menjembatani keduanya. Tanpa itu, rekaman stdlib tidak pernah melewati prosesor structlog.

Diverifikasi lewat grep: **tidak ada satu pun `structlog.get_logger()` di seluruh `backend/`**. Kemunculan structlog hanya di `logging_config.py` (`:8`, `:14-27`) dan `main.py` (`:9`, `:96-97`).

Konsekuensinya, `main.py:96-97` mengikat `request_id` ke contextvars structlog — tetapi karena setiap baris log ditulis lewat `logging.getLogger(__name__)` (`main.py:30`, `rag_pipeline.py:60`), prosesor `merge_contextvars` (`logging_config.py:16`) tidak pernah berjalan. `request_id` hanya sampai ke output karena di-f-string secara manual ke dalam pesan di `main.py:105`. **Tidak ada baris log dari `rag_pipeline.py` yang membawa `request_id`.**

#### Yang benar-benar di-log di jalur query

Semua lewat stdlib `logger`, sebagai teks datar dengan `key=value` ad hoc di dalam f-string:

| Peristiwa | Lokasi | Level |
|---|---|---|
| `http_request method=... path=... status=... duration_ms=... request_id=...` | `main.py:103-106` | INFO |
| `stream_complete` / `stream_aborted ... duration_ms=... request_id=...` | `main.py:82-85` | INFO |
| `L2_bypassed reason=...` | `rag_pipeline.py:1111`, `:1376` | WARNING |
| `L2_moderation_blocked reason=... l1_flags=...` | `:1113-1117`, `:1378` | WARNING |
| `L3_identity_override → chitchat question=...` | `:1127`, `:1386` | INFO |
| `L3_using_fallback intent=...` | `:1137` | WARNING |
| `L3_intent=... conf=...` | `:1138`, `:1398` | INFO |
| `condense_fallback reason=...` | `:548`, `:551`, `:557` | INFO/WARNING |
| `condense_llm_error error=...` | `:577` | ERROR |
| `Condensed: '...' → '...'` | `:597` | INFO |
| `Cache hit for: '...'` | `:1209`; juga `cache.py:63` | INFO |
| `Query expanded: '...' → '...'` | `:1223` | INFO |
| `neighbor_expansion retrieved=... expanded=... radius=...` | `:832-835` | INFO — **hanya bila ekspansi menambah node** (`:831`) |
| `neighbor_expansion_error file=... error=...` | `:795` | WARNING |
| `tei_rerank_timeout timeout_sec=... n_nodes=...` | `:679-682` | WARNING |
| `tei_rerank_http_error error=...` / `tei_rerank_unexpected error=...` | `:686-688`, `:692-694` | ERROR |
| `Top score ... < ... — low-relevance fallback` | `:1230` | INFO |
| `L5e_marginal_confidence top_score=... threshold=... buffer=...` | `:247-250` | INFO |
| `RAG done in ...s — top_score=..., sources=..., band=...` | `:1266-1269`, `:1524-1527` | INFO |
| `output_filter_redacted pattern=... session=...` | `output_filter.py:93`, `:110` | WARNING |
| Baris setup satu kali (`embed_provider=`, `reranker_provider=`, `Retriever ready.`) | `:613`, `:622`, `:633`, `:643`, `:721-724`, `:732` | INFO |

#### Cukupkah untuk merekonstruksi satu run dari log saja?

Tidak. Yang **ada**: intent dan confidence, apakah condensation berjalan dan hasilnya, cache hit/miss, top_score, jumlah sumber, confidence band, durasi total, dan kegagalan rerank.

Yang **kurang**, dan tanpa itu rekonstruksi tidak mungkin:

| Kekurangan | Akibat |
|---|---|
| **Tidak ada `request_id` di baris `rag_pipeline`** | Di bawah konkurensi, baris dari request berbeda saling menyisip tanpa cara memisahkannya. Satu-satunya `request_id` ada di baris `http_request` milik middleware |
| **Tidak ada identitas chunk apa pun** | Baris log tidak pernah menyebut `file_name`, `chunk_index`, apalagi id titik. Himpunan yang di-retrieve tidak dapat direkonstruksi |
| **Tidak ada peringkat maupun skor per node** | Hanya `top_score` agregat |
| **Baris ekspansi tetangga bersyarat** | `:831` menekan baris log bila ekspansi tidak menambah apa pun — "nol tetangga" tidak dapat dibedakan dari "layer tidak berjalan" |
| **Tidak ada `timings_ms` di log** | Hanya masuk response (`schemas.py:88`), tidak pernah di-log |
| **Tidak ada versi model** | Lihat F2 |
| **Tidak ada teks pertanyaan lengkap** | Dipotong 50–60 karakter di `:597`, `:1127`, `:1209` |
| **Jalur bahagia terlalu senyap** | Query normal tanpa condensation, tanpa cache, tanpa ekspansi, tanpa kegagalan rerank hanya menghasilkan dua baris: `L3_intent=` dan `RAG done in` |

Untuk eksperimen, log **bukan** medium yang layak. Response `debug` jauh lebih kaya (`schemas.py:57-101`) — di situlah instrumentasi tambahan sebaiknya diletakkan, bukan di logging.

---

## I. PERBANDINGAN ADIL DUA STRATEGI

### I16. Yang HARUS identik antara dua fork

Agar perbedaan skor hanya berasal dari strategi indexing, semua yang berikut harus sama persis.

**Ekstraksi PDF — `backend/services/preprocessing.py`, seluruhnya:**

| Fungsi | Baris | Mengapa |
|---|---|---|
| `file_sha256` | `:91-97` | Menentukan berkas mana yang dianggap berubah |
| `detect_strategy` | `:102-125` | Menentukan `fast` vs `hi_res` per berkas. Ambang `> 1.0` (`:122`) dan sampel 5 halaman (`:112`) harus sama |
| `_get_ocr_engine` | `:40-52` | `lang` dan `device` PaddleOCR |
| `_extract_texts_from_ocr_result` | `:55-86` | Parsing keluaran OCR |
| `_extract_text_from_page_fast` | `:130-153` | Ambang OCR 50 karakter (`:133`), DPI 300 (`:137`) |
| `_extract_fast` | `:156-173` | — |
| `_extract_hi_res` | `:189-256` | Argumen `partition_pdf` (`:201-208`), termasuk `languages=["ind","eng"]` (`:207`) |
| `_html_table_to_markdown` | `:178-186` | — |
| `_chunk_elements` | `:308-407` | Semua logika pemotongan |

**Embedding — dan ini yang paling mudah terlewat:**

| Item | Lokasi |
|---|---|
| `_configure_settings` | `rag_pipeline.py:603-625` |
| `_configure_embed` (duplikat) | `index_narratives.py:183-201` |
| `EMBED_PROVIDER`, `EMBED_MODEL`, `EMBED_BATCH_SIZE`, `EMBED_DEVICE`, `EMBED_DIMENSION` | `config.py:40-46` |
| `--model-id` TEI | `docker-compose.poc.yml:95` |
| `Settings.chunk_size` / `chunk_overlap` | `rag_pipeline.py:624-625` — karena mengendalikan re-chunking LlamaIndex (G10) |

**Penyimpanan:**

| Item | Lokasi |
|---|---|
| Konstruksi metadata `Document` | `indexing.py:249-261` |
| `_embed_and_store` | `indexing.py:149-164` |
| `_ensure_collection` (`VectorParams`) | `indexing.py:119-146` |

**Retrieval — seluruh konfigurasinya, karena perbandingannya di sisi indexing:**

`SIMILARITY_TOP_K`, `SCORE_THRESHOLD`, `NEIGHBOR_EXPANSION_*`, `MAX_EXPANDED_CHUNKS`, `RERANKER_PROVIDER`, `RERANKER_MODEL`, `RERANKER_TOP_N`, `RERANKER_TIMEOUT` (`config.py:55-125`), beserta `_expand_with_neighbors` (`rag_pipeline.py:736-837`), `_retrieve_and_rerank` (`:840-888`), `_TEIRerankPostprocessor` (`:647-707`), `SourceLabelPostprocessor` (`:272-285`), `_expand_query` + `_ACADEMIC_ACRONYMS` (`:290-299`, `:176-189`).

`RERANKER_TIMEOUT` layak disorot: pada nilai 5 detik (`config.py:60`) di mesin bersama, satu fork bisa mengalami lebih banyak fallback daripada yang lain semata karena beban mesin — memasukkan perbedaan yang tidak berasal dari strategi mana pun.

**Deskripsi gambar, bila kedua fork memakainya:** `_DESCRIPTION_PROMPT` (`image_describer.py:24-34`), `max_tokens`/`temperature` (`:150-151`), `_MAX_IMAGE_DIM` (`config.py:107`), dan `LLM_MODEL`.

### I17. Yang aman berbeda

| Aman berbeda | Alasan |
|---|---|
| **Isi teks chunk gambar** | Inilah variabel yang dibandingkan. Titik divergensi alaminya `preprocessing.py:296-301` dan `:381-388` |
| **Field metadata tambahan** | Payload Qdrant tidak berskema kaku; menambah key tidak mengganggu `_build_sources` (`rag_pipeline.py:891-902`) yang membaca lewat `.get()` |
| **Apakah gambar disimpan ke disk dan bagaimana dinamai** | Selama teks chunk yang di-embed tetap dikendalikan |
| **Nilai `element_type`** | Bebas-nilai (`indexing.py:256`) — asal jangan menabrak `"Table"` yang dipakai `_chunk_elements:367` |
| **`QDRANT_COLLECTION`** | **Wajib berbeda.** Lihat I19 |
| **Prompt tambahan untuk ringkasan terstruktur** | Selama prompt naratif tetap identik bila kedua fork juga memproduksi naratif |
| **Skrip analisis dan pelaporan** | Di luar pipeline |

Satu yang **tidak** aman meski terlihat begitu: `PDF_EXTRACTION_STRATEGY`. Bernilai `auto` (`config.py:101`), routing-nya bergantung isi PDF (`preprocessing.py:428`). Selama kedua fork memakai korpus yang sama dan `detect_strategy` yang sama, hasilnya sama — tetapi bila salah satu fork mengubah ambang di `:122`, seluruh perbandingan runtuh. Aman hanya karena berada di daftar I16, bukan karena sifatnya sendiri.

### I18. Kelayakan memisahkan bagian identik jadi paket bersama

**Penilaian: layak, ukuran sedang.** Tidak diperlukan refactor besar.

**Yang memudahkan:**

- **Tidak ada import melingkar** (dikonfirmasi laporan 1, B8).
- Klaster indexing hampir bersih terpisah: `preprocessing.py` mengimpor hanya `config` (`:23`) plus `image_describer` secara lazy (`:274`); `image_describer.py` hanya `config` (`:17`); `indexing.py` hanya `config` (`:20`) dan `preprocessing` (`:26`).
- Semua import berat sudah lazy: `paddleocr` (`preprocessing.py:43`), `unstructured` (`:194`), `markdownify` (`:181`), PIL (`image_describer.py:48`), `httpx` (`:138`, `:164`).

**Tiga penghalang, semuanya berbatas jelas:**

| # | Penghalang | Ukuran |
|---|---|---|
| 1 | `config.py` monolitik. `preprocessing.py:23-31` mengimpor 10 nama yang semuanya relevan-indexing, tetapi modulnya juga mengevaluasi env DB/Redis/JWT saat import (`config.py:149-167`) | **Kecil** — memindahkan konstanta indexing ke modul terpisah bersifat mekanis |
| 2 | `scripts/index_documents.py:20` mengimpor `_configure_settings` dari `rag_pipeline`, sehingga indexing menarik seluruh stack retrieval | **Kecil** — fungsi itu (`rag_pipeline.py:603-625`) hanya memakai `EMBED_*`, `get_llm()`, dan `CHUNK_*`. Memindahkannya ke modul bersama adalah pemindahan yang berdiri sendiri |
| 3 | `_configure_embed` di `index_narratives.py:183-201` adalah duplikat yang sudah melenceng (`Settings.llm = None` di `:201` vs `get_llm()` di `rag_pipeline.py:623`) | **Kecil** — hapus duplikat setelah #2 |

**Batas alami paket bersama:** `config` indexing + `preprocessing` + `image_describer` + `indexing` + `_configure_settings` yang dipindahkan. Itu memenuhi hampir seluruh daftar I16 kecuali bagian retrieval.

**Peringatan yang lebih penting daripada refactor-nya.** Paket bersama menjamin *kode* identik, bukan *konfigurasi* identik. `_chunk_elements` berperilaku beda pada `CHUNK_SIZE` beda; `detect_strategy` berperilaku beda pada ambang beda; `Settings.chunk_size` mengubah re-chunking LlamaIndex (G10). Semuanya nilai runtime, bukan kode. Yang sebenarnya dibutuhkan kedua fork adalah **manifest konfigurasi yang tercatat di dalam artefak** — nilai efektif dari daftar "Kelompok 1" laporan pertama, ditulis ke metadata chunk atau ke berkas pendamping saat indexing. Paket bersama tanpa manifest hanya memindahkan risiko, tidak menghilangkannya.

Bagian retrieval dari I16 lebih sulit dipisah karena tertanam di `rag_pipeline.py` bersama moderasi, intent, dan vision. Menariknya keluar berukuran **besar**. Alternatif yang jauh lebih murah: kunci nilai konfigurasinya lewat berkas `.env` bersama dan verifikasi kesamaannya, alih-alih berbagi kode.

### I19. Dua proses indexing bersamaan ke collection berbeda

| Kategori | Item | Status |
|---|---|---|
| **Variabel global** | `_ocr_engine` (`preprocessing.py:37`), `_description_cache` (`image_describer.py:37`), `_retriever`/`_reranker` (`rag_pipeline.py:63-64`), `Settings.*` | **Aman.** Semuanya per-proses. Dua proses OS terpisah tidak berbagi apa pun |
| **Direktori temporer** | `tempfile.NamedTemporaryFile(suffix=".png", delete=False)` (`preprocessing.py:140`) | **Aman.** Nama acak dari stdlib. Tetapi `delete=False` dengan `os.unlink` di `finally` (`:153`) berarti **crash antara `:142` dan `:153` meninggalkan PNG yatim** di direktori temp sistem. Dua proses menggandakan lajunya |
| **Nama collection** | `QDRANT_COLLECTION` (`config.py:75`) | **Aman bila benar-benar berbeda** — dan ini bergantung pada kedua proses memuat `.env` berbeda. `config.py:7` memanggil `load_dotenv()` yang mencari `.env` relatif ke direktori kerja. Dua proses di direktori kerja yang sama akan memuat `.env` yang sama dan menulis ke collection yang sama tanpa peringatan |
| **`_ensure_collection`** | `indexing.py:135-143` | **Berbahaya bila nama bertabrakan.** Pada error yang memuat `"already exists"`, kode memanggil `client.delete_collection(QDRANT_COLLECTION_NAME)` (`:141`) lalu mencoba lagi — sampai 5 kali (`:126`). Dua proses pada collection yang sama dapat saling menghapus data. Jalur ini destruktif secara desain |
| **`clear_collection`** | `indexing.py:37-53`, dipicu `force=True` di `:189` | Sama: menghapus seluruh titik. Aman hanya selama nama collection benar-benar terpisah |
| **Server Qdrant** | Satu instans (`docker-compose.poc.yml:36-42`) | Aman antar-collection. `get_indexed_file_hashes` men-scroll per collection (`indexing.py:77-83`), tidak lintas |
| **Cache HuggingFace** | **Tidak dikonfigurasi di repo mana pun** | Grep tidak menemukan `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, maupun `TRANSFORMERS_CACHE` di kode, Dockerfile, atau `.env*`. Jadi memakai default per-pengguna. `huggingface_hub` memakai berkas kunci saat mengunduh, sehingga unduhan bersamaan umumnya aman — **tetapi ini perilaku library, bukan jaminan dari repo.** Verifikasi di V5 |
| **Volume `hf_cache` (Docker)** | `docker-compose.poc.yml:78`, `:99`, `:115` | Volume bernama sama di-mount ke `vllm` (`/root/.cache/huggingface`), `tei-embed` (`/data`), dan `tei-rerank` (`/data`) — **titik mount berbeda untuk volume yang sama**. Container `backend` (`:164-168`) **tidak** me-mount-nya sama sekali, sehingga mengunduh model HF ke lapisan container-nya sendiri setiap kali dibangun ulang |
| **Cache model PaddleOCR** | **Tidak dikonfigurasi** | Tidak ada env var maupun argumen yang menyetel direktori model Paddle (`preprocessing.py:47-51` hanya mengirim `use_textline_orientation`, `lang`, `device`). PaddleOCR mengunduh model deteksi/rekognisi ke direktori default pada pemakaian pertama. Dua proses yang menginisialisasi engine bersamaan pada mesin dingin dapat mengunduh ke lokasi yang sama secara serentak. Verifikasi di V6 |
| **`./models` bind mount** | `docker-compose.poc.yml:165` (`:ro`) | Aman — read-only |
| **Memori GPU** | Tidak ada batas di mana pun | Dua proses memuat embedding (jalur `huggingface`) plus dua engine PaddleOCR pada GPU yang sama. Digabung dengan temuan unified-memory laporan pertama (D14), ini menggandakan tekanan pada kolam 128 GB yang juga dipakai OS dan peneliti lain |
| **Server vLLM / Ollama bersama** | `docker-compose.poc.yml:61-86`, `:127-138` | Bila `PDF_DESCRIBE_IMAGES` aktif, kedua proses indexing mengantre di server yang sama. Tidak merusak, tetapi menambah latensi dan meningkatkan peluang timeout — yang menurut G10 dapat menggeser `chunk_index` |
| **Redis** | Tidak dipakai jalur indexing | Aman |
| **PostgreSQL** | Tidak dipakai jalur indexing | Aman |

**Ringkas:** dengan nama collection yang benar-benar berbeda, tidak ada state bersama yang merusak data. Risiko nyatanya ada di dua tempat: (a) `load_dotenv()` di `config.py:7` yang membuat "collection berbeda" bergantung pada disiplin direktori kerja, dan (b) jalur pemulihan `_ensure_collection` di `indexing.py:141` yang menghapus collection — tidak berbahaya selama namanya terpisah, tetapi tidak memaafkan bila tidak.

---

## YANG HARUS DIUBAH

Diurutkan berdasarkan apa yang memblokir apa. Butir 1–3 adalah prasyarat; sisanya sebagian besar independen setelah ketiganya selesai.

| # | Perubahan | File tersentuh | Ukuran | Memblokir |
|---|---|---|---|---|
| **1** | **Hentikan pemecahan ulang oleh LlamaIndex.** `indexing.py:160` tidak mengirim override, sehingga node parser default memecah Document yang melebihi `Settings.chunk_size` dan beberapa titik Qdrant berbagi satu `chunk_index`. Selama ini berlaku, `chunk_id` apa pun yang kamu rancang tidak akan menjadi kunci unik | `indexing.py` (`:160`), berpotensi `rag_pipeline.py` (`:624-625`) | **Kecil** | #2, dan seluruh `chunks.jsonl` |
| **2** | **Bangun identitas berlapis tiga: `document_id`, `chunk_id` per-halaman, `image_id`.** Skema mengandaikan rantai rujukan ini; kode hanya punya `file_name` dan pencacah per berkas. Termasuk mengubah granularitas pencacah dari per-berkas ke per-halaman, dan menyediakan slug dokumen | `preprocessing.py` (`:308-407`), `indexing.py` (`:249-261`), sumber pemetaan slug baru | **Sedang** | #3, #4, #5, seluruh field `gold_*` |
| **3** | **Simpan gambar ke disk sesuai `file_path` skema.** Prasyarat seluruh `images.jsonl`, dan prasyarat memproses ulang gambar tanpa mengulang ekstraksi PDF. `IMAGES_DIR` (`config.py:12`) sudah ada dan menganggur; `_hash_image` (`image_describer.py:40-41`) sudah menghitung hash pra-resize sebagai jaring pengaman | `preprocessing.py` (`:232-242`, `:296-301`, `:381-388`), `indexing.py` (`:251-260`), `config.py` | **Sedang** | #4, `visual_type`, `structured_summary` |
| **4** | **Ubah kontrak `describe_image` menjadi struktur bervarian, dan tambahkan varian ke kunci cache.** Selama `image_describer.py:95` mengembalikan `str`, `structured_summary` dan `narrative_summary` tidak bisa hidup berdampingan — dan `_description_cache` (`:37`) akan mengembalikan varian yang salah | `image_describer.py` (`:37`, `:95`, `:105-133`), `preprocessing.py` (`:261-303`) | **Sedang** | Perbandingan varian (b) vs (c) |
| **5** | **Pertahankan `raw_html` tabel sampai payload.** Mengisi `chunks.jsonl.text_as_html` **dan** `images.jsonl.structured_summary` untuk tabel sekaligus — deck menyatakan keduanya sama. Sudah hidup di `preprocessing.py:229`, mati 130 baris kemudian. Rasio imbalan-terhadap-usaha tertinggi di daftar ini | `preprocessing.py` (`:355-374`), `indexing.py` (`:251-260`) | **Kecil** | `structured_summary` tabel, RCAA |
| **6** | **Tambahkan metadata temporal dan kemampuan memfilternya.** `effective_start`/`effective_end`/`superseded_by` di lapis 1 dan `validity_context` di `qa_pairs` menopang seluruh lapis 4. Tidak ada padanan apa pun di repo, dan retrieval tidak punya filter berbasis tanggal — `_expand_with_neighbors` hanya memfilter `file_name`/`chunk_index`/`section` (`rag_pipeline.py:779-784`) | `indexing.py` (`:249-261`), `rag_pipeline.py` (`:736-837`, `:840-888`), sumber metadata dokumen baru | **Besar** | Seluruh lapis 4 |
| **7** | **Kembalikan hasil ketiga tahap dari `_retrieve_and_rerank` beserta id chunk.** Datanya masih hidup di `:871` dan `:878` saat dibuang di `:888`. Tanpa ini Precision@k / Recall@k / MRR@k / nDCG@k tidak dapat dihitung per tahap | `rag_pipeline.py` (`:840-888`, `:891-902`), `schemas.py` (`:48-54`) | **Sedang** | Seluruh lapis 1 |
| **8** | **Pisahkan metrik retrieval teks dari retrieval gambar.** Deck (hal. 8) meminta lapis 1 dijalankan terpisah untuk teks, gambar, lalu gabungan. Pipeline punya satu jalur atas satu collection (`rag_pipeline.py:636-642`) dan deskripsi gambar dilebur jadi chunk teks (`preprocessing.py:381-388`). Paling murah: bawa `element_type`/`image_id` sampai ke hasil dan pisahkan saat analisis, bukan membangun retriever kedua | `rag_pipeline.py` (`:891-902`), skrip analisis | **Kecil** setelah #7 | Pelaporan lapis 1 |
| **9** | **Tambahkan entry point retrieval-only.** Lapis 1 tidak butuh LLM; saat ini setiap jalur berakhir di generasi, dan mengimpor `rag_pipeline` menarik torch, prometheus, dan YAML keyword | `routers/query.py` atau skrip baru; berpotensi memindahkan `_retrieve_and_rerank` keluar dari `rag_pipeline.py` | **Sedang** | Efisiensi eksperimen, bukan kebenarannya |
| **10** | **Propagasikan `visual_type` ke chunk.** Label berasal dari anotasi manusia (`images.jsonl`), jadi tidak perlu diturunkan model — tetapi metrik lapis 1 diukur atas chunk, sehingga chunk harus membawa `image_id` atau `visual_type` agar bisa di-join | `indexing.py` (`:251-260`), `preprocessing.py` (`:381-388`) | **Kecil** setelah #3 | Stratifikasi per tipe visual |
| **11** | **Tangkap `bbox`.** Unstructured menyediakan koordinat di metadata element; `_extract_hi_res` (`:213-254`) tidak pernah membacanya. Field opsional di skema, tapi murah selagi menyentuh fungsi yang sama | `preprocessing.py` (`:213-254`) | **Kecil** | `chunks.jsonl.bbox` |
| **12** | **Catat manifest konfigurasi dan model ke dalam artefak.** Saat ini hanya `LLM_MODEL` yang terekam (F2). Tanpa ini korpus terproses yang dipublikasikan tidak dapat dibuktikan asal-usulnya — padahal itu justru fungsi lapis 2 dataset menurut deck | `indexing.py` (`:249-261`) atau berkas pendamping saat indexing | **Kecil** | Klaim reproduksibilitas di paper |
| **13** | **Pisahkan `_configure_settings` dari `rag_pipeline` dan hapus duplikat di `index_narratives`.** Prasyarat praktis untuk paket bersama dua fork (I18) | Modul baru; `rag_pipeline.py` (`:603-625`), `index_narratives.py` (`:183-201`), `scripts/index_documents.py` (`:20`) | **Kecil** | Perbandingan adil dua fork |
| **14** | **Samakan konfigurasi embedding kedua provider secara eksplisit** — pooling, normalisasi, instruksi, panjang maksimum. Keduanya kini diserahkan ke default library yang berbeda (F3) | `rag_pipeline.py` (`:603-625`), `docker-compose.poc.yml` (`:95`) | **Kecil** | Kesetaraan lintas provider |
| **15** | **Beri sinyal fallback rerank ke pemanggil.** `score` berubah makna tanpa penanda apa pun di response (F4, H12) | `rag_pipeline.py` (`:678-707`, `:840-888`), `schemas.py` (`:57-101`) | **Kecil** | Validitas ambang skor |
| **16** | **Kecualikan jalur eksperimen dari `output_filter`, atau persempit polanya.** `\bmeta\b` di `output_filter.py:17` mencocoki "meta-analisis"; `transformers` dan `bert` di `:14`/`:20` juga bisa muncul di teks akademik. Jawaban yang dievaluasi lapis 2 saat ini adalah teks pasca-redaksi | `output_filter.py` (`:12-27`), pemanggil di `rag_pipeline.py:1263`, `routers/query.py:82`, `routers/chat.py:113` | **Kecil** | Kebersihan metrik lapis 2 |
| **17** | **Perbaiki jembatan structlog atau berhenti mengklaimnya.** Docstring `logging_config.py:3-4` menyatakan log stdlib menjadi JSON; tidak ada `ProcessorFormatter` yang mewujudkannya, dan `request_id` tidak pernah sampai ke baris `rag_pipeline` | `logging_config.py` (`:14-33`) | **Kecil** | Kejelasan; log tetap bukan medium eksperimen yang layak |

**Di luar cakupan kode, tapi menghambat lapis 4.** `expected_behavior="abstain_or_flag_conflict"` menuntut perilaku yang tidak ada di pipeline mana pun. Yang paling mendekati adalah `_low_relevance_response` (`rag_pipeline.py:484-524`), tetapi ia dipicu oleh skor rendah (`:1229`) — bukan oleh terdeteksinya dua sumber yang saling bertentangan. Contoh `qa-0455` di deck (narasi "maksimal 2 kali" vs tabel lampiran "3 kali") mengandaikan sistem mengenali konflik lalu melaporkannya. Membangun itu adalah pekerjaan riset tersendiri, bukan perbaikan pipeline, dan sebaiknya diperlakukan sebagai kontribusi yang dirancang — bukan sebagai gap yang ditambal.

Dua catatan yang tidak masuk tabel karena bukan perubahan kode:

- **Nama collection per fork harus dipisahkan lewat mekanisme yang tidak bergantung direktori kerja** (I19). `load_dotenv()` di `config.py:7` membuat pemisahan bergantung disiplin operasional.
- **`is_likely_informative`** (`image_describer.py:71-92`) adalah fungsi mati — tidak ada pemanggil di `backend/` maupun `scripts/`. Ambang `PDF_MIN_IMAGE_SIZE_KB` yang dirujuknya (`:76`) karenanya tidak berefek. Keputusan penyaringan gambar perlu dibuat eksplisit di #2, bukan diwarisi dari sini.

---

## MENUNGGU KORPUS

Tidak semua butir di bawah butuh PDF. Empat yang pertama dapat dijalankan **sekarang** dengan masukan sintetis; sisanya menunggu korpus.

### Dapat dijalankan sekarang (tanpa PDF)

**V1 — Apakah LlamaIndex memecah ulang chunk?** Ini menopang seluruh temuan G10 konsekuensi 3. Bangun satu `Document` sintetis yang lebih panjang dari `CHUNK_SIZE`, index-kan lewat jalur yang sama, lalu hitung titik yang dihasilkan.

```bash
python - <<'PY'
from llama_index.core import Document
from backend.services.rag_pipeline import _configure_settings
from backend.services.indexing import _embed_and_store, get_collection_count
_configure_settings()
before = get_collection_count()
doc = Document(text="A"*2000, metadata={"file_name":"probe.pdf","chunk_index":0,
                                        "file_hash":"probe","page":1,
                                        "element_type":"Table","section":"",
                                        "extraction_strategy":"probe","source_type":"pdf"})
_embed_and_store([doc])
print("titik bertambah:", get_collection_count() - before, "(1 = tidak dipecah, >1 = dipecah ulang)")
PY
```

Pakai collection sekali-pakai: `QDRANT_COLLECTION=probe_rechunk python - <<'PY' ...`

**V2 — Apakah vektor TEI dan HuggingFace identik?** Menopang F3.

```bash
EMBED_PROVIDER=tei python -c "
from backend.services.rag_pipeline import _configure_settings
from llama_index.core import Settings
_configure_settings(); import json
print(json.dumps(Settings.embed_model.get_text_embedding('prosedur cuti akademik')))" > /tmp/vec_tei.json

EMBED_PROVIDER=huggingface python -c "
from backend.services.rag_pipeline import _configure_settings
from llama_index.core import Settings
_configure_settings(); import json
print(json.dumps(Settings.embed_model.get_text_embedding('prosedur cuti akademik')))" > /tmp/vec_hf.json

python -c "
import json,math
a=json.load(open('/tmp/vec_tei.json')); b=json.load(open('/tmp/vec_hf.json'))
print('dim:',len(a),len(b))
print('norm:',math.sqrt(sum(x*x for x in a)), math.sqrt(sum(x*x for x in b)))
print('cosine:',sum(x*y for x,y in zip(a,b))/(math.sqrt(sum(x*x for x in a))*math.sqrt(sum(y*y for y in b))))
print('maxdiff:',max(abs(x-y) for x,y in zip(a,b)))"
```

Norma ≠ 1.0 pada salah satu sisi berarti normalisasi berbeda. Cosine < 0.999 berarti pooling atau truncation berbeda.

**V3 — Apakah skala skor kedua reranker sama?** Menopang F4. Jalankan kueri yang sama pada kedua provider dan bandingkan `debug.top_score`.

```bash
for p in tei sentence_transformers; do
  RERANKER_PROVIDER=$p curl -s localhost:8000/api/query \
    -H 'Content-Type: application/json' \
    -d '{"question":"bagaimana prosedur cuti akademik","history":[]}' \
  | python -c "import sys,json; d=json.load(sys.stdin)['debug']; print('$p', d['top_score'], [s for s in json.load(open('/dev/null'))] if False else '')"
done
```

Perlu restart backend antar iterasi karena `_reranker` adalah singleton tanpa reset (`rag_pipeline.py:64`, `:710-713`). Skor di luar rentang (0,1) pada salah satu sisi menunjukkan aktivasi berbeda.

**V4 — Apakah konstruksi klien LLM melakukan panggilan jaringan?** Menopang H13 butir 2.

```bash
# Jalankan dengan LLM_BASE_URL menunjuk ke port mati
LLM_PROVIDER=vllm LLM_BASE_URL=http://127.0.0.1:1 python -c "
from backend.services.llm_factory import get_llm
llm = get_llm(); print('konstruksi berhasil tanpa jaringan:', type(llm).__name__)"
```

Bila ini gagal, retrieval-only lewat `rag_pipeline` mensyaratkan endpoint LLM hidup.

**V5 — Apakah unduhan HF bersamaan aman?** Menopang I19.

```bash
rm -rf ~/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3
( python scripts/download_models.py --model reranker & \
  python scripts/download_models.py --model reranker & wait )
find ~/.cache/huggingface -name "*.incomplete" -o -name "*.lock" | head
```

**V6 — Di mana PaddleOCR menaruh model, dan apakah aman diakses bersamaan?** Menopang I19.

```bash
python -c "
from backend.services.preprocessing import _get_ocr_engine
_get_ocr_engine(); print('engine siap')"
find ~ -maxdepth 3 -type d \( -name ".paddleocr" -o -name ".paddlex" -o -name "*paddle*" \) 2>/dev/null
```

Setelah lokasinya diketahui, jalankan dua proses inisialisasi bersamaan pada cache kosong dan periksa berkas parsial.

### Menunggu PDF tersedia

**V7 — Distribusi `fast` vs `hi_res` di korpus nyata.** Menentukan berapa banyak dokumen yang sama sekali tidak melewati ekstraksi tabel/gambar.

```bash
python -c "
from pathlib import Path
from backend.services.preprocessing import detect_strategy
from collections import Counter
c = Counter()
for p in sorted(Path('data/pdfs').glob('*.pdf')):
    s = detect_strategy(p); c[s] += 1; print(f'{s:7} {p.name}')
print(c)"
```

Bila `fast` mendominasi, temuan A2/A4 laporan pertama menjadi masalah utama, bukan catatan pinggir.

**V8 — Berapa banyak halaman yang memicu OCR fallback.** Menentukan seberapa keras ketergantungan PaddleOCR (laporan 1, D15).

```bash
python -m scripts.index_documents 2>&1 | grep -c "page_ocr_fallback"
python -m scripts.index_documents 2>&1 | grep "pdf_extract_failed"
```

Baris kedua mengungkap berkas yang dibuang seluruhnya karena satu halaman gagal.

**V9 — Apakah `chunk_index` stabil pada dua kali indexing berkas yang tidak berubah.** Uji langsung atas G10 konsekuensi 2.

```bash
QDRANT_COLLECTION=probe_a python -m scripts.index_documents --force
QDRANT_COLLECTION=probe_b python -m scripts.index_documents --force
python -c "
from qdrant_client import QdrantClient
c = QdrantClient(url='http://localhost:6333')
def snap(col):
    out, off = {}, None
    while True:
        pts, off = c.scroll(col, limit=500, offset=off, with_payload=True, with_vectors=False)
        for p in pts:
            k = (p.payload.get('file_name'), p.payload.get('chunk_index'))
            out.setdefault(k, []).append((p.payload.get('text') or '')[:80])
        if off is None: return out
a, b = snap('probe_a'), snap('probe_b')
print('kunci hanya di A:', len(set(a)-set(b)))
print('kunci hanya di B:', len(set(b)-set(a)))
print('kunci sama tapi teks beda:', sum(1 for k in set(a)&set(b) if a[k]!=b[k]))
print('kunci dengan >1 titik (bukti pecah ulang):', sum(1 for v in a.values() if len(v)>1))"
```

Baris terakhir memberi bukti langsung atas G10 konsekuensi 3 pada data nyata.

**V10 — Apakah verdict `DEKORATIF` konsisten antar run.** Menopang G10 konsekuensi 2 pemicu pertama.

```bash
for i in 1 2 3; do
  QDRANT_COLLECTION=probe_vl_$i python -m scripts.index_documents --force 2>&1 \
    | grep -c "image_describer_error"
  python -c "
from qdrant_client import QdrantClient
c=QdrantClient(url='http://localhost:6333')
n=0; off=None
while True:
    pts,off=c.scroll('probe_vl_$i',limit=500,offset=off,with_payload=True,with_vectors=False)
    n+=sum(1 for p in pts if p.payload.get('element_type')=='ImageDescription')
    if off is None: break
print('run $i chunk ImageDescription:', n)"
done
```

Jumlah yang berbeda antar run membuktikan `chunk_index` tidak stabil bahkan pada berkas identik.

**V11 — Apakah `output_filter` merusak teks korpus.** Menopang F6 dan butir #12.

```bash
python -c "
from pathlib import Path
import fitz, re
from backend.services.output_filter import _PATTERNS
hits = {}
for p in sorted(Path('data/pdfs').glob('*.pdf')):
    txt = ' '.join(page.get_text() for page in fitz.open(str(p)))
    for pat, repl, label in _PATTERNS:
        for m in re.findall(pat, txt, flags=re.IGNORECASE):
            hits.setdefault(label, set()).add(str(m)[:40])
for k, v in hits.items(): print(k, sorted(v)[:15])"
```

Kemunculan `meta`, `bert`, atau `transformers` di korpus akademik mengonfirmasi risiko redaksi.

**V12 — Apakah korpus memenuhi target minimal per strata.** Deck (hal. 6) menetapkan ambang: ≥15–20 dokumen flowchart prosedural, ≥15–20 tabel persyaratan, ≥10–15 formulir/template, ≥10 figur deskriptif. Ini pengecekan **kelayakan korpus**, dan harus dijalankan sebelum anotasi Q&A dimulai — kalau salah satu strata kurang, jumlah pertanyaan per strata (target deck: 25 masing-masing) tidak akan tercapai tanpa pertanyaan berulang atas dokumen yang sama.

Langkah pertamanya otomatis — hitung berapa dokumen yang punya gambar sama sekali:

```bash
python -c "
import fitz
from pathlib import Path
from collections import Counter
c = Counter()
for p in sorted(Path('data/pdfs').glob('*.pdf')):
    d = fitz.open(str(p))
    n = sum(len(pg.get_images()) for pg in d)
    c['dengan_gambar' if n else 'tanpa_gambar'] += 1
    print(f'{n:4d}  {p.name}')
    d.close()
print(c)"
```

Langkah keduanya manual: setelah butir #3 diterapkan dan gambar tersimpan ke `IMAGES_DIR`, anotasi `visual_type` per gambar sesuai empat enum di skema. Tidak ada perintah otomatis untuk ini — deck sendiri menyatakan label berasal dari anotasi manusia.

**V13 — Apakah `chunk_id` per-halaman benar-benar unik pada korpus nyata.** Menopang butir #1 dan #2. Setelah keduanya diterapkan, hitung tabrakan:

```bash
python -c "
from qdrant_client import QdrantClient
from collections import Counter
c = QdrantClient(url='http://localhost:6333')
ids, off = Counter(), None
while True:
    pts, off = c.scroll('unhas_docs', limit=500, offset=off, with_payload=True, with_vectors=False)
    for p in pts: ids[p.payload.get('chunk_id')] += 1
    if off is None: break
dup = {k: v for k, v in ids.items() if v > 1}
print('total chunk_id:', len(ids), '| tabrakan:', len(dup))
for k, v in list(dup.items())[:10]: print(' ', k, '→', v, 'titik')"
```

Hasil bukan-nol berarti butir #1 belum benar-benar selesai, dan `gold_chunk_ids` masih akan ambigu.

---

## Catatan

`.gitignore:43` memuat pola `*.md`, sehingga `INSPECTION_REPORT.md` maupun `INSPECTION_REPORT_2.md` **tidak ter-track git**. Untuk menyimpannya:

```bash
git add -f INSPECTION_REPORT.md INSPECTION_REPORT_2.md
```
