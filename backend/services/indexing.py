"""Document indexing service — preprocess PDFs, embed, store to Qdrant.

Production-grade features:
- Content hashing (SHA256): skip file yang sudah di-index dengan hash sama
- Incremental re-index: hanya proses file baru atau yang berubah
- Per-element metadata: file_name, file_hash, page, section, element_type
- Hi-res extraction untuk PDF rich content (tabel + gambar)
"""

import logging
import time
from pathlib import Path

from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client.models import Distance, VectorParams

from backend import config
from backend.config import (
    QDRANT_URL,
    QDRANT_COLLECTION_NAME,
    EMBED_DIMENSION,
    DATA_DIR,
    ALLOW_INCOMPLETE_IMAGE_CORPUS,
    DOCUMENT_REGISTRY_PATH,
    INDEX_DISABLE_NODE_PARSER,
    INDEX_EXCLUDE_METADATA_FROM_EMBED,
    INDEX_PERSIST_IMAGES,
    INDEX_STRUCTURAL_METADATA,
    PDF_EXTRACTION_STRATEGY,
    EMBED_EXCLUDED_METADATA_KEYS,
)
from backend.services import (
    chunk_dump, document_registry, image_describer, index_verify, vision_cache,
)
from backend.services.node_passthrough import build_transformations
from backend.services.preprocessing import (
    extract_from_pdf, chunk_documents, file_sha256,
)

logger = logging.getLogger(__name__)

# Field yang diproduksi _chunk_elements saat INDEX_STRUCTURAL_METADATA aktif.
# Diteruskan ke payload apa adanya bila ada; tidak semua chunk punya semuanya
# (mis. raw_html hanya pada chunk Table, bbox hanya bila koordinat tersedia).
_STRUCTURAL_METADATA_KEYS = (
    "document_id",
    "chunk_id",
    "text_sha",
    "raw_html",
    "table_format",
    "bbox",
    "image_id",
)


def _check_flag_consistency() -> None:
    """Tolak kombinasi flag yang pasti menjatuhkan indexing di tengah jalan.

    `raw_html` sebuah tabel bisa mencapai ratusan token. Bila ia ikut masuk
    metadata_str DAN node parser masih berjalan, SentenceSplitter melempar
    ValueError("Metadata length (N) is longer than chunk size") — terukur 987
    token pada tabel 30 baris, jauh di atas CHUNK_SIZE 512.

    Aman bila salah satu terpenuhi: raw_html dikecualikan dari metadata_str,
    atau node parser dimatikan sehingga tidak ada yang menghitung metadata_len.
    """
    if not INDEX_STRUCTURAL_METADATA:
        return
    if INDEX_EXCLUDE_METADATA_FROM_EMBED or INDEX_DISABLE_NODE_PARSER:
        return
    raise ValueError(
        "Kombinasi flag tidak aman: INDEX_STRUCTURAL_METADATA=true menambahkan "
        "raw_html ke metadata chunk, tapi INDEX_EXCLUDE_METADATA_FROM_EMBED dan "
        "INDEX_DISABLE_NODE_PARSER dua-duanya mati. SentenceSplitter akan "
        "melempar ValueError saat metadata_len melewati CHUNK_SIZE. "
        "Nyalakan salah satu (untuk riset: keduanya)."
    )


def _effective_research_flags() -> dict[str, object]:
    """Nilai efektif tiap flag riset, dibaca dari config yang sudah dimuat."""
    return {nama: getattr(config, nama, "<TIDAK ADA>")
            for nama in config.RESEARCH_EXPECTED_FLAGS}


def print_effective_flags() -> None:
    """Cetak nilai efektif tiap flag riset ke stdout saat start.

    Sengaja print(), bukan logger: harus terlihat apa pun konfigurasi logging,
    dan harus muncul SEBELUM indexing berjalan berjam-jam. Salah lokasi .env
    membuat seluruh flag jatuh ke default tanpa peringatan lain.
    """
    aktual = _effective_research_flags()
    print("=" * 72)
    print(f"FLAG RISET  (RESEARCH_MODE={config.RESEARCH_MODE})")
    print(f"  .env dicari dari lokasi backend/config.py ke atas, BUKAN dari")
    print(f"  direktori kerja. Repo: {Path(__file__).resolve().parent.parent.parent}")
    print("-" * 72)
    for nama, diharapkan in config.RESEARCH_EXPECTED_FLAGS.items():
        nilai = aktual[nama]
        tanda = "  " if nilai == diharapkan else "  <-- BEDA, diharapkan %r" % (diharapkan,)
        print(f"  {nama:<36}{nilai!r}{tanda}")
    print(f"  {'QDRANT_COLLECTION':<36}{config.QDRANT_COLLECTION!r}"
          f"  (kirim eksplisit tiap run)")
    print(f"  {'QDRANT_URL':<36}{config.QDRANT_URL!r}")
    print(f"  {'VISION_MODEL':<36}{config.VISION_MODEL!r}")
    print("=" * 72)


def _check_research_mode() -> None:
    """Tolak run bila RESEARCH_MODE aktif tapi ada flag yang tidak sesuai daftar beku.

    Mode kegagalan yang ditangani: load_dotenv() mencari .env dari lokasi
    config.py KE ATAS, bukan dari direktori kerja. `.env` di direktori kerja lain
    diabaikan tanpa peringatan, seluruh flag jatuh ke default, dan indexing tetap
    berjalan — menghasilkan chunk yang salah setelah berjam-jam.
    """
    if not config.RESEARCH_MODE:
        return

    aktual = _effective_research_flags()
    salah = {n: (aktual[n], d) for n, d in config.RESEARCH_EXPECTED_FLAGS.items()
             if aktual[n] != d}
    if not salah:
        return

    rincian = "\n".join(
        f"    {n:<36}sekarang={a!r}  diharapkan={d!r}" for n, (a, d) in sorted(salah.items())
    )
    raise ValueError(
        f"RESEARCH_MODE=true tapi {len(salah)} flag riset tidak sesuai daftar beku "
        f"di CHANGES.md:\n{rincian}\n"
        f"  Penyebab paling sering: .env tidak berada di ROOT repo. load_dotenv() "
        f"mencari dari lokasi backend/config.py ke atas, BUKAN dari direktori "
        f"kerja — .env di tempat lain diabaikan tanpa peringatan.\n"
        f"  Perbaiki: cp .env.research <root-repo>/.env\n"
        f"  Atau matikan gerbang ini bila memang menjalankan konfigurasi lain: "
        f"RESEARCH_MODE=false"
    )


def _check_vision_cache() -> None:
    """Tolak run bila cache memuat deskripsi dari konfigurasi vision LAIN.

    Kunci cache sudah memuat digest model dan hash prompt, jadi entri dari
    konfigurasi lain tidak akan pernah dikembalikan sebagai hit — tidak ada
    risiko salah-pakai. Yang ditolak di sini masalah berbeda: cache yang memuat
    lebih dari satu konfigurasi berarti korpus dideskripsikan oleh model atau
    prompt yang berbeda-beda, sehingga sebagian chunk berasal dari model A dan
    sebagian dari model B. Perbandingan varian (b) vs (c) tidak lagi mengukur
    strategi indexing.

    Hanya aktif saat RESEARCH_MODE — di luar riset, cache campuran tidak
    bermasalah.
    """
    if not config.RESEARCH_MODE or not vision_cache.enabled():
        return

    prov = image_describer.vision_provenance()
    digest_kini = prov["vision_model_digest"]
    prompt_kini = prov["prompt_sha256"]

    asing = [
        c for c in vision_cache.configurations()
        if c["vision_model_digest"] != digest_kini or c["prompt_sha256"] != prompt_kini
    ]
    if not asing:
        return

    rincian = "\n".join(
        f"    {c['entries']:>6} entri  model={c['vision_model']!r} "
        f"digest={(c['vision_model_digest'] or '<tidak ada>')[:24]} "
        f"prompt={c['prompt_sha256'][:16]} terakhir={c['last_written']}"
        for c in asing
    )
    raise ValueError(
        f"Cache deskripsi gambar memuat {len(asing)} konfigurasi vision yang "
        f"BERBEDA dari yang sedang dipakai:\n{rincian}\n"
        f"  sedang dipakai: model={prov['vision_model']!r} "
        f"digest={(digest_kini or '<tidak ada>')[:24]} prompt={prompt_kini[:16]}\n"
        f"  Artinya korpus dideskripsikan oleh model atau prompt yang "
        f"berbeda-beda — sebagian chunk dari konfigurasi lama, sebagian dari "
        f"yang baru. Perbandingan varian tidak lagi mengukur strategi indexing.\n"
        f"  Perbaiki: kembalikan VISION_MODEL/prompt ke konfigurasi yang sudah "
        f"ada di cache, atau mulai cache baru di VISION_CACHE_PATH lain — "
        f"TAPI ingat itu membatalkan seluruh anotasi gold yang menempel pada "
        f"chunk_id.\n"
        f"  Periksa isinya: python scripts/inspect_vision_cache.py"
    )


def _check_vision_reachable() -> None:
    """Tolak bila gambar disimpan tapi tidak akan pernah dideskripsikan.

    should_describe di preprocessing._describe_image_elements bernilai:
        PDF_DESCRIBE_IMAGES == "true"
        or (PDF_DESCRIBE_IMAGES == "auto" and LLM_SUPPORTS_VISION)

    LLM_SUPPORTS_VISION default false. Dikombinasikan dengan PDF_DESCRIBE_IMAGES
    ="auto" (juga default), SETIAP element Image dibuang tanpa satu pun error —
    narrative_summary null di seluruh images.jsonl, dan varian (b) tidak dapat
    direplikasi. Mode kegagalan sekelas PDF_EXTRACTION_STRATEGY=auto.
    """
    if not INDEX_PERSIST_IMAGES or ALLOW_INCOMPLETE_IMAGE_CORPUS:
        return

    akan_deskripsi = (
        config.PDF_DESCRIBE_IMAGES == "true"
        or (config.PDF_DESCRIBE_IMAGES == "auto" and config.LLM_SUPPORTS_VISION)
    )
    if akan_deskripsi:
        return

    raise ValueError(
        f"INDEX_PERSIST_IMAGES=true tapi tidak ada gambar yang akan "
        f"dideskripsikan.\n"
        f"    PDF_DESCRIBE_IMAGES={config.PDF_DESCRIBE_IMAGES!r}\n"
        f"    LLM_SUPPORTS_VISION={config.LLM_SUPPORTS_VISION}\n"
        f"  Dengan kombinasi ini SETIAP element Image dibuang di "
        f"_describe_image_elements tanpa satu pun error, narrative_summary null "
        f"di seluruh images.jsonl, dan varian (b) tidak dapat direplikasi.\n"
        f"  Perbaiki: LLM_SUPPORTS_VISION=true (plus VISION_MODEL yang multimodal "
        f"dan endpoint yang hidup), atau PDF_DESCRIBE_IMAGES=true.\n"
        f"  Atau terima korpus tidak lengkap secara sadar: "
        f"ALLOW_INCOMPLETE_IMAGE_CORPUS=true"
    )


def _check_image_strategy() -> None:
    """Tolak strategi ekstraksi yang membuat gambar hilang tanpa jejak.

    `_extract_fast` tidak mengekstrak gambar sama sekali — ia hanya memanggil
    page.get_text() per halaman. Dokumen yang diproses lewat jalur itu TIDAK
    menghasilkan satu baris pun di images.jsonl, bahkan bila halamannya penuh
    diagram.

    Yang membuat ini berbahaya: ketiadaan baris tidak dapat dibedakan dari
    "dokumen ini memang tidak punya gambar" dengan melihat berkas hasil saja.
    Dengan "auto" keputusannya per dokumen dan senyap; dengan "fast" seluruh
    korpus kehilangan gambarnya.
    """
    if not INDEX_PERSIST_IMAGES or ALLOW_INCOMPLETE_IMAGE_CORPUS:
        return
    if PDF_EXTRACTION_STRATEGY == "hi_res":
        return
    raise ValueError(
        f"PDF_EXTRACTION_STRATEGY={PDF_EXTRACTION_STRATEGY!r} tidak boleh "
        f"dipakai bersama INDEX_PERSIST_IMAGES=true.\n"
        f"  Jalur `fast` tidak mengekstrak gambar sama sekali, sehingga dokumen "
        f"yang dirutekan ke sana tidak menghasilkan baris apa pun di "
        f"images.jsonl — dan ketiadaannya TIDAK dapat dibedakan dari 'dokumen "
        f"ini memang tanpa gambar' dengan melihat berkas hasil.\n"
        f"  Dengan 'auto', detect_strategy merutekan setiap PDF hasil scan ke "
        f"`fast` (ambang > 1.0 strict; halaman scan = tepat 1 gambar/halaman).\n"
        f"  Perbaiki: PDF_EXTRACTION_STRATEGY=hi_res\n"
        f"  Atau terima korpus tidak lengkap secara sadar: "
        f"ALLOW_INCOMPLETE_IMAGE_CORPUS=true"
    )


def _check_no_silent_fallback(reports: dict[str, dict]) -> None:
    """Gagalkan run bila ada dokumen yang jatuh diam-diam dari hi_res ke fast.

    Efeknya identik dengan memakai strategi `fast`: dokumen itu kehilangan
    seluruh gambarnya, dan itu tidak terbaca dari images.jsonl. Menolak "auto"
    saja tidak cukup karena fallback terjadi saat runtime, setelah pemeriksaan
    konfigurasi lewat.

    Dipanggil SETELAH dump ditulis (buktinya tersimpan) dan SEBELUM
    _embed_and_store (tidak ada yang masuk Qdrant).
    """
    if not INDEX_PERSIST_IMAGES or ALLOW_INCOMPLETE_IMAGE_CORPUS:
        return
    jatuh = sorted(
        name for name, r in reports.items() if r.get("hi_res_fallback_reason")
    )
    if not jatuh:
        return

    rincian = "\n".join(
        f"    - {name}: {reports[name]['hi_res_fallback_reason']}" for name in jatuh
    )
    raise RuntimeError(
        f"{len(jatuh)} dokumen jatuh dari hi_res ke fast dan karenanya "
        f"KEHILANGAN SELURUH GAMBARNYA:\n{rincian}\n"
        f"  Tidak ada yang di-index. Dump sudah ditulis — periksa "
        f"`degraded_documents` di run_manifest.json.\n"
        f"  Ketiadaan gambar dokumen-dokumen ini tidak akan terbaca dari "
        f"images.jsonl, jadi run digagalkan alih-alih hanya dicatat.\n"
        f"  Perbaiki penyebab fallback-nya, atau terima korpus tidak lengkap "
        f"secara sadar: ALLOW_INCOMPLETE_IMAGE_CORPUS=true"
    )


def _resolve_document_id(pdf_path) -> str | None:
    """document_id dari registry, atau None bila berkas tidak layak di-index.

    Kegagalan memuat registry diperlakukan sebagai "tidak terdaftar" untuk
    SELURUH berkas — bukan crash — supaya pesannya muncul sekali per berkas
    lewat ringkasan skipped_unregistered, bukan sebagai traceback.
    """
    try:
        return document_registry.get_document_id(pdf_path.name)
    except document_registry.RegistryError as e:
        logger.error("document_registry_error error=%s", e)
        return None


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def clear_collection() -> bool:
    """Hapus semua points di collection (keep collection structure)."""
    try:
        client = get_qdrant_client()
        collections = [c.name for c in client.get_collections().collections]
        if QDRANT_COLLECTION_NAME not in collections:
            logger.info(f"collection_not_found name={QDRANT_COLLECTION_NAME}")
            return True
        client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=Filter(),
        )
        logger.info(f"collection_cleared name={QDRANT_COLLECTION_NAME}")
        return True
    except Exception as e:
        logger.warning(f"collection_clear_failed error={e}")
        return False


def get_collection_count() -> int:
    """Cek jumlah dokumen di collection."""
    try:
        client = get_qdrant_client()
        info = client.get_collection(QDRANT_COLLECTION_NAME)
        return info.points_count
    except Exception:
        return 0


def get_indexed_file_hashes() -> dict[str, str]:
    """Ambil mapping {file_name: file_hash} dari semua chunk di Qdrant.

    Dipakai untuk deteksi incremental: file dengan hash beda artinya berubah,
    harus di-delete + re-index. File dengan hash sama → skip.
    """
    try:
        client = get_qdrant_client()
        hashes: dict[str, str] = {}
        offset = None
        while True:
            results, offset = client.scroll(
                collection_name=QDRANT_COLLECTION_NAME,
                limit=200,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in results:
                fn = point.payload.get("file_name")
                fh = point.payload.get("file_hash")
                if fn and fh:
                    hashes[fn] = fh
            if offset is None:
                break
        return hashes
    except Exception as e:
        logger.debug(f"get_indexed_hashes_failed error={e}")
        return {}


def get_indexed_files() -> set[str]:
    """Backward-compat: ambil daftar file_name yang sudah di-index."""
    return set(get_indexed_file_hashes().keys())


def delete_file_chunks(file_name: str) -> int:
    """Hapus semua chunks dari file tertentu di Qdrant."""
    try:
        client = get_qdrant_client()
        client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="file_name", match=MatchValue(value=file_name))]
            ),
        )
        logger.info(f"chunks_deleted file={file_name}")
        return 1
    except Exception as e:
        logger.warning(f"chunks_delete_failed file={file_name} error={e}")
        return 0


def _ensure_collection(client: QdrantClient):
    """Pastikan collection ada di Qdrant, buat jika belum."""
    collections = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION_NAME in collections:
        return

    logger.info(f"collection_create name={QDRANT_COLLECTION_NAME} dim={EMBED_DIMENSION}")
    max_retries = 5
    for attempt in range(max_retries):
        try:
            client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBED_DIMENSION, distance=Distance.COSINE),
            )
            return
        except Exception as e:
            if "already exists" in str(e).lower() and attempt < max_retries - 1:
                wait = 2 * (attempt + 1)
                logger.warning(
                    f"orphaned_storage_retry wait={wait}s attempt={attempt + 1}/{max_retries}"
                )
                try:
                    client.delete_collection(QDRANT_COLLECTION_NAME)
                except Exception:
                    pass
                time.sleep(wait)
            else:
                raise


def _embed_and_store(documents: list[Document]):
    """Embed documents dan simpan ke Qdrant."""
    qdrant_client = get_qdrant_client()
    _ensure_collection(qdrant_client)

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=QDRANT_COLLECTION_NAME,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # build_transformations() mengembalikan [PassthroughNodeParser()] saat
    # INDEX_DISABLE_NODE_PARSER aktif, dan None saat tidak — bukan daftar kosong.
    # Daftar kosong bersifat falsy dan JATUH KEMBALI ke SentenceSplitter default
    # (llama_index/core/indices/base.py:109). Lihat node_passthrough.py.
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=build_transformations(),
        show_progress=True,
    )


def index_documents(data_dir: str | None = None, force: bool = False) -> int:
    """Index PDF dari data_dir ke Qdrant dengan incremental content hashing.

    Pipeline (production-grade):
        PDF → strategy detection (fast/hi_res)
            → text + table + image extraction
            → image description via vision LLM (kalau aktif)
            → smart chunking dengan section context
            → embed + store ke Qdrant

    Args:
        data_dir: Path folder PDF (default: DATA_DIR)
        force: Hapus collection lama, re-index semua

    Returns:
        Jumlah chunks yang berhasil di-index
    """
    print_effective_flags()
    _check_research_mode()
    _check_flag_consistency()
    _check_image_strategy()
    _check_vision_reachable()
    _check_vision_cache()

    target_dir = Path(data_dir or DATA_DIR)

    if force:
        existing = get_collection_count()
        logger.info(f"index_force_clear existing_chunks={existing}")
        clear_collection()

    pdf_files = sorted(target_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"no_pdf_found dir={target_dir}")
        return 0

    # Incremental detection: bandingkan content hash
    indexed_hashes = {} if force else get_indexed_file_hashes()

    new_files: list[Path] = []
    changed_files: list[Path] = []
    unchanged: list[Path] = []

    for f in pdf_files:
        current_hash = file_sha256(f)
        prev_hash = indexed_hashes.get(f.name)

        if prev_hash is None:
            new_files.append(f)
        elif prev_hash != current_hash:
            changed_files.append(f)
        else:
            unchanged.append(f)

    if unchanged:
        logger.info(f"pdf_skip_unchanged count={len(unchanged)}")

    if changed_files:
        logger.info(f"pdf_changed count={len(changed_files)} files={[f.name for f in changed_files]}")
        # Hapus chunks lama untuk file yang berubah
        for f in changed_files:
            delete_file_chunks(f.name)

    files_to_process = new_files + changed_files
    if not files_to_process:
        logger.info("no_new_or_changed_pdf")
        return 0

    logger.info(f"pdf_processing total={len(files_to_process)} new={len(new_files)} changed={len(changed_files)}")

    # Step 1: Extract + chunk semua file
    all_documents: list[Document] = []

    excluded_keys = (
        list(EMBED_EXCLUDED_METADATA_KEYS)
        if INDEX_EXCLUDE_METADATA_FROM_EMBED
        else []
    )
    skipped_unregistered: list[str] = []
    sha_mismatch: list[str] = []
    extraction_reports: dict[str, dict] = {}
    all_images: list[dict] = []

    for pdf_path in files_to_process:
        document_id = None
        if INDEX_STRUCTURAL_METADATA:
            document_id = _resolve_document_id(pdf_path)
            if document_id is None:
                skipped_unregistered.append(pdf_path.name)
                continue
            if not document_registry.check_sha256(pdf_path.name, file_sha256(pdf_path)):
                sha_mismatch.append(pdf_path.name)

        try:
            result = extract_from_pdf(pdf_path, document_id=document_id)
        except Exception as e:
            logger.error(f"pdf_extract_failed file={pdf_path.name} error={e}", exc_info=True)
            continue

        chunks = chunk_documents(result, document_id=document_id)
        if not chunks:
            logger.warning(f"pdf_no_chunks file={pdf_path.name}")
            continue

        file_hash = result["file_hash"]
        strategy_used = result["strategy"]

        report = result.get("report")
        if report is not None:
            extraction_reports[pdf_path.name] = report.as_dict()
        all_images.extend(result.get("images") or [])

        for chunk in chunks:
            metadata = {
                "file_name": chunk["file_name"],
                "file_hash": file_hash,
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
                "element_type": chunk.get("element_type", "text"),
                "section": chunk.get("section", ""),
                "extraction_strategy": strategy_used,
                "source_type": "pdf",
            }
            # Field struktural Tahap 2 hanya ada bila _chunk_elements
            # menghasilkannya (INDEX_STRUCTURAL_METADATA aktif).
            for key in _STRUCTURAL_METADATA_KEYS:
                if key in chunk:
                    metadata[key] = chunk[key]

            all_documents.append(Document(
                text=chunk["text"],
                metadata=metadata,
                excluded_embed_metadata_keys=list(excluded_keys),
                excluded_llm_metadata_keys=list(excluded_keys),
            ))

        logger.info(
            f"pdf_chunked file={pdf_path.name} chunks={len(chunks)} strategy={strategy_used}"
        )

    if skipped_unregistered:
        logger.error(
            "pdf_skipped_unregistered count=%d files=%s — tambahkan document_id di %s "
            "(buat kerangkanya: python scripts/scaffold_document_registry.py)",
            len(skipped_unregistered), skipped_unregistered, DOCUMENT_REGISTRY_PATH,
        )
    if sha_mismatch:
        logger.error(
            "pdf_sha256_mismatch count=%d files=%s — nama berkas dipakai ulang untuk "
            "isi berbeda; anotasi gold bisa menunjuk dokumen yang salah",
            len(sha_mismatch), sha_mismatch,
        )

    if not all_documents:
        logger.warning("no_chunks_produced")
        return 0

    # Step 1b: Dump untuk tinjauan tim evaluasi — SEBELUM embedding, dengan
    # Document yang sama persis yang akan dikirim ke _embed_and_store.
    # Kegagalan menulis dump tidak boleh menjatuhkan indexing.
    try:
        chunk_dump.write_run(all_documents, extraction_reports, all_images)
    except Exception as e:
        logger.error("chunk_dump_failed error=%s", e, exc_info=True)

    # Gerbang terakhir sebelum apa pun masuk Qdrant. Sengaja SETELAH dump:
    # buktinya tersimpan walau run digagalkan.
    _check_no_silent_fallback(extraction_reports)

    # Step 2: Embed + store batch
    logger.info(f"embedding_start total_chunks={len(all_documents)}")
    _embed_and_store(all_documents)

    # Step 3: Verifikasi HASIL, bukan niat. Gerbang di atas memeriksa daftar
    # Document sebelum dikirim; yang ini membaca kembali titik yang benar-benar
    # tersimpan. chunk_id duplikat lahir DI DALAM LlamaIndex, setelah gerbang
    # terakhir, jadi tidak ada pemeriksaan sebelum-kirim yang bisa melihatnya.
    if INDEX_STRUCTURAL_METADATA:
        index_verify.check_unique_chunk_ids(
            get_qdrant_client(), QDRANT_COLLECTION_NAME, strict=config.RESEARCH_MODE
        )

    logger.info(f"indexing_complete total_chunks={len(all_documents)} files={len(files_to_process)}")

    return len(all_documents)
