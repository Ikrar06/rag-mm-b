r"""
index_narratives.py
===================
Index file narasi .txt (hasil preprocess-template.py) ke Qdrant.

File .txt di-split menjadi chunks jika melebihi batas token embedding model.
Paragraf pendek digabung (greedy), paragraf panjang di-split per kalimat.
Default: MAX_CHUNK_TOKENS=400 (aman untuk model 512-token).

Bisa dijalankan bersama indexing.py (PDF) karena keduanya
store ke collection Qdrant yang sama.

Usage:
    # Dari root project
    python backend\services\index_narratives.py

    # Dengan opsi
    python backend\services\index_narratives.py --input data/narratives --force
    python backend\services\index_narratives.py --input data/narratives --endpoint fakultas
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# Pastikan root project ada di sys.path
# ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    Filter, FieldCondition, MatchValue,
)

from config import (
    QDRANT_URL,
    QDRANT_COLLECTION_NAME,
    EMBED_PROVIDER,
    EMBED_MODEL,
    EMBED_BASE_URL,
    EMBED_DEVICE,
    EMBED_BATCH_SIZE,
    EMBED_TIMEOUT,
    EMBED_DIMENSION,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Endpoint yang valid (subfolder di data/narratives)
VALID_ENDPOINTS = [
    "fakultas", "prodi", "jenjang", "kurikulum",
    "mata-kuliah", "prasyarat", "rps", "kelas",
    "jadwal", "fasilitas", "pmb", "pengumuman", "mahasiswa",
]


# ─────────────────────────────────────────────────────────────────
# Qdrant helpers (mirip indexing.py agar konsisten)
# ─────────────────────────────────────────────────────────────────

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def _ensure_collection(client: QdrantClient):
    collections = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION_NAME in collections:
        return

    logger.info(f"Creating collection '{QDRANT_COLLECTION_NAME}' (dim={EMBED_DIMENSION})...")
    max_retries = 5
    for attempt in range(max_retries):
        try:
            client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBED_DIMENSION, distance=Distance.COSINE),
            )
            logger.info("Collection created.")
            return
        except Exception as e:
            if "already exists" in str(e).lower() and attempt < max_retries - 1:
                wait = 2 * (attempt + 1)
                logger.warning(f"Orphaned storage, retry in {wait}s...")
                try:
                    client.delete_collection(QDRANT_COLLECTION_NAME)
                except Exception:
                    pass
                time.sleep(wait)
            else:
                raise


def get_indexed_source_files() -> set[str]:
    """Ambil set source_file yang sudah di-index (untuk incremental update)."""
    try:
        client = get_qdrant_client()
        indexed = set()
        offset = None
        while True:
            results, offset = client.scroll(
                collection_name=QDRANT_COLLECTION_NAME,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in results:
                sf = point.payload.get("source_file")
                if sf:
                    indexed.add(sf)
            if offset is None:
                break
        return indexed
    except Exception:
        return set()


def delete_endpoint_chunks(endpoint: str) -> None:
    """Hapus semua chunks dari endpoint tertentu (untuk re-index sebagian)."""
    try:
        client = get_qdrant_client()
        client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(
                    key="endpoint",
                    match=MatchValue(value=endpoint),
                )]
            ),
        )
        logger.info(f"Deleted all chunks for endpoint='{endpoint}'")
    except Exception as e:
        logger.warning(f"Could not delete chunks for endpoint='{endpoint}': {e}")


# ─────────────────────────────────────────────────────────────────
# Embed setup
# ─────────────────────────────────────────────────────────────────

def _configure_embed():
    if EMBED_PROVIDER == "tei":
        # POC: panggil TEI embedding service via HTTP (container ga butuh GPU)
        from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
        Settings.embed_model = TextEmbeddingsInference(
            model_name=EMBED_MODEL,
            base_url=EMBED_BASE_URL,
            embed_batch_size=EMBED_BATCH_SIZE,
            timeout=float(EMBED_TIMEOUT),
        )
        logger.info(f"embed_provider=tei base_url={EMBED_BASE_URL} batch={EMBED_BATCH_SIZE} timeout={EMBED_TIMEOUT}s")
    else:
        # Dev: in-process via HuggingFace (CUDA atau CPU sesuai EMBED_DEVICE)
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=EMBED_MODEL,
            device=EMBED_DEVICE,
            trust_remote_code=True,
            embed_batch_size=EMBED_BATCH_SIZE,
        )
        logger.info(f"embed_provider=huggingface device={EMBED_DEVICE}")
    # LLM tidak dipakai saat indexing — matikan agar tidak load model
    Settings.llm = None



# ─────────────────────────────────────────────────────────────────
# Token-aware chunking
# ─────────────────────────────────────────────────────────────────

# Batas token aman — di bawah limit model (512) agar ada ruang untuk
# special tokens dan overlap.  Ubah sesuai model kamu.
MAX_CHUNK_TOKENS = 400
# Estimasi: 1 token ≈ 3.5 karakter untuk teks Bahasa Indonesia
CHARS_PER_TOKEN  = 3.5


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def _split_long_paragraph(para: str, max_tokens: int) -> list[str]:
    """Split 1 paragraf panjang per kalimat jika masih terlalu panjang."""
    if _estimate_tokens(para) <= max_tokens:
        return [para]
    # Split per kalimat (titik/tanda tanya/seru diikuti spasi atau newline)
    import re
    sentences = re.split(r"(?<=[.!?])\s+", para.strip())
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        candidate = (current + " " + sent).strip() if current else sent
        if _estimate_tokens(candidate) <= max_tokens:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # Kalimat tunggal yang masih terlalu panjang → paksa masuk 1 chunk
            current = sent
    if current:
        chunks.append(current)
    return chunks if chunks else [para]


def chunk_narrative(text: str, max_tokens: int = MAX_CHUNK_TOKENS) -> list[str]:
    """
    Pecah teks narasi menjadi chunks yang aman untuk embedding.

    Strategi:
    1. Split teks per paragraf (\n\n sebagai batas alami).
    2. Gabungkan paragraf-paragraf pendek secara greedy sampai mendekati batas.
    3. Paragraf yang masih terlalu panjang sendiri → split per kalimat.

    Setiap chunk menyertakan baris pertama file (judul/identitas item)
    sebagai prefix agar retrieval tetap punya konteks meskipun di-split.
    """
    if not text.strip():
        return []

    # Jika seluruh teks masih di bawah batas → langsung return 1 chunk
    if _estimate_tokens(text) <= max_tokens:
        return [text]

    # Ambil baris pertama sebagai "header" yang di-repeat di tiap chunk
    header_line = text.split("\n")[0].strip()
    # Maksimal 120 karakter untuk header agar tidak makan terlalu banyak token
    header = header_line[:120] + ("..." if len(header_line) > 120 else "")

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        # Pecah dulu kalau paragraf itu sendiri lebih panjang dari batas
        sub_paras = _split_long_paragraph(para, max_tokens - _estimate_tokens(header) - 10)

        for sub in sub_paras:
            sub_tokens = _estimate_tokens(sub)
            if current_tokens + sub_tokens > max_tokens and current_parts:
                # Flush chunk yang ada
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_tokens = 0
            current_parts.append(sub)
            current_tokens += sub_tokens

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    # Tambahkan header ke chunk ke-2 dst agar tetap ada konteks
    result = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            result.append(chunk)
        else:
            # Cek apakah header sudah ada di awal chunk
            if not chunk.startswith(header[:40]):
                result.append(f"{header}\n\n{chunk}")
            else:
                result.append(chunk)
    return result

# ─────────────────────────────────────────────────────────────────
# Core indexing
# ─────────────────────────────────────────────────────────────────

def _load_txt_files(
    narratives_dir: Path,
    endpoints: list[str],
    force: bool,
    indexed_files: set[str],
) -> list[Document]:
    """
    Baca semua .txt dari subfolder endpoint, buat Document objects.

    Metadata per Document:
        source_file  : nama file .txt (untuk incremental check)
        endpoint     : nama kategori data (fakultas, prodi, dst.)
        item_id      : nama file tanpa ekstensi (e.g. "0000_FEB")
        file_name    : sama dengan source_file (kompatibel dengan indexing.py)
    """
    documents = []
    stats = {}

    for endpoint in endpoints:
        ep_dir = narratives_dir / endpoint
        if not ep_dir.exists():
            logger.warning(f"  [SKIP] Subfolder tidak ditemukan: {ep_dir}")
            continue

        txt_files = sorted(ep_dir.glob("*.txt"))
        if not txt_files:
            logger.warning(f"  [SKIP] Tidak ada .txt di {ep_dir}")
            continue

        new_files = [
            f for f in txt_files
            if force or f.name not in indexed_files
        ]
        skipped = len(txt_files) - len(new_files)

        stats[endpoint] = {"total": len(txt_files), "new": len(new_files), "skip": skipped}
        _ep_doc_start = len(documents)  # track chunks added for this endpoint

        for txt_path in new_files:
            try:
                text = txt_path.read_text(encoding="utf-8").strip()
                if not text:
                    logger.warning(f"    [!] File kosong: {txt_path.name}")
                    continue

                chunks = chunk_narrative(text)
                n_chunks = len(chunks)

                for chunk_idx, chunk_text in enumerate(chunks):
                    doc = Document(
                        text=chunk_text,
                        metadata={
                            "source_file":  txt_path.name,
                            "file_name":    txt_path.name,  # kompatibel dengan indexing.py
                            "endpoint":     endpoint,
                            "item_id":      txt_path.stem,
                            "page":         None,
                            "chunk_index":  chunk_idx,
                            "total_chunks": n_chunks,
                            "element_type": "narrative",
                        },
                    )
                    documents.append(doc)

                if n_chunks > 1:
                    logger.debug(f"    split: {txt_path.name} → {n_chunks} chunks")

            except Exception as e:
                logger.error(f"    [!] Gagal baca {txt_path.name}: {e}")

    # Print ringkasan per endpoint
    print()
    for ep, s in stats.items():
        msg = f"  {ep:<15} {s['total']:>4} file"
        if s["skip"]:
            msg += f"  ({s['skip']} sudah di-index, {s['new']} baru)"
        else:
            msg += f"  ({s['new']} akan di-index)"
        print(msg)

    return documents


def _embed_and_store(documents: list[Document]):
    qdrant_client = get_qdrant_client()
    _ensure_collection(qdrant_client)

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=QDRANT_COLLECTION_NAME,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    logger.info(f"Embedding {len(documents)} documents...")
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Index narasi .txt ke Qdrant untuk RAG UNHAS"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/narratives",
        help="Folder narasi (default: data/narratives)",
    )
    parser.add_argument(
        "--endpoint", "-e",
        default=None,
        help=(
            "Index hanya endpoint tertentu, pisah koma jika lebih dari satu. "
            f"Pilihan: {', '.join(VALID_ENDPOINTS)}. "
            "Default: semua endpoint."
        ),
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Hapus dan re-index ulang endpoint yang dipilih (default: incremental).",
    )
    args = parser.parse_args()

    narratives_dir = Path(args.input)
    if not narratives_dir.exists():
        print(f"[ERROR] Folder tidak ditemukan: {narratives_dir.resolve()}")
        print(f"  Jalankan dulu: python scripts/preprocess-template.py")
        sys.exit(1)

    # Tentukan endpoint yang akan di-index
    if args.endpoint:
        requested = [e.strip() for e in args.endpoint.split(",")]
        invalid = [e for e in requested if e not in VALID_ENDPOINTS]
        if invalid:
            print(f"[ERROR] Endpoint tidak valid: {invalid}")
            print(f"  Pilihan valid: {VALID_ENDPOINTS}")
            sys.exit(1)
        endpoints = requested
    else:
        endpoints = VALID_ENDPOINTS

    print("=" * 60)
    print("  RAG Narrative Indexer — UNHAS")
    print(f"  Input     : {narratives_dir.resolve()}")
    print(f"  Endpoints : {', '.join(endpoints)}")
    print(f"  Mode      : {'force (re-index)' if args.force else 'incremental'}")
    print(f"  Qdrant    : {QDRANT_URL}  →  collection '{QDRANT_COLLECTION_NAME}'")
    print("=" * 60)

    # ── Hapus chunks lama jika force ──────────────────────────────
    if args.force:
        print("\n  Menghapus chunks lama...")
        for ep in endpoints:
            delete_endpoint_chunks(ep)

    # ── Load embed model ──────────────────────────────────────────
    print("\n  Memuat embedding model...")
    _configure_embed()

    # ── Cek file yang sudah di-index (incremental) ────────────────
    indexed_files: set[str] = set()
    if not args.force:
        print("  Mengecek file yang sudah di-index...")
        indexed_files = get_indexed_source_files()
        logger.info(f"  {len(indexed_files)} file sudah di-index sebelumnya.")

    # ── Load .txt files ───────────────────────────────────────────
    print("\n  Memuat file narasi:")
    documents = _load_txt_files(narratives_dir, endpoints, args.force, indexed_files)

    if not documents:
        print("\n  ✅ Tidak ada file baru untuk di-index.")
        sys.exit(0)

    # ── Embed + store ─────────────────────────────────────────────
    print(f"\n  Total: {len(documents)} dokumen akan di-embed dan disimpan ke Qdrant.")
    t_start = time.time()
    _embed_and_store(documents)
    elapsed = round(time.time() - t_start, 1)

    print("\n" + "=" * 60)
    print(f"  ✅ Selesai dalam {elapsed}s")
    print(f"     {len(documents)} dokumen berhasil di-index ke Qdrant")
    print(f"     Collection: {QDRANT_COLLECTION_NAME}")
    print("=" * 60)


if __name__ == "__main__":
    main()