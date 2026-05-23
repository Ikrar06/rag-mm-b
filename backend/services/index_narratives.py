r"""
index_narratives.py
===================
Index file narasi .txt (hasil preprocess-template.py) ke Qdrant.

Usage:
    python backend/services/index_narratives.py
    python backend/services/index_narratives.py --input data/narratives --force
    python backend/services/index_narratives.py --input data/narratives --endpoint fakultas
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

# ── Matikan log HTTP yang spam sebelum import apapun ─────────────
logging.basicConfig(
    level=logging.WARNING,                      # default WARNING
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
# Matikan httpx / httpcore (penyebab baris "HTTP Request: PUT ..." yang berulang)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("hpack").setLevel(logging.WARNING)
# Tetap tampilkan log penting dari app sendiri
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

VALID_ENDPOINTS = [
    "fakultas", "prodi", "jenjang", "kurikulum",
    "mata-kuliah", "prasyarat", "rps", "kelas",
    "jadwal", "fasilitas", "pmb", "pengumuman", "mahasiswa",
]

BATCH_SIZE = 2048   # dokumen per batch embed+store


# ─────────────────────────────────────────────────────────────────
# Progress helper
# ─────────────────────────────────────────────────────────────────

def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}j {m}m {s}d"
    if m:
        return f"{m}m {s}d"
    return f"{s}d"


def _progress_bar(done: int, total: int, width: int = 30) -> str:
    pct   = done / total if total else 0
    filled = int(width * pct)
    bar   = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct*100:5.1f}%"


def _print_progress(done: int, total: int, elapsed: float,
                    batch_num: int, total_batches: int,
                    docs_per_sec: float):
    eta = (total - done) / docs_per_sec if docs_per_sec > 0 else 0
    bar = _progress_bar(done, total)
    line = (
        f"\r  {bar}  "
        f"{done:>7,}/{total:,} dok  │  "
        f"batch {batch_num}/{total_batches}  │  "
        f"{docs_per_sec:,.0f} dok/s  │  "
        f"ETA {_fmt_duration(eta)}  │  "
        f"elapsed {_fmt_duration(elapsed)}"
    )
    # Potong agar tidak wrap di terminal 80-col
    print(line[:160], end="", flush=True)


# ─────────────────────────────────────────────────────────────────
# Qdrant helpers
# ─────────────────────────────────────────────────────────────────

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def _ensure_collection(client: QdrantClient):
    collections = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION_NAME in collections:
        return
    logger.info(f"Membuat collection '{QDRANT_COLLECTION_NAME}' (dim={EMBED_DIMENSION})...")
    max_retries = 5
    for attempt in range(max_retries):
        try:
            client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBED_DIMENSION, distance=Distance.COSINE),
            )
            logger.info("Collection dibuat.")
            return
        except Exception as e:
            if "already exists" in str(e).lower() and attempt < max_retries - 1:
                wait = 2 * (attempt + 1)
                logger.warning(f"Storage orphaned, retry dalam {wait}s...")
                try:
                    client.delete_collection(QDRANT_COLLECTION_NAME)
                except Exception:
                    pass
                time.sleep(wait)
            else:
                raise


def get_indexed_source_files() -> set[str]:
    try:
        client = get_qdrant_client()
        indexed = set()
        offset = None
        while True:
            results, offset = client.scroll(
                collection_name=QDRANT_COLLECTION_NAME,
                limit=500, offset=offset,
                with_payload=True, with_vectors=False,
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
    try:
        client = get_qdrant_client()
        client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="endpoint", match=MatchValue(value=endpoint))]
            ),
        )
        logger.info(f"Dihapus semua chunks endpoint='{endpoint}'")
    except Exception as e:
        logger.warning(f"Gagal hapus chunks endpoint='{endpoint}': {e}")


# ─────────────────────────────────────────────────────────────────
# Embed setup
# ─────────────────────────────────────────────────────────────────

def _configure_embed():
    if EMBED_PROVIDER == "tei":
        from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
        Settings.embed_model = TextEmbeddingsInference(
            model_name=EMBED_MODEL,
            base_url=EMBED_BASE_URL,
            embed_batch_size=EMBED_BATCH_SIZE,
            timeout=float(EMBED_TIMEOUT),
        )
        logger.info(f"embed=tei  base_url={EMBED_BASE_URL}  batch={EMBED_BATCH_SIZE}")
    else:
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=EMBED_MODEL,
            device=EMBED_DEVICE,
            trust_remote_code=True,
            embed_batch_size=EMBED_BATCH_SIZE,
        )
        logger.info(f"embed=huggingface  device={EMBED_DEVICE}")
    Settings.llm = None


# ─────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────

MAX_CHUNK_TOKENS = 400
CHARS_PER_TOKEN  = 3.5


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def _split_long_paragraph(para: str, max_tokens: int) -> list[str]:
    if _estimate_tokens(para) <= max_tokens:
        return [para]
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
            current = sent
    if current:
        chunks.append(current)
    return chunks if chunks else [para]


def chunk_narrative(text: str, max_tokens: int = MAX_CHUNK_TOKENS) -> list[str]:
    if not text.strip():
        return []
    if _estimate_tokens(text) <= max_tokens:
        return [text]
    header_line = text.split("\n")[0].strip()
    header = header_line[:120] + ("..." if len(header_line) > 120 else "")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0
    for para in paragraphs:
        sub_paras = _split_long_paragraph(para, max_tokens - _estimate_tokens(header) - 10)
        for sub in sub_paras:
            sub_tokens = _estimate_tokens(sub)
            if current_tokens + sub_tokens > max_tokens and current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_tokens = 0
            current_parts.append(sub)
            current_tokens += sub_tokens
    if current_parts:
        chunks.append("\n\n".join(current_parts))
    result = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            result.append(chunk)
        else:
            result.append(chunk if chunk.startswith(header[:40]) else f"{header}\n\n{chunk}")
    return result


# ─────────────────────────────────────────────────────────────────
# Load .txt files → Documents
# ─────────────────────────────────────────────────────────────────

def _load_txt_files(
    narratives_dir: Path,
    endpoints: list[str],
    force: bool,
    indexed_files: set[str],
) -> list[Document]:
    documents = []
    stats = {}

    for endpoint in endpoints:
        ep_dir = narratives_dir / endpoint
        if not ep_dir.exists():
            print(f"  [SKIP] subfolder tidak ditemukan: {ep_dir}")
            continue
        txt_files = sorted(ep_dir.glob("*.txt"))
        if not txt_files:
            print(f"  [SKIP] tidak ada .txt di {ep_dir}")
            continue
        new_files = [f for f in txt_files if force or f.name not in indexed_files]
        stats[endpoint] = {
            "total": len(txt_files),
            "new":   len(new_files),
            "skip":  len(txt_files) - len(new_files),
        }
        for txt_path in new_files:
            try:
                text = txt_path.read_text(encoding="utf-8").strip()
                if not text:
                    continue
                chunks = chunk_narrative(text)
                for chunk_idx, chunk_text in enumerate(chunks):
                    documents.append(Document(
                        text=chunk_text,
                        metadata={
                            "source_file":  txt_path.name,
                            "file_name":    txt_path.name,
                            "endpoint":     endpoint,
                            "item_id":      txt_path.stem,
                            "page":         None,
                            "chunk_index":  chunk_idx,
                            "total_chunks": len(chunks),
                            "element_type": "narrative",
                        },
                    ))
            except Exception as e:
                print(f"   [!] Gagal baca {txt_path.name}: {e}")

    print()
    for ep, s in stats.items():
        skip_info = f"  ({s['skip']} skip, {s['new']} baru)" if s["skip"] else f"  ({s['new']} akan di-index)"
        print(f"  {ep:<15} {s['total']:>6,} file{skip_info}")

    return documents


# ─────────────────────────────────────────────────────────────────
# Embed + store — dengan progress bar per batch
# ─────────────────────────────────────────────────────────────────

def _embed_and_store(documents: list[Document]):
    qdrant_client = get_qdrant_client()
    _ensure_collection(qdrant_client)

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=QDRANT_COLLECTION_NAME,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    total         = len(documents)
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    done          = 0
    t_start       = time.time()
    t_batch_start = t_start

    print(f"\n  Mulai embedding {total:,} dokumen dalam {total_batches} batch (@{BATCH_SIZE:,}/batch)\n")

    for batch_num in range(1, total_batches + 1):
        start_idx = (batch_num - 1) * BATCH_SIZE
        end_idx   = min(batch_num * BATCH_SIZE, total)
        batch     = documents[start_idx:end_idx]

        # Embed & store batch ini (show_progress=False karena kita buat sendiri)
        VectorStoreIndex.from_documents(
            batch,
            storage_context=storage_context,
            show_progress=False,
        )

        done     += len(batch)
        elapsed   = time.time() - t_start
        docs_per_sec = done / elapsed if elapsed > 0 else 0

        _print_progress(done, total, elapsed, batch_num, total_batches, docs_per_sec)

    # Newline setelah progress bar selesai
    print()


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Index narasi .txt ke Qdrant untuk RAG UNHAS"
    )
    parser.add_argument("--input",    "-i", default="data/narratives")
    parser.add_argument("--endpoint", "-e", default=None,
        help=f"Endpoint tertentu (pisah koma). Pilihan: {', '.join(VALID_ENDPOINTS)}")
    parser.add_argument("--force",    "-f", action="store_true",
        help="Hapus dan re-index ulang endpoint yang dipilih")
    args = parser.parse_args()

    narratives_dir = Path(args.input)
    if not narratives_dir.exists():
        print(f"[ERROR] Folder tidak ditemukan: {narratives_dir.resolve()}")
        print(f"  Jalankan dulu: python scripts/preprocess-template.py")
        sys.exit(1)

    if args.endpoint:
        requested = [e.strip() for e in args.endpoint.split(",")]
        invalid = [e for e in requested if e not in VALID_ENDPOINTS]
        if invalid:
            print(f"[ERROR] Endpoint tidak valid: {invalid}")
            sys.exit(1)
        endpoints = requested
    else:
        endpoints = VALID_ENDPOINTS

    print("=" * 65)
    print("  RAG Narrative Indexer — UNHAS")
    print(f"  Input     : {narratives_dir.resolve()}")
    print(f"  Endpoints : {', '.join(endpoints)}")
    print(f"  Mode      : {'force (re-index)' if args.force else 'incremental'}")
    print(f"  Qdrant    : {QDRANT_URL}  →  '{QDRANT_COLLECTION_NAME}'")
    print("=" * 65)

    if args.force:
        print("\n  Menghapus chunks lama...")
        for ep in endpoints:
            delete_endpoint_chunks(ep)

    print("\n  Memuat embedding model...")
    _configure_embed()

    indexed_files: set[str] = set()
    if not args.force:
        print("  Mengecek file yang sudah di-index...")
        indexed_files = get_indexed_source_files()
        print(f"  → {len(indexed_files):,} file sudah di-index sebelumnya.")

    print("\n  Memuat file narasi:")
    documents = _load_txt_files(narratives_dir, endpoints, args.force, indexed_files)

    if not documents:
        print("\n  ✅ Tidak ada file baru untuk di-index.")
        sys.exit(0)

    print(f"\n  Total: {len(documents):,} dokumen siap di-embed.")

    t_start = time.time()
    _embed_and_store(documents)
    elapsed = round(time.time() - t_start, 1)

    print("\n" + "=" * 65)
    print(f"  ✅ Selesai dalam {_fmt_duration(elapsed)}")
    print(f"     {len(documents):,} dokumen berhasil di-index ke Qdrant")
    print(f"     Collection: {QDRANT_COLLECTION_NAME}")
    print("=" * 65)


if __name__ == "__main__":
    main()