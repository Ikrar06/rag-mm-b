"""retrieval_dump.py — jalankan RETRIEVAL SAJA atas daftar pertanyaan.

Tidak memanggil LLM generation, tidak butuh endpoint HTTP. Menulis hasil ketiga
tahap retrieval — dense, ekspansi tetangga, rerank — ke JSONL yang siap dipakai
tim evaluasi menghitung metrik lapis 1 (Precision@k, Recall@k, MRR@k, nDCG@k).

Memakai `_retrieve_and_rerank` yang SAMA dengan jalur produksi, bukan salinannya.
Menulis ulang logikanya akan mengukur pipeline yang berbeda dari yang dilayani ke
pengguna — persis kesalahan yang membuat probe Tahap 0 lolos padahal kode
produksinya rusak.

Ongkos yang diterima (H13): mengimpor rag_pipeline menarik torch, prometheus, dan
keyword_filter YAML, dan `_get_retriever()` mengonstruksi klien LLM lewat
`_configure_settings()`. Klien itu dibangun, tidak dipakai — tidak ada panggilan
generation yang terjadi.

Masukan
-------
Berkas teks, satu pertanyaan per baris (baris kosong dan diawali `#` dilewati),
ATAU JSONL dengan field `question` (opsional `question_id`, `expected_behavior`,
dan field lain yang ikut dibawa apa adanya ke keluaran).

Keluaran
--------
`retrieval_nodes.jsonl`   satu baris per (pertanyaan, tahap, peringkat) — tabel
                          datar untuk perhitungan metrik
`retrieval_runs.jsonl`    satu baris per pertanyaan — ringkasan + galat
`retrieval_manifest.json` konfigurasi, flag RESEARCH_*, koleksi, model

Pemisahan teks vs gambar (deck meminta lapis 1 dijalankan terpisah untuk teks,
gambar, lalu gabungan) memakai kolom `element_type` dan `image_id` pada
`retrieval_nodes.jsonl`:

    teks    : element_type not in ("ImageDescription",) and image_id is null
    gambar  : image_id is not null
    gabungan: seluruh baris

Usage:
    python scripts/retrieval_dump.py --questions soal.txt
    python scripts/retrieval_dump.py --questions qa_pairs.jsonl --out hasil/
    python scripts/retrieval_dump.py --questions soal.txt --collection rag_mm_b_varian_b_v2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("retrieval_dump")

TAHAP = ("dense", "expansion", "rerank")


def baca_pertanyaan(path: Path) -> list[dict]:
    """Baca berkas pertanyaan. Mendukung teks polos dan JSONL.

    Deteksi berdasarkan isi, bukan ekstensi: baris pertama yang tidak kosong
    dicoba di-parse sebagai JSON object. Berkas .jsonl yang ternyata teks polos
    tetap terbaca, begitu pula sebaliknya.
    """
    baris = [b.rstrip("\n") for b in path.read_text(encoding="utf-8").splitlines()]
    isi = [b for b in baris if b.strip() and not b.lstrip().startswith("#")]
    if not isi:
        raise ValueError(f"{path}: tidak ada pertanyaan")

    jsonl = False
    try:
        jsonl = isinstance(json.loads(isi[0]), dict)
    except json.JSONDecodeError:
        jsonl = False

    out: list[dict] = []
    for i, b in enumerate(isi):
        if jsonl:
            try:
                rec = json.loads(b)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} baris {i + 1}: JSON tidak valid — {e}") from e
            q = (rec.get("question") or "").strip()
            if not q:
                raise ValueError(f"{path} baris {i + 1}: field 'question' kosong")
            rec = {k: v for k, v in rec.items() if k != "question"}
        else:
            q, rec = b.strip(), {}
        out.append({
            "question_id": str(rec.pop("question_id", None) or f"q{i + 1:04d}"),
            "question": q,
            "extra": rec,
        })

    ids = [r["question_id"] for r in out]
    if len(set(ids)) != len(ids):
        rangkap = sorted({x for x in ids if ids.count(x) > 1})
        raise ValueError(
            f"{path}: question_id duplikat: {rangkap}. Setiap pertanyaan butuh id "
            f"unik supaya baris keluaran dapat dipetakan balik ke test set."
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Retrieval tanpa generation → JSONL")
    ap.add_argument("--questions", required=True, help="Berkas pertanyaan (teks / JSONL)")
    ap.add_argument("--out", default="retrieval_dump", help="Direktori keluaran")
    ap.add_argument("--collection", default=None, help="Override QDRANT_COLLECTION")
    ap.add_argument("--limit", type=int, default=0, help="Proses N pertanyaan pertama saja")
    ap.add_argument("--role", default="public")
    ap.add_argument("--preview-chars", type=int, default=300,
                    help="Panjang text_preview (0 = tanpa teks)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")

    src = Path(args.questions)
    if not src.exists():
        print(f"GAGAL: {src} tidak ditemukan")
        return 2
    try:
        pertanyaan = baca_pertanyaan(src)
    except ValueError as e:
        print(f"GAGAL: {e}")
        return 2
    if args.limit > 0:
        pertanyaan = pertanyaan[: args.limit]

    # Override koleksi SEBELUM rag_pipeline diimpor — modul itu membaca
    # QDRANT_COLLECTION_NAME saat impor, jadi menyetelnya sesudah tidak berefek.
    from backend import config
    if args.collection:
        config.QDRANT_COLLECTION = args.collection
        config.QDRANT_COLLECTION_NAME = args.collection

    from backend.services import rag_pipeline as rp

    koleksi = rp.QDRANT_COLLECTION_NAME
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Qdrant     : {config.QDRANT_URL}")
    print(f"Koleksi    : {koleksi}")
    print(f"Pertanyaan : {len(pertanyaan)} dari {src}")
    print(f"Keluaran   : {out_dir.resolve()}\n")

    f_nodes = (out_dir / "retrieval_nodes.jsonl").open("w", encoding="utf-8")
    f_runs = (out_dir / "retrieval_runs.jsonl").open("w", encoding="utf-8")

    n_node = 0
    n_gagal = 0
    per_tahap = {t: 0 for t in TAHAP}
    t0 = time.time()

    try:
        for i, item in enumerate(pertanyaan, 1):
            qid, q = item["question_id"], item["question"]
            stages: dict = {}
            timings: dict = {}
            galat = None
            top_score = 0.0

            # _expand_query dipakai persis seperti query(): kalau tidak, retrieval
            # di sini memakai string yang berbeda dari produksi.
            diperluas = rp._expand_query(q)

            try:
                _, top_score = rp._retrieve_and_rerank(
                    diperluas, role=args.role, timings=timings, stages=stages
                )
            except Exception as e:
                galat = f"{type(e).__name__}: {e}"
                n_gagal += 1
                logger.error("retrieval_gagal qid=%s error=%s", qid, e, exc_info=True)

            for tahap in TAHAP:
                for rec in stages.get(tahap, []):
                    if args.preview_chars <= 0:
                        rec = {k: v for k, v in rec.items() if k != "text_preview"}
                    else:
                        rec = {**rec, "text_preview": rec["text_preview"][: args.preview_chars]}
                    f_nodes.write(json.dumps(
                        {"question_id": qid, "question": q, **rec},
                        ensure_ascii=False,
                    ) + "\n")
                    n_node += 1
                    per_tahap[tahap] += 1

            f_runs.write(json.dumps({
                "question_id": qid,
                "question": q,
                "question_expanded": diperluas if diperluas != q else None,
                "collection": koleksi,
                "top_score": round(top_score, 6),
                "rerank_fallback": stages.get("rerank_fallback"),
                "n_dense": len(stages.get("dense", [])),
                "n_expansion": len(stages.get("expansion", [])),
                "n_rerank": len(stages.get("rerank", [])),
                "timings_ms": timings,
                "error": galat,
                **item["extra"],
            }, ensure_ascii=False) + "\n")

            if i % 10 == 0 or i == len(pertanyaan):
                print(f"  {i}/{len(pertanyaan)} pertanyaan, {n_node} node"
                      f"{f', {n_gagal} gagal' if n_gagal else ''}")
    finally:
        f_nodes.close()
        f_runs.close()

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tool": "scripts/retrieval_dump.py",
        "questions_file": str(src.resolve()),
        "n_questions": len(pertanyaan),
        "n_failed": n_gagal,
        "n_nodes": n_node,
        "nodes_per_stage": per_tahap,
        "elapsed_s": round(time.time() - t0, 1),
        "collection": koleksi,
        "qdrant_url": config.QDRANT_URL,
        "retrieval": {
            "SIMILARITY_TOP_K": config.SIMILARITY_TOP_K,
            "RERANKER_TOP_N": config.RERANKER_TOP_N,
            "RERANKER_PROVIDER": config.RERANKER_PROVIDER,
            "RERANKER_MODEL": config.RERANKER_MODEL,
            "RERANKER_TIMEOUT": config.RERANKER_TIMEOUT,
            "SCORE_THRESHOLD": config.SCORE_THRESHOLD,
            "EMBED_PROVIDER": config.EMBED_PROVIDER,
            "EMBED_MODEL": config.EMBED_MODEL,
            "NEIGHBOR_EXPANSION_ENABLED": config.NEIGHBOR_EXPANSION_ENABLED,
            "NEIGHBOR_EXPANSION_RADIUS": config.NEIGHBOR_EXPANSION_RADIUS,
            "MAX_EXPANDED_CHUNKS": config.MAX_EXPANDED_CHUNKS,
        },
        # Sebagian flag ini mengubah angka yang dilaporkan, jadi hasil tanpa
        # penyertaannya tidak dapat ditafsirkan.
        "research_flags": config.research_query_flags(),
        "catatan": {
            "stages": "dense = hasil vector search; expansion = dense + tetangga "
                      "(is_neighbor=true, score=0.0); rerank = top RERANKER_TOP_N",
            "text_preview": "teks chunk apa adanya — direkam SEBELUM "
                            "SourceLabelPostprocessor menyisipkan '[nama_file]\\n'",
            "pemisahan_teks_gambar": "gambar = image_id tidak null; teks = image_id "
                                     "null dan element_type != 'ImageDescription'",
            "rerank_fallback": "true berarti rerank GAGAL dan skor pada tahap "
                               "rerank adalah skor dense, bukan skor reranker",
        },
    }
    (out_dir / "retrieval_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nSelesai dalam {manifest['elapsed_s']}s")
    for t in TAHAP:
        print(f"  {t:<12}{per_tahap[t]:>8} node")
    if n_gagal:
        print(f"\n  {n_gagal} pertanyaan GAGAL — lihat field 'error' di retrieval_runs.jsonl")
    return 1 if n_gagal else 0


if __name__ == "__main__":
    sys.exit(main())
