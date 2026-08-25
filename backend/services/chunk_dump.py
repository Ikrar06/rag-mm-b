"""Dump chunk ke berkas untuk ditinjau tim evaluasi, sebelum masuk Qdrant.

Sekaligus lapis 2 dataset publikasi riset, jadi `chunks.jsonl` mengikuti skema
`chunks.jsonl` di deck ("Publikasi Dataset", hal. 19) — bukan dump ad hoc.

Tiga berkas per run, di bawah CHUNK_DUMP_DIR/<run_id>/:

    chunks.jsonl        kanonik, satu baris JSON per chunk
    chunks_review.csv   untuk dibaca manusia + kolom kosong untuk anotasi
    run_manifest.json   konfigurasi efektif yang menghasilkan chunk itu
    images.jsonl        satu baris per gambar (hanya bila ada gambar tersimpan)
    images_review.csv   untuk anotasi visual_type oleh manusia

KESETIAAN DUMP
--------------
Titik dump berada antara konstruksi Document dan _embed_and_store. Teks di titik
itu identik dengan yang masuk Qdrant HANYA bila INDEX_DISABLE_NODE_PARSER=true.
Tanpa flag itu node parser LlamaIndex masih berjalan dan (a) memecah Document
yang kebesaran, (b) mem-strip trailing whitespace bahkan pada Document yang
tidak dipecah — terverifikasi. `run_manifest.json` mencatat status ini di
`dump_faithful`; jangan pakai dump yang `dump_faithful: false` sebagai dataset.

Saat INDEX_EXCLUDE_METADATA_FROM_EMBED aktif, TIDAK ADA metadata yang
divektorkan — `text_content` sama persis dengan string yang di-embed, sehingga
dataset lapis 2 cukup untuk mereproduksi index. Manifest mencatatnya di
`embedded_text_shape`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from backend import config
from backend.services import image_describer, preprocessing

logger = logging.getLogger(__name__)

# Kolom CSV. `text_content` sengaja ditaruh setelah kolom ringkas supaya
# spreadsheet tetap terbaca; `raw_html` TIDAK ikut (lihat _row).
_CSV_COLUMNS = [
    "chunk_id",
    "document_id",
    "page_number",
    "chunk_type",
    "element_type",
    "chunk_index",
    "image_id",
    "text_sha",
    "n_chars",
    "has_table_html",
    "table_html_chars",
    "table_format",
    "bbox",
    "section",
    "file_name",
    "text_preview",
    "text_content",
    # ── kolom kosong untuk reviewer ──
    "visual_type",
    "verdict",
    "catatan_reviewer",
]

# Enum visual_type dari skema images.jsonl, dicantumkan di header CSV sebagai
# pengingat nilai yang sah.
VISUAL_TYPE_ENUM = ("flowchart", "tabel_sebagai_gambar", "formulir", "figur_deskriptif")

_PREVIEW_CHARS = 180


def dump_dir() -> Path | None:
    """Direktori dump, atau None bila CHUNK_DUMP_DIR kosong (perilaku lama)."""
    raw = (config.CHUNK_DUMP_DIR or "").strip()
    return Path(raw) if raw else None


def new_run_id() -> str:
    """Id unik per eksekusi: stempel waktu UTC + 8 hex acak."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


# ─── Pemetaan Document -> baris skema ────────────────────────────────────────

def _chunk_type(element_type: str) -> str:
    """Petakan element_type internal ke enum `chunk_type` skema ("text"/"table").

    Perhatian: jalur `fast` menandai SETIAP element "NarrativeText"
    (preprocessing.py `_extract_fast`), termasuk tabel yang tergilas jadi teks
    datar. Chunk semacam itu akan terpetakan "text" walau isinya tabel.
    """
    return "table" if element_type == "Table" else "text"


def _record(doc) -> dict:
    """Satu baris chunks.jsonl, mengikuti skema lapis 2."""
    meta = doc.metadata
    page = meta.get("page")
    return {
        # ── skema chunks.jsonl ──
        "chunk_id": meta.get("chunk_id"),
        "document_id": meta.get("document_id"),
        # page 0 adalah penanda "tidak diketahui" dari jalur hi_res, bukan
        # halaman nol — dinormalkan jadi null di dataset.
        "page_number": page if isinstance(page, int) and page > 0 else None,
        "chunk_type": _chunk_type(meta.get("element_type", "")),
        "text_content": doc.text,
        "text_as_html": meta.get("raw_html") or None,
        "bbox": meta.get("bbox"),
        # ── kolom tambahan di luar skema minimum ──
        # image_id menautkan chunk ImageDescription ke barisnya di images.jsonl.
        # Tanpa ini metrik lapis 1 tidak dapat distratifikasi per tipe visual,
        # karena visual_type hidup di images.jsonl sementara retrieval
        # menghasilkan chunk.
        "image_id": meta.get("image_id"),
        "text_sha": meta.get("text_sha"),
        "chunk_index": meta.get("chunk_index"),
        "element_type": meta.get("element_type"),
        "table_format": meta.get("table_format"),
        "section": meta.get("section"),
        "file_name": meta.get("file_name"),
        "file_hash": meta.get("file_hash"),
        "extraction_strategy": meta.get("extraction_strategy"),
    }


def _row(rec: dict) -> dict:
    """Satu baris chunks_review.csv.

    `text_as_html` sengaja TIDAK dimasukkan: HTML tabel bisa ribuan karakter
    berisi newline dan tanda kutip, yang walau ter-quote dengan benar membuat
    satu sel meledak dan spreadsheet tak terbaca. Yang dibawa hanya penanda
    keberadaan dan panjangnya; HTML utuhnya ada di chunks.jsonl.
    """
    html = rec.get("text_as_html") or ""
    text = rec.get("text_content") or ""
    preview = " ".join(text.split())[:_PREVIEW_CHARS]
    bbox = rec.get("bbox")
    return {
        "chunk_id": rec.get("chunk_id") or "",
        "document_id": rec.get("document_id") or "",
        "page_number": "" if rec.get("page_number") is None else rec["page_number"],
        "chunk_type": rec.get("chunk_type") or "",
        "element_type": rec.get("element_type") or "",
        "chunk_index": rec.get("chunk_index"),
        "image_id": rec.get("image_id") or "",
        "text_sha": rec.get("text_sha") or "",
        "n_chars": len(text),
        "has_table_html": "ya" if html else "",
        "table_html_chars": len(html) if html else "",
        "table_format": rec.get("table_format") or "",
        "bbox": ",".join(str(v) for v in bbox) if bbox else "",
        "section": rec.get("section") or "",
        "file_name": rec.get("file_name") or "",
        "text_preview": preview,
        "text_content": text,
        "visual_type": "",
        "verdict": "",
        "catatan_reviewer": "",
    }


# ─── Gambar: images.jsonl + images_review.csv ────────────────────────────────

_IMAGE_CSV_COLUMNS = [
    "image_id",
    "document_id",
    "page_number",
    "file_path",
    "mime_type",
    "size_kb",
    "width",
    "height",
    "sha256",
    "has_narrative",
    "narrative_preview",
    "source_file",
    # ── kolom kosong untuk anotasi manusia ──
    "visual_type",
    "verdict",
    "catatan_reviewer",
]


def _image_record(img: dict) -> dict:
    """Satu baris images.jsonl, mengikuti skema lapis 2.

    `visual_type` dan `structured_summary` sengaja null: yang pertama berasal
    dari anotasi manusia (isi lewat images_review.csv), yang kedua menunggu
    kontrak describe_image bervarian.
    """
    return {
        # ── skema images.jsonl ──
        "image_id": img.get("image_id"),
        "document_id": img.get("document_id"),
        "page_number": img.get("page_number"),
        "file_path": img.get("file_path"),
        "visual_type": img.get("visual_type"),
        "structured_summary": img.get("structured_summary"),
        "narrative_summary": img.get("narrative_summary"),
        # ── kolom tambahan di luar skema minimum ──
        "sha256": img.get("sha256"),
        "mime_type": img.get("mime_type"),
        "size_bytes": img.get("size_bytes"),
        "width": img.get("width"),
        "height": img.get("height"),
        "source_file": img.get("source_file"),
    }


def _image_row(img: dict) -> dict:
    narrative = img.get("narrative_summary") or ""
    size = img.get("size_bytes") or 0
    return {
        "image_id": img.get("image_id") or "",
        "document_id": img.get("document_id") or "",
        "page_number": "" if img.get("page_number") is None else img["page_number"],
        "file_path": img.get("file_path") or "",
        "mime_type": img.get("mime_type") or "",
        "size_kb": round(size / 1024, 1) if size else "",
        "width": img.get("width") or "",
        "height": img.get("height") or "",
        "sha256": (img.get("sha256") or "")[:16],
        # Kosong berarti gambar TIDAK dideskripsikan — bisa karena dinilai
        # DEKORATIF, vision mati, atau panggilan gagal. Gambarnya tetap ada
        # di file_path dan tetap layak dianotasi.
        "has_narrative": "ya" if narrative else "",
        "narrative_preview": " ".join(narrative.split())[:_PREVIEW_CHARS],
        "source_file": img.get("source_file") or "",
        "visual_type": "",
        "verdict": "",
        "catatan_reviewer": "",
    }


# ─── Manifest ────────────────────────────────────────────────────────────────

def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent.parent,
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _registry_sha() -> str | None:
    try:
        path = Path(config.DOCUMENT_REGISTRY_PATH)
        if not path.exists():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return None


def build_manifest(run_id: str, reports: dict[str, dict], n_chunks: int,
                   images: list[dict] | None = None) -> dict:
    """Konfigurasi efektif yang menghasilkan chunk pada run ini."""
    faithful = config.INDEX_DISABLE_NODE_PARSER
    # Saat exclusion aktif, TIDAK ADA metadata yang divektorkan — `text_content`
    # di chunks.jsonl sama persis dengan string yang di-embed, sehingga dataset
    # lapis 2 cukup untuk mereproduksi index tanpa mereplikasi pipeline.
    embedded_shape = (
        "<text_content>"
        if config.INDEX_EXCLUDE_METADATA_FROM_EMBED
        else "<seluruh metadata>\\n\\n<text_content>"
    )
    degraded = {k: v for k, v in reports.items()
                if v.get("hi_res_fallback_reason") or v.get("ocr_failed_pages")}

    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_chunks": n_chunks,
        "n_images": len(images or []),
        "n_images_with_narrative": sum(
            1 for i in (images or []) if i.get("narrative_summary")
        ),
        # file_path di images.jsonl relatif terhadap direktori ini.
        "images_root": str(Path(config.IMAGES_DIR).parent),

        # Apakah dump ini boleh dipakai sebagai dataset — lihat docstring modul.
        "dump_faithful": faithful,
        "dump_faithful_reason": (
            "INDEX_DISABLE_NODE_PARSER=true: 1 Document = 1 node, teks identik"
            if faithful else
            "INDEX_DISABLE_NODE_PARSER=false: node parser LlamaIndex masih "
            "memecah dan mem-strip whitespace, jadi text_content di dump TIDAK "
            "identik dengan yang masuk Qdrant"
        ),
        "embedded_text_shape": embedded_shape,

        # ── Kelompok 1 (INSPECTION_REPORT.md E17): menentukan isi chunk ──
        "chunking": {
            "PDF_EXTRACTION_STRATEGY": config.PDF_EXTRACTION_STRATEGY,
            "CHUNK_SIZE": config.CHUNK_SIZE,
            "CHUNK_OVERLAP": config.CHUNK_OVERLAP,
            "PDF_TABLE_MAX_CHARS": config.PDF_TABLE_MAX_CHARS,
            "PDF_EXTRACT_TABLES": config.PDF_EXTRACT_TABLES,
            "PDF_EXTRACT_IMAGES": config.PDF_EXTRACT_IMAGES,
            "PDF_DESCRIBE_IMAGES": config.PDF_DESCRIBE_IMAGES,
            "PDF_MIN_IMAGE_SIZE_KB": config.PDF_MIN_IMAGE_SIZE_KB,
            "PDF_MAX_IMAGE_DIM": config.PDF_MAX_IMAGE_DIM,
            "LLM_SUPPORTS_VISION": config.LLM_SUPPORTS_VISION,
            "OCR_LANG": config.OCR_LANG,
            "OCR_USE_GPU": config.OCR_USE_GPU,
        },

        # ── Flag riset Tahap 1 & 2 ──
        "research_flags": {
            "INDEX_EXCLUDE_METADATA_FROM_EMBED": config.INDEX_EXCLUDE_METADATA_FROM_EMBED,
            "INDEX_MAX_CHUNK_TOKENS": config.INDEX_MAX_CHUNK_TOKENS,
            "INDEX_MIN_CHUNK_TOKENS": config.INDEX_MIN_CHUNK_TOKENS,
            "INDEX_DISABLE_NODE_PARSER": config.INDEX_DISABLE_NODE_PARSER,
            "INDEX_STRUCTURAL_METADATA": config.INDEX_STRUCTURAL_METADATA,
            # Tidak diminta eksplisit, tapi WAJIB dicatat: menggeser batas chunk
            # teks di seluruh dokumen, bukan sekadar menambah chunk tabel.
            "INDEX_TABLES_AS_OWN_CHUNKS": config.INDEX_TABLES_AS_OWN_CHUNKS,
            "INDEX_PERSIST_IMAGES": config.INDEX_PERSIST_IMAGES,
        },

        # ── Konstanta hardcoded (E17), dibaca dari sumbernya ──
        "hardcoded_constants": {
            "ocr_text_threshold_chars": preprocessing.OCR_TEXT_THRESHOLD_CHARS,
            "ocr_render_dpi": preprocessing.OCR_RENDER_DPI,
            "detect_strategy_image_threshold": preprocessing.STRATEGY_IMAGE_THRESHOLD,
            "detect_strategy_sample_pages": preprocessing.STRATEGY_SAMPLE_PAGES,
            "image_description_max_tokens": image_describer.DESCRIPTION_MAX_TOKENS,
            "image_description_temperature": image_describer.DESCRIPTION_TEMPERATURE,
            "image_description_prompt_sha256": hashlib.sha256(
                image_describer._DESCRIPTION_PROMPT.encode("utf-8")
            ).hexdigest(),
        },

        # ── Model ──
        "models": {
            "embedding": {
                "provider": config.EMBED_PROVIDER,
                "model": config.EMBED_MODEL,
                "device": config.EMBED_DEVICE,
                "batch_size": config.EMBED_BATCH_SIZE,
                "dimension": config.EMBED_DIMENSION,
                # Pada provider "tei", bobot sebenarnya ditentukan --model-id di
                # docker-compose, bukan nilai ini.
                "authoritative": config.EMBED_PROVIDER != "tei",
            },
            "vision": {
                "provider": config.LLM_PROVIDER,
                "model": config.LLM_MODEL,
                "base_url": config.LLM_BASE_URL,
            },
            "reranker": {
                "provider": config.RERANKER_PROVIDER,
                "model": config.RERANKER_MODEL,
                # Jalur TEI mengabaikan RERANKER_MODEL sepenuhnya.
                "authoritative": config.RERANKER_PROVIDER != "tei",
            },
        },

        # ── Lingkungan ──
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: _package_version(name)
                for name in ("unstructured", "llama-index-core", "tiktoken", "pymupdf")
            },
        },

        # ── Provenance ──
        "provenance": {
            "git_commit": _git_commit(),
            "qdrant_collection": config.QDRANT_COLLECTION,
            "document_registry_path": str(config.DOCUMENT_REGISTRY_PATH),
            "document_registry_sha256": _registry_sha(),
        },

        # ── Degradasi per dokumen ──
        "extraction_reports": reports,
        "degraded_documents": sorted(degraded),
        "n_degraded_documents": len(degraded),
    }


# ─── Penulisan ───────────────────────────────────────────────────────────────

def write_run(
    documents,
    reports: dict[str, dict],
    images: list[dict] | None = None,
    run_id: str | None = None,
) -> Path | None:
    """Tulis berkas dump untuk satu run. None bila dump tidak aktif.

    `documents` adalah list[Document] persis seperti yang akan dikirim ke
    _embed_and_store. `images` adalah record dari `_persist_image_elements`.
    """
    base = dump_dir()
    if base is None:
        return None

    images = images or []
    run_id = run_id or new_run_id()
    out = base / run_id
    out.mkdir(parents=True, exist_ok=True)

    records = [_record(d) for d in documents]

    with (out / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with (out / "chunks_review.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(_row(rec))

    if images:
        with (out / "images.jsonl").open("w", encoding="utf-8") as f:
            for img in images:
                f.write(json.dumps(_image_record(img), ensure_ascii=False) + "\n")

        with (out / "images_review.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_IMAGE_CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for img in images:
                writer.writerow(_image_row(img))

    manifest = build_manifest(run_id, reports, len(records), images)
    (out / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if not manifest["dump_faithful"]:
        logger.error(
            "chunk_dump_not_faithful run_id=%s — INDEX_DISABLE_NODE_PARSER mati, "
            "text_content TIDAK identik dengan yang masuk Qdrant. Jangan pakai "
            "dump ini sebagai dataset.", run_id,
        )
    logger.info(
        "chunk_dump_written run_id=%s dir=%s chunks=%d degraded_docs=%d faithful=%s",
        run_id, out, len(records), manifest["n_degraded_documents"],
        manifest["dump_faithful"],
    )
    return out
