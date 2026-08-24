"""Registry pemetaan nama berkas PDF -> document_id.

Skema dataset riset menuntut `document_id` berupa slug yang memuat penanda versi
(contoh: "sop-izin-ujian-online-v2"). Nama berkas tidak cukup: sufiks angka pada
nama berkas tidak dapat dibedakan antara nomor urut dan nomor versi, dan menebak
versi dari nama berkas akan menghasilkan identitas yang salah secara diam-diam.

Karena itu pemetaan dikurasi manual di berkas JSON. Registry ini juga menjadi
titik tumbuh untuk 13 field lain yang dibutuhkan corpus_metadata.jsonl (title,
source_unit, effective_start, superseded_by, dan seterusnya) — semuanya hanya
bisa datang dari kurasi manusia.

Bentuk berkas (data/document_registry.json):

    {
      "sop_pengurusan_izin_ujian_akhir_online_2.pdf": {
        "document_id": "sop-izin-ujian-online-v2",
        "sha256": "9f2b1c..."
      }
    }

`document_id` wajib. `sha256` opsional — bila diisi, dipakai untuk mendeteksi
nama berkas yang dipakai ulang untuk isi berbeda.

Buat kerangkanya dengan:
    python scripts/scaffold_document_registry.py
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from backend.config import DOCUMENT_REGISTRY_PATH

logger = logging.getLogger(__name__)

# Slug: huruf kecil, angka, dan tanda hubung. Tidak boleh diawali/diakhiri hubung.
_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

_registry_cache: dict[str, dict] | None = None


class RegistryError(Exception):
    """Registry tidak dapat dimuat atau isinya tidak valid."""


def _validate(raw: object, path: Path) -> dict[str, dict]:
    """Validasi bentuk registry. Melempar RegistryError bila rusak.

    Entri dengan document_id kosong DIBIARKAN LOLOS — itu keluaran normal
    scaffolder yang belum diisi. `get_document_id` akan mengembalikan None
    untuk entri semacam itu, sehingga berkasnya dilewati saat indexing.
    """
    if not isinstance(raw, dict):
        raise RegistryError(f"{path}: akar JSON harus object, bukan {type(raw).__name__}")

    seen_ids: dict[str, str] = {}
    entries: dict[str, dict] = {}

    for file_name, entry in raw.items():
        if not isinstance(entry, dict):
            raise RegistryError(f"{path}: entri {file_name!r} harus object")

        doc_id = entry.get("document_id", "")
        if not isinstance(doc_id, str):
            raise RegistryError(f"{path}: document_id {file_name!r} harus string")

        doc_id = doc_id.strip()
        if doc_id:
            if not _SLUG_RE.match(doc_id):
                raise RegistryError(
                    f"{path}: document_id {doc_id!r} pada {file_name!r} bukan slug valid "
                    f"(huruf kecil, angka, tanda hubung)"
                )
            if doc_id in seen_ids:
                raise RegistryError(
                    f"{path}: document_id {doc_id!r} dipakai dua berkas: "
                    f"{seen_ids[doc_id]!r} dan {file_name!r}"
                )
            seen_ids[doc_id] = file_name

        entries[file_name] = {**entry, "document_id": doc_id}

    return entries


def _load() -> dict[str, dict]:
    global _registry_cache
    if _registry_cache is not None:
        return _registry_cache

    path = Path(DOCUMENT_REGISTRY_PATH)
    if not path.exists():
        raise RegistryError(
            f"{path} tidak ditemukan. Buat kerangkanya dengan: "
            f"python scripts/scaffold_document_registry.py"
        )

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RegistryError(f"{path}: JSON tidak valid — {e}") from e

    entries = _validate(raw, path)
    terisi = sum(1 for e in entries.values() if e["document_id"])
    logger.info(
        "document_registry_loaded path=%s total=%d terisi=%d kosong=%d",
        path, len(entries), terisi, len(entries) - terisi,
    )
    _registry_cache = entries
    return _registry_cache


def get_entry(file_name: str) -> dict | None:
    """Entri registry untuk sebuah nama berkas, atau None bila tidak terdaftar."""
    return _load().get(file_name)


def get_document_id(file_name: str) -> str | None:
    """document_id untuk sebuah nama berkas.

    Mengembalikan None bila berkas tidak terdaftar ATAU terdaftar dengan
    document_id kosong (keluaran scaffolder yang belum diisi). Pemanggil
    memperlakukan keduanya sama: berkas dilewati, tidak diberi id provisional.
    """
    entry = get_entry(file_name)
    if entry is None:
        return None
    return entry["document_id"] or None


def check_sha256(file_name: str, actual_sha256: str) -> bool:
    """True bila sha256 cocok atau tidak dicatat di registry.

    False berarti nama berkas dipakai ulang untuk isi yang berbeda — kondisi
    yang membuat anotasi gold menunjuk ke dokumen yang salah.
    """
    entry = get_entry(file_name)
    if entry is None:
        return True
    expected = (entry.get("sha256") or "").strip()
    return not expected or expected == actual_sha256


def reload() -> None:
    """Buang cache. Dipakai setelah registry diedit tanpa restart proses."""
    global _registry_cache
    _registry_cache = None
