"""Registry pemetaan nama berkas PDF -> document_id.

`document_id` menentukan `chunk_id`, `image_id`, DAN nama direktori gambar di
disk sekaligus. Karena itu registry diperlakukan sebagai ARTEFAK BERSAMA:
di-generate sekali, disimpan di lokasi bersama, dan dipakai apa adanya oleh
kedua fork. Konsistensi datang dari berkas yang sama, bukan dari dua algoritma
yang kebetulan menghasilkan keluaran sama.

`document_id` diisi otomatis oleh scaffolder lewat `slugify()`. Satu-satunya
bagian yang manual adalah penyelesaian tabrakan slug — dan itu memang gagal
keras, bukan diselesaikan dengan sufiks otomatis.

Registry juga menjadi titik tumbuh untuk 13 field lain yang dibutuhkan
corpus_metadata.jsonl (title, source_unit, effective_start, superseded_by, dan
seterusnya). Field-field itu tetap butuh kurasi manusia dan menyusul bersama
metadata temporal di lapis 4.

Bentuk berkas:

    {
      "_meta": {
        "generated_at": "2026-08-25T09:00:00+00:00",
        "slug_rule_version": 1,
        "tool": "scripts/scaffold_document_registry.py"
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

Bentuk datar lama (tanpa `_meta`/`documents`) tetap dibaca.

Generate atau perbarui dengan:
    python scripts/scaffold_document_registry.py
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path

from backend.config import DOCUMENT_REGISTRY_PATH

logger = logging.getLogger(__name__)

# Slug: huruf kecil, angka, dan tanda hubung. Tidak boleh diawali/diakhiri hubung.
_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

# Versi aturan slug. Dicatat per entri saat di-generate, sehingga registry yang
# memuat entri dari beberapa versi aturan tetap dapat ditelusuri — entri lama
# TIDAK PERNAH ditimpa walau aturannya berubah.
SLUG_RULE_VERSION = 1

# Prefix penomoran di AWAL nama: "1.-", "01_", "12)", "(3) ".
#
# Dibatasi DUA digit, bukan tiga. Asimetri kerugiannya: prefix bermakna yang
# terpangkas menghasilkan slug yang terlihat normal — kesalahannya senyap —
# sedangkan nomor urut tiga digit yang gagal terpangkas menghasilkan slug jelek
# tapi jelas dan bisa diedit. Untuk registry yang menentukan chunk_id dan
# image_id sekaligus, kesalahan yang terlihat lebih baik daripada yang tidak.
#
# Batas ini juga melindungi tahun empat digit: "2024_Kalender.pdf" tetap
# menjadi "2024-kalender", tidak tergerus jadi "kalender" yang akan bertabrakan
# dengan berkas tahun lain.
_NUMBER_PREFIX_RE = re.compile(r"^\s*\(?\d{1,2}\)?\s*[.)_-]+\s*")


def slugify(file_name: str) -> str:
    """Nama berkas -> slug document_id.

    Enam langkah berurutan: buang ekstensi, buang prefix penomoran, normalisasi
    NFKD + buang non-ASCII, lowercase, non-alfanumerik jadi tanda hubung,
    rapatkan tanda hubung berulang.

    TIDAK menebak penanda versi. Angka di akhir nama seperti "..._online_2"
    ambigu antara urutan dan versi; menerjemahkannya jadi "-v2" berarti
    mengklaim dua dokumen adalah revisi satu sama lain. Angka itu dipertahankan
    apa adanya sebagai bagian nama. Versi sebenarnya menyusul bersama metadata
    temporal di lapis 4.

    TIDAK dipotong panjangnya. Dokumen akademik sering berbeda hanya di ujung
    nama ("Program Sarjana" vs "Program Magister"); memotong akan menabrakkan
    keduanya, dan tabrakan adalah hal yang justru harus dihindari.

    Mengembalikan string kosong bila tidak ada karakter yang tersisa — pemanggil
    memperlakukan itu sebagai kegagalan.
    """
    s = Path(file_name).stem
    s = _NUMBER_PREFIX_RE.sub("", s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9]+", "-", s.lower())
    return re.sub(r"-{2,}", "-", s).strip("-")

_registry_cache: dict[str, dict] | None = None


class RegistryError(Exception):
    """Registry tidak dapat dimuat atau isinya tidak valid."""


def split_document_block(raw: object) -> tuple[dict, dict]:
    """Pisahkan header dari blok dokumen.

    Mendukung dua bentuk:
      {"_meta": {...}, "documents": {...}}   <- bentuk sekarang
      {"nama.pdf": {...}, ...}               <- bentuk datar lama
    """
    if isinstance(raw, dict) and isinstance(raw.get("documents"), dict):
        return raw.get("_meta") or {}, raw["documents"]
    return {}, raw if isinstance(raw, dict) else {}


def _validate(raw: object, path: Path) -> dict[str, dict]:
    """Validasi bentuk registry. Melempar RegistryError bila rusak.

    Entri dengan document_id kosong DIBIARKAN LOLOS — itu keluaran normal
    scaffolder yang belum diisi. `get_document_id` akan mengembalikan None
    untuk entri semacam itu, sehingga berkasnya dilewati saat indexing.
    """
    if not isinstance(raw, dict):
        raise RegistryError(f"{path}: akar JSON harus object, bukan {type(raw).__name__}")

    _, raw = split_document_block(raw)

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
