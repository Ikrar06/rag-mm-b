"""Cache persisten deskripsi gambar — ARTEFAK EKSPERIMEN, bukan optimasi.

Isi cache ini adalah SUMBER ISI CHUNK. Dua run yang menghasilkan chunk sama
hanya dapat dibuktikan berasal dari deskripsi yang sama bila cache-nya sama,
jadi ia diperlakukan setara `document_registry.json`: dibagikan berkasnya antar
fork, dicatat hash isinya di manifest, dan tidak dihapus di tengah eksperimen.

KUNCI
-----
    sha256( image_sha256 | variant | prompt_sha256 | vision_model_digest )

`image_sha256` dihitung atas bytes ASLI sebelum resize (lihat
preprocessing._persist_image_elements) sehingga tidak bergantung versi Pillow.

Model atau prompt yang berbeda menghasilkan kunci berbeda, jadi entri dari
konfigurasi lain tidak akan pernah dikembalikan sebagai hit. Gerbang di
indexing._check_vision_cache menolak run bila cache MEMUAT entri dari
konfigurasi lain — bukan karena bisa salah-pakai, tapi karena artinya korpus
dideskripsikan oleh dua model berbeda.

VARIAN (b) DAN (c) BERBAGI CACHE
--------------------------------
Disengaja. `narrative_summary` yang sama wajib dipakai kedua varian supaya
perbedaan skor berasal dari strategi indexing, bukan dari dua panggilan model
yang kebetulan berbeda. Komponen `variant` di kunci membedakan JENIS ringkasan
(naratif vs terstruktur), bukan run.

YANG DI-CACHE DAN YANG TIDAK
----------------------------
Di-cache — keputusan model yang sah:
    described   model mengembalikan teks yang bisa dipakai
    decorative  model menjawab DEKORATIF
    unclear     model menjawab TIDAK JELAS

TIDAK PERNAH di-cache — kegagalan transient:
    timeout, HTTP error, respons kosong, bentuk respons tak terduga

Kalau kegagalan transient ikut masuk, satu timeout menjadi permanen: gambar itu
hilang dari SELURUH eksperimen dan tidak dapat dipulihkan kecuali cache dihapus
manual — yang sendirinya membatalkan anotasi gold.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from backend.config import VISION_CACHE_ENABLED, VISION_CACHE_PATH

logger = logging.getLogger(__name__)

# Verdict yang boleh disimpan. `failed` sengaja TIDAK ada di sini.
VERDICT_DESCRIBED = "described"
VERDICT_DECORATIVE = "decorative"
VERDICT_UNCLEAR = "unclear"
CACHEABLE_VERDICTS = (VERDICT_DESCRIBED, VERDICT_DECORATIVE, VERDICT_UNCLEAR)

VARIANT_NARRATIVE = "narrative"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS descriptions (
    cache_key           TEXT PRIMARY KEY,
    image_sha256        TEXT NOT NULL,
    variant             TEXT NOT NULL,
    prompt_sha256       TEXT NOT NULL,
    vision_model        TEXT NOT NULL,
    vision_model_digest TEXT,
    verdict             TEXT NOT NULL,
    description         TEXT,
    created_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cfg ON descriptions(vision_model_digest, prompt_sha256);
CREATE INDEX IF NOT EXISTS idx_img ON descriptions(image_sha256);
"""

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None
_readonly = False
_opened = False

# Penghitung per-run, dicatat di manifest.
_stats = {"hit": 0, "miss": 0, "stored": 0, "skipped_failure": 0}


@dataclass(frozen=True)
class CachedDescription:
    verdict: str
    description: str | None

    @property
    def usable_text(self) -> str | None:
        """Teks yang dipakai sebagai isi chunk, atau None bila bukan `described`."""
        return self.description if self.verdict == VERDICT_DESCRIBED else None


def cache_path() -> Path:
    return Path(os.path.expanduser(VISION_CACHE_PATH))


def enabled() -> bool:
    return VISION_CACHE_ENABLED and bool((VISION_CACHE_PATH or "").strip())


def make_key(image_sha256: str, variant: str, prompt_sha256: str,
             vision_model_digest: str | None) -> str:
    """Kunci cache. `vision_model_digest` None diberi penanda eksplisit supaya
    entri tanpa digest tidak diam-diam disamakan dengan entri ber-digest."""
    digest = vision_model_digest or "<no-digest>"
    raw = "|".join((image_sha256, variant, prompt_sha256, digest))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _connect() -> sqlite3.Connection | None:
    """Buka koneksi sekali. Read-only bila berkas tidak dapat ditulis.

    Read-only BUKAN kegagalan: itu mode yang disengaja untuk pihak kedua saat
    cache dimiliki satu user (lihat CHANGES.md, opsi berbagi cache).
    """
    global _conn, _readonly, _opened
    if _opened:
        return _conn
    _opened = True

    if not enabled():
        return None

    path = cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error("vision_cache_dir_failed path=%s error=%s — cache dimatikan", path, e)
        return None

    exists = path.exists()
    writable = os.access(path, os.W_OK) if exists else os.access(path.parent, os.W_OK)

    try:
        if writable:
            conn = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")   # aman untuk dua proses
            conn.execute("PRAGMA busy_timeout=30000")
            conn.executescript(_SCHEMA)
            conn.commit()
            _readonly = False
        else:
            if not exists:
                logger.error(
                    "vision_cache_unwritable path=%s — direktori tidak dapat ditulis "
                    "dan berkas belum ada; cache dimatikan", path,
                )
                return None
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30,
                                   check_same_thread=False)
            _readonly = True
            logger.warning(
                "vision_cache_readonly path=%s — berkas tidak dapat ditulis. "
                "Hit tetap dipakai, tapi deskripsi BARU tidak tersimpan; gambar "
                "yang belum ada di cache akan dipanggilkan model tiap run.", path,
            )
    except sqlite3.Error as e:
        logger.error("vision_cache_open_failed path=%s error=%s — cache dimatikan", path, e)
        return None

    _conn = conn
    logger.info("vision_cache_opened path=%s readonly=%s entries=%d",
                path, _readonly, count_entries())
    return _conn


def get(key: str) -> CachedDescription | None:
    conn = _connect()
    if conn is None:
        return None
    try:
        with _lock:
            row = conn.execute(
                "SELECT verdict, description FROM descriptions WHERE cache_key=?", (key,)
            ).fetchone()
    except sqlite3.Error as e:
        logger.warning("vision_cache_read_failed error=%s", e)
        return None

    if row is None:
        _stats["miss"] += 1
        return None
    _stats["hit"] += 1
    return CachedDescription(verdict=row[0], description=row[1])


def put(key: str, *, image_sha256: str, variant: str, prompt_sha256: str,
        vision_model: str, vision_model_digest: str | None,
        verdict: str, description: str | None) -> None:
    """Simpan HANYA keputusan model yang sah.

    Verdict di luar CACHEABLE_VERDICTS ditolak dan dihitung terpisah — kegagalan
    transient tidak boleh menjadi permanen.
    """
    if verdict not in CACHEABLE_VERDICTS:
        _stats["skipped_failure"] += 1
        logger.debug("vision_cache_skip_failure verdict=%s sha=%s", verdict, image_sha256[:12])
        return

    conn = _connect()
    if conn is None or _readonly:
        return
    try:
        with _lock:
            conn.execute(
                "INSERT OR REPLACE INTO descriptions "
                "(cache_key, image_sha256, variant, prompt_sha256, vision_model, "
                " vision_model_digest, verdict, description, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (key, image_sha256, variant, prompt_sha256, vision_model,
                 vision_model_digest, verdict, description,
                 datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
        _stats["stored"] += 1
    except sqlite3.Error as e:
        logger.warning("vision_cache_write_failed error=%s", e)


# ─── Inspeksi & provenance ───────────────────────────────────────────────────

def count_entries() -> int:
    conn = _conn
    if conn is None:
        return 0
    try:
        with _lock:
            return conn.execute("SELECT COUNT(*) FROM descriptions").fetchone()[0]
    except sqlite3.Error:
        return 0


def configurations() -> list[dict]:
    """Kombinasi (model, digest, prompt) yang ADA di cache, beserta jumlahnya.

    Dipakai gerbang: lebih dari satu kombinasi berarti korpus dideskripsikan
    oleh konfigurasi berbeda.
    """
    conn = _connect()
    if conn is None:
        return []
    try:
        with _lock:
            rows = conn.execute(
                "SELECT vision_model, vision_model_digest, prompt_sha256, "
                "       variant, COUNT(*), MAX(created_at) "
                "FROM descriptions "
                "GROUP BY vision_model, vision_model_digest, prompt_sha256, variant "
                "ORDER BY COUNT(*) DESC"
            ).fetchall()
    except sqlite3.Error as e:
        logger.warning("vision_cache_inspect_failed error=%s", e)
        return []
    return [
        {"vision_model": r[0], "vision_model_digest": r[1], "prompt_sha256": r[2],
         "variant": r[3], "entries": r[4], "last_written": r[5]}
        for r in rows
    ]


def verdict_counts() -> dict[str, int]:
    conn = _connect()
    if conn is None:
        return {}
    try:
        with _lock:
            rows = conn.execute(
                "SELECT verdict, COUNT(*) FROM descriptions GROUP BY verdict"
            ).fetchall()
    except sqlite3.Error:
        return {}
    return {r[0]: r[1] for r in rows}


def content_sha256() -> str | None:
    """Hash ISI cache, bukan byte berkasnya.

    Berkas SQLite memuat halaman bebas dan jurnal WAL yang berubah tanpa isi
    berubah, dan proses lain yang menulis bersamaan mengubah bytenya. Hash atas
    baris yang diurutkan stabil dan bermakna: dua cache dengan hash sama memuat
    deskripsi yang sama persis.
    """
    conn = _connect()
    if conn is None:
        return None
    h = hashlib.sha256()
    try:
        with _lock:
            for row in conn.execute(
                "SELECT cache_key, verdict, COALESCE(description,'') "
                "FROM descriptions ORDER BY cache_key"
            ):
                h.update("\x1f".join(row).encode("utf-8"))
                h.update(b"\x1e")
    except sqlite3.Error as e:
        logger.warning("vision_cache_hash_failed error=%s", e)
        return None
    return h.hexdigest()


def stats() -> dict:
    return dict(_stats)


def provenance() -> dict:
    """Blok untuk run_manifest.json."""
    if not enabled():
        return {"enabled": False}
    _connect()
    return {
        "enabled": True,
        "path": str(cache_path()),
        "readonly": _readonly,
        "entries_total": count_entries(),
        "content_sha256": content_sha256(),
        "verdicts": verdict_counts(),
        **stats(),
    }


def reset_stats() -> None:
    for k in _stats:
        _stats[k] = 0


def close() -> None:
    """Tutup koneksi. Dipakai skrip inspeksi dan pengujian."""
    global _conn, _opened, _readonly
    if _conn is not None:
        try:
            _conn.close()
        except sqlite3.Error:
            pass
    _conn, _opened, _readonly = None, False, False
