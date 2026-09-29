"""Gabungkan dump run utama dengan dump run susulan. Fungsi murni.

Run susulan terjadi ketika sebagian dokumen gagal di run utama lalu diindeks
ulang secara inkremental ke collection yang SAMA. Collection lalu berisi
keduanya, jadi dump yang menjadi dataset juga harus berisi keduanya — dan
hanya sah bila konfigurasi yang menentukan isi chunk identik di kedua run.
"""

from __future__ import annotations

# Bagian manifest yang menentukan isi chunk; wajib sama antar run.
BAGIAN_KONFIGURASI = ("chunking", "research_flags", "hardcoded_constants", "models")
# Nilai yang wajar berbeda antar run (hitungan, bukan konfigurasi).
KUNCI_HITUNGAN = frozenset({"hitungan", "batas_gambar"})


def _tanpa_hitungan(x):
    if isinstance(x, dict):
        return {k: _tanpa_hitungan(v) for k, v in x.items() if k not in KUNCI_HITUNGAN}
    return x


def beda_konfigurasi(m_utama: dict, m_susulan: dict) -> list[str]:
    """Jalur kunci manifest yang berbeda di bagian penentu isi chunk."""
    beda: list[str] = []

    def telusur(a, b, jalur):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                telusur(a.get(k), b.get(k), f"{jalur}.{k}")
        elif a != b:
            beda.append(f"{jalur}: {a!r} != {b!r}")

    for bagian in BAGIAN_KONFIGURASI:
        telusur(_tanpa_hitungan(m_utama.get(bagian)), _tanpa_hitungan(m_susulan.get(bagian)), bagian)
    k_u = (m_utama.get("provenance") or {}).get("qdrant_collection")
    k_s = (m_susulan.get("provenance") or {}).get("qdrant_collection")
    if k_u != k_s:
        beda.append(f"provenance.qdrant_collection: {k_u!r} != {k_s!r}")
    return beda


def gabung(utama: list[dict], susulan: list[dict], kunci: str) -> list[dict]:
    """Rekaman utama + susulan. ValueError bila dokumen atau `kunci` bertabrakan."""
    dok_u = {r.get("document_id") for r in utama} - {None}
    tabrak_dok = sorted({r.get("document_id") for r in susulan} & dok_u)
    if tabrak_dok:
        raise ValueError(f"dokumen ada di kedua run: {tabrak_dok[:5]}")
    id_u = {r.get(kunci) for r in utama}
    tabrak = sorted({r.get(kunci) for r in susulan} & id_u - {None})
    if tabrak:
        raise ValueError(f"{kunci} ganda: {tabrak[:5]}")
    return [*utama, *susulan]


def manifest_gabungan(m_utama: dict, m_susulan: dict, n_chunks: int, n_images: int,
                      n_narasi: int, dokumen_susulan: list[str]) -> dict:
    """Manifest utama, ditambah catatan run susulan dan jumlah gabungan."""
    ringkas = lambda m: {k: m.get(k) for k in ("run_id", "created_at", "n_chunks", "n_images")} | {  # noqa: E731
        "git_commit": (m.get("provenance") or {}).get("git_commit"),
        "table_transcription_hitungan": (((m.get("research_flags") or {})
                                          .get("table_transcription") or {}).get("hitungan")),
        "vision_cache": m.get("vision_cache"),
    }
    return {
        **m_utama,
        "n_chunks": n_chunks, "n_images": n_images, "n_images_with_narrative": n_narasi,
        "extraction_reports": {**(m_utama.get("extraction_reports") or {}),
                               **(m_susulan.get("extraction_reports") or {})},
        "degraded_documents": [*(m_utama.get("degraded_documents") or []),
                               *(m_susulan.get("degraded_documents") or [])],
        "gabungan": {
            "runs": [{"peran": "utama", **ringkas(m_utama)},
                     {"peran": "susulan", "dokumen": dokumen_susulan, **ringkas(m_susulan)}],
            "catatan": "Run susulan mengindeks dokumen yang gagal di run utama ke collection "
                       "yang sama; konfigurasi penentu isi chunk diverifikasi identik.",
        },
    }
