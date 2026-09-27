"""Fungsi murni untuk alat ukur transkripsi tabel (scripts/ukur_transkripsi.py).

Tidak memanggil model, tidak membuka PDF. Semua yang dapat diuji tanpa server
ada di sini: pembersihan keluaran model, penguraian tabel Markdown, penanda
pengecekan silang, sebaran tumpang tindih gambar-tabel, dan proyeksi waktu.

Fungsi yang juga dipakai indexing diimpor dari backend/services/transkripsi_murni.py;
yang tersisa di sini khusus pengukuran.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

# Fungsi yang juga dipakai indexing tinggal di backend (satu sumber).
from backend.services.transkripsi_murni import (  # noqa: E402,F401
    PROMPT_TRANSKRIPSI, PROMPT_TRANSKRIPSI_SEMUA, RencanaUkuran, TINGGI_KATA_MAKS,
    angka_dalam, bersihkan, bersihkan_semua, jenis_halaman, markdown_dari_baris,
    median, perluas_bbox, peringatan, rasio_tumpang, rencana_ukuran, urai_markdown,
)

_ORDINAL = re.compile(r"_c(\d+)$")


def cakupan_angka(transkripsi: str, teks_bbox: str) -> float | None:
    """Bagian angka lapisan teks area tabel yang muncul di transkripsi.

    Proksi recall tanpa anotasi: baris yang hilang menurunkannya. None bila
    area tidak punya angka (tidak dapat dinilai).
    """
    rujukan = angka_dalam(teks_bbox)
    if not rujukan:
        return None
    return len(rujukan & angka_dalam(transkripsi)) / len(rujukan)


def ketepatan_angka(transkripsi: str, teks_halaman: str) -> float | None:
    """Bagian angka transkripsi yang ada di lapisan teks halaman (proksi presisi)."""
    milik = angka_dalam(transkripsi)
    if not milik:
        return None
    return len(milik & angka_dalam(teks_halaman)) / len(milik)


@dataclass(frozen=True)
class Tumpang:
    image_chunk: str
    table_chunk: str
    document_id: str
    halaman: int
    rasio: float


def sebaran_tumpang(rows: list[dict]) -> list[Tumpang]:
    """Untuk tiap chunk gambar, tabel sehalaman dengan tumpang tindih terbesar (> 0)."""
    tabel = defaultdict(list)
    for r in rows:
        if r.get("element_type") == "Table" and r.get("bbox"):
            tabel[(r.get("document_id"), r.get("page_number"))].append(r)
    hasil = []
    for r in rows:
        if r.get("element_type") != "ImageDescription" or not r.get("bbox"):
            continue
        kunci = (r.get("document_id"), r.get("page_number"))
        calon = [(rasio_tumpang(r["bbox"], t["bbox"]), t["chunk_id"]) for t in tabel[kunci]]
        calon = [c for c in calon if c[0] > 0]
        if calon:
            rasio, tid = max(calon)
            hasil.append(Tumpang(r["chunk_id"], tid, kunci[0], kunci[1], rasio))
    return sorted(hasil, key=lambda t: (t.rasio, t.image_chunk))


def celah_terbesar(nilai: list[float]) -> tuple[float, float] | None:
    """Pasangan nilai berurutan dengan selisih terbesar. None bila < 2 nilai."""
    urut = sorted(nilai)
    if len(urut) < 2:
        return None
    i = max(range(1, len(urut)), key=lambda k: urut[k] - urut[k - 1])
    return urut[i - 1], urut[i]


def dampak_serapan(rows: list[dict], tumpang: list[Tumpang], ambang: float) -> dict:
    """Halaman dan chunk yang chunk_id-nya bergeser bila gambar >= ambang diserap."""
    serap = {t.image_chunk for t in tumpang if t.rasio >= ambang}
    per_halaman = defaultdict(list)
    for r in rows:
        m = _ORDINAL.search(r.get("chunk_id") or "")
        if m:
            per_halaman[(r.get("document_id"), r.get("page_number"))].append(
                (int(m.group(1)), r["chunk_id"]))
    halaman, geser = set(), 0
    for kunci, isi in per_halaman.items():
        hilang = [o for o, cid in isi if cid in serap]
        if not hilang:
            continue
        halaman.add(kunci)
        geser += sum(1 for o, cid in isi if cid not in serap and o > min(hilang))
    return {"ambang": ambang, "gambar_diserap": len(serap),
            "halaman_terdampak": len(halaman), "chunk_bergeser": geser}


def regresi_linear(x: list[float], y: list[float]) -> tuple[float, float] | None:
    """Kuadrat terkecil y = a + b*x. None bila x tak bervariasi."""
    n = len(x)
    if n < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    if sxx == 0:
        return None
    b = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / sxx
    return my - b * mx, b
