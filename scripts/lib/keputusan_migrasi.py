"""Petakan ulang kunci berkas keputusan antar penomoran chunk_id.

chunk_id adalah POSISI, dan posisi bergeser ketika logika chunking berubah.
Perbaikan current_page (af85fdc) memindahkan chunk teks ke halaman yang benar,
sehingga pencacah per halaman bergeser dan tabel yang sama berganti nama:
rubrik_p28_c00 di v2 menjadi rubrik_p28_c01 di v3, sementara rubrik_p28_c00 di
v3 kini chunk TEKS. Berkas keputusan yang dibuat dari koleksi v2 menunjuk
chunk yang salah di v3.

Jangkar pemetaan: (dokumen, halaman, urutan tabel di halaman itu). Ekstraksi
tabel tidak berubah antara kedua run — yang bergeser hanya chunk teks — jadi
tabel ke-k di halaman h tetap tabel ke-k di halaman h. Setiap pemetaan
DIVERIFIKASI dengan sidik `text_as_html`; yang sidiknya tidak cocok tidak
dipetakan otomatis.

Fungsi murni.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CHUNK_ID_RE = re.compile(r"^(?P<doc>.+)_p(?:\d+|NA)_c\d+$")


def _slug(cid: str) -> str | None:
    m = _CHUNK_ID_RE.match(cid or "")
    return m.group("doc") if m else None


def posisi_tabel(rows) -> tuple[dict, dict]:
    """(chunk_id -> posisi, posisi -> chunk_id) untuk chunk tabel.

    posisi = (dokumen, halaman, urutan tabel di halaman menurut chunk_index).
    """
    per_hal: dict[tuple, list] = {}
    for r in rows:
        if r.get("element_type") != "Table":
            continue
        cid = r.get("chunk_id")
        dok = _slug(cid)
        if not dok:
            continue
        per_hal.setdefault((dok, r.get("page_number")), []).append(r)
    ke_pos, dari_pos = {}, {}
    for (dok, hal), daftar in per_hal.items():
        daftar.sort(key=lambda r: r.get("chunk_index") if isinstance(r.get("chunk_index"), int) else 0)
        for k, r in enumerate(daftar):
            pos = (dok, hal, k)
            ke_pos[r["chunk_id"]] = pos
            dari_pos[pos] = r["chunk_id"]
    return ke_pos, dari_pos


@dataclass(frozen=True)
class HasilKunci:
    lama: str
    baru: str | None
    status: str          # id_sama | dipetakan | sidik_beda | tak_ditemukan
    alasan: str = ""


def petakan_id(cid: str, pos_lama: dict, dari_pos_baru: dict,
               sidik_lama: dict, sidik_baru: dict) -> HasilKunci:
    pos = pos_lama.get(cid)
    if pos is None:
        return HasilKunci(cid, None, "tak_ditemukan",
                          "bukan chunk tabel di dump lama")
    baru = dari_pos_baru.get(pos)
    if baru is None:
        return HasilKunci(cid, None, "tak_ditemukan",
                          f"tidak ada tabel ke-{pos[2]} di halaman {pos[1]} dump baru")
    sl, sb = sidik_lama.get(cid, ""), sidik_baru.get(baru, "")
    if sl and sb and sl != sb:
        return HasilKunci(cid, None, "sidik_beda",
                          f"posisi cocok ({baru}) tapi isi tabel berbeda")
    return HasilKunci(cid, baru, "id_sama" if baru == cid else "dipetakan",
                      "" if sl and sb else "sidik tidak tersedia di salah satu dump")
