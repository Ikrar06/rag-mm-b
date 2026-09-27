"""Fungsi murni untuk scripts/uji_transkripsi_pipeline.py: pemilihan sampel,
pemasangan chunk v4 dengan v5, dan penyusunan laporan berdampingan."""

from __future__ import annotations

import re

_ORDINAL = re.compile(r"_c(\d+)$")

# Sampel bawaan: sepuluh tabel campuran yang diminta. Pemilih berbentuk
# chunk_id persis, "awalan_document_id:halaman:element_type", atau "lanjutan"
# (potongan tabel lanjutan pertama yang header-nya diulang di v4).
SAMPEL_BAWAAN = (
    "ukt-tahun-2025:5:Table",                                                   # cap BSrE
    "jadwal-retensi-arsip:50:Table",                                            # sel gabungan
    "standar-biaya-universitas-hasanuddin-tahun-anggaran-2025:9:Table",         # tabel panjang
    "pedoman-penyusunan-laporan-keuangan-perguruan-tinggi-negeri-badan-hukum-universitas-haanuddin"
    ":23:ImageDescription",                                                     # narasi+tabel
    "manual-dosen:35:ImageDescription",                                         # tangkapan layar
    "lanjutan",                                                                 # potongan lanjutan
    "bagan-akun-standar-dan-kodefikasi-akun:10:Table",                          # pindai lapisan OCR
    "sop12-pelaksanaan-kerja-praktek:10:ImageDescription",                      # gambar-tabel
    "standar-biaya-universitas-hasanuddin-tahun-anggaran-2026:19:Table",        # kolom BESARAN
    "unhas-academic-calendar-20252026:10:Table",                                # teks vertikal
)


def _ordinal(cid: str) -> int:
    m = _ORDINAL.search(cid or "")
    return int(m.group(1)) if m else -1


def pilih(rows: list[dict], pemilih: str) -> dict | None:
    """Chunk v4 untuk satu pemilih, atau None. Deterministik (urut chunk_id)."""
    urut = sorted(rows, key=lambda r: r.get("chunk_id") or "")
    if pemilih == "lanjutan":
        return next((r for r in urut if (r.get("table_part") or 0) >= 1
                     and r.get("table_header_repeated")), None)
    if pemilih.count(":") == 2:
        awalan, hal, jenis = pemilih.split(":")
        return next((r for r in urut if (r.get("document_id") or "").startswith(awalan)
                     and r.get("page_number") == int(hal) and r.get("element_type") == jenis), None)
    return next((r for r in urut if r.get("chunk_id") == pemilih), None)


def halaman_uji(target: dict) -> set[int]:
    """Halaman yang harus diproses: halaman target, ditambah halaman kepala
    rantai sampai target bila ia potongan lanjutan (header rantai berasal dari
    transkripsi kepala)."""
    hal = target.get("page_number")
    kepala = target.get("table_group_id") or ""
    m = re.search(r"_p(\d+)_c\d+$", kepala)
    awal = int(m.group(1)) if m and (target.get("table_part") or 0) >= 1 else hal
    return set(range(min(awal, hal), hal + 1))


def pasangkan(v4: dict, v5_rows: list[dict], dibuang: set[str]) -> tuple[dict | None, str]:
    """Chunk v5 padanan chunk v4. -> (chunk, cara).

    Utamakan chunk_id sama DAN jenis cocok; bila tidak, urutan chunk berjenis
    sama di halaman yang sama (chunk_id bergeser setelah cap dibuang).
    Gambar yang ditranskripsi jadi Table dianggap sejenis dengan
    ImageDescription asalnya lewat image_id.
    """
    if v4.get("image_id") and v4["image_id"] in dibuang:
        return None, "dibuang"
    if v4.get("image_id"):
        for r in v5_rows:
            if r.get("image_id") == v4["image_id"]:
                return r, "image_id"
    for r in v5_rows:
        if r.get("chunk_id") == v4.get("chunk_id") and r.get("element_type") == v4.get("element_type"):
            return r, "chunk_id"
    sejenis = [r for r in v5_rows if r.get("page_number") == v4.get("page_number")
               and r.get("element_type") == v4.get("element_type")]
    return (sejenis[0], "urutan") if sejenis else (None, "tidak_ada")


def md_berdampingan(no: int, pemilih: str, v4: dict, v5: dict | None, cara: str,
                    info_gambar: dict | None) -> str:
    """Satu berkas Markdown: metadata v5, teks v4, teks v5."""
    meta = {k: (v5 or {}).get(k) for k in (
        "chunk_id", "element_type", "table_source", "table_origin", "image_content",
        "transkripsi_peringatan", "render_bbox", "table_group_id", "table_part",
        "table_header_repeated")}
    baris_meta = "\n".join(f"- {k}: `{v}`" for k, v in meta.items() if v not in (None, "", []))
    gambar = f"\n- klasifikasi gambar: `{info_gambar}`" if info_gambar else ""
    teks_v5 = (v5 or {}).get("text_content") or "(tidak ada — " + cara + ")"
    return (f"# {no}. {pemilih}\n\nv4: `{v4.get('chunk_id')}` ({v4.get('element_type')})  \n"
            f"v5: dipasangkan lewat `{cara}`\n{baris_meta}{gambar}\n\n"
            f"## v4\n\n{v4.get('text_content') or ''}\n\n## v5\n\n{teks_v5}\n")


def baris_ringkas(no: int, v4: dict, v5: dict | None, cara: str, info_gambar: dict | None) -> dict:
    return {
        "no": no, "chunk_v4": v4.get("chunk_id"), "chunk_v5": (v5 or {}).get("chunk_id"),
        "pasangan": cara, "jenis_v4": v4.get("element_type"),
        "jenis_v5": (v5 or {}).get("element_type"),
        "sumber": (v5 or {}).get("table_source") or (v5 or {}).get("image_content") or "",
        "perlakuan_gambar": (info_gambar or {}).get("perlakuan", ""),
        "peringatan": "; ".join((v5 or {}).get("transkripsi_peringatan") or []),
        "header_diulang": (v5 or {}).get("table_header_repeated"),
        "karakter_v4": len(v4.get("text_content") or ""),
        "karakter_v5": len((v5 or {}).get("text_content") or ""),
    }
