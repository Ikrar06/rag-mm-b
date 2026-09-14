"""Penanganan tabel yang terpotong di batas halaman.

`partition_pdf` mendeteksi layout PER HALAMAN, jadi tabel yang melintasi halaman
sudah tiba sebagai DUA element Table terpisah. Tidak ada penanda apa pun dari
Unstructured yang menautkannya — `is_continuation` di pustaka itu berarti hal
lain (potongan ke-2+ dari satu element yang dipotong agar muat jendela chunk,
`documents/elements.py:180`).

Modul ini mengumpulkan sinyal yang dipakai memutuskan apakah dua potongan
memang satu tabel. Keputusan akhirnya TIDAK ditanam di sini: ia dibaca dari
`table_continuation.json`, berkas terkurasi yang ditinjau manusia — pola yang
sama dengan `document_registry.json`.

Seluruhnya di belakang `INDEX_TABLE_CONTINUATION`, default mati.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from backend.config import INDEX_TABLE_CONTINUATION

logger = logging.getLogger(__name__)

# Nama env var milik Unstructured. Dibaca `env_config.EXTRACT_TABLE_AS_CELLS`
# (partition/utils/config.py:123) pada SETIAP akses properti, jadi menyetelnya
# sebelum partition_pdf sudah cukup — tidak ada cache yang perlu diakali.
ENV_TABLE_AS_CELLS = "EXTRACT_TABLE_AS_CELLS"

# Toleransi "menempel batas area teks", dalam satuan tinggi halaman
# ternormalisasi. Dipakai setelah area teks dipersempit oleh Header/Footer,
# jadi jauh lebih ketat daripada ambang kasar terhadap halaman penuh.
TOLERANSI_MARGIN = 0.05

# Kategori element yang BUKAN isi dokumen tapi menandai batas area teks.
KATEGORI_PENANDA = frozenset({"Header", "Footer", "PageNumber", "PageBreak"})


def siapkan_ekstraksi() -> dict[str, str]:
    """Setel env var Unstructured sebelum `partition_pdf`, dan laporkan apa yang disetel.

    Disetel DI SINI, bukan di shell, karena nilainya menentukan isi metadata
    element dan karenanya wajib tercatat di `run_manifest.json`. Env var yang
    hanya hidup di shell peneliti tidak akan pernah masuk artefak eksperimen.

    Nilai yang sudah ada di lingkungan TIDAK ditimpa — peneliti yang sengaja
    menyetelnya berbeda tetap menang, dan manifest merekam nilai efektifnya.
    """
    if not INDEX_TABLE_CONTINUATION:
        return {}
    sebelum = os.environ.get(ENV_TABLE_AS_CELLS)
    if sebelum is None:
        os.environ[ENV_TABLE_AS_CELLS] = "true"
    efektif = os.environ[ENV_TABLE_AS_CELLS]
    if sebelum is not None and sebelum.lower() not in ("true", "1", "t"):
        logger.warning(
            "%s sudah disetel %r di lingkungan dan TIDAK ditimpa — table_as_cells "
            "tidak akan terisi, deteksi tabel lintas halaman jatuh ke raw_html",
            ENV_TABLE_AS_CELLS, sebelum,
        )
    return {ENV_TABLE_AS_CELLS: efektif}


@dataclass(frozen=True)
class AreaTeks:
    """Batas atas dan bawah area teks sebuah halaman, ternormalisasi 0..1."""

    atas: float = 0.0
    bawah: float = 1.0

    def di_puncak(self, bbox, toleransi: float = TOLERANSI_MARGIN) -> bool:
        """Apakah element menempel batas ATAS area teks."""
        return bool(bbox) and len(bbox) == 4 and bbox[1] <= self.atas + toleransi

    def di_dasar(self, bbox, toleransi: float = TOLERANSI_MARGIN) -> bool:
        """Apakah element menempel batas BAWAH area teks."""
        return bool(bbox) and len(bbox) == 4 and bbox[3] >= self.bawah - toleransi


def area_teks_halaman(bbox_penanda) -> AreaTeks:
    """Persempit area teks berdasarkan bbox Header/Footer/PageNumber halaman itu.

    Penanda di paruh ATAS halaman mendorong batas atas ke bawah; penanda di
    paruh BAWAH mendorong batas bawah ke atas. Tanpa penanda, area teks adalah
    seluruh halaman — itu perilaku lama, dan hasilnya sama dengan sebelumnya.

    Ini yang menggantikan ambang tetap 0,72/0,3 terhadap halaman penuh: dokumen
    dengan kop surat tinggi dan dokumen tanpa kop tidak lagi diukur dengan
    penggaris yang sama.
    """
    atas, bawah = 0.0, 1.0
    for b in bbox_penanda:
        if not b or len(b) != 4:
            continue
        y0, y1 = b[1], b[3]
        tengah = (y0 + y1) / 2
        if tengah < 0.5:
            atas = max(atas, min(y1, 0.5))
        else:
            bawah = min(bawah, max(y0, 0.5))
    return AreaTeks(atas=round(atas, 4), bawah=round(bawah, 4))


def ulangi_header(teks_header: str, teks_potongan: str) -> str:
    """Sisipkan baris header di depan potongan lanjutan.

    Bentuk (b) dari rancangan: tiap potongan berdiri sendiri saat di-retrieve,
    tanpa bergantung pada ekspansi tetangga atau reranker. `raw_html` TIDAK
    disentuh — ia catatan ekstraksi yang setia dan jadi rujukan RCAA.

    Mengembalikan teks apa adanya bila headernya kosong, supaya pemanggil tidak
    perlu menjaga kasus itu.
    """
    header = (teks_header or "").strip()
    isi = (teks_potongan or "").strip()
    if not header or not isi:
        return teks_potongan
    if isi.startswith(header):
        return teks_potongan          # sudah punya header, jangan digandakan
    return f"{header}\n{isi}"


def lengkapi_sinyal(elements: list[dict], penanda_per_halaman: dict, pdf_path) -> None:
    """Tambahkan sinyal lintas-halaman ke metadata tiap element Table.

    Mengisi `area_teks` (batas area teks halaman itu) dan `batas_kolom`
    (dipulihkan dari posisi kata lewat PyMuPDF). Keduanya hanya dipakai untuk
    MEMUTUSKAN kelanjutan; tidak satu pun masuk teks chunk.

    Satu-satunya fungsi di modul ini yang mengubah argumennya. Dilakukan di
    tempat karena `elements` bisa memuat ribuan entri dan menyalinnya hanya
    untuk menambah dua kunci tidak sepadan; pemanggilnya adalah `_extract_hi_res`
    yang memang sedang membangun daftar itu.

    Tidak pernah melempar: kegagalan membaca PDF membuat `batas_kolom` menjadi
    "tidak tersedia", yang berarti pasangan itu jatuh ke adjudikasi vision —
    bukan dianggap tidak cocok.
    """
    from backend.services.table_columns import batas_dari_halaman

    area_per_halaman = {
        hal: area_teks_halaman(bboxes)
        for hal, bboxes in penanda_per_halaman.items()
    }
    bawaan = AreaTeks()

    for el in elements:
        if el.get("category") != "Table":
            continue
        meta = el.setdefault("metadata", {})
        area = area_per_halaman.get(el.get("page"), bawaan)
        meta["area_teks"] = [area.atas, area.bawah]

        bbox = meta.get("bbox")
        meta["di_dasar_halaman"] = area.di_dasar(bbox)
        meta["di_puncak_halaman"] = area.di_puncak(bbox)

        try:
            bk = batas_dari_halaman(pdf_path, el.get("page"), bbox)
            meta["batas_kolom"] = list(bk.batas) if bk.tersedia else None
            meta["batas_kolom_tersedia"] = bk.tersedia
        except Exception as e:      # pragma: no cover — jaring pengaman terakhir
            logger.warning("batas_kolom_gagal page=%s error=%s", el.get("page"), e)
            meta["batas_kolom"] = None
            meta["batas_kolom_tersedia"] = False
