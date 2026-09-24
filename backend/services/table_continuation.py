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

# Nilai kolom `keputusan` yang dikenali. Hanya "terima" yang memicu penggabungan.
KEPUTUSAN_TERIMA = "terima"
KEPUTUSAN_TOLAK = "tolak"
# Kosong = belum ditinjau. Dilewati, dan itu BUKAN kelalaian yang perlu
# diperingatkan — berkas memang lahir dengan kolom ini kosong.
KEPUTUSAN_DIKENALI = frozenset({KEPUTUSAN_TERIMA, KEPUTUSAN_TOLAK, ""})


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


# ─── Berkas keputusan terkurasi ──────────────────────────────────────────────

_PEMISAH_RE = None      # diisi saat pertama dipakai, lihat baris_header_markdown
_keputusan_cache: frozenset | None = None


def html_sha(html) -> str:
    """Sidik isi tabel dari `raw_html`, tahan terhadap perbedaan spasi.

    Dipakai sebagai PENJAGA di samping kunci chunk_id. chunk_id adalah posisi,
    dan posisi bergeser: perbaikan current_page memindahkan chunk teks ke
    halaman yang benar, sehingga pencacah per halaman bergeser dan tabel yang
    sama berganti nama (rubrik_p28_c00 di v2 menjadi rubrik_p28_c01 di v3,
    sementara rubrik_p28_c00 di v3 kini chunk TEKS). Kunci yang cocok tapi
    sidiknya tidak berarti berkas keputusan menunjuk tabel lain.
    """
    import hashlib
    norma = " ".join(str(html or "").split())
    return hashlib.sha256(norma.encode("utf-8")).hexdigest()[:16] if norma else ""


def kunci_pasangan(chunk_id_a: str, chunk_id_b: str) -> str:
    """Kunci stabil sebuah pasangan potongan.

    Memakai chunk_id, yang TIDAK bergeser saat header diulang — mengulang header
    tidak menambah atau mengurangi chunk. Berkas keputusan karenanya tetap sahih
    setelah re-index.
    """
    return f"{chunk_id_a}__{chunk_id_b}"


def baris_header_markdown(teks: str, maks_baris: int = 3) -> str:
    """Ambil baris header dari teks tabel Markdown.

    Teks chunk sudah berupa Markdown (`_html_table_to_markdown`) dan bisa
    didahului prefiks `## {section}`, jadi pencarian dimulai dari baris pertama
    yang diawali pipa. Header = baris pertama plus baris pemisah `| --- |`.

    Mengembalikan string kosong bila tidak ada bentuk tabel Markdown yang
    dikenali — pemanggil memperlakukannya sebagai "tidak ada header untuk
    diulang" dan membiarkan potongan apa adanya.
    """
    import re as _re

    baris = (teks or "").splitlines()
    mulai = next((i for i, b in enumerate(baris) if b.lstrip().startswith("|")), None)
    if mulai is None or mulai + 1 >= len(baris):
        return ""
    pemisah = baris[mulai + 1].strip()
    if not _re.fullmatch(r"\|[\s:|-]+\|", pemisah):
        return ""
    return "\n".join(baris[mulai:mulai + min(maks_baris, 2)])


def muat_keputusan(path=None) -> frozenset[str]:
    """Kunci pasangan yang DISETUJUI manusia untuk digabung.

    Hanya entri dengan `keputusan == "terima"` yang dikembalikan. Entri bernilai
    lain — termasuk yang ditolak otomatis karena headernya dari OCR — sengaja
    tetap ada di berkas supaya terlihat saat ditinjau, tapi tidak diproses.

    Berkas tidak ada berarti himpunan kosong, bukan galat: menyalakan flag tanpa
    berkas keputusan menghasilkan perilaku identik dengan flag mati, dan itu
    aman. Ketiadaannya dicatat sebagai peringatan.
    """
    global _keputusan_cache
    if _keputusan_cache is not None:
        return _keputusan_cache

    from backend.config import TABLE_CONTINUATION_PATH
    import json
    from pathlib import Path

    p = Path(os.path.expanduser(str(path or TABLE_CONTINUATION_PATH)))
    if not p.exists():
        logger.warning(
            "table_continuation_tidak_ada path=%s — tidak ada pasangan yang "
            "digabung. Buat dengan scripts/analisis_tabel_lintas_halaman.py, "
            "lalu tinjau kolom 'keputusan'.", p,
        )
        _keputusan_cache = frozenset()
        return _keputusan_cache

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        pasangan = (raw or {}).get("pasangan") if isinstance(raw, dict) else None
        if not isinstance(pasangan, dict):
            raise ValueError(
                f"kunci 'pasangan' harus object, bukan {type(pasangan).__name__}"
            )
    except (json.JSONDecodeError, OSError, ValueError) as e:
        logger.error(
            "table_continuation_rusak path=%s error=%s — diperlakukan kosong "
            "supaya indexing tidak menggabung berdasarkan berkas yang tidak "
            "dapat dibaca", p, e,
        )
        _keputusan_cache = frozenset()
        return _keputusan_cache

    disetujui: set[str] = set()
    tak_dikenali: dict[str, str] = {}
    belum_ditinjau = 0
    for k, v in pasangan.items():
        if not isinstance(v, dict):
            continue
        nilai = str(v.get("keputusan", "")).strip().lower()
        if nilai == KEPUTUSAN_TERIMA:
            disetujui.add(k)
        elif nilai == "":
            belum_ditinjau += 1
        elif nilai not in KEPUTUSAN_DIKENALI:
            tak_dikenali[k] = nilai

    # Nilai tak dikenali DILEWATI — sikap aman, karena menebak bahwa "ok"
    # berarti terima akan menggabung tabel atas dasar tebakan. Tapi melewatinya
    # DIAM-DIAM membuang niat peninjau tanpa jejak, jadi selalu diperingatkan.
    if tak_dikenali:
        contoh = ", ".join(f"{k}={v!r}" for k, v in list(tak_dikenali.items())[:5])
        logger.warning(
            "table_continuation_keputusan_tak_dikenali jumlah=%d contoh=%s — "
            "DILEWATI. Nilai yang dikenali hanya %r dan %r; isi ulang entri ini "
            "bila memang dimaksudkan diterima.",
            len(tak_dikenali), contoh, KEPUTUSAN_TERIMA, KEPUTUSAN_TOLAK,
        )
    if belum_ditinjau:
        logger.info(
            "table_continuation_belum_ditinjau jumlah=%d — dilewati, bukan diterima",
            belum_ditinjau,
        )
    disetujui = frozenset(disetujui)
    logger.info("table_continuation_dimuat path=%s total=%d disetujui=%d "
                "belum_ditinjau=%d tak_dikenali=%d",
                p, len(pasangan), len(disetujui), belum_ditinjau, len(tak_dikenali))
    _keputusan_cache = disetujui
    return _keputusan_cache


_sidik_cache: dict | None = None


def sidik_keputusan(path=None) -> dict[str, tuple[str, str]]:
    """Kunci pasangan disetujui -> (html_sha_a, html_sha_b) dari berkas.

    String kosong berarti entri itu tidak membawa sidik — berkas lama yang
    belum dimigrasi. Pemanggil memperlakukannya sebagai TIDAK dapat diverifikasi.
    """
    global _sidik_cache
    if _sidik_cache is not None:
        return _sidik_cache
    from backend.config import TABLE_CONTINUATION_PATH
    import json
    from pathlib import Path

    p = Path(os.path.expanduser(str(path or TABLE_CONTINUATION_PATH)))
    hasil: dict[str, tuple[str, str]] = {}
    try:
        pasangan = (json.loads(p.read_text(encoding="utf-8")) or {}).get("pasangan") or {}
    except (OSError, ValueError, AttributeError):
        pasangan = {}
    for k in muat_keputusan(path):
        v = pasangan.get(k) or {}
        m = v.get("_migrasi") or {}
        hasil[k] = (str(v.get("html_sha_a") or m.get("html_sha_a") or ""),
                    str(v.get("html_sha_b") or m.get("html_sha_b") or ""))
    _sidik_cache = hasil
    return hasil


def reload_keputusan() -> None:
    """Buang cache keputusan. Dipakai uji dan setelah berkas disunting."""
    global _keputusan_cache, _sidik_cache
    _keputusan_cache = None
    _sidik_cache = None


# ─── Putusan per potongan tabel ──────────────────────────────────────────────
#
# Dipisah dari _chunk_elements supaya dapat diuji tanpa pipeline. Tiga hal yang
# membentuknya, masing-masing dari kegagalan terukur di run v3:
#
# 1. Header diambil dari HTML dan HARUS lolos header_tabel.deteksi_header.
#    Fallback lama "pakai baris pertama" menyalin baris data 921112 ke 8 chunk.
# 2. Header rantai dibawa EKSPLISIT dari potongan terurai pertama, bukan
#    diturunkan ulang dari teks gabungan potongan sebelumnya.
# 3. Kunci chunk_id yang cocok belum cukup: sidik html kedua sisi harus cocok
#    juga. chunk_id tabel bergeser antara v2 dan v3, sehingga kunci lama bisa
#    menunjuk tabel lain.

from dataclasses import dataclass as _dataclass


@_dataclass(frozen=True)
class TabelRantai:
    """Yang dibawa dari satu tabel ke tabel berikutnya dalam dokumen."""

    chunk_id: str
    sidik: str
    keadaan: object          # header_tabel.KeadaanRantai
    group_id: str
    part: int


@_dataclass(frozen=True)
class PutusanPotongan:
    header_md: str           # kosong = tidak ada yang diulang
    group_id: str
    part: int
    lanjutan: bool
    alasan: str
    rantai: TabelRantai


def header_markdown(sel) -> str:
    """Baris header tabel Markdown yang sah dan berdiri sendiri."""
    isi = [str(s).strip().replace("|", "\\|") for s in sel]
    return "| " + " | ".join(isi) + " |\n| " + " | ".join("---" for _ in isi) + " |"


def sisipkan_header(prefiks: str, header_md: str, teks_tabel: str) -> str:
    """Header masuk SETELAH prefiks `## {section}`, bukan sebelumnya.

    Versi lama menempelkan header di depan teks yang sudah berprefiks, sehingga
    `## 6. Revenues` terdorong ke tengah teks chunk.
    """
    if not header_md:
        return prefiks + teks_tabel
    return f"{prefiks}{header_md}\n{teks_tabel}"


def putuskan_potongan(sebelumnya: "TabelRantai | None", calon_id: str, raw_html,
                      disetujui: frozenset, sidik: dict,
                      maks_panjang_sel: int | None = None) -> PutusanPotongan:
    from backend.services.header_tabel import (
        STATUS_DIKETAHUI, deteksi_header, lanjutkan_rantai,
    )
    from backend.services.tabel_html import urai

    tabel = urai(raw_html)
    putusan = (deteksi_header(tabel.baris, tabel.ada_th, tabel.ada_thead,
                              maks_panjang_sel=maks_panjang_sel)
               if tabel is not None else None)
    sid = html_sha(raw_html)

    lanjutan, alasan = False, ""
    if sebelumnya is not None:
        kunci = kunci_pasangan(sebelumnya.chunk_id, calon_id)
        if kunci in disetujui:
            sa, sb = sidik.get(kunci, ("", ""))
            if not (sa and sb):
                alasan = ("kunci disetujui tapi berkas keputusan tanpa sidik html — "
                          "tidak dapat diverifikasi; jalankan scripts/migrasi_keputusan.py")
            elif sa != sebelumnya.sidik or sb != sid:
                alasan = "kunci cocok tapi sidik html berbeda — kunci menunjuk tabel lain"
            else:
                lanjutan = True

    if lanjutan:
        k_prev = sebelumnya.keadaan
        header_md = ""
        if k_prev.status == STATUS_DIKETAHUI:
            baris_b = [c.strip() for c in tabel.baris[0]] if tabel and tabel.baris else None
            if baris_b != [c.strip() for c in k_prev.header]:
                header_md = header_markdown(k_prev.header)
            else:
                alasan = "potongan B sudah diawali header rantai — tidak digandakan"
        elif k_prev.status != STATUS_DIKETAHUI:
            alasan = f"rantai tanpa header sah ({k_prev.status})"
        keadaan = lanjutkan_rantai(k_prev, calon_id, tabel, putusan)
        group, part = sebelumnya.group_id, sebelumnya.part + 1
    else:
        header_md = ""
        keadaan = lanjutkan_rantai(None, calon_id, tabel, putusan)
        group, part = calon_id, 0

    return PutusanPotongan(
        header_md=header_md, group_id=group, part=part, lanjutan=lanjutan,
        alasan=alasan,
        rantai=TabelRantai(calon_id, sid, keadaan, group, part),
    )
