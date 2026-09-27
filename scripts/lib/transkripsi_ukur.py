"""Fungsi murni untuk alat ukur transkripsi tabel (scripts/ukur_transkripsi.py).

Tidak memanggil model, tidak membuka PDF. Semua yang dapat diuji tanpa server
ada di sini: pembersihan keluaran model, penguraian tabel Markdown, penanda
pengecekan silang, sebaran tumpang tindih gambar-tabel, dan proyeksi waktu.

Prompt di sini DRAF untuk pengukuran. Setelah disetujui, prompt produksi hidup
di backend dan prompt_sha256-nya masuk manifest.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

PROMPT_TRANSKRIPSI = """Transkripsikan tabel pada gambar ini menjadi SATU tabel Markdown.
1. Salin teks dan angka PERSIS. Jangan menghitung, menjumlah, membulatkan,
   atau menambah nilai yang tidak terlihat.
2. Sel gabungan yang berlaku untuk beberapa baris: salin nilainya ke SETIAP baris.
3. Header bertingkat: gabungkan dari induk ke anak, dipisah spasi.
   Contoh: "JANGKA WAKTU PENYIMPANAN AKTIF".
4. Hierarki butir (8, a., 1), -) dipertahankan di kolom labelnya.
5. Abaikan cap, logo, stempel, catatan tanda tangan elektronik.
6. Jika gambar ini potongan tabel TANPA baris judul kolom, tulis baris header
   dengan sel kosong. JANGAN mengarang nama kolom.
7. Sel kosong ditulis kosong. Karakter | di dalam sel ditulis \\|.
Keluaran HANYA tabel Markdown. Tanpa penjelasan, tanpa pagar kode."""

# Tiga jenis dalam satu panggilan. Condong ke tabel antara tabel/lainnya:
# salah ke arah tabel hanya menambah satu transkripsi (lalu jatuh ke deskripsi
# bila tak terurai). "cap" didahulukan karena potongan cap BSrE di UKT memuat
# baris tabel di belakangnya dan tanpa aturan ini terjawab "tabel".
PROMPT_KLASIFIKASI = """Gambar ini diambil dari dokumen PDF. Tentukan jenisnya:
- "cap": cap atau segel tanda tangan elektronik (misalnya logo Balai Sertifikasi
  Elektronik / BSrE) atau catatan "dokumen ini telah ditandatangani secara
  elektronik". Jawab cap bila unsur itu terlihat, walaupun ada potongan tabel
  di belakangnya.
- "tabel": gambar yang HAMPIR SELURUHNYA berupa tabel data (baris dan kolom),
  termasuk tabel hasil pindai.
- "lainnya": selain itu, termasuk tangkapan layar aplikasi, halaman berisi
  paragraf dan tabel sekaligus, diagram, bagan alir, foto, dan logo lembaga.
Jika ragu antara tabel dan lainnya, jawab tabel.
Jawab HANYA dengan JSON satu baris: {"jenis": "tabel"}, {"jenis": "cap"}, atau {"jenis": "lainnya"}"""

JENIS_GAMBAR = ("tabel", "cap", "lainnya")
_JENIS_RE = re.compile(r'"jenis"\s*:\s*"([a-z]+)"')

_PAGAR = re.compile(r"^\s*```[a-zA-Z]*\s*$")
_BARIS_PEMISAH = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_ANGKA = re.compile(r"\d[\d.,]*\d|\d")
# Lapisan OCR pada halaman pindai kerap memberi spasi antardigit: kode akun
# 426111 terbaca "4 2 6 1 1 1" (bagan-akun p10). Deret digit tunggal berspasi
# disatukan sebelum angka diekstrak.
_DIGIT_BERSPASI = re.compile(r"(?<![\d.,])\d(?: \d(?![\d.,])){2,}")
_ORDINAL = re.compile(r"_c(\d+)$")


def bersihkan(raw: str | None) -> str:
    """Ambil blok baris berpipa berurutan terpanjang; buang pagar dan teks luar."""
    if not raw:
        return ""
    blok: list[list[str]] = []
    kini: list[str] = []
    for baris in raw.splitlines():
        if _PAGAR.match(baris):
            continue
        if baris.strip().startswith("|"):
            kini.append(baris.rstrip())
        elif kini:
            blok.append(kini)
            kini = []
    if kini:
        blok.append(kini)
    return "\n".join(max(blok, key=len)) if blok else ""


def _pecah_sel(baris: str) -> list[str]:
    isi = baris.strip()
    isi = isi[1:] if isi.startswith("|") else isi
    isi = isi[:-1] if isi.endswith("|") and not isi.endswith("\\|") else isi
    sel = re.split(r"(?<!\\)\|", isi)
    return [s.strip().replace("\\|", "|") for s in sel]


def urai_markdown(md: str) -> tuple[tuple[str, ...], ...] | None:
    """Tabel Markdown -> baris sel (header di indeks 0). None bila bukan tabel.

    Sah bila ada baris pemisah tepat setelah header, minimal 2 kolom dan
    minimal satu baris data.
    """
    baris = [b for b in (md or "").splitlines() if b.strip()]
    if len(baris) < 3 or not _BARIS_PEMISAH.match(baris[1]):
        return None
    hasil = tuple(tuple(_pecah_sel(b)) for b in [baris[0], *baris[2:]])
    if len(hasil[0]) < 2:
        return None
    return hasil


def angka_dalam(teks: str | None) -> frozenset[str]:
    """Angka >= 2 digit, pemisah ribuan/desimal dibuang (1.500.000 -> 1500000)."""
    hasil = set()
    teks = _DIGIT_BERSPASI.sub(lambda m: m.group(0).replace(" ", ""), teks or "")
    for m in _ANGKA.findall(teks):
        digit = re.sub(r"[.,]", "", m)
        if len(digit) >= 2:
            hasil.add(digit)
    return frozenset(hasil)


def peringatan(baris, teks_rujukan: str) -> tuple[str, ...]:
    """Penanda pengecekan silang. Tidak pernah mengubah transkripsi.

    - angka_tak_ditemukan: angka transkripsi yang tidak ada di teks rujukan.
    - baris_berturut_identik: seluruh baris sama berturutan — tanda model
      berulang. (label_berturut_sama dibuang: terukur di standar biaya p9,
      53-54 kemunculan, semuanya sel gabungan yang sah disalin ke tiap baris.)
    """
    if not baris:
        return ()
    data = baris[1:]
    rujukan = angka_dalam(teks_rujukan)
    tak_ada = sorted(angka_dalam(" ".join(" ".join(b) for b in baris)) - rujukan)
    hasil = []
    if tak_ada:
        hasil.append("angka_tak_ditemukan:" + ",".join(tak_ada[:10]))
    identik = [i for i in range(1, len(data)) if any(data[i]) and data[i] == data[i - 1]]
    if identik:
        hasil.append(f"baris_berturut_identik:{len(identik)}")
    return tuple(hasil)


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


def rasio_tumpang(gambar, tabel) -> float:
    """Luas irisan dibagi luas GAMBAR. 1.0 = gambar seluruhnya di dalam tabel."""
    if not gambar or not tabel or len(gambar) != 4 or len(tabel) != 4:
        return 0.0
    luas = max(0.0, gambar[2] - gambar[0]) * max(0.0, gambar[3] - gambar[1])
    if luas <= 0:
        return 0.0
    lebar = min(gambar[2], tabel[2]) - max(gambar[0], tabel[0])
    tinggi = min(gambar[3], tabel[3]) - max(gambar[1], tabel[1])
    return max(0.0, lebar) * max(0.0, tinggi) / luas


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


def median(nilai: list[float]) -> float | None:
    urut = sorted(nilai)
    if not urut:
        return None
    t = len(urut) // 2
    return urut[t] if len(urut) % 2 else (urut[t - 1] + urut[t]) / 2


def urai_klasifikasi(raw: str | None) -> str | None:
    """Jawaban klasifikasi -> "tabel" | "cap" | "lainnya". None bila tak dikenali."""
    m = _JENIS_RE.search((raw or "").lower())
    return m.group(1) if m and m.group(1) in JENIS_GAMBAR else None


def jenis_halaman(ada_kata: bool, rasio_gambar_terbesar: float) -> str:
    """Asal lapisan teks halaman.

    Halaman yang tertutup satu gambar >= 90% luasnya adalah pindaian; lapisan
    teksnya (bila ada) hasil OCR yang tak terlihat, bukan teks asli. Angka dari
    lapisan itu bukan rujukan kebenaran: di laporan-keuangan p23 lapisan OCR
    kehilangan baris JUMLAH EKUITAS yang dibaca model dengan benar.
    """
    if not ada_kata:
        return "pindai_tanpa_lapisan"
    return "pindai_lapisan_ocr" if rasio_gambar_terbesar >= 0.9 else "digital_asli"


def perluas_bbox(bbox, kata_bbox, margin: float = 0.0) -> list[float]:
    """Perluas bbox ke kata lapisan teks yang BERIRISAN dengannya, lalu margin.

    Kata yang terpotong tepi bbox ditarik utuh: di ukt p5 kolom KELOMPOK VIII
    (x 0,92-0,97) terpotong di 0,95. Hanya kata yang beririsan, bukan paragraf
    di sekitarnya. Koordinat ternormalisasi, dijepit ke [0, 1].
    """
    x0, y0, x1, y1 = bbox
    for k in kata_bbox:
        if k[0] < x1 and k[2] > x0 and k[1] < y1 and k[3] > y0:
            x0, y0, x1, y1 = min(x0, k[0]), min(y0, k[1]), max(x1, k[2]), max(y1, k[3])
    return [max(0.0, x0 - margin), max(0.0, y0 - margin),
            min(1.0, x1 + margin), min(1.0, y1 + margin)]


@dataclass(frozen=True)
class RencanaUkuran:
    """Ukuran gambar setelah diperkecil (lebar, tinggi) dan kanvas setelah padding."""
    lebar: int
    tinggi: int
    kanvas_lebar: int
    kanvas_tinggi: int

    @property
    def dipadding(self) -> bool:
        return (self.kanvas_lebar, self.kanvas_tinggi) != (self.lebar, self.tinggi)


def rencana_ukuran(lebar: int, tinggi: int, sisi_maks: int | None,
                   sisi_min: int, rasio_maks: int) -> RencanaUkuran:
    """Ukuran aman untuk image processor Qwen-VL di Ollama.

    SmartResize Ollama panic (HTTP 500) bila sisi < patch_size * merge_size
    atau max(sisi) // min(sisi) > 200. Aturannya:
    - pengecilan ke sisi_maks tidak boleh membuat sisi pendek < sisi_min;
      skala berhenti di situ (manual_p23_c03 2087x118 -> 512x29 memicu 500);
    - gambar tidak pernah diperbesar; sisi yang masih kurang ditambal putih,
      juga untuk memenuhi rasio_maks. Tidak ada peregangan.
    """
    panjang, pendek = max(lebar, tinggi), min(lebar, tinggi)
    skala = 1.0
    if sisi_maks and panjang > sisi_maks:
        skala = min(1.0, max(sisi_maks / panjang, sisi_min / max(pendek, 1)))
    w, h = max(1, round(lebar * skala)), max(1, round(tinggi * skala))
    minimum = max(sisi_min, -(-max(w, h) // rasio_maks))
    return RencanaUkuran(w, h, max(w, minimum), max(h, minimum))
