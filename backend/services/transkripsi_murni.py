"""Fungsi murni transkripsi tabel oleh vision: prompt, pembersihan keluaran
model, penguraian Markdown, penanda pengecekan silang, geometri area render,
dan ukuran gambar yang aman bagi model.

Tidak memanggil model, tidak membuka PDF. Dipakai jalur indexing
(backend/services/table_transcription.py) DAN alat ukur di scripts/, supaya
yang diukur sebelum implementasi persis yang dijalankan saat indexing.
"""

from __future__ import annotations

import re
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

# Varian untuk gambar narasi+tabel: tangkapan layar atau halaman campuran bisa
# memuat LEBIH DARI SATU tabel (manual-dosen p35: dua panel). Prompt satu tabel
# akan membuang salah satunya.
PROMPT_TRANSKRIPSI_SEMUA = """Salin SETIAP tabel data pada gambar ini, masing-masing sebagai tabel
Markdown terpisah, dipisah satu baris kosong. Abaikan bagian gambar yang bukan
tabel data: tombol, menu, paragraf, panah, logo.
1. Salin teks dan angka PERSIS. Jangan menghitung, menjumlah, membulatkan,
   atau menambah nilai yang tidak terlihat.
2. Sel gabungan yang berlaku untuk beberapa baris: salin nilainya ke SETIAP baris.
3. Header bertingkat: gabungkan dari induk ke anak, dipisah spasi.
4. Hierarki butir (8, a., 1), -) dipertahankan di kolom labelnya.
5. Abaikan cap, logo, stempel, catatan tanda tangan elektronik.
6. Tabel TANPA baris judul kolom: tulis baris header dengan sel kosong.
   JANGAN mengarang nama kolom.
7. Sel kosong ditulis kosong. Karakter | di dalam sel ditulis \\|.
Keluaran HANYA tabel Markdown. Tanpa penjelasan, tanpa pagar kode."""


# Prompt koreksi: dipakai SEKALI bila jumlah sel header tidak sama dengan baris
# data. Terukur di uji 10 tabel: UKT p5 (header induk "UKT PER SEMESTER" jadi
# kolom sendiri, semua nilai bergeser satu kolom), jadwal-retensi p50 (AKTIF/
# INAKTIF tidak dipecah, sub-butir a. / 1) jadi sel sendiri), standar-biaya
# 2025 p9 (kode 1.7. dan huruf a-r jadi kolom sendiri). Templat; {x} dan {y}
# diisi per tabel. prompt_sha256 di cache dan manifest adalah sha TEMPLAT —
# nilai x/y ditentukan transkripsi pertama atas gambar yang sama.
PROMPT_KOREKSI_KOLOM = """Transkripsikan tabel pada gambar ini menjadi SATU tabel Markdown.
Transkripsi sebelumnya SALAH: baris header punya {x} sel, tetapi baris data punya {y} sel.
Kolom bergeser sehingga nilai jatuh di bawah judul kolom yang salah. Perbaiki:
A. Header bertingkat: JANGAN jadikan header induk kolom tersendiri. Gabungkan
   induk dengan SETIAP anaknya dalam satu sel, dipisah spasi. Contoh: induk
   "UKT PER SEMESTER" di atas "KELOMPOK I" dan "KELOMPOK II" menjadi dua sel
   "UKT PER SEMESTER KELOMPOK I" dan "UKT PER SEMESTER KELOMPOK II".
B. Hierarki butir (1.7., a, 1), -) ditulis di DALAM sel kolom uraiannya,
   di depan teksnya, BUKAN sebagai kolom tersendiri.
C. Setiap baris, termasuk header, harus punya jumlah sel yang SAMA. Sel tanpa
   isi ditulis kosong.
Aturan lain tetap berlaku:
1. Salin teks dan angka PERSIS. Jangan menghitung, menjumlah, membulatkan,
   atau menambah nilai yang tidak terlihat.
2. Sel gabungan yang berlaku untuk beberapa baris: salin nilainya ke SETIAP baris.
3. Abaikan cap, logo, stempel, catatan tanda tangan elektronik.
4. Potongan tabel TANPA baris judul kolom: baris header berisi sel kosong.
5. Karakter | di dalam sel ditulis \\|.
Keluaran HANYA tabel Markdown. Tanpa penjelasan, tanpa pagar kode."""


def prompt_koreksi_kolom(x: int, ys) -> str:
    return PROMPT_KOREKSI_KOLOM.format(x=x, y=" atau ".join(str(y) for y in ys))


def kolom_tidak_konsisten(baris) -> tuple[int, tuple[int, ...]] | None:
    """(sel header, jumlah sel baris data yang berbeda) bila tidak konsisten."""
    if not baris:
        return None
    x = len(baris[0])
    beda = sorted({len(b) for b in baris[1:] if len(b) != x})
    return (x, tuple(beda)) if beda else None


def baris_meleset(baris) -> int:
    """Jumlah baris data yang jumlah selnya tidak sama dengan header."""
    return sum(len(b) != len(baris[0]) for b in baris[1:]) if baris else 0


_PAGAR = re.compile(r"^\s*```[a-zA-Z]*\s*$")
_BARIS_PEMISAH = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_ANGKA = re.compile(r"\d[\d.,]*\d|\d")
# Lapisan OCR pada halaman pindai kerap memberi spasi antardigit: kode akun
# 426111 terbaca "4 2 6 1 1 1" (bagan-akun p10). Deret digit tunggal berspasi
# disatukan sebelum angka diekstrak.
_DIGIT_BERSPASI = re.compile(r"(?<![\d.,])\d(?: \d(?![\d.,])){2,}")
# Kata dengan tinggi > kelipatan ini dari median tinggi kata halaman diabaikan
# saat memperluas area render.
TINGGI_KATA_MAKS = 3.0
# Gambar yang menutup >= bagian ini dari halaman menandai halaman pindaian.
RASIO_GAMBAR_PINDAI = 0.9


def bersihkan(raw: str | None) -> str:
    """Ambil blok baris berpipa berurutan terpanjang; buang pagar dan teks luar."""
    blok = _blok_berpipa(raw)
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


def angka_sel(teks: str | None) -> frozenset[str]:
    """Angka >= 2 digit di SATU sel, tanpa menyatukan digit berspasi.

    Digit tunggal di sel berbeda tidak boleh tersambung: nomor kolom "1 | 2 | 3
    | 4 | 5 | 7" (jadwal-retensi p50) dan tanggal 1-9 (academic-calendar p10)
    pernah menjadi "123457" dan "123456789" karena sel digabung dulu baru
    digit tunggal berspasi disatukan.
    """
    hasil = set()
    for m in _ANGKA.findall(teks or ""):
        digit = re.sub(r"[.,]", "", m)
        if len(digit) >= 2:
            hasil.add(digit)
    return frozenset(hasil)


def ada_di_rujukan(angka: str, rujukan: str, rujukan_angka: frozenset[str]) -> bool:
    """Angka ada di rujukan, dengan spasi dan pemisah antar digit rujukan diabaikan.

    Lapisan OCR pindaian menulis kode 426111 sebagai "4 2 6 1 1 1" (bagan-akun
    p10) dan 28.111.676.194 sebagai "2 8 1 1 1 6 7 6 1 9 4" (laporan-keuangan
    p23). Yang dilonggarkan RUJUKANNYA, bukan transkripsinya.
    """
    if angka in rujukan_angka:
        return True
    pola = r"(?<![\d])" + r"[\s.,]*".join(angka) + r"(?![\d])"
    return re.search(pola, rujukan or "") is not None


def peringatan(baris, teks_rujukan: str) -> tuple[str, ...]:
    """Penanda pengecekan silang. Tidak pernah mengubah transkripsi.

    - angka_tak_ditemukan: angka sel transkripsi yang tidak ada di rujukan.
    - baris_berturut_identik: seluruh baris sama berturutan — tanda model
      berulang. (label_berturut_sama dibuang: terukur di standar biaya p9,
      53-54 kemunculan, semuanya sel gabungan yang sah disalin ke tiap baris.)
    - kolom_tidak_konsisten:x/y: jumlah sel header x, baris data y — nilai
      bisa berada di bawah judul kolom yang salah.
    """
    if not baris:
        return ()
    data = baris[1:]
    rujukan_angka = angka_sel(teks_rujukan)
    milik = set().union(*(angka_sel(s) for b in baris for s in b))
    tak_ada = sorted(n for n in milik if not ada_di_rujukan(n, teks_rujukan, rujukan_angka))
    hasil = []
    if tak_ada:
        hasil.append("angka_tak_ditemukan:" + ",".join(tak_ada[:10]))
    identik = [i for i in range(1, len(data)) if any(data[i]) and data[i] == data[i - 1]]
    if identik:
        hasil.append(f"baris_berturut_identik:{len(identik)}")
    kolom = kolom_tidak_konsisten(baris)
    if kolom:
        hasil.append(f"kolom_tidak_konsisten:{kolom[0]}/{','.join(map(str, kolom[1]))}")
    return tuple(hasil)


def _blok_berpipa(raw: str | None) -> list[list[str]]:
    blok: list[list[str]] = []
    kini: list[str] = []
    for baris in (raw or "").splitlines():
        if _PAGAR.match(baris):
            continue
        if baris.strip().startswith("|"):
            kini.append(baris.rstrip())
        elif kini:
            blok.append(kini)
            kini = []
    if kini:
        blok.append(kini)
    return blok


def bersihkan_semua(raw: str | None) -> list[str]:
    """Setiap blok berpipa yang merupakan tabel Markdown sah, urut kemunculan."""
    return [md for md in ("\n".join(b) for b in _blok_berpipa(raw)) if urai_markdown(md)]


def markdown_dari_baris(baris) -> str:
    """Susun ulang tabel Markdown dari baris sel (header di indeks 0)."""
    def satu(sel) -> str:
        return "| " + " | ".join(str(s).replace("|", "\\|") for s in sel) + " |"
    kepala, *data = baris
    return "\n".join([satu(kepala), "| " + " | ".join("---" for _ in kepala) + " |",
                      *(satu(b) for b in data)])


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


def median(nilai: list[float]) -> float | None:
    urut = sorted(nilai)
    if not urut:
        return None
    t = len(urut) // 2
    return urut[t] if len(urut) % 2 else (urut[t - 1] + urut[t]) / 2


def jenis_halaman(ada_kata: bool, rasio_gambar_terbesar: float) -> str:
    """Asal lapisan teks halaman.

    Halaman yang tertutup satu gambar >= 90% luasnya adalah pindaian; lapisan
    teksnya (bila ada) hasil OCR yang tak terlihat, bukan teks asli. Angka dari
    lapisan itu bukan rujukan kebenaran: di laporan-keuangan p23 lapisan OCR
    kehilangan baris JUMLAH EKUITAS yang dibaca model dengan benar.
    """
    if not ada_kata:
        return "pindai_tanpa_lapisan"
    return "pindai_lapisan_ocr" if rasio_gambar_terbesar >= RASIO_GAMBAR_PINDAI else "digital_asli"


def perluas_bbox(bbox, kata_bbox, margin: float = 0.0, milik_lain=()) -> list[float]:
    """Perluas bbox ke kata lapisan teks yang BERIRISAN dengannya, lalu margin.

    Kata yang terpotong tepi bbox ditarik utuh: di ukt p5 kolom KELOMPOK VIII
    (x 0,92-0,97) terpotong di 0,95. Hanya kata yang beririsan, bukan paragraf
    di sekitarnya. Koordinat ternormalisasi, dijepit ke [0, 1].
    """
    x0, y0, x1, y1 = bbox
    tinggi = median([k[3] - k[1] for k in kata_bbox])
    # Kata yang titik tengahnya di dalam bbox elemen LAIN (tabel lain, chunk
    # teks, gambar) sudah dimiliki elemen itu. Menariknya menggandakan isi:
    # bagan-akun p6, baris kode di bawah tabel tersimpan sebagai chunk teks.
    lain = [b for b in milik_lain if b and len(b) == 4]

    def dimiliki(k) -> bool:
        cx, cy = (k[0] + k[2]) / 2, (k[1] + k[3]) / 2
        return any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in lain)

    for k in kata_bbox:
        # Kata jauh lebih tinggi dari kata biasa bukan isi sel: teks vertikal
        # raksasa ("February" di academic-calendar p10, menarik 34% halaman)
        # atau kotak sampah lapisan OCR (bagan-akun p6).
        if tinggi and k[3] - k[1] > TINGGI_KATA_MAKS * tinggi:
            continue
        if k[0] < x1 and k[2] > x0 and k[1] < y1 and k[3] > y0 and not dimiliki(k):
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


class AreaTidakSah(ValueError):
    """Area render yang akan menghasilkan pixmap nol piksel."""


# Sisi minimum area render, dalam poin PDF. Di bawah ini pixmap bisa 0 piksel
# dan PyMuPDF gagal saat menulis PNG ("Invalid bandwriter header dimensions") —
# galat yang menjatuhkan UKT dan Pedoman Tesis di reindex v5 pertama.
SISI_KLIP_MIN_PT = 1.0


def klip_aman(bbox, lebar_hal: float, tinggi_hal: float) -> tuple[float, float, float, float]:
    """bbox ternormalisasi -> klip dalam poin, dijepit ke halaman. AreaTidakSah bila
    bbox bukan 4 angka hingga, terbalik, di luar halaman, atau lebih sempit dari
    SISI_KLIP_MIN_PT setelah dijepit."""
    import math
    if not bbox or len(bbox) != 4:
        raise AreaTidakSah(f"bbox bukan 4 angka: {bbox!r}")
    try:
        x0, y0, x1, y1 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        raise AreaTidakSah(f"bbox bukan angka: {bbox!r}") from None
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
        raise AreaTidakSah(f"bbox tidak hingga: {bbox!r}")
    if x1 <= x0 or y1 <= y0:
        raise AreaTidakSah(f"bbox terbalik atau nol: {bbox!r}")
    kx0, ky0 = max(0.0, x0) * lebar_hal, max(0.0, y0) * tinggi_hal
    kx1, ky1 = min(1.0, x1) * lebar_hal, min(1.0, y1) * tinggi_hal
    if kx1 - kx0 < SISI_KLIP_MIN_PT or ky1 - ky0 < SISI_KLIP_MIN_PT:
        raise AreaTidakSah(f"area setelah dijepit ke halaman {kx1 - kx0:.2f}x{ky1 - ky0:.2f} pt: {bbox!r}")
    return kx0, ky0, kx1, ky1

