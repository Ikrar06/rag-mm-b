"""Kriteria "baris ini header kolom sungguhan, bukan baris data".

Dipakai memutuskan baris mana yang boleh diulang ke potongan tabel lanjutan.
Salah memilih di sini MERUSAK: di run v3, baris data `921112 | BELANJA
PENGADAAN BAHAN MAKANAN` tersalin ke delapan chunk berbeda, sehingga pertanyaan
tentang kode akun itu cocok ke delapan tempat dan tujuh di antaranya salah.

Asimetri kerugian yang membentuk kriteria ini:

    salah terima (data jadi header) -> menyisipkan isi dokumen ke chunk lain,
                                        senyap, dan merusak metrik RCAA
    salah tolak  (header terlewat)  -> potongan B dibiarkan apa adanya, persis
                                        seperti sebelum fitur ini ada

Karena itu kriterianya KONSERVATIF: ragu berarti tolak.

Fungsi murni: tidak menyentuh berkas, tidak mengubah argumennya.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Sel yang isinya hanya angka, pemisah, mata uang, atau tanda kutip pembuka.
# Tanda kutip ikut karena OCR menghasilkan sel seperti "'00" untuk nomor urut.
_NUMERIK = re.compile(r"^['\"`]?\s*[\d.,%()\-\s/:]+$")
# Sel tanpa huruf maupun angka sama sekali — mis. "." dari OCR kolom kosong.
_TANPA_ALNUM = re.compile(r"^[^0-9A-Za-z]+$")
# Diawali tiga digit atau lebih: kode akun (921112), kode alokasi (4141).
_DIGIT_AWAL = re.compile(r"^['\"`]?\d{3,}")

# Proporsi sel tubuh yang harus numerik agar sebuah kolom dianggap "kontras".
AMBANG_KONTRAS = 0.5


def sel_numerik(s: str) -> bool:
    t = (s or "").strip()
    return bool(t) and bool(_NUMERIK.match(t)) and any(c.isdigit() for c in t)


def sel_tanpa_alnum(s: str) -> bool:
    t = (s or "").strip()
    return bool(t) and bool(_TANPA_ALNUM.match(t))


def sel_kode_panjang(s: str) -> bool:
    return bool(_DIGIT_AWAL.match((s or "").strip()))


def n_kolom_modal(baris) -> int:
    hitung: dict[int, int] = {}
    for b in baris:
        hitung[len(b)] = hitung.get(len(b), 0) + 1
    if not hitung:
        return 0
    return max(hitung.items(), key=lambda kv: (kv[1], -kv[0]))[0]


@dataclass(frozen=True)
class Putusan:
    """Hasil penilaian satu baris kandidat header."""

    header: bool
    aturan: str          # aturan yang MENENTUKAN putusan
    alasan: str = ""

    def __bool__(self) -> bool:
        return self.header


def deteksi_header(baris, ada_th: bool = False, ada_thead: bool = False,
                   pakai_kontras: bool = True) -> Putusan:
    """Apakah `baris[0]` header kolom sungguhan? `baris[1:]` tubuh tabel.

    `pakai_kontras=False` memberi Varian B — tanpa aturan (d). Disediakan
    supaya kedua varian dapat dibandingkan atas data yang sama, bukan supaya
    dipakai di produksi.
    """
    # Markup eksplisit bersifat otoritatif: kalau penghasil HTML menandai
    # barisnya sebagai header, tidak ada gunanya menebak ulang.
    if ada_th or ada_thead:
        return Putusan(True, "markup", "<th>/<thead> eksplisit")

    if not baris:
        return Putusan(False, "bentuk", "tabel kosong")
    h = list(baris[0])

    # (a) bentuk baris
    if len(h) < 2:
        return Putusan(False, "a-bentuk", f"hanya {len(h)} sel — bukan baris berkolom")
    if any(not str(s).strip() for s in h):
        return Putusan(False, "a-bentuk", "ada sel kosong")
    modal = n_kolom_modal(baris)
    if len(h) != modal:
        return Putusan(False, "a-bentuk", f"jumlah sel {len(h)} != kolom modal {modal}")

    # (b) dan (c) isi sel
    for i, s in enumerate(h):
        t = str(s).strip()
        if sel_numerik(t):
            return Putusan(False, "b-numerik", f"sel {i} {t!r} seluruhnya numerik")
        if sel_tanpa_alnum(t):
            return Putusan(False, "b-numerik", f"sel {i} {t!r} tanpa huruf/angka")
        if sel_kode_panjang(t):
            return Putusan(False, "c-kode", f"sel {i} {t!r} diawali >=3 digit")

    if not pakai_kontras:
        return Putusan(True, "b-numerik", "lolos saringan bentuk dan isi sel")

    # (d) kontras dengan tubuh tabel
    tubuh = [list(b) for b in baris[1:]]
    if not tubuh:
        return Putusan(False, "d-kontras", "tidak ada baris tubuh untuk dibandingkan")
    for j in range(len(h)):
        kolom = [str(b[j]) for b in tubuh if j < len(b) and str(b[j]).strip()]
        if not kolom:
            continue
        rasio = sum(sel_numerik(s) for s in kolom) / len(kolom)
        if rasio >= AMBANG_KONTRAS:
            return Putusan(True, "d-kontras",
                           f"kolom {j} kontras: header teks, {rasio:.0%} tubuh numerik")
    return Putusan(False, "d-kontras",
                   "tak satu pun kolom kontras — baris pertama sejenis dengan tubuh")
