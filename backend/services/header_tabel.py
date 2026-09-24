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


def rasio_terisi(sel) -> float:
    sel = list(sel)
    return sum(1 for s in sel if str(s).strip()) / len(sel) if sel else 0.0


def panjang_sel_terpanjang(sel) -> int:
    return max((len(str(s).strip()) for s in sel), default=0)


def deteksi_header(baris, ada_th: bool = False, ada_thead: bool = False,
                   pakai_kontras: bool = True,
                   modal_untuk_markup: bool = False,
                   maks_panjang_sel: int | None = None) -> Putusan:
    """Apakah `baris[0]` header kolom sungguhan? `baris[1:]` tubuh tabel.

    Markup `<th>`/`<thead>` BUKAN otoritatif. Terukur di korpus: table
    transformer menandai baris pertama sebagai `<th>` apa pun isinya — kode akun
    5341, nomor urut 78, baris kosong, semuanya lolos sebagai "header" bila
    markup dipercaya begitu saja.

    Urutan aturan, dan kepada siapa berlaku:

        a1  >= 2 sel dan LEBIH dari separuh terisi     markup & non-markup
        b   tak ada sel terisi yang numerik/tanpa alnum markup & non-markup
        c   tak ada sel diawali >= 3 digit             markup & non-markup
        a2  jumlah sel = kolom modal tubuh             non-markup (markup: opsional)
        d   ada kolom kontras dengan tubuh             non-markup saja

    (d) sengaja TIDAK diterapkan ke markup: header jadwal retensi arsip
    ("SERIES/JENIS ARSIP | AKTIF | INAKTIF | KETERANGAN") bertubuh teks, dan
    (d) akan membuang kedelapannya. Markup tetap punya nilai sebagai sinyal —
    ia menggantikan (d), bukan menggantikan (a1)-(c).

    `pakai_kontras=False` memberi Varian B. `modal_untuk_markup=True` menguji
    aturan (a2) pada markup; disediakan untuk DIUKUR, bukan dinyalakan.

    `maks_panjang_sel` (aturan e) menolak baris yang punya sel lebih panjang
    dari batas itu — judul kolom pendek, isi sel tubuh bisa berupa kalimat.
    None berarti mati. Batasnya TIDAK dikarang di sini: ia harus diambil dari
    celah terukur antara header sungguhan dan baris data yang bocor.
    """
    markup = bool(ada_th or ada_thead)
    jalur = "markup" if markup else "fallback"
    if not baris:
        return Putusan(False, "bentuk", "tabel kosong")
    h = [str(s) for s in baris[0]]

    # (a1) bentuk dan keterisian
    if len(h) < 2:
        return Putusan(False, f"a-bentuk/{jalur}", f"hanya {len(h)} sel — bukan baris berkolom")
    terisi = [s.strip() for s in h if s.strip()]
    if len(terisi) * 2 <= len(h):
        return Putusan(False, f"a-terisi/{jalur}",
                       f"hanya {len(terisi)} dari {len(h)} sel terisi — tidak lebih dari separuh")

    # (b) dan (c) isi sel terisi
    for i, s in enumerate(h):
        t = s.strip()
        if not t:
            continue
        if sel_numerik(t):
            return Putusan(False, f"b-numerik/{jalur}", f"sel {i} {t!r} seluruhnya numerik")
        if sel_tanpa_alnum(t):
            return Putusan(False, f"b-numerik/{jalur}", f"sel {i} {t!r} tanpa huruf/angka")
        if sel_kode_panjang(t):
            return Putusan(False, f"c-kode/{jalur}", f"sel {i} {t!r} diawali >=3 digit")

    # (e) panjang sel — hanya bila batasnya sudah diukur
    if maks_panjang_sel:
        terpanjang = panjang_sel_terpanjang(h)
        if terpanjang > maks_panjang_sel:
            return Putusan(False, f"e-panjang/{jalur}",
                           f"sel terpanjang {terpanjang} karakter > batas {maks_panjang_sel}")

    modal = n_kolom_modal(baris)
    if markup:
        if modal_untuk_markup and len(baris) > 1 and len(h) != modal:
            return Putusan(False, "a-modal/markup", f"jumlah sel {len(h)} != kolom modal {modal}")
        return Putusan(True, "markup", "<th>/<thead> dan lolos (a1)-(c)")

    # (a2) jumlah sel cocok dengan tubuh
    if len(h) != modal:
        return Putusan(False, "a-bentuk/fallback", f"jumlah sel {len(h)} != kolom modal {modal}")
    if not pakai_kontras:
        return Putusan(True, "b-numerik/fallback", "lolos saringan bentuk dan isi sel")

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


# ─── Keadaan rantai ──────────────────────────────────────────────────────────
#
# Tiga keadaan, bukan dua. "Kepala tidak terurai" berarti TIDAK DIKETAHUI, bukan
# terbukti salah: HTML-nya kosong atau rusak, sehingga tidak ada bukti bahwa
# baris pertamanya data. "Kepala ditolak" berarti TERBUKTI — baris pertamanya
# kode akun, nomor urut, atau baris kosong.

# Aturan yang penolakannya MEMBUKTIKAN baris itu data. Penolakan lain — satu sel
# (a-bentuk), mayoritas kosong (a-terisi), jumlah sel tak cocok modal — hanya
# membuktikan baris itu BUKAN header, bukan bahwa ia data: di rubrik, "Tahap 1:
# Pembinaan dan Penyusunan Usulan Konsep Desain" adalah judul tahap satu sel di
# dalam tabel aktivitas, dan mematikan rantai karenanya membuang header sah
# "Aktivitas/Subaktivitas | Volume | Menit | Total | Bobot" di potongan berikutnya.
ATURAN_BUKTI_DATA = frozenset({"b-numerik", "c-kode", "d-kontras", "e-panjang"})


def membuktikan_data(putusan) -> bool:
    """True bila penolakan `putusan` membuktikan baris pertamanya baris data."""
    return (putusan is not None and not putusan.header
            and putusan.aturan.split("/")[0] in ATURAN_BUKTI_DATA)


STATUS_TIDAK_DIKETAHUI = "tidak_diketahui"
STATUS_DIKETAHUI = "diketahui"
STATUS_MATI = "mati"


@dataclass(frozen=True)
class KeadaanRantai:
    status: str
    header: tuple[str, ...] | None = None
    sumber: str | None = None       # chunk_id yang menetapkan keadaan ini


def lanjutkan_rantai(keadaan, chunk_id: str, tabel, putusan) -> KeadaanRantai:
    """Keadaan rantai SETELAH potongan `chunk_id` dinilai.

    `keadaan` None berarti potongan ini kepala rantai baru. Header yang diulang
    ke potongan BERIKUTNYA adalah `header` dari keadaan yang dikembalikan.

    Aturannya: potongan pertama di rantai yang MEMBERI BUKTI yang memutuskan.
    Lolos kriteria -> headernya jadi header rantai. Ditolak dengan bukti data
    (ATURAN_BUKTI_DATA) -> rantai mati dan tidak mewarisi apa pun sampai ujung.
    Potongan tak terurai, atau yang ditolak tanpa bukti data (satu sel,
    mayoritas kosong), bersifat netral: menyerahkan keputusan ke potongan
    berikutnya.

    Sengaja BUKAN "header sah pertama di mana pun dalam rantai": kalau potongan
    terurai pertama ternyata baris data, header asli tabel itu ada di kepala
    yang tak terurai, dan baris yang tampak sah di tengah rantai lebih mungkin
    kebetulan lolos daripada header sungguhan.
    """
    if keadaan is not None and keadaan.status != STATUS_TIDAK_DIKETAHUI:
        return keadaan
    if tabel is None or putusan is None:
        return keadaan or KeadaanRantai(STATUS_TIDAK_DIKETAHUI)
    if putusan.header:
        return KeadaanRantai(STATUS_DIKETAHUI, tuple(tabel.baris[0]), chunk_id)
    if membuktikan_data(putusan):
        return KeadaanRantai(STATUS_MATI, None, chunk_id)
    return keadaan or KeadaanRantai(STATUS_TIDAK_DIKETAHUI)


def kepala_rantai(pasangan) -> dict[str, str]:
    """Peta chunk_id -> chunk_id kepala rantainya.

    `pasangan` adalah iterable (a, b) yang disetujui. Rantai A->B->C terbentuk
    bila B dari satu pasangan adalah A pasangan berikutnya. Header yang diulang
    ke seluruh rantai diambil dari KEPALA, bukan dari potongan tepat sebelumnya:
    di jadwal KKN, potongan tengah rantai baris pertamanya kosong karena sudah
    lanjutan, sementara header aslinya ada di potongan pertama.

    Kepala yang tidak punya header sah berarti SELURUH rantai tidak mewarisi
    apa pun — kepala tidak digantikan potongan di tengah rantai.
    """
    sebelum = {b: a for a, b in pasangan}
    hasil: dict[str, str] = {}
    for a, b in pasangan:
        for cid in (a, b):
            if cid in hasil:
                continue
            kepala, terlihat = cid, set()
            while kepala in sebelum and kepala not in terlihat:
                terlihat.add(kepala)
                kepala = sebelum[kepala]
            hasil[cid] = kepala
    return hasil
