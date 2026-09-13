"""Penguraian tabel HTML dan deteksi pola potongan lintas halaman.

Fungsi murni tanpa I/O: tidak membaca berkas, tidak menyentuh Qdrant, tidak
mengubah argumennya. Dipisah dari skrip analisis supaya dapat diuji sendiri —
angkanya dipakai untuk keputusan yang menyentuh 1.300 item gold.

Sumbernya `raw_html` yang sudah tersimpan di payload, yaitu `text_as_html` dari
Unstructured. Tidak butuh ekstraksi ulang.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

# Kata sambung Indonesia yang lazim mengawali lanjutan kalimat. Dipakai sebagai
# indikator sel yang terpenggal, bukan sebagai bukti.
KONJUNGSI = frozenset({
    "dan", "atau", "serta", "yang", "dengan", "untuk", "dalam", "pada", "dari",
    "ke", "oleh", "sebagai", "maupun", "tersebut", "kepada", "tentang", "agar",
    "sehingga", "apabila", "bila", "jika", "karena", "namun", "tetapi", "serta",
    "melampirkan", "disertai", "beserta",
})

# Tanda baca yang menandai akhir isi sel yang utuh.
_PENUTUP = tuple(".!?:;)]}”\"'%")


class _Pengurai(HTMLParser):
    """Kumpulkan baris tabel sebagai list-of-list, plus tahu ada penanda header."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.baris: list[list[str]] = []
        self.ada_th = False
        self.ada_thead = False
        self._baris: list[str] | None = None
        self._sel: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "thead":
            self.ada_thead = True
        elif tag == "tr":
            self._baris = []
        elif tag in ("td", "th"):
            if tag == "th":
                self.ada_th = True
            self._sel = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._sel is not None and self._baris is not None:
            self._baris.append(" ".join("".join(self._sel).split()))
            self._sel = None
        elif tag == "tr" and self._baris is not None:
            # Sel yang masih terbuka saat </tr> tiba tetap diambil. HTML dari
            # OCR kadang kehilangan </td>; membuang barisnya berarti kehilangan
            # data yang sebenarnya terbaca.
            if self._sel is not None:
                self._baris.append(" ".join("".join(self._sel).split()))
                self._sel = None
            if self._baris:
                self.baris.append(self._baris)
            self._baris = None

    def handle_data(self, data):
        if self._sel is not None:
            self._sel.append(data)


@dataclass(frozen=True)
class Tabel:
    """Tabel hasil urai. Immutable: seluruh operasi mengembalikan nilai baru."""

    baris: tuple[tuple[str, ...], ...] = ()
    ada_th: bool = False
    ada_thead: bool = False

    @property
    def n_kolom(self) -> int:
        """Jumlah kolom modal — tahan terhadap baris judul yang di-colspan."""
        if not self.baris:
            return 0
        hitung: dict[int, int] = {}
        for b in self.baris:
            hitung[len(b)] = hitung.get(len(b), 0) + 1
        return max(hitung.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    @property
    def punya_header(self) -> bool:
        return self.ada_th or self.ada_thead


def urai(html: str | None) -> Tabel | None:
    """HTML -> Tabel. None bila bukan tabel atau tidak ada baris sama sekali."""
    if not html or "<t" not in html.lower():
        return None
    p = _Pengurai()
    try:
        p.feed(html)
        p.close()
    except Exception:
        return None
    if not p.baris:
        return None
    return Tabel(
        baris=tuple(tuple(b) for b in p.baris),
        ada_th=p.ada_th,
        ada_thead=p.ada_thead,
    )


def _norm(sel) -> str:
    return "|".join(re.sub(r"\s+", " ", s).strip().lower() for s in sel)


def rasio_angka(sel) -> float:
    """Proporsi sel yang berupa angka. Baris data tinggi, baris header rendah."""
    if not sel:
        return 0.0
    n = sum(1 for s in sel if s.strip() and re.fullmatch(r"[\d.,%()\-\s/]+", s.strip()))
    return n / len(sel)


def bandingkan_header(a: Tabel | None, b: Tabel | None) -> str:
    """'b_tanpa_header' | 'header_diulang' | 'header_berbeda' | 'tak_tentu'."""
    if a is None or b is None or not a.baris or not b.baris:
        return "tak_tentu"
    if _norm(a.baris[0]) == _norm(b.baris[0]):
        return "header_diulang"
    if a.punya_header and not b.punya_header:
        return "b_tanpa_header"
    if rasio_angka(b.baris[0]) >= 0.5 and rasio_angka(a.baris[0]) < 0.5:
        return "b_tanpa_header"
    if a.punya_header and b.punya_header:
        return "header_berbeda"
    return "tak_tentu"


# ─── Kasus 3: sel terpotong di tengah ────────────────────────────────────────

@dataclass(frozen=True)
class SelTerpotong:
    """Satu kolom yang isinya diduga terpenggal antara potongan A dan B."""

    kolom: int
    ekor_a: str
    kepala_b: str
    tanpa_tanda_baca: bool
    lanjutan_huruf_kecil: bool
    lanjutan_konjungsi: bool
    sel_lain_kosong_a: bool
    sel_lain_kosong_b: bool

    @property
    def sel_lain_kosong(self) -> bool:
        """Sel kosong di baris mana pun dari pasangan.

        Dalam praktik justru muncul di B: nomor baris tidak diulang pada
        potongan lanjutan, sehingga kolom pertamanya kosong. Memeriksa A saja
        melewatkan kasus yang paling lazim.
        """
        return self.sel_lain_kosong_a or self.sel_lain_kosong_b

    @property
    def skor(self) -> int:
        return sum((self.tanpa_tanda_baca,
                    self.lanjutan_huruf_kecil or self.lanjutan_konjungsi,
                    self.sel_lain_kosong))


@dataclass(frozen=True)
class HasilSel:
    kandidat: tuple[SelTerpotong, ...] = ()
    baris_terakhir_a: tuple[str, ...] = ()
    baris_pertama_b: tuple[str, ...] = ()
    catatan: str = ""

    @property
    def kuat(self) -> tuple[SelTerpotong, ...]:
        return tuple(k for k in self.kandidat if k.skor >= 2)


def _berakhir_tanpa_penutup(teks: str) -> bool:
    t = teks.strip()
    return bool(t) and not t.endswith(_PENUTUP)


def _mulai_lanjutan(teks: str) -> tuple[bool, bool]:
    """(diawali huruf kecil, diawali konjungsi)."""
    t = teks.strip()
    if not t:
        return False, False
    kata = re.split(r"\s+", t)[0].strip(".,;:()").lower()
    return t[0].islower() and t[0].isalpha(), kata in KONJUNGSI


def deteksi_sel_terpotong(a: Tabel | None, b: Tabel | None) -> HasilSel:
    """Cari kolom yang isinya terpenggal di batas halaman.

    Berbeda dari baris terpotong: pemetaan baris-kolom benar, tapi NILAI selnya
    terbelah jadi dua fragmen yang masing-masing tidak lengkap. Header yang
    diulang tidak menolong — sistem mengambil setengah jawaban dan mengira utuh.

    Tiga indikator per kolom:
      - sel terakhir A berakhir tanpa tanda baca penutup
      - sel pertama B di kolom yang sama diawali huruf kecil ATAU konjungsi
      - ada sel lain yang kosong di baris terakhir A ATAU baris pertama B
        (nomor baris lazimnya tidak diulang di potongan lanjutan)

    Mengembalikan KANDIDAT, bukan putusan. `kuat` menyaring skor >= 2.
    """
    if a is None or b is None or not a.baris or not b.baris:
        return HasilSel(catatan="raw_html tidak dapat diurai pada salah satu potongan")

    ekor = a.baris[-1]
    kepala = b.baris[0]
    if len(ekor) != len(kepala):
        return HasilSel(
            baris_terakhir_a=ekor, baris_pertama_b=kepala,
            catatan=f"lebar baris berbeda ({len(ekor)} vs {len(kepala)}), "
                    f"perbandingan per kolom tidak sahih",
        )

    kosong_a = any(not s.strip() for s in ekor)
    kosong_b = any(not s.strip() for s in kepala)
    kandidat = []
    for i, (sa, sb) in enumerate(zip(ekor, kepala)):
        if not sa.strip() or not sb.strip():
            continue
        kecil, konj = _mulai_lanjutan(sb)
        tanpa_baca = _berakhir_tanpa_penutup(sa)
        if not (tanpa_baca or kecil or konj):
            continue
        kandidat.append(SelTerpotong(
            kolom=i, ekor_a=sa, kepala_b=sb,
            tanpa_tanda_baca=tanpa_baca,
            lanjutan_huruf_kecil=kecil,
            lanjutan_konjungsi=konj,
            sel_lain_kosong_a=kosong_a,
            sel_lain_kosong_b=kosong_b,
        ))
    return HasilSel(kandidat=tuple(kandidat),
                    baris_terakhir_a=ekor, baris_pertama_b=kepala)
