"""Pemetaan gold lama ke index baru, berjangkar `text_sha`.

Kenapa ini sebagian besar VERIFIKASI, bukan penyelamatan
---------------------------------------------------------
Tahap B tidak menggeser `chunk_id`: mengulang baris header tidak menambah atau
mengurangi chunk, dan `ordinal` di `preprocessing.emit` hanya naik saat sebuah
chunk benar-benar di-`append`. Jadi mayoritas item gold seharusnya terpetakan
identik tanpa perubahan apa pun.

Yang tetap harus dideteksi, bukan diasumsikan tidak ada:

1. Potongan lanjutan yang headernya diulang — `text_sha` BERUBAH walau
   `chunk_id` tetap. Itemnya tetap sahih, tapi isinya bukan lagi yang dibaca
   peninjau, jadi wajib masuk daftar tinjau ulang.
2. Chunk yang sebelumnya dibuang filter `INDEX_MIN_CHUNK_TOKENS` lalu lolos
   setelah ditambahi header. Ia mulai memakan ordinal dan MENGGESER seluruh
   `chunk_id` sesudahnya di halaman itu. Di sinilah `text_sha` benar-benar
   bekerja sebagai jangkar.

Fungsi murni: tidak menyentuh berkas, tidak mengubah argumennya.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# chunk_id berbentuk "{document_id}_p{N}_c{NN}". Jangkar di UJUNG supaya slug
# yang memuat "_p" tidak terpotong di tempat yang salah.
_CHUNK_ID_RE = re.compile(r"^(?P<doc>.+)_p(?:\d+|NA)_c\d+$")


def slug_dokumen(chunk_id: str) -> str | None:
    """document_id sebuah chunk, dibaca dari chunk_id-nya sendiri.

    Sengaja dari chunk_id, bukan dari field `document_id` gold: yang pertama
    intrinsik pada chunk itu, yang kedua bisa saja tidak sinkron. Kalau keduanya
    berbeda, itu masalah data tersendiri yang tidak boleh disamarkan oleh
    pemetaan yang diam-diam memilih salah satu.
    """
    m = _CHUNK_ID_RE.match(chunk_id or "")
    return m.group("doc") if m else None

STATUS_IDENTIK = "identik"
STATUS_ISI_BERUBAH = "isi_berubah"
STATUS_PINDAH = "pindah"
STATUS_AMBIGU = "ambigu"
STATUS_HILANG = "hilang"

# Status yang masih menghasilkan chunk_id sahih di index baru.
STATUS_TERPETAKAN = (STATUS_IDENTIK, STATUS_ISI_BERUBAH, STATUS_PINDAH)


@dataclass(frozen=True)
class IndeksBaru:
    """Ringkasan index baru: peta chunk_id <-> text_sha, dan himpunan image_id."""

    sha_per_chunk: dict[str, str]
    chunk_per_sha: dict[str, tuple[str, ...]]
    image_ids: frozenset[str] = frozenset()
    # chunk_id -> teks. Hanya terisi bila dump baru membawa `text_content`.
    # Dipakai sebagai jangkar KETIGA, lihat petakan_chunk.
    teks_per_chunk: dict[str, str] = field(default_factory=dict)

    @property
    def n_chunk(self) -> int:
        return len(self.sha_per_chunk)


def bangun_indeks(baris_chunk, image_ids=()) -> IndeksBaru:
    """Bangun IndeksBaru dari rekaman chunk index baru.

    Tiap rekaman cukup punya `chunk_id` dan `text_sha`. `text_content` opsional
    tapi mengaktifkan jangkar ketiga. Rekaman tanpa `chunk_id` dilewati — chunk
    tanpa id memang tidak dapat dirujuk gold.
    """
    sha_per_chunk: dict[str, str] = {}
    per_sha: dict[str, list[str]] = {}
    teks: dict[str, str] = {}
    for r in baris_chunk:
        cid, sha = r.get("chunk_id"), r.get("text_sha")
        if not isinstance(cid, str) or not cid:
            continue
        sha_per_chunk[cid] = sha if isinstance(sha, str) else ""
        if isinstance(sha, str) and sha:
            per_sha.setdefault(sha, []).append(cid)
        isi = r.get("text_content") or r.get("text")
        if isinstance(isi, str) and isi:
            teks[cid] = isi
    return IndeksBaru(
        sha_per_chunk=sha_per_chunk,
        chunk_per_sha={k: tuple(v) for k, v in per_sha.items()},
        image_ids=frozenset(i for i in image_ids if isinstance(i, str) and i),
        teks_per_chunk=teks,
    )


@dataclass(frozen=True)
class HasilChunk:
    """Nasib satu chunk_id gold."""

    lama: str
    baru: str | None
    status: str
    alasan: str = ""

    @property
    def terpetakan(self) -> bool:
        return self.status in STATUS_TERPETAKAN


def _cocok_sufiks(teks_lama: str, indeks: IndeksBaru) -> tuple[str, ...]:
    """Chunk baru yang teksnya BERAKHIR dengan teks lama.

    Jangkar ketiga, untuk potongan tabel lanjutan: pengulangan header menambah
    baris di DEPAN, sehingga teks baru = header + teks lama. text_sha berubah
    dan chunk_id bergeser, jadi dua jangkar pertama sama-sama gagal — tapi
    ekornya masih utuh.
    """
    ekor = " ".join((teks_lama or "").split())
    if len(ekor) < 40:
        return ()                     # terlalu pendek untuk jadi bukti
    return tuple(
        cid for cid, isi in indeks.teks_per_chunk.items()
        if " ".join(isi.split()).endswith(ekor)
    )


def petakan_chunk(chunk_id: str, sha_lama: str | None, indeks: IndeksBaru,
                  teks_lama: str | None = None) -> HasilChunk:
    """Petakan satu chunk_id gold ke index baru.

    URUTAN JANGKAR: text_sha DULU, chunk_id belakangan. Ini bukan detail gaya —
    `chunk_id` adalah POSISI, dan saat penomoran halaman diperbaiki, sebuah id
    lama bisa tetap ada di index baru tapi kini ditempati chunk yang BERBEDA.
    Memeriksa chunk_id lebih dulu akan memetakan gold ke isi yang keliru, dan
    kekeliruannya senyap karena id-nya tampak sahih. Terukur di uji skala: 70
    chunk terpetakan "isi berubah" padahal hanya 23 yang isinya benar-benar
    berubah.

    Tiga jangkar berurutan:
      1. text_sha — identitas ISI, tahan terhadap pergeseran penomoran
      2. sufiks teks — untuk potongan tabel yang headernya diulang: teks baru
         berakhir dengan teks lama, sehingga sha berubah tapi ekornya utuh
      3. chunk_id — hanya sebagai upaya terakhir, dan hasilnya ditandai perlu
         ditinjau karena isinya sudah tidak sama

    `sha_lama` boleh None bila dump lama tidak tersedia; pemetaan lalu hanya
    dapat mengandalkan keberadaan `chunk_id`, dan perubahan isi tidak terdeteksi.
    Itu dilaporkan apa adanya, bukan disamarkan jadi "identik".
    """
    if not sha_lama:
        if chunk_id in indeks.sha_per_chunk:
            return HasilChunk(chunk_id, chunk_id, STATUS_IDENTIK,
                              "chunk_id masih ada; isi TIDAK dapat dibandingkan "
                              "karena dump lama tidak diberikan")
        return HasilChunk(
            chunk_id, None, STATUS_HILANG,
            "chunk_id tidak ada di index baru dan text_sha lama tidak diketahui, "
            "sehingga tidak ada jangkar untuk mencarinya",
        )

    # ── Jangkar 1: text_sha ──
    kandidat = indeks.chunk_per_sha.get(sha_lama, ())
    if chunk_id in kandidat:
        # Id yang sama DAN isi yang sama — tidak ada keraguan, walau ada chunk
        # lain berisi teks identik di tempat lain.
        return HasilChunk(chunk_id, chunk_id, STATUS_IDENTIK)
    if len(kandidat) == 1:
        return HasilChunk(chunk_id, kandidat[0], STATUS_PINDAH,
                          "chunk_id bergeser; dikenali lewat text_sha yang sama")
    if len(kandidat) > 1:
        # Jangkar pembeda: boilerplate identik lazimnya tersebar di BANYAK
        # dokumen, jadi menyaring kandidat ke dokumen yang sama memangkas
        # sebagian besar tabrakan. Yang tersisa — teks identik berulang di
        # dalam satu dokumen — memang tidak dapat dipilih tanpa menebak.
        dok = slug_dokumen(chunk_id)
        sedokumen = tuple(c for c in kandidat if slug_dokumen(c) == dok) if dok else ()
        if len(sedokumen) == 1:
            return HasilChunk(
                chunk_id, sedokumen[0], STATUS_PINDAH,
                f"text_sha cocok dengan {len(kandidat)} chunk baru, tapi hanya "
                f"satu berada di dokumen yang sama ({dok})",
            )
        if len(sedokumen) > 1:
            return HasilChunk(
                chunk_id, None, STATUS_AMBIGU,
                f"text_sha {sha_lama[:12]} cocok dengan {len(sedokumen)} chunk "
                f"di dokumen yang SAMA ({', '.join(sedokumen[:4])}) — teks "
                f"identik berulang di dalam satu dokumen, tidak dapat dipilih "
                f"tanpa menebak",
            )
        return HasilChunk(
            chunk_id, None, STATUS_AMBIGU,
            f"text_sha {sha_lama[:12]} cocok dengan {len(kandidat)} chunk baru "
            f"({', '.join(kandidat[:4])}) dan TIDAK SATU PUN di dokumen yang "
            f"sama — chunk aslinya kemungkinan tidak lagi diproduksi",
        )

    # ── Jangkar 2: sufiks teks (pola pengulangan header) ──
    if teks_lama:
        sufiks = _cocok_sufiks(teks_lama, indeks)
        if len(sufiks) == 1:
            return HasilChunk(
                chunk_id, sufiks[0], STATUS_ISI_BERUBAH,
                "isi berubah; dikenali karena teks baru BERAKHIR dengan teks "
                "lama — pola pengulangan baris header. Perlu ditinjau ulang",
            )
        if len(sufiks) > 1:
            return HasilChunk(
                chunk_id, None, STATUS_AMBIGU,
                f"teks lama cocok sebagai ekor {len(sufiks)} chunk baru — "
                f"tidak dipilih otomatis",
            )

    # ── Jangkar 3: chunk_id, upaya terakhir ──
    if chunk_id in indeks.sha_per_chunk:
        return HasilChunk(
            chunk_id, chunk_id, STATUS_ISI_BERUBAH,
            "chunk_id masih ada TAPI isinya berbeda dan tidak ada jangkar isi "
            "yang cocok. Id ini bisa saja kini ditempati chunk LAIN — wajib "
            "ditinjau manusia sebelum dipakai",
        )
    return HasilChunk(
        chunk_id, None, STATUS_HILANG,
        "chunk_id tidak ada dan tidak ada chunk baru ber-text_sha sama maupun "
        "berteks berakhiran sama — isinya berubah ATAU chunk-nya tidak lagi "
        "diproduksi",
    )


@dataclass(frozen=True)
class HasilItem:
    """Nasib satu item gold."""

    item: dict
    chunks: tuple[HasilChunk, ...] = ()
    image_hilang: tuple[str, ...] = ()
    masalah: tuple[str, ...] = ()

    @property
    def status(self) -> str:
        """Status terburuk di antara chunk-nya, plus validasi gambar."""
        if any(c.status == STATUS_HILANG for c in self.chunks):
            return STATUS_HILANG
        if any(c.status == STATUS_AMBIGU for c in self.chunks):
            return STATUS_AMBIGU
        if self.image_hilang:
            return STATUS_HILANG
        if any(c.status == STATUS_ISI_BERUBAH for c in self.chunks):
            return STATUS_ISI_BERUBAH
        if any(c.status == STATUS_PINDAH for c in self.chunks):
            return STATUS_PINDAH
        return STATUS_IDENTIK

    @property
    def terpetakan(self) -> bool:
        return self.status in STATUS_TERPETAKAN


def petakan_item(item: dict, sha_lama_per_chunk: dict, indeks: IndeksBaru,
                 teks_lama_per_chunk: dict | None = None) -> HasilItem:
    """Petakan satu item gold. TIDAK mengubah `item`.

    `relevant_images` tidak dimigrasi — penamaan image_id tidak tersentuh
    perubahan ini — tapi keberadaannya tetap divalidasi, karena gold yang
    menunjuk gambar hilang sama tidak sahihnya dengan yang menunjuk chunk hilang.
    """
    chunks = tuple(
        petakan_chunk(cid, sha_lama_per_chunk.get(cid), indeks,
                      (teks_lama_per_chunk or {}).get(cid))
        for cid in (item.get("relevant_text_chunks") or [])
        if isinstance(cid, str) and cid
    )
    gambar = [i for i in (item.get("relevant_images") or [])
              if isinstance(i, str) and i]
    hilang = tuple(i for i in gambar if indeks.image_ids and i not in indeks.image_ids)

    masalah: list[str] = []
    if chunks and gambar:
        # Invarian yang dinyatakan tim eval: tidak ada item yang mengisi
        # keduanya. Dilaporkan, bukan diasumsikan.
        masalah.append("mengisi relevant_text_chunks DAN relevant_images sekaligus")
    if not chunks and not gambar:
        masalah.append("tidak merujuk chunk maupun gambar")
    return HasilItem(item=item, chunks=chunks, image_hilang=hilang,
                     masalah=tuple(masalah))


def terapkan(hasil: HasilItem) -> dict:
    """Salinan item dengan chunk_id yang sudah dipetakan.

    Mengembalikan objek BARU; item asli tidak disentuh. Seluruh kolom lain
    terbawa apa adanya, termasuk `human_verdict`, `catatan`, dan
    `structural_annotation`.

    `query_id` sengaja TIDAK diubah walau memuat chunk_id lama: ia identitas
    item, bukan rujukan ke chunk. Mengubahnya memutus jejak ke hasil tinjauan
    manusia yang sudah ada. Item yang query_id-nya memuat chunk_id bergeser
    dilaporkan terpisah supaya tim eval sadar.
    """
    baru = dict(item_baru for item_baru in hasil.item.items())
    if hasil.chunks:
        baru["relevant_text_chunks"] = [
            c.baru for c in hasil.chunks if c.baru is not None
        ]
    return baru


def query_id_usang(hasil: HasilItem) -> bool:
    """True bila `query_id` memuat chunk_id yang bergeser."""
    qid = hasil.item.get("query_id")
    if not isinstance(qid, str):
        return False
    return any(c.status == STATUS_PINDAH and c.lama in qid for c in hasil.chunks)
