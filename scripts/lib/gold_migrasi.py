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

from dataclasses import dataclass

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

    @property
    def n_chunk(self) -> int:
        return len(self.sha_per_chunk)


def bangun_indeks(baris_chunk, image_ids=()) -> IndeksBaru:
    """Bangun IndeksBaru dari rekaman chunk index baru.

    Tiap rekaman cukup punya `chunk_id` dan `text_sha`. Rekaman tanpa salah
    satunya dilewati — chunk tanpa `chunk_id` memang tidak dapat dirujuk gold.
    """
    sha_per_chunk: dict[str, str] = {}
    per_sha: dict[str, list[str]] = {}
    for r in baris_chunk:
        cid, sha = r.get("chunk_id"), r.get("text_sha")
        if not isinstance(cid, str) or not cid:
            continue
        sha_per_chunk[cid] = sha if isinstance(sha, str) else ""
        if isinstance(sha, str) and sha:
            per_sha.setdefault(sha, []).append(cid)
    return IndeksBaru(
        sha_per_chunk=sha_per_chunk,
        chunk_per_sha={k: tuple(v) for k, v in per_sha.items()},
        image_ids=frozenset(i for i in image_ids if isinstance(i, str) and i),
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


def petakan_chunk(chunk_id: str, sha_lama: str | None, indeks: IndeksBaru) -> HasilChunk:
    """Petakan satu chunk_id gold ke index baru.

    `sha_lama` boleh None bila dump lama tidak tersedia; pemetaan lalu hanya
    dapat mengandalkan keberadaan `chunk_id`, dan perubahan isi tidak terdeteksi.
    Itu dilaporkan apa adanya, bukan disamarkan jadi "identik".
    """
    sha_baru = indeks.sha_per_chunk.get(chunk_id)

    if sha_baru is not None:
        if sha_lama is None:
            return HasilChunk(chunk_id, chunk_id, STATUS_IDENTIK,
                              "chunk_id masih ada; isi tidak dapat dibandingkan "
                              "karena dump lama tidak diberikan")
        if sha_lama == sha_baru:
            return HasilChunk(chunk_id, chunk_id, STATUS_IDENTIK)
        return HasilChunk(
            chunk_id, chunk_id, STATUS_ISI_BERUBAH,
            "chunk_id sama tapi text_sha berubah — kemungkinan besar potongan "
            "tabel lanjutan yang headernya diulang; perlu ditinjau ulang manusia",
        )

    # chunk_id hilang. Jangkar text_sha.
    if not sha_lama:
        return HasilChunk(
            chunk_id, None, STATUS_HILANG,
            "chunk_id tidak ada di index baru dan text_sha lama tidak diketahui, "
            "sehingga tidak ada jangkar untuk mencarinya",
        )

    kandidat = indeks.chunk_per_sha.get(sha_lama, ())
    if len(kandidat) == 1:
        return HasilChunk(
            chunk_id, kandidat[0], STATUS_PINDAH,
            f"chunk_id bergeser; dikenali lewat text_sha yang sama",
        )
    if len(kandidat) > 1:
        return HasilChunk(
            chunk_id, None, STATUS_AMBIGU,
            f"text_sha {sha_lama[:12]} cocok dengan {len(kandidat)} chunk baru "
            f"({', '.join(kandidat[:4])}) — tidak dipilih otomatis",
        )
    return HasilChunk(
        chunk_id, None, STATUS_HILANG,
        "chunk_id tidak ada dan tidak ada chunk baru ber-text_sha sama — isinya "
        "berubah ATAU chunk-nya tidak lagi diproduksi",
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


def petakan_item(item: dict, sha_lama_per_chunk: dict, indeks: IndeksBaru) -> HasilItem:
    """Petakan satu item gold. TIDAK mengubah `item`.

    `relevant_images` tidak dimigrasi — penamaan image_id tidak tersentuh
    perubahan ini — tapi keberadaannya tetap divalidasi, karena gold yang
    menunjuk gambar hilang sama tidak sahihnya dengan yang menunjuk chunk hilang.
    """
    chunks = tuple(
        petakan_chunk(cid, sha_lama_per_chunk.get(cid), indeks)
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
