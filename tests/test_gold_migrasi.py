"""Uji pemetaan gold ke index baru.

Bentuk item mengikuti implementasi nyata tim evaluasi:
`relevant_text_chunks` dan `relevant_images` — BUKAN `gold_chunk_ids` /
`gold_image_ids` yang ada di skema deck.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from lib.gold_migrasi import (  # noqa: E402
    STATUS_AMBIGU, STATUS_HILANG, STATUS_IDENTIK, STATUS_ISI_BERUBAH,
    STATUS_PINDAH, bangun_indeks, petakan_chunk, petakan_item, query_id_usang,
    terapkan,
)


def item(**kw):
    dasar = {
        "query_id": "sop_text_only_sop_p17_c01",
        "question": "Berapa IPK minimum untuk S2?",
        "reference_answer": "3,00",
        "query_type": "text_only",
        "visual_type": "",
        "document_id": "sop",
        "relevant_text_chunks": ["sop_p17_c01"],
        "relevant_images": [],
        "structural_annotation": {"type": "none"},
        "generator": "qwen2.5:7b seed=1337",
        "needs_human_review": True,
        "human_verdict": "ok",
        "catatan": "",
    }
    dasar.update(kw)
    return dasar


IDX = bangun_indeks(
    [{"chunk_id": "sop_p17_c01", "text_sha": "aaa"},
     {"chunk_id": "sop_p17_c02", "text_sha": "bbb"},
     {"chunk_id": "sop_p18_c00", "text_sha": "ccc"}],
    image_ids=["sop_p3_img01"],
)


# ─── petakan_chunk ───────────────────────────────────────────────────────────

@pytest.mark.unit
def test_chunk_identik():
    h = petakan_chunk("sop_p17_c01", "aaa", IDX)
    assert h.status == STATUS_IDENTIK and h.baru == "sop_p17_c01" and h.terpetakan


@pytest.mark.unit
def test_isi_berubah_chunk_id_tetap():
    """chunk_id masih ada tapi isinya beda, dan tidak ada jangkar isi.

    Dipetakan ke id yang sama sebagai UPAYA TERAKHIR, dengan peringatan bahwa
    id itu bisa saja kini ditempati chunk lain."""
    h = petakan_chunk("sop_p17_c01", "sha_lama", IDX)
    assert h.status == STATUS_ISI_BERUBAH and h.baru == "sop_p17_c01"
    assert h.terpetakan and "ditempati chunk LAIN" in h.alasan


@pytest.mark.unit
def test_sufiks_menang_atas_chunk_id():
    """Pengulangan header: isi lama jadi EKOR isi baru di chunk yang bergeser.

    Jangkar isi harus menang atas chunk_id, karena id lama bisa saja kini
    ditempati chunk yang sama sekali berbeda."""
    idx = bangun_indeks([
        {"chunk_id": "sop_p40_c02", "text_sha": "milik_chunk_lain",
         "text_content": "Paragraf yang sama sekali berbeda isinya di sini."},
        {"chunk_id": "sop_p40_c03", "text_sha": "sha_baru",
         "text_content": "| Uraian | 2024 |\n| --- | --- |\n| Kegiatan A | 1.250.000 |"},
    ])
    h = petakan_chunk("sop_p40_c02", "sha_lama", idx,
                      teks_lama="| Kegiatan A | 1.250.000 |" + " " * 0 + " pelengkap agar cukup panjang")
    # Teks lama bukan ekor -> jatuh ke jangkar ketiga.
    assert h.baru == "sop_p40_c02"

    h2 = petakan_chunk("sop_p40_c02", "sha_lama", idx,
                       teks_lama="| Uraian | 2024 |\n| --- | --- |\n| Kegiatan A | 1.250.000 |")
    assert h2.baru == "sop_p40_c03" and h2.status == STATUS_ISI_BERUBAH
    assert "BERAKHIR dengan teks lama" in h2.alasan


@pytest.mark.unit
def test_chunk_bergeser_dikenali_lewat_text_sha():
    """Kasus tepi: chunk yang dulu dibuang filter min-token kini lolos,
    sehingga seluruh chunk_id sesudahnya di halaman itu bergeser."""
    h = petakan_chunk("sop_p18_c99", "ccc", IDX)
    assert h.status == STATUS_PINDAH and h.baru == "sop_p18_c00" and h.terpetakan


@pytest.mark.unit
def test_text_sha_ganda_tidak_dipilih_otomatis():
    """Dua kandidat di dokumen yang SAMA — jangkar dokumen tidak memangkas apa
    pun, jadi tetap ambigu."""
    idx = bangun_indeks([{"chunk_id": "a_p1_c00", "text_sha": "dup"},
                         {"chunk_id": "a_p2_c00", "text_sha": "dup"}])
    h = petakan_chunk("a_p9_c00", "dup", idx)
    assert h.status == STATUS_AMBIGU and h.baru is None and not h.terpetakan
    assert "2 chunk di dokumen yang SAMA" in h.alasan


@pytest.mark.unit
def test_hilang_tanpa_jangkar():
    h = petakan_chunk("sop_p99_c00", None, IDX)
    assert h.status == STATUS_HILANG and "tidak ada jangkar" in h.alasan


@pytest.mark.unit
def test_hilang_dengan_jangkar_yang_tidak_cocok():
    h = petakan_chunk("sop_p99_c00", "zzz", IDX)
    assert h.status == STATUS_HILANG and "tidak lagi diproduksi" in h.alasan


@pytest.mark.unit
def test_tanpa_dump_lama_perubahan_isi_tidak_disamarkan():
    h = petakan_chunk("sop_p17_c01", None, IDX)
    assert h.status == STATUS_IDENTIK
    assert "TIDAK dapat dibandingkan" in h.alasan


# ─── petakan_item ────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_item_identik():
    h = petakan_item(item(), {"sop_p17_c01": "aaa"}, IDX)
    assert h.status == STATUS_IDENTIK and h.terpetakan and not h.masalah


@pytest.mark.unit
def test_status_item_mengambil_yang_terburuk():
    it = item(relevant_text_chunks=["sop_p17_c01", "sop_p99_c00"])
    h = petakan_item(it, {"sop_p17_c01": "aaa"}, IDX)
    assert h.status == STATUS_HILANG and not h.terpetakan


@pytest.mark.unit
def test_isi_berubah_kalah_dari_hilang_tapi_menang_dari_identik():
    it = item(relevant_text_chunks=["sop_p17_c01", "sop_p17_c02"])
    h = petakan_item(it, {"sop_p17_c01": "aaa", "sop_p17_c02": "beda"}, IDX)
    assert h.status == STATUS_ISI_BERUBAH and h.terpetakan


@pytest.mark.unit
def test_item_image_only_divalidasi_bukan_dimigrasi():
    it = item(query_type="image_only", relevant_text_chunks=[],
              relevant_images=["sop_p3_img01"])
    h = petakan_item(it, {}, IDX)
    assert h.status == STATUS_IDENTIK and not h.image_hilang


@pytest.mark.unit
def test_image_hilang_membuat_item_tidak_terpetakan():
    it = item(query_type="image_only", relevant_text_chunks=[],
              relevant_images=["sop_p9_img99"])
    h = petakan_item(it, {}, IDX)
    assert h.image_hilang == ("sop_p9_img99",) and not h.terpetakan


@pytest.mark.unit
def test_invarian_tim_eval_dilaporkan_bukan_diasumsikan():
    """Tidak ada item yang mengisi keduanya — kalau ada, itu dilaporkan."""
    it = item(relevant_images=["sop_p3_img01"])
    h = petakan_item(it, {"sop_p17_c01": "aaa"}, IDX)
    assert any("sekaligus" in m for m in h.masalah)


@pytest.mark.unit
def test_item_tanpa_rujukan_apa_pun_dilaporkan():
    h = petakan_item(item(relevant_text_chunks=[], relevant_images=[]), {}, IDX)
    assert any("tidak merujuk" in m for m in h.masalah)


# ─── terapkan ────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_seluruh_kolom_terbawa_apa_adanya():
    it = item(human_verdict="revisi", catatan="ditulis manual tim eval",
              structural_annotation={"type": "steps",
                                     "ground_truth_steps": ["a", "b"]})
    hasil = terapkan(petakan_item(it, {"sop_p17_c01": "aaa"}, IDX))
    for kunci in it:
        assert kunci in hasil, f"kolom {kunci} hilang"
    assert hasil["human_verdict"] == "revisi"
    assert hasil["catatan"] == "ditulis manual tim eval"
    assert hasil["structural_annotation"]["ground_truth_steps"] == ["a", "b"]


@pytest.mark.unit
def test_item_asli_tidak_diubah():
    it = item()
    salinan = dict(it)
    hasil = terapkan(petakan_item(it, {"sop_p17_c01": "beda"}, IDX))
    assert it == salinan
    assert hasil is not it


@pytest.mark.unit
def test_chunk_id_bergeser_ditulis_ulang():
    it = item(relevant_text_chunks=["sop_p18_c99"])
    hasil = terapkan(petakan_item(it, {"sop_p18_c99": "ccc"}, IDX))
    assert hasil["relevant_text_chunks"] == ["sop_p18_c00"]


@pytest.mark.unit
def test_relevant_images_tidak_disentuh():
    it = item(query_type="image_only", relevant_text_chunks=[],
              relevant_images=["sop_p3_img01"])
    hasil = terapkan(petakan_item(it, {}, IDX))
    assert hasil["relevant_images"] == ["sop_p3_img01"]


# ─── query_id ────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_query_id_usang_terdeteksi_tapi_tidak_diubah():
    """query_id adalah identitas, bukan rujukan. Mengubahnya memutus jejak ke
    hasil tinjauan manusia yang sudah ada."""
    it = item(query_id="sop_text_only_sop_p18_c99",
              relevant_text_chunks=["sop_p18_c99"])
    h = petakan_item(it, {"sop_p18_c99": "ccc"}, IDX)
    assert query_id_usang(h)
    assert terapkan(h)["query_id"] == "sop_text_only_sop_p18_c99"


@pytest.mark.unit
def test_query_id_tidak_usang_bila_chunk_tidak_bergeser():
    h = petakan_item(item(), {"sop_p17_c01": "aaa"}, IDX)
    assert not query_id_usang(h)


# ─── bangun_indeks ───────────────────────────────────────────────────────────

@pytest.mark.unit
def test_chunk_tanpa_id_dilewati():
    idx = bangun_indeks([{"text_sha": "x"}, {"chunk_id": "", "text_sha": "y"},
                         {"chunk_id": "ok_p1_c00", "text_sha": "z"}])
    assert idx.n_chunk == 1


@pytest.mark.unit
def test_chunk_tanpa_sha_tetap_terdaftar_tapi_bukan_jangkar():
    idx = bangun_indeks([{"chunk_id": "a_p1_c00"}])
    assert petakan_chunk("a_p1_c00", None, idx).status == STATUS_IDENTIK
    assert idx.chunk_per_sha == {}


# ─── jangkar pembeda document_id ─────────────────────────────────────────────

@pytest.mark.unit
def test_boilerplate_lintas_dokumen_diselesaikan_jangkar_dokumen():
    """Paragraf baku yang muncul SEKALI di tiap dokumen: tabrakan text_sha
    terselesaikan karena hanya satu kandidat berada di dokumen yang sama."""
    idx = bangun_indeks([
        {"chunk_id": "dok-a_p9_c00", "text_sha": "baku"},
        {"chunk_id": "dok-b_p3_c00", "text_sha": "baku"},
        {"chunk_id": "dok-c_p7_c00", "text_sha": "baku"},
    ])
    h = petakan_chunk("dok-b_p2_c00", "baku", idx)
    assert h.status == STATUS_PINDAH and h.baru == "dok-b_p3_c00"
    assert "dokumen yang sama" in h.alasan


@pytest.mark.unit
def test_teks_identik_berulang_dalam_satu_dokumen_tetap_ambigu():
    """Jangkar dokumen tidak menolong di sini — tidak dapat dipilih tanpa menebak."""
    idx = bangun_indeks([
        {"chunk_id": "dok-a_p3_c00", "text_sha": "baku"},
        {"chunk_id": "dok-a_p9_c00", "text_sha": "baku"},
        {"chunk_id": "dok-b_p1_c00", "text_sha": "baku"},
    ])
    h = petakan_chunk("dok-a_p2_c00", "baku", idx)
    assert h.status == STATUS_AMBIGU and h.baru is None
    assert "dokumen yang SAMA" in h.alasan


@pytest.mark.unit
def test_tidak_satu_pun_kandidat_di_dokumen_yang_sama():
    """Chunk aslinya kemungkinan tidak lagi diproduksi — jangan asal pilih."""
    idx = bangun_indeks([
        {"chunk_id": "dok-x_p1_c00", "text_sha": "baku"},
        {"chunk_id": "dok-y_p1_c00", "text_sha": "baku"},
    ])
    h = petakan_chunk("dok-z_p1_c00", "baku", idx)
    assert h.status == STATUS_AMBIGU and "TIDAK SATU PUN" in h.alasan


@pytest.mark.unit
def test_id_persis_menang_atas_tabrakan():
    """Kalau id lama ADA di antara kandidat, itu jawabannya — tanpa keraguan."""
    idx = bangun_indeks([
        {"chunk_id": "dok-a_p3_c00", "text_sha": "baku"},
        {"chunk_id": "dok-a_p9_c00", "text_sha": "baku"},
    ])
    h = petakan_chunk("dok-a_p3_c00", "baku", idx)
    assert h.status == STATUS_IDENTIK and h.baru == "dok-a_p3_c00"


@pytest.mark.unit
@pytest.mark.parametrize("cid,harapan", [
    ("standar-biaya-2026_p12_c03", "standar-biaya-2026"),
    ("dok_p1_c00", "dok"),
    ("dok_pNA_c07", "dok"),
    ("bentuk-lain", None),
    ("", None),
    (None, None),
])
def test_slug_dokumen_dari_chunk_id(cid, harapan):
    from lib.gold_migrasi import slug_dokumen
    assert slug_dokumen(cid) == harapan
