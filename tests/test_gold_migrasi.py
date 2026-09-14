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
    """Potongan lanjutan yang headernya diulang: chunk_id sama, text_sha beda."""
    h = petakan_chunk("sop_p17_c01", "sha_lama", IDX)
    assert h.status == STATUS_ISI_BERUBAH and h.baru == "sop_p17_c01"
    assert h.terpetakan and "ditinjau ulang" in h.alasan


@pytest.mark.unit
def test_chunk_bergeser_dikenali_lewat_text_sha():
    """Kasus tepi: chunk yang dulu dibuang filter min-token kini lolos,
    sehingga seluruh chunk_id sesudahnya di halaman itu bergeser."""
    h = petakan_chunk("sop_p18_c99", "ccc", IDX)
    assert h.status == STATUS_PINDAH and h.baru == "sop_p18_c00" and h.terpetakan


@pytest.mark.unit
def test_text_sha_ganda_tidak_dipilih_otomatis():
    idx = bangun_indeks([{"chunk_id": "a_p1_c00", "text_sha": "dup"},
                         {"chunk_id": "a_p2_c00", "text_sha": "dup"}])
    h = petakan_chunk("a_p9_c00", "dup", idx)
    assert h.status == STATUS_AMBIGU and h.baru is None and not h.terpetakan
    assert "2 chunk baru" in h.alasan


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
    assert "tidak dapat dibandingkan" in h.alasan


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
