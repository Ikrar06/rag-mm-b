"""Uji orkestrasi transkripsi tabel dan klasifikasi gambar (Tahap T).

Model dan cache dipalsukan; PDF dibuat dengan fitz. Yang dibuktikan:
  1. Flag mati: element tidak berubah dan model tidak dipanggil.
  2. Tabel: teks diganti transkripsi, teks OCR ke teks_ocr, raw_html utuh,
     area render ditarik ke kata yang terpotong tepi bbox.
  3. Gagal (jaringan, terpotong, tak terurai): jatuh ke OCR, beralasan, tidak
     di-cache.
  4. Gambar: buang / transkripsi / narasi+tabel / narasi sesuai klasifikasi.
  5. Cache: jawaban sah dipakai ulang tanpa memanggil model.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

fitz = pytest.importorskip("fitz")

from backend import config  # noqa: E402
from backend.services import table_transcription as tt  # noqa: E402
from backend.services import vision_cache, vision_io  # noqa: E402
from backend.services.klasifikasi_gambar import PROMPT_KLASIFIKASI  # noqa: E402
from backend.services.transkripsi_murni import (  # noqa: E402
    PROMPT_TRANSKRIPSI, PROMPT_TRANSKRIPSI_SEMUA,
)

BATAS = {"model": "m", "ollama": "0.13.0", "patch_size": 16, "spatial_merge_size": 2,
         "sisi_min": 32, "rasio_maks": 200}
MD = "| No | Uraian | Biaya |\n|---|---|---|\n| 1 | UKT I | 1.500.000 |\n| 2 | UKT II | 8.000.000 |"


class Model:
    """Model palsu: jawaban per jenis prompt, diambil berurutan."""

    def __init__(self):
        self.jawab = {PROMPT_KLASIFIKASI: [], PROMPT_TRANSKRIPSI: [], PROMPT_TRANSKRIPSI_SEMUA: []}
        self.panggilan = []

    def __call__(self, png, prompt, *, num_predict, num_ctx):
        self.panggilan.append((prompt, num_predict))
        x = self.jawab[prompt].pop(0)
        if isinstance(x, Exception):
            raise x
        teks, alasan = x if isinstance(x, tuple) else (x, "stop")
        return {"response": teks, "done_reason": alasan, "eval_count": 1, "prompt_eval_count": 1,
                "detik": 1.0, "percobaan": 1, "dipadding": False}


@pytest.fixture
def lingkungan(monkeypatch):
    model, simpanan = Model(), {}
    monkeypatch.setattr(tt, "panggil", model)
    monkeypatch.setattr(vision_io, "batas_model", lambda: BATAS)
    import backend.services.image_describer as idesc
    monkeypatch.setattr(idesc, "vision_provenance", lambda: {"vision_model_digest": "sha256:d"})
    monkeypatch.setattr(vision_cache, "enabled", lambda: True)
    monkeypatch.setattr(vision_cache, "get", lambda k: simpanan.get(k))
    monkeypatch.setattr(vision_cache, "put", lambda k, **kw: simpanan.__setitem__(
        k, vision_cache.CachedDescription(kw["verdict"], kw["description"])))
    for nama, nilai in (("INDEX_TABLE_TRANSCRIPTION", True), ("INDEX_IMAGE_TABLE_TRANSCRIPTION", True),
                        ("TABLE_TRANSCRIPTION_DPI", 72), ("IMAGE_CLASSIFICATION_DPI", 72)):
        monkeypatch.setattr(config, nama, nilai)
    tt.reset_stats()
    return model, simpanan


@pytest.fixture
def pdf(tmp_path):
    doc = fitz.open()
    hal = doc.new_page()                       # 595 x 842, digital
    hal.insert_text((72, 100), "No Uraian Biaya")
    hal.insert_text((72, 120), "1 UKT I 1.500.000")
    hal.insert_text((400, 140), "8.000.000")   # terpotong tepi kanan bbox tabel
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 40), 0)
    hal.insert_image(fitz.Rect(300, 400, 400, 500), pixmap=pix)
    doc.save(tmp_path / "ukt.pdf")
    return tmp_path / "ukt.pdf"


# bbox tabel berakhir di tengah "8.000.000" (x 400-450 pt dari 595).
BBOX_TABEL = [0.1, 0.1, 0.72, 0.2]


def tabel_el(teks="OCR 1 UKT I 1.500.000"):
    return {"category": "Table", "text": teks, "page": 1,
            "metadata": {"raw_html": "<table><tr><td>OCR</td></tr></table>",
                         "table_format": "markdown", "bbox": list(BBOX_TABEL)}}


def gambar_el(bbox, image_id="ukt_p1_img00"):
    return {"category": "Image", "text": "", "page": 1,
            "metadata": {"image_base64": "x", "bbox": bbox, "image_id": image_id}}


def test_flag_mati_tidak_mengubah_apa_pun(lingkungan, pdf, monkeypatch):
    model, _ = lingkungan
    monkeypatch.setattr(config, "INDEX_TABLE_TRANSCRIPTION", False)
    monkeypatch.setattr(config, "INDEX_IMAGE_TABLE_TRANSCRIPTION", False)
    els = [tabel_el(), gambar_el([0.5, 0.47, 0.67, 0.6])]
    assert tt.proses(els, pdf) == (els, {})
    assert model.panggilan == [] and tt.provenance() is None


def test_tabel_ditranskripsi_raw_html_utuh_area_diperluas(lingkungan, pdf):
    model, _ = lingkungan
    model.jawab[PROMPT_TRANSKRIPSI] = [f"Berikut:\n```markdown\n{MD}\n```"]
    (el,), _ = tt.proses([tabel_el()], pdf)
    m = el["metadata"]
    assert el["text"] == MD and m["teks_ocr"] == "OCR 1 UKT I 1.500.000"
    assert m["raw_html"] == "<table><tr><td>OCR</td></tr></table>"
    assert m["table_source"] == "vision_transcription" and m["table_format"] == "markdown"
    assert m["transkripsi_baris"][0] == ("No", "Uraian", "Biaya")
    assert m["render_bbox"][2] > BBOX_TABEL[2]          # kata terpotong ditarik utuh
    assert m["render_bbox"][0] == BBOX_TABEL[0]
    assert m["transkripsi_peringatan"] == []            # kedua angka ada di lapisan teks
    assert model.panggilan == [(PROMPT_TRANSKRIPSI, config.TABLE_TRANSCRIPTION_NUM_PREDICT)]


def test_angka_tak_ditemukan_ditandai_dengan_rujukan(lingkungan, pdf):
    model, _ = lingkungan
    model.jawab[PROMPT_TRANSKRIPSI] = [MD.replace("8.000.000", "9.999.999")]
    (el,), _ = tt.proses([tabel_el()], pdf)
    assert el["metadata"]["transkripsi_peringatan"] == [
        "rujukan=lapisan_teks", "angka_tak_ditemukan:9999999"]


@pytest.mark.parametrize("jawaban,alasan", [
    ((MD, "length"), "gagal:terpotong"),
    ("maaf, tidak ada tabel", "gagal:tak_terurai"),
    (vision_io.GagalVision("4 percobaan; terakhir HTTPStatusError"), "gagal:gagal_vision:4 percobaan"),
])
def test_tabel_gagal_jatuh_ke_ocr_dan_tidak_dicache(lingkungan, pdf, jawaban, alasan):
    model, simpanan = lingkungan
    model.jawab[PROMPT_TRANSKRIPSI] = [jawaban]
    (el,), _ = tt.proses([tabel_el()], pdf)
    assert el["text"] == "OCR 1 UKT I 1.500.000" and "teks_ocr" not in el["metadata"]
    assert el["metadata"]["table_source"] == "ocr_fallback"
    assert el["metadata"]["transkripsi_peringatan"][0].startswith(alasan)
    assert simpanan == {}
    assert tt.provenance()["hitungan"]["tabel_fallback_ocr"] == 1


def test_cache_dipakai_ulang(lingkungan, pdf):
    model, simpanan = lingkungan
    model.jawab[PROMPT_TRANSKRIPSI] = [MD]
    tt.proses([tabel_el()], pdf)
    (el,), _ = tt.proses([tabel_el()], pdf)
    assert el["text"] == MD and len(model.panggilan) == 1 and len(simpanan) == 1


BBOX_GAMBAR = [0.5, 0.47, 0.67, 0.6]


def test_cap_di_atas_tabel_dibuang(lingkungan, pdf):
    model, _ = lingkungan
    model.jawab[PROMPT_KLASIFIKASI] = ['{"jenis": "cap", "memuat_tabel_data": true}']
    model.jawab[PROMPT_TRANSKRIPSI] = [MD]
    tabel = {**tabel_el(), "metadata": {**tabel_el()["metadata"], "bbox": [0.1, 0.1, 0.9, 0.9]}}
    keluar, info = tt.proses([tabel, gambar_el(BBOX_GAMBAR)], pdf)
    assert [e["category"] for e in keluar] == ["Table"]
    assert info["ukt_p1_img00"]["perlakuan"] == "buang"
    assert info["ukt_p1_img00"]["rasio_tumpang_tabel"] == 1.0


def test_gambar_tabel_jadi_element_table(lingkungan, pdf):
    model, _ = lingkungan
    model.jawab[PROMPT_KLASIFIKASI] = ['{"jenis": "tabel", "memuat_tabel_data": true}']
    model.jawab[PROMPT_TRANSKRIPSI] = [MD]
    (el,), info = tt.proses([gambar_el(BBOX_GAMBAR)], pdf)
    m = el["metadata"]
    assert el["category"] == "Table" and el["text"] == MD
    assert m["table_origin"] == "image" and m["image_content"] == "tabel"
    assert m["image_id"] == "ukt_p1_img00" and m["raw_html"] == ""
    assert info["ukt_p1_img00"] == {"klasifikasi_jenis": "tabel", "memuat_tabel_data": True,
                                    "rasio_tumpang_tabel": 0.0, "perlakuan": "transkripsi",
                                    "image_content": "tabel"}


def test_narasi_tabel_memakai_prompt_setiap_tabel(lingkungan, pdf):
    model, _ = lingkungan
    model.jawab[PROMPT_KLASIFIKASI] = ['{"jenis": "lainnya", "memuat_tabel_data": true}']
    kedua = MD.replace("UKT", "SPP")
    model.jawab[PROMPT_TRANSKRIPSI_SEMUA] = [f"{MD}\n\nteks tombol\n\n{kedua}"]
    (el,), info = tt.proses([gambar_el(BBOX_GAMBAR)], pdf)
    assert el["category"] == "Image" and el["metadata"]["image_content"] == "narasi+tabel"
    assert el["metadata"]["transkripsi_tabel"] == f"{MD}\n\n{kedua}"
    assert info["ukt_p1_img00"]["perlakuan"] == "narasi+tabel"


def test_narasi_tabel_gagal_tetap_narasi(lingkungan, pdf):
    model, _ = lingkungan
    model.jawab[PROMPT_KLASIFIKASI] = ['{"jenis": "lainnya", "memuat_tabel_data": true}']
    model.jawab[PROMPT_TRANSKRIPSI_SEMUA] = ["tidak ada tabel"]
    (el,), info = tt.proses([gambar_el(BBOX_GAMBAR)], pdf)
    assert el["category"] == "Image" and el["metadata"]["image_content"] == "narasi"
    assert el["metadata"]["transkripsi_peringatan"] == ["gagal:tak_terurai"]
    assert "transkripsi_tabel" not in el["metadata"]


@pytest.mark.parametrize("jawaban", ['{"jenis": "lainnya", "memuat_tabel_data": false}',
                                     "entahlah", vision_io.GagalVision("x")])
def test_flowchart_atau_klasifikasi_gagal_tetap_narasi(lingkungan, pdf, jawaban):
    model, simpanan = lingkungan
    model.jawab[PROMPT_KLASIFIKASI] = [jawaban]
    (el,), info = tt.proses([gambar_el(BBOX_GAMBAR)], pdf)
    assert el["category"] == "Image" and el["metadata"]["image_content"] == "narasi"
    assert info["ukt_p1_img00"]["perlakuan"] == "narasi"
    assert len(model.panggilan) == 1


def test_gambar_bertumpang_tabel_tetap_narasi(lingkungan, pdf):
    """Keputusan 3: gambar berjenis tabel di dalam Table tidak ditranskripsi ulang."""
    model, _ = lingkungan
    model.jawab[PROMPT_KLASIFIKASI] = ['{"jenis": "tabel", "memuat_tabel_data": true}']
    model.jawab[PROMPT_TRANSKRIPSI] = [MD]
    tabel = {**tabel_el(), "metadata": {**tabel_el()["metadata"], "bbox": [0.1, 0.1, 0.9, 0.9]}}
    keluar, info = tt.proses([tabel, gambar_el(BBOX_GAMBAR)], pdf)
    assert [e["category"] for e in keluar] == ["Table", "Image"]
    assert info["ukt_p1_img00"]["perlakuan"] == "narasi"


def test_provenance_mencatat_prompt_dan_hitungan(lingkungan, pdf):
    model, _ = lingkungan
    model.jawab[PROMPT_TRANSKRIPSI] = [MD]
    tt.proses([tabel_el()], pdf)
    p = tt.provenance()
    assert set(p["prompt_sha256"]) == {tt.VARIANT_TABEL, tt.VARIANT_TABEL_SEMUA, tt.VARIANT_KLASIFIKASI}
    assert p["render_dpi"] == 72 and p["cap_min_overlap"] == config.IMAGE_CAP_MIN_OVERLAP
    assert p["hitungan"]["tabel_ditranskripsi"] == 1 and p["batas_gambar"] == BATAS


def test_halaman_pindai_tanpa_lapisan_memakai_margin(lingkungan, tmp_path):
    doc = fitz.open()
    doc.new_page()
    doc.save(tmp_path / "pindai.pdf")
    model, _ = lingkungan
    model.jawab[PROMPT_TRANSKRIPSI] = [MD]
    (el,), _ = tt.proses([tabel_el()], tmp_path / "pindai.pdf")
    assert el["metadata"]["render_bbox"] == pytest.approx(
        [0.1 - 0.025, 0.1 - 0.025, 0.72 + 0.025, 0.2 + 0.025])
    # rujukan teks OCR: 8000000 tidak ada di teks OCR -> ditandai
    assert el["metadata"]["transkripsi_peringatan"][0] == "rujukan=teks_ocr"


def test_halaman_terpilih_membatasi_pemrosesan(lingkungan, pdf):
    model, _ = lingkungan
    els = [tabel_el(), {**tabel_el(), "page": 2}]
    keluar, _ = tt.proses(els, pdf, halaman_terpilih={2})
    assert keluar == els and model.panggilan == []     # halaman 2 tidak ada di PDF 1 halaman
