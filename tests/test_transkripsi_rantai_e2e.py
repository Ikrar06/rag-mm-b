"""Uji rantai tabel lanjutan di atas hasil transkripsi, lewat _chunk_elements.

Yang dibuktikan:
  1. Header rantai berasal dari TRANSKRIPSI kepala, dan MENGGANTI baris header
     kosong potongan B (aturan 6 prompt) — satu baris header, bukan dua.
  2. Baris pertama B yang terisi data diturunkan jadi baris data.
  3. Jumlah kolom berbeda: tidak disisipkan, table_header_repeated False.
  4. Sidik tetap dari raw_html OCR: berkas keputusan lama tetap berlaku.
  5. Gambar-tabel di antara A dan B tidak memutus pasangan A->B dan tidak
     masuk rantai.
  6. Kunci Tahap T sampai ke chunk; baris transkripsi internal tidak.
  7. Chunk narasi+tabel: narasi dulu, lalu tabel.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import test_table_continuation_e2e as e2e  # noqa: E402
from backend.services.transkripsi_murni import markdown_dari_baris, urai_markdown  # noqa: E402

HDR = ("Persyaratan", "S1", "S2")


def tr_el(rows_ocr, rows_tr, page):
    """Element Table yang sudah ditranskripsi: raw_html OCR, teks = transkripsi."""
    md = markdown_dari_baris(rows_tr)
    el = e2e.tabel_el(rows_ocr, page)
    return {**el, "text": md, "metadata": {
        **el["metadata"], "teks_ocr": el["text"], "table_source": "vision_transcription",
        "render_bbox": [0.1, 0.1, 0.9, 0.9], "transkripsi_peringatan": [],
        "transkripsi_baris": urai_markdown(md)}}


def els_tr(b_tr, sisipan=()):
    a_tr = (HDR, ("IPK minimum", "2,75", "3,00"))
    return [{"category": "Title", "text": e2e.SECTION, "page": 4, "metadata": {}},
            tr_el(e2e.A_ROWS, a_tr, 4), *sisipan, tr_el(e2e.B_ROWS, b_tr, 5)]


def dua_kali(monkeypatch, tmp_path, els):
    tanpa = e2e.jalankan(monkeypatch, True, els, tmp_path=tmp_path)
    ocr = [c for c in tanpa if c.get("table_origin") != "image"]
    dengan = e2e.jalankan(monkeypatch, True, els, e2e.setujui(ocr, [(0, 1)]), tmp_path)
    return tanpa, dengan


@pytest.mark.integration
def test_header_rantai_mengganti_header_kosong(monkeypatch, tmp_path):
    b_tr = (("", "", ""), ("Masa studi", "14", "8"), ("TOEFL", "450", "500"))
    tanpa, dengan = dua_kali(monkeypatch, tmp_path, els_tr(b_tr))
    b = dengan[1]["text"]
    assert b.startswith(f"## {e2e.SECTION}\n\n| Persyaratan | S1 | S2 |\n| --- | --- | --- |\n"
                        "| Masa studi | 14 | 8 |"), b
    assert sum(l.startswith("| ---") for l in b.splitlines()) == 1 and "|  |  |  |" not in b
    assert dengan[1]["table_header_repeated"] is True and dengan[1]["table_part"] == 1
    assert dengan[1]["raw_html"] == tanpa[1]["raw_html"]


@pytest.mark.integration
def test_baris_pertama_terisi_diturunkan_jadi_data(monkeypatch, tmp_path):
    b_tr = (("Masa studi", "14", "8"), ("TOEFL", "450", "500"))
    _, dengan = dua_kali(monkeypatch, tmp_path, els_tr(b_tr))
    baris = urai_markdown(dengan[1]["text"].split("\n\n", 1)[1])
    assert baris == (HDR, ("Masa studi", "14", "8"), ("TOEFL", "450", "500"))


@pytest.mark.integration
def test_jumlah_kolom_berbeda_tidak_disisipkan(monkeypatch, tmp_path):
    b_tr = (("", ""), ("Masa studi", "14"), ("TOEFL", "450"))
    _, dengan = dua_kali(monkeypatch, tmp_path, els_tr(b_tr))
    assert "| Persyaratan |" not in dengan[1]["text"]
    assert dengan[1]["table_header_repeated"] is False
    assert dengan[1]["table_group_id"] == dengan[0]["chunk_id"], "tautan tetap diisi"


@pytest.mark.integration
def test_gambar_tabel_tidak_memutus_pasangan(monkeypatch, tmp_path):
    gambar_tabel = {"category": "Table", "text": "| x | y |\n|---|---|\n| 1 | 2 |", "page": 5,
                    "metadata": {"raw_html": "", "table_format": "markdown",
                                 "bbox": [0.1, 0.0, 0.9, 0.05], "image_id": "sop_p5_img00",
                                 "table_origin": "image", "image_content": "tabel",
                                 "table_source": "vision_transcription",
                                 "transkripsi_baris": (("x", "y"), ("1", "2"))}}
    b_tr = (("", "", ""), ("Masa studi", "14", "8"))
    _, dengan = dua_kali(monkeypatch, tmp_path, els_tr(b_tr, sisipan=[gambar_tabel]))
    a, g, b = dengan
    assert g["table_origin"] == "image" and "table_group_id" not in g
    assert g["image_id"] == "sop_p5_img00" and g["image_content"] == "tabel"
    assert b["table_group_id"] == a["chunk_id"] and b["table_header_repeated"] is True


@pytest.mark.integration
def test_kunci_tahap_t_sampai_ke_chunk(monkeypatch, tmp_path):
    b_tr = (("", "", ""), ("Masa studi", "14", "8"))
    t = e2e.jalankan(monkeypatch, True, els_tr(b_tr), tmp_path=tmp_path)
    assert t[0]["table_source"] == "vision_transcription"
    assert t[0]["teks_ocr"].startswith("| Persyaratan") and t[0]["render_bbox"]
    assert all("transkripsi_baris" not in c for c in t)


@pytest.mark.integration
def test_chunk_narasi_tabel(monkeypatch, tmp_path):
    import importlib
    e2e.jalankan(monkeypatch, True, [], tmp_path=tmp_path)     # muat ulang config & modul
    import backend.services.preprocessing as pre
    importlib.reload(pre)
    el = {"category": "ImageDescription", "text": "Tangkapan layar menu nilai.", "page": 3,
          "metadata": {"bbox": [0, 0, 1, 1], "image_id": "m_p3_img00",
                       "image_content": "narasi+tabel",
                       "transkripsi_tabel": "| a | b |\n|---|---|\n| 1 | 2 |",
                       "transkripsi_peringatan": []}}
    (c,) = pre._chunk_elements([el], "m.pdf", document_id="m")
    assert c["text"] == ("[Deskripsi Gambar] Tangkapan layar menu nilai.\n\n"
                         "[Tabel dalam Gambar]\n| a | b |\n|---|---|\n| 1 | 2 |")
    assert c["image_content"] == "narasi+tabel" and c["image_id"] == "m_p3_img00"
