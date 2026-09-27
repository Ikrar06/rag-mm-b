"""Uji alat ukur transkripsi tabel. Model dipalsukan; PDF dibuat dengan fitz."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.transkripsi_ukur import (  # noqa: E402
    angka_dalam, bersihkan, cakupan_angka, celah_terbesar, dampak_serapan,
    ketepatan_angka, median, peringatan, rasio_tumpang, regresi_linear,
    sebaran_tumpang, urai_markdown,
)

MD = "| No | Uraian | Biaya |\n|---|---|---|\n| 1 | UKT I | 1.500.000 |\n| 2 | UKT II | 2.400.000 |"


class TestBersihkan:
    def test_pagar_dan_teks_luar_dibuang(self):
        raw = f"Berikut tabelnya:\n```markdown\n{MD}\n```\nSemoga membantu."
        assert bersihkan(raw) == MD

    def test_blok_terpanjang_dipilih(self):
        raw = "| a | b |\n\nteks\n" + MD
        assert bersihkan(raw) == MD

    def test_kosong(self):
        assert bersihkan(None) == "" and bersihkan("tidak ada tabel") == ""


class TestUraiMarkdown:
    def test_tabel_sah(self):
        b = urai_markdown(MD)
        assert b[0] == ("No", "Uraian", "Biaya") and len(b) == 3

    def test_pipa_ter_escape_tetap_satu_sel(self):
        b = urai_markdown("| a | b |\n|---|---|\n| x \\| y | z |")
        assert b[1] == ("x | y", "z")

    def test_header_kosong_aturan_6(self):
        b = urai_markdown("|  |  |\n|---|---|\n| 9 | lanjutan |")
        assert b[0] == ("", "")

    @pytest.mark.parametrize("md", ["", "| a | b |", "| a |\n|---|\n| 1 |",
                                    "| a | b |\n| 1 | 2 |\n| 3 | 4 |"])
    def test_bukan_tabel(self, md):
        assert urai_markdown(md) is None


def test_angka_dinormalkan():
    assert angka_dalam("Rp1.500.000 dan 2,5 serta 7") == {"1500000", "25"}


class TestPeringatan:
    def test_angka_karangan_ditandai(self):
        p = peringatan(urai_markdown(MD), "UKT I 1.500.000 UKT II 2.400.001")
        assert p == ("angka_tak_ditemukan:2400000",)

    def test_label_dan_baris_berulang(self):
        md = "| a | b |\n|---|---|\n| X | 1 |\n| X | 2 |\n| Y | 3 |\n| Y | 3 |"
        p = peringatan(urai_markdown(md), "")
        assert "label_berturut_sama:2" in p and "baris_berturut_identik:1" in p

    def test_bersih(self):
        assert peringatan(urai_markdown(MD), "1.500.000 2.400.000") == ()
        assert peringatan(None, "") == ()


def test_cakupan_dan_ketepatan():
    assert cakupan_angka(MD, "1.500.000 2.400.000 9.999") == pytest.approx(2 / 3)
    assert cakupan_angka(MD, "tanpa angka") is None
    assert ketepatan_angka(MD, "1500000") == pytest.approx(1 / 2)
    assert ketepatan_angka("| a | b |", "x") is None


class TestTumpang:
    def test_rasio(self):
        assert rasio_tumpang([0.1, 0.1, 0.2, 0.2], [0, 0, 1, 1]) == pytest.approx(1.0)
        assert rasio_tumpang([0, 0, 0.2, 0.2], [0.1, 0, 1, 1]) == pytest.approx(0.5)
        assert rasio_tumpang([0, 0, 0.1, 0.1], [0.5, 0.5, 1, 1]) == 0.0
        assert rasio_tumpang(None, [0, 0, 1, 1]) == 0.0
        assert rasio_tumpang([0, 0, 0, 1], [0, 0, 1, 1]) == 0.0

    def rows(self):
        return [
            {"chunk_id": "ukt_p5_c00", "document_id": "ukt", "page_number": 5,
             "element_type": "Table", "bbox": [0, 0.2, 1, 0.9]},
            {"chunk_id": "ukt_p5_c01", "document_id": "ukt", "page_number": 5,
             "element_type": "ImageDescription", "bbox": [0.7, 0.8, 0.8, 0.85]},
            {"chunk_id": "ukt_p5_c02", "document_id": "ukt", "page_number": 5,
             "element_type": "NarrativeText", "bbox": [0, 0.9, 1, 1]},
            {"chunk_id": "ukt_p6_c00", "document_id": "ukt", "page_number": 6,
             "element_type": "ImageDescription", "bbox": [0, 0, 0.1, 0.1]},
            {"chunk_id": "lain_p5_c00", "document_id": "lain", "page_number": 5,
             "element_type": "ImageDescription", "bbox": [0.1, 0.3, 0.2, 0.4]},
        ]

    def test_hanya_tabel_sehalaman_sedokumen(self):
        t = sebaran_tumpang(self.rows())
        assert [(x.image_chunk, x.table_chunk, x.rasio) for x in t] == [
            ("ukt_p5_c01", "ukt_p5_c00", pytest.approx(1.0))]

    def test_dampak_menghitung_chunk_setelahnya(self):
        rows = self.rows()
        d = dampak_serapan(rows, sebaran_tumpang(rows), 0.8)
        assert d == {"ambang": 0.8, "gambar_diserap": 1,
                     "halaman_terdampak": 1, "chunk_bergeser": 1}
        assert dampak_serapan(rows, sebaran_tumpang(rows), 1.01)["gambar_diserap"] == 0


def test_celah_regresi_median():
    assert celah_terbesar([0.1, 0.15, 0.9, 1.0]) == (0.15, 0.9)
    assert celah_terbesar([0.5]) is None
    a, b = regresi_linear([0, 10, 20], [5, 15, 25])
    assert (a, b) == (pytest.approx(5), pytest.approx(1))
    assert regresi_linear([1, 1], [2, 3]) is None and regresi_linear([1], [1]) is None
    assert median([3, 1, 2]) == 2 and median([1, 2, 3, 4]) == 2.5 and median([]) is None


# ── skrip CLI dengan model palsu ────────────────────────────────────────────

fitz = pytest.importorskip("fitz")
import ukur_transkripsi as uk  # noqa: E402


@pytest.fixture
def korpus(tmp_path):
    doc = fitz.open()
    hal = doc.new_page()
    hal.insert_text((72, 100), "No Uraian Biaya")
    hal.insert_text((72, 120), "1 UKT I 1.500.000")
    hal.insert_text((72, 140), "2 UKT II 2.400.000")
    doc.new_page()      # halaman 2 tanpa lapisan teks = "pindai"
    doc.save(tmp_path / "ukt.pdf")
    rows = [
        {"chunk_id": "ukt_p1_c00", "document_id": "ukt", "page_number": 1, "file_name": "ukt.pdf",
         "element_type": "Table", "bbox": [0, 0, 1, 0.5], "text_content": "No Uraian Biaya " * 5},
        {"chunk_id": "ukt_p2_c00", "document_id": "ukt", "page_number": 2, "file_name": "ukt.pdf",
         "element_type": "Table", "bbox": [0, 0, 1, 0.5], "text_content": "x" * 30, "table_part": 1},
        {"chunk_id": "ukt_p2_c01", "document_id": "ukt", "page_number": 2, "file_name": "ukt.pdf",
         "element_type": "ImageDescription", "bbox": [0, 0.6, 0.5, 0.9], "text_content": "gambar"},
    ]
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("\n".join(json.dumps(r) for r in rows))
    return tmp_path, chunks


def test_alur_penuh_model_palsu(korpus, monkeypatch, capsys):
    tmp, chunks = korpus
    panggilan = []

    def palsu(png, prompt, num_predict):
        panggilan.append((prompt[:12], num_predict))
        jawab = '{"tabel": true}' if "Apakah" in prompt else f"```markdown\n{MD}\n```"
        return {"detik": 20.0, "response": jawab, "prompt_eval_count": 1500,
                "eval_count": 60, "done_reason": "stop"}

    monkeypatch.setattr(uk, "panggil", palsu)
    out = tmp / "out"
    monkeypatch.setattr(sys, "argv", [
        "x", "--chunks", str(chunks), "--pdf-dir", str(tmp), "--sampel", "3",
        "--dpi", "72,100", "--pilih", "ukt_p2_c01,tidak_ada", "--klasifikasi", "1",
        "--out", str(out)])
    uk.main()
    lap = json.loads((out / "laporan.json").read_text())
    recs = lap["transkripsi"]
    assert {(r["chunk_id"], r["dpi"]) for r in recs} == {
        (c, d) for c in ("ukt_p1_c00", "ukt_p2_c00", "ukt_p2_c01") for d in (72, 100)}
    digital = next(r for r in recs if r["chunk_id"] == "ukt_p1_c00")
    assert digital["digital"] and digital["cakupan_angka"] == 1.0 and digital["peringatan"] == []
    pindai = next(r for r in recs if r["chunk_id"] == "ukt_p2_c00")
    assert not pindai["digital"] and pindai["cakupan_angka"] is None
    assert pindai["peringatan"] == ["angka_tak_ditemukan:1500000,2400000"]
    assert (out / "ukt_p1_c00_72dpi.md").read_text().count("## Teks v4 (OCR)") == 1
    ring = lap["ringkas"]["per_dpi"]["72"]
    assert ring["median_detik"] == 20.0 and ring["terpotong"] == 0
    assert lap["klasifikasi"]["jumlah_gambar_dideskripsi"] == 1
    assert "tidak ada di dump" in capsys.readouterr().out


def test_hanya_dampak_tanpa_pdf(korpus, monkeypatch):
    tmp, chunks = korpus
    monkeypatch.setattr(uk, "panggil", lambda *a: pytest.fail("model tidak boleh dipanggil"))
    monkeypatch.setattr(sys, "argv", ["x", "--chunks", str(chunks), "--hanya-dampak",
                                      "--out", str(tmp / "o")])
    uk.main()
    lap = json.loads((tmp / "o" / "laporan.json").read_text())
    assert lap["dampak"]["n_gambar"] == 1 and "transkripsi" not in lap
