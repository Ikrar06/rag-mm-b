"""Uji alat ukur transkripsi tabel. Model dipalsukan; PDF dibuat dengan fitz."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.transkripsi_ukur import (  # noqa: E402
    angka_dalam, bersihkan, cakupan_angka, celah_terbesar, dampak_serapan,
    jenis_halaman, ketepatan_angka, median, peringatan, perluas_bbox,
    rasio_tumpang, regresi_linear, sebaran_tumpang, urai_klasifikasi,
    urai_markdown,
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


def test_digit_berspasi_lapisan_ocr_disatukan():
    # bagan-akun p10: kode 426111 di lapisan OCR terbaca "4 2 6 1 1 1".
    assert angka_dalam("4 2 6 1 1 1 PENDAPATAN APBD") == {"426111"}
    # laporan-keuangan p23: 28.111.676.194 berspasi per digit.
    assert angka_dalam("2 8 1 1 1 6 7 6 1 9 4") == {"28111676194"}
    # Dua digit tunggal tidak disatukan; angka bertitik tidak disentuh.
    assert angka_dalam("kolom 1 2 dan 1.500.000 3") == {"1500000"}


class TestPeringatan:
    def test_angka_karangan_ditandai(self):
        p = peringatan(urai_markdown(MD), "UKT I 1.500.000 UKT II 2.400.001")
        assert p == ("angka_tak_ditemukan:2400000",)

    def test_hanya_baris_identik_yang_ditandai(self):
        # Label sama berturut (sel gabungan disalin) SAH, tidak ditandai.
        md = "| a | b |\n|---|---|\n| X | 1 |\n| X | 2 |\n| Y | 3 |\n| Y | 3 |"
        assert peringatan(urai_markdown(md), "") == ("baris_berturut_identik:1",)

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
        jawab = '{"jenis": "tabel"}' if "jenisnya" in prompt else f"```markdown\n{MD}\n```"
        return {"detik": 20.0, "response": jawab, "prompt_eval_count": 1500,
                "eval_count": 60, "done_reason": "stop"}

    monkeypatch.setattr(uk, "panggil", palsu)
    out = tmp / "out"
    monkeypatch.setattr(sys, "argv", [
        "x", "--chunks", str(chunks), "--pdf-dir", str(tmp), "--sampel", "3",
        "--dpi", "72,100", "--pilih", "ukt_p2_c01,tidak_ada", "--out", str(out)])
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
    assert "tidak ada di dump" in capsys.readouterr().out


def test_hanya_dampak_tanpa_pdf(korpus, monkeypatch):
    tmp, chunks = korpus
    monkeypatch.setattr(uk, "panggil", lambda *a: pytest.fail("model tidak boleh dipanggil"))
    monkeypatch.setattr(sys, "argv", ["x", "--chunks", str(chunks), "--hanya-dampak",
                                      "--out", str(tmp / "o")])
    uk.main()
    lap = json.loads((tmp / "o" / "laporan.json").read_text())
    assert lap["dampak"]["n_gambar"] == 1 and "transkripsi" not in lap


# ── klasifikasi, jenis halaman, perluasan area ──────────────────────────────

@pytest.mark.parametrize("raw,jenis", [
    ('{"jenis": "tabel"}', "tabel"), ('```json\n{"jenis": "Cap"}\n```', "cap"),
    ('{"jenis":"lainnya"}', "lainnya"), ('{"jenis": "foto"}', None), ("", None), (None, None)])
def test_urai_klasifikasi(raw, jenis):
    assert urai_klasifikasi(raw) == jenis


def test_jenis_halaman():
    assert jenis_halaman(False, 1.0) == "pindai_tanpa_lapisan"
    assert jenis_halaman(True, 0.95) == "pindai_lapisan_ocr"
    assert jenis_halaman(True, 0.3) == "digital_asli"


def test_perluas_bbox_menarik_kata_terpotong_saja():
    # ukt p5: kolom terakhir x 0,92-0,971 terpotong bbox di 0,9499.
    bbox = [0.03, 0.28, 0.9499, 0.7491]
    kata = [(0.92, 0.40, 0.971, 0.42),      # beririsan -> ditarik
            (0.975, 0.40, 0.99, 0.42),      # di luar, tidak beririsan
            (0.10, 0.80, 0.30, 0.82)]       # paragraf di bawah
    assert perluas_bbox(bbox, kata) == [0.03, 0.28, 0.971, 0.7491]
    assert perluas_bbox([0.01, 0.5, 0.99, 0.6], [], 0.025) == pytest.approx([0, 0.475, 1, 0.625])


def test_buka_area_kata_utuh_dan_lapisan_ocr(tmp_path):
    from lib.ukur_io import buka_area
    doc = fitz.open()
    hal = doc.new_page()
    hal.insert_text((400, 100), "8,000,000")
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 10, 10), 0)
    pindai = doc.new_page()
    pindai.insert_image(pindai.rect, pixmap=pix)
    pindai.insert_text((72, 100), "4 2 6 1 1 1")
    r = doc[0].rect
    doc.save(tmp_path / "a.pdf")
    # bbox berakhir di tengah angka: klip karakter memotongnya, kata utuh tidak.
    a = buka_area(tmp_path / "a.pdf", 1, [0, 0, 440 / r.width, 0.5], 36)
    assert "8,000,000" in a.teks_area and a.jenis_halaman == "digital_asli" and a.png
    b = buka_area(tmp_path / "a.pdf", 2, [0, 0, 1, 1], None)
    assert b.jenis_halaman == "pindai_lapisan_ocr" and b.png == b""


def test_perluas_lewat_cli(korpus, monkeypatch):
    tmp, _ = korpus
    r = {"file_name": "ukt.pdf", "page_number": 1, "bbox": [0.1, 0.1, 0.2, 0.2]}
    diperluas = uk.area_render(tmp / "ukt.pdf", r, True)
    assert diperluas[0] < 0.1 or diperluas[2] > 0.2
    assert uk.area_render(tmp / "ukt.pdf", r, False) == [0.1, 0.1, 0.2, 0.2]
    pindai = {"file_name": "ukt.pdf", "page_number": 2, "bbox": [0.1, 0.1, 0.2, 0.2]}
    assert uk.area_render(tmp / "ukt.pdf", pindai, True) == pytest.approx(
        [0.075, 0.075, 0.225, 0.225])


def test_hanya_dampak_menampilkan_deskripsi_rasio_tinggi(tmp_path, monkeypatch, capsys):
    rows = TestTumpang().rows()
    rows[1]["text_content"] = "logo Balai Sertifikasi Elektronik"
    c = tmp_path / "c.jsonl"
    c.write_text("\n".join(json.dumps(r) for r in rows))
    monkeypatch.setattr(sys, "argv", ["x", "--chunks", str(c), "--hanya-dampak", "--out", str(tmp_path)])
    uk.main()
    assert "Balai Sertifikasi Elektronik" in capsys.readouterr().out


import klasifikasi_gambar as kg  # noqa: E402


def test_klasifikasi_penuh_dan_lanjut(korpus, monkeypatch):
    tmp, chunks = korpus
    rows = [json.loads(l) for l in chunks.read_text().splitlines()]
    rows.append({**rows[2], "chunk_id": "ukt_p1_c05", "page_number": 1,
                 "bbox": [0.05, 0.1, 0.6, 0.3]})          # bertumpang tabel p1
    chunks.write_text("\n".join(json.dumps(r) for r in rows))
    jawab = {"ukt_p1_c05": '{"jenis": "cap"}', "ukt_p2_c01": "tidak tahu"}
    dipanggil = []

    def palsu(png, prompt, num_predict):
        dipanggil.append(num_predict)
        cid = getattr(kg, "_kini", None)
        return {"detik": 3.0, "response": jawab.get(cid, '{"jenis": "tabel"}'),
                "prompt_eval_count": 200, "eval_count": 5, "done_reason": "stop"}

    asli = kg.klasifikasi_satu

    def dengan_id(r, pdf_dir, sisi):
        kg._kini = r["chunk_id"]
        return asli(r, pdf_dir, sisi)

    monkeypatch.setattr(kg, "_kini", None, raising=False)
    monkeypatch.setattr(kg, "panggil", palsu)
    monkeypatch.setattr(kg, "klasifikasi_satu", dengan_id)
    out = tmp / "k"
    argv = ["x", "--chunks", str(chunks), "--pdf-dir", str(tmp), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv + ["--batas", "1"])
    kg.main()
    monkeypatch.setattr(sys, "argv", argv)
    kg.main()
    hasil = [json.loads(l) for l in (out / "klasifikasi.jsonl").read_text().splitlines()]
    assert [h["chunk_id"] for h in hasil] == ["ukt_p1_c05", "ukt_p2_c01"]   # tanpa ulang
    assert [h["jenis"] for h in hasil] == ["cap", None]
    ring = json.loads((out / "ringkasan.json").read_text())
    assert ring["per_jenis"] == {"cap": 1, "TAK_DIKENALI": 1}
    assert ring["cap_tanpa_tumpang_tabel"] == 0
    tinjau = (out / "klasifikasi_tinjau.csv").read_text().splitlines()
    assert tinjau[1].startswith("cap,ukt_p1_c05") and "ukt_p1_c00" in tinjau[1]
    assert (out / "gambar" / "ukt_p1_c05.png").is_file()


def test_klasifikasi_menolak_campur_prompt(tmp_path):
    with pytest.raises(SystemExit, match="prompt lain"):
        kg.sudah_selesai([{"chunk_id": "a", "prompt_sha256": "lama"}])
    assert kg.sudah_selesai([{"chunk_id": "a", "prompt_sha256": kg.PROMPT_SHA}]) == {"a"}
