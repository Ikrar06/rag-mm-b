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
    rasio_tumpang, regresi_linear, sebaran_tumpang,
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
from lib import ukur_io  # noqa: E402
from lib.klasifikasi import Klasifikasi, urai_klasifikasi  # noqa: E402

# Fungsi asli, diambil sebelum fixture autouse menimpanya.
_BATAS_ASLI = ukur_io.batas_model.__wrapped__

BATAS = {"model": "qwen3-vl:8b-instruct", "ollama": "0.13.0", "patch_size": 16,
         "spatial_merge_size": 2, "sisi_min": 32, "rasio_maks": 200}


@pytest.fixture(autouse=True)
def batas_tetap(monkeypatch):
    """Tanpa server: batas model dipatok seperti qwen3-vl (16 x 2)."""
    monkeypatch.setattr(ukur_io, "batas_model", lambda: BATAS)


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

@pytest.mark.parametrize("raw,jenis,memuat", [
    ('{"jenis": "tabel", "memuat_tabel_data": true}', "tabel", True),
    ('```json\n{"jenis": "Cap", "memuat_tabel_data": false}\n```', "cap", False),
    ('{"jenis":"lainnya","memuat_tabel_data":true}', "lainnya", True),
    ('{"jenis": "lainnya"}', "lainnya", None),
    ('{"jenis": "foto", "memuat_tabel_data": true}', None, True), ("", None, None), (None, None, None)])
def test_urai_klasifikasi(raw, jenis, memuat):
    assert urai_klasifikasi(raw) == Klasifikasi(jenis, memuat)


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
    monkeypatch.setattr(kg, "batas_model", lambda: BATAS)
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
    assert tinjau[1].startswith("buang,cap,,ukt_p1_c05") and "ukt_p1_c00" in tinjau[1]
    assert ring["per_perlakuan"] == {"buang": 1, "narasi": 1}
    assert ring["rasio_cap_bertumpang"] == [[1.0, "ukt_p1_c05"]]
    assert (out / "gambar" / "ukt_p1_c05.png").is_file()


def test_klasifikasi_menolak_campur_prompt(tmp_path):
    with pytest.raises(SystemExit, match="prompt lain"):
        kg.sudah_selesai([{"chunk_id": "a", "prompt_sha256": "lama"}], 1024)
    assert kg.sudah_selesai([{"chunk_id": "a", "prompt_sha256": kg.PROMPT_SHA,
                              "sisi": 1024}], 1024) == {"a"}


# ── batas ukuran gambar dan ketahanan panggilan ─────────────────────────────

from lib.transkripsi_ukur import rencana_ukuran  # noqa: E402


class TestRencanaUkuran:
    def test_manual_p23_tidak_dikecilkan_di_bawah_batas(self):
        # 2087x118 -> 512 px memberi tinggi 29 (< 32, Ollama 500).
        r = rencana_ukuran(2087, 118, 512, 32, 200)
        assert r.tinggi == 32 and r.lebar == 566 and not r.dipadding

    def test_gambar_kecil_ditambal_bukan_diregangkan(self):
        r = rencana_ukuran(20, 10, None, 32, 200)
        assert (r.lebar, r.tinggi, r.kanvas_lebar, r.kanvas_tinggi) == (20, 10, 32, 32)
        assert r.dipadding

    def test_rasio_ekstrem_ditambal_sampai_batas_rasio(self):
        r = rencana_ukuran(7000, 33, None, 32, 200)
        assert (r.kanvas_lebar, r.kanvas_tinggi) == (7000, 35)
        assert r.kanvas_lebar // r.kanvas_tinggi <= 200

    def test_gambar_aman_tidak_berubah(self):
        r = rencana_ukuran(800, 600, None, 32, 200)
        assert (r.kanvas_lebar, r.kanvas_tinggi) == (800, 600) and not r.dipadding
        r = rencana_ukuran(2000, 1000, 512, 32, 200)
        assert (r.lebar, r.tinggi) == (512, 256)


def _png(w, h, warna="black"):
    from PIL import Image
    import io
    buf = io.BytesIO()
    Image.new("RGB", (w, h), warna).save(buf, format="PNG")
    return buf.getvalue()


def test_siapkan_png_menambal_di_tengah_dengan_putih():
    from PIL import Image
    import io
    asli = _png(10, 10)
    assert ukur_io.siapkan_png(_png(100, 100)) == (_png(100, 100), False)
    hasil, dipadding = ukur_io.siapkan_png(asli)
    img = Image.open(io.BytesIO(hasil)).convert("RGB")
    assert dipadding and img.size == (32, 32)
    assert img.getpixel((0, 0)) == (255, 255, 255) and img.getpixel((16, 16)) == (0, 0, 0)
    kecil, _ = ukur_io.siapkan_png(_png(2087, 118), 512)
    assert Image.open(io.BytesIO(kecil)).size == (566, 32)


class TestPanggil:
    @pytest.fixture(autouse=True)
    def lingkungan(self, monkeypatch):
        import backend.config as cfg
        monkeypatch.setattr(cfg, "LLM_PROVIDER", "ollama")
        self.jeda = []
        monkeypatch.setattr(ukur_io.time, "sleep", self.jeda.append)

    def galat(self, kode):
        import httpx
        req = httpx.Request("POST", "http://x/api/generate")
        return httpx.HTTPStatusError("x", request=req, response=httpx.Response(kode, request=req))

    def test_5xx_dicoba_ulang_lalu_berhasil(self, monkeypatch):
        urutan = [self.galat(500), self.galat(503), {"response": "ok", "eval_count": 1}]

        def kirim(payload):
            x = urutan.pop(0)
            if isinstance(x, Exception):
                raise x
            return x

        monkeypatch.setattr(ukur_io, "_kirim", kirim)
        h = ukur_io.panggil(_png(10, 10), "p", 5)
        assert h["response"] == "ok" and h["percobaan"] == 3 and h["dipadding"]
        assert self.jeda == [5.0, 15.0]

    def test_tetap_gagal_setelah_tiga_ulang(self, monkeypatch):
        import httpx

        def kirim(payload):
            raise httpx.ReadTimeout("lambat")

        monkeypatch.setattr(ukur_io, "_kirim", kirim)
        with pytest.raises(ukur_io.GagalVision, match="4 percobaan; terakhir ReadTimeout"):
            ukur_io.panggil(_png(64, 64), "p", 5)
        assert self.jeda == [5.0, 15.0, 45.0]

    def test_4xx_tidak_diulang(self, monkeypatch):
        def kirim(payload):
            raise self.galat(400)

        monkeypatch.setattr(ukur_io, "_kirim", kirim)
        with pytest.raises(ukur_io.GagalVision, match="1 percobaan"):
            ukur_io.panggil(_png(64, 64), "p", 5)
        assert self.jeda == []


class TestBatasModel:
    def klien(self, monkeypatch, info):
        import httpx

        class Jawab:
            def __init__(self, d):
                self.d = d

            def raise_for_status(self):
                return self

            def json(self):
                return self.d

        class Klien:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def post(self, url, json):
                return Jawab({"model_info": info})

            def get(self, url):
                return Jawab({"version": "0.13.0"})

        monkeypatch.setattr(httpx, "Client", Klien)

    def test_dibaca_dari_model_info(self, monkeypatch):
        self.klien(monkeypatch, {"qwen3vl.vision.patch_size": 16,
                                 "qwen3vl.vision.spatial_merge_size": 2})
        b = _BATAS_ASLI()
        assert b["sisi_min"] == 32 and b["ollama"] == "0.13.0"

    def test_tanpa_kunci_berhenti(self, monkeypatch):
        self.klien(monkeypatch, {"general.architecture": "qwen3vl"})
        with pytest.raises(SystemExit, match="tidak dapat dipastikan"):
            _BATAS_ASLI()


def test_klasifikasi_gagal_dicatat_lalu_dicoba_lagi(korpus, monkeypatch):
    tmp, chunks = korpus
    percobaan = {"n": 0}

    def palsu(png, prompt, num_predict):
        if num_predict == kg.NUM_PREDICT:       # bukan panggilan pemanasan
            percobaan["n"] += 1
            if percobaan["n"] == 1:
                raise kg.GagalVision("4 percobaan; terakhir HTTPStatusError: 500")
        return {"detik": 1.0, "response": '{"jenis": "lainnya"}', "prompt_eval_count": 9,
                "eval_count": 3, "done_reason": "stop", "percobaan": 1, "dipadding": False}

    monkeypatch.setattr(kg, "panggil", palsu)
    monkeypatch.setattr(kg, "batas_model", lambda: BATAS)
    out = tmp / "k"
    monkeypatch.setattr(sys, "argv", ["x", "--chunks", str(chunks), "--pdf-dir", str(tmp),
                                      "--out", str(out)])
    kg.main()
    ring = json.loads((out / "ringkasan.json").read_text())
    assert ring["gagal_belum_terklasifikasi"] == 1 and ring["jumlah"] == 0
    assert not (out / "klasifikasi.jsonl").read_text().strip()
    kg.main()
    ring = json.loads((out / "ringkasan.json").read_text())
    assert ring["gagal_belum_terklasifikasi"] == 0 and ring["jumlah"] == 1


def test_perluas_mengabaikan_kata_raksasa():
    # academic-calendar p10: "February" vertikal setinggi ~0,4 halaman menarik
    # area render 34% ke bawah. Kata biasa tingginya ~0,015.
    biasa = [(0.1, y, 0.2, y + 0.015) for y in (0.1, 0.2, 0.3, 0.4)]
    raksasa = (0.05, 0.3, 0.12, 0.7)
    assert perluas_bbox([0.1, 0.1, 0.5, 0.35], biasa + [raksasa]) == [0.1, 0.1, 0.5, 0.35]
    assert perluas_bbox([0.1, 0.1, 0.5, 0.35], biasa) == [0.1, 0.1, 0.5, 0.35]
    assert perluas_bbox([0.1, 0.1, 0.5, 0.41], biasa)[3] == pytest.approx(0.415)


# ── klasifikasi versi 2 dan pembanding ──────────────────────────────────────

import bandingkan_klasifikasi as bk  # noqa: E402
from lib.klasifikasi import (  # noqa: E402
    PROMPT_KLASIFIKASI, Klasifikasi, perlakuan_gambar, urai_klasifikasi,
)
def test_prompt_v3_dua_jawaban():
    assert "Jika ragu, jawab lainnya." in PROMPT_KLASIFIKASI
    assert '"memuat_tabel_data"' in PROMPT_KLASIFIKASI and "APA PUN jenisnya" in PROMPT_KLASIFIKASI
    for frasa in ("tangkapan layar", "swimlane", "matriks logo", "garis kotak", "BSrE", "UKT",
                  "lajur swimlane", "tabel SWOT", "tanpa garis"):
        assert frasa in PROMPT_KLASIFIKASI


def test_resume_menolak_sisi_berbeda():
    with pytest.raises(SystemExit, match="--sisi"):
        kg.sudah_selesai([{"chunk_id": "a", "prompt_sha256": kg.PROMPT_SHA, "sisi": 512}], 1024)


def test_cocokkan_awalan_dan_persis():
    ids = ["manual_p16_c00", "manual-dosen_p16_c00", "draft-panduan-teknis-kkn-covid_p89_c01",
           bk.LK.rstrip("$") + "_p23_c00",
           "pedoman-penyusunan-laporan-keuangan-perguruan-tinggi-negeri-badan-hukum-universitas-hasanuddin_p23_c00"]
    assert bk.cocokkan(ids, "manual$", "_p16_c00") == ["manual_p16_c00"]
    assert bk.cocokkan(ids, "draft-panduan-teknis-kkn", "_p89_c01") == [ids[2]]
    assert bk.cocokkan(ids, bk.LK, "_p23_c00") == [ids[3]]
    assert len(bk.cocokkan(ids, "pedoman-penyusunan-laporan-keuangan", "_p23_c00")) == 2


def test_bandingkan_melaporkan_perpindahan_dan_harapan(tmp_path, monkeypatch, capsys):
    lama = [{"chunk_id": "ukt-tahun-2025_p5_c02", "jenis": "tabel"},
            {"chunk_id": "manual_p16_c00", "jenis": "tabel"},
            {"chunk_id": "rubrik-2024_p50_c02", "jenis": "tabel"},
            {"chunk_id": "v2-sop-evaluasi-empat-semester-3e_p6_c02", "jenis": "tabel"},
            {"chunk_id": "sop-final-project-x_p7_c01", "jenis": "tabel"}]
    baru = [{"chunk_id": "ukt-tahun-2025_p5_c02", "jenis": "cap", "memuat_tabel_data": True},
            {"chunk_id": "manual_p16_c00", "jenis": "lainnya", "memuat_tabel_data": True},
            {"chunk_id": "rubrik-2024_p50_c02", "jenis": "lainnya", "jawaban_mentah": "x"},
            {"chunk_id": "v2-sop-evaluasi-empat-semester-3e_p6_c02", "jenis": "lainnya",
             "memuat_tabel_data": True},
            {"chunk_id": "sop-final-project-x_p7_c01", "jenis": "lainnya", "memuat_tabel_data": True}]
    (tmp_path / "c").write_text(json.dumps({"chunk_id": "manual_p16_c00", "text_content": "layar  login"}))
    for nama, isi in (("l", lama), ("b", baru)):
        (tmp_path / nama).write_text("\n".join(json.dumps(x) for x in isi))
    monkeypatch.setattr(sys, "argv", ["x", "--lama", str(tmp_path / "l"), "--baru", str(tmp_path / "b"),
                                      "--out", str(tmp_path / "o.csv"), "--chunks", str(tmp_path / "c")])
    with pytest.raises(SystemExit) as e:
        bk.main()
    keluar = capsys.readouterr().out
    assert e.value.code == 1
    baris = {l.split()[-1]: l for l in keluar.splitlines() if l.startswith("  OK") or l.startswith("  MELESET")}
    assert baris["ukt-tahun-2025_p5_c02"].startswith("  OK")                 # cap+tabel diterima
    assert baris["manual_p16_c00"].startswith("  OK")                        # lainnya+tabel diterima
    assert baris["sop-final-project-x_p7_c01"].startswith("  OK")            # tabel via narasi+tabel
    assert baris["rubrik-2024_p50_c02"].startswith("  MELESET")              # tabel hilang
    assert baris["v2-sop-evaluasi-empat-semester-3e_p6_c02"].startswith("  MELESET")  # flowchart
    assert "0 kecocokan" in keluar
    csv_isi = (tmp_path / "o.csv").read_text()
    assert csv_isi.count("\n") == 6 and "tabel,lainnya+tabel,,layar login" in csv_isi


def test_perluas_tidak_menarik_kata_milik_elemen_lain():
    # bagan-akun p6: baris kode di bawah tabel tersimpan sebagai chunk teks.
    kata = [(0.2, 0.78, 0.6, 0.80)]                 # memotong tepi bawah tabel
    tabel = [0.1, 0.1, 0.9, 0.79]
    assert perluas_bbox(tabel, kata)[3] == pytest.approx(0.80)
    assert perluas_bbox(tabel, kata, milik_lain=[[0.15, 0.785, 0.7, 0.84]]) == tabel
    assert perluas_bbox(tabel, kata, milik_lain=[None, [0, 0, 0.05, 0.05]])[3] == pytest.approx(0.80)


@pytest.mark.parametrize("jenis,memuat,rasio,harap", [
    ("cap", False, 0.966, "buang"),          # cap BSrE di atas tabel UKT
    ("cap", True, 0.948, "buang"),
    ("cap", False, 0.023, "narasi"),         # pengelolaan-dana p9: tidak menutupi tabel
    ("cap", True, 0.0, "narasi+tabel"),      # cap berdiri sendiri yang memuat tabel
    ("cap", False, 0.0, "narasi"),
    ("tabel", True, 0.3, "narasi"),          # bertumpang Table: tetap narasi seperti v4
    ("lainnya", True, 0.9, "narasi"),
    ("tabel", True, 0.0, "transkripsi"),
    ("tabel", False, 0.0, "transkripsi"),
    ("lainnya", True, 0.0, "narasi+tabel"),  # tangkapan layar berisi tabel data
    ("lainnya", False, 0.0, "narasi"),       # flowchart
    ("lainnya", None, 0.0, "narasi"),
    (None, True, 0.0, "narasi"),             # jawaban tak dikenali
])
def test_perlakuan_gambar(jenis, memuat, rasio, harap):
    assert perlakuan_gambar(Klasifikasi(jenis, memuat), rasio) == harap


def test_label_klasifikasi():
    assert Klasifikasi("lainnya", True).label == "lainnya+tabel"
    assert Klasifikasi("tabel", False).label == "tabel"
    assert Klasifikasi(None, None).label == "None"


# ── penanda angka per sel (uji 10 tabel) ────────────────────────────────────

from backend.services.transkripsi_murni import ada_di_rujukan, angka_sel  # noqa: E402


def test_nomor_kolom_tidak_tersambung_jadi_angka_palsu():
    # jadwal-retensi p50: baris nomor kolom -> dulu "angka_tak_ditemukan:123457"
    baris = (("No.", "SERIES", "AKTIF", "INAKTIF", "KET", "X"), ("1", "2", "3", "4", "5", "7"))
    assert peringatan(baris, "No. SERIES/JENIS ARSIP AKTIF INAKTIF 1 2 3 4 5") == ()
    # academic-calendar p10: tanggal 1-9 -> dulu "angka_tak_ditemukan:123456789"
    kal = (("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"), ("1", "2", "3", "4", "5", "6", "7"),
           ("8", "9", "10", "11", "12", "13", "14"))
    assert peringatan(kal, "Sun Mon 1 2 3 4 5 6 7 8 9 10 11 12 13 14") == ()


def test_rujukan_ocr_berspasi_tetap_cocok():
    # bagan-akun p10: lapisan OCR "4 2 6 1 1 1"; laporan-keuangan p23 berspasi per digit
    assert peringatan((("Kode", "Uraian"), ("426111", "PENDAPATAN APBD")),
                      "4 2 6 1 1 1 PENDAPATAN APBD PROVINS!") == ()
    assert ada_di_rujukan("28111676194", "Jumlah 2 8 . 1 1 1 . 6 7 6 . 1 9 4", frozenset())
    assert not ada_di_rujukan("426111", "4261119 x", frozenset())       # bukan bagian angka lain
    assert peringatan((("A", "B"), ("426112", "x")), "4 2 6 1 1 1") == ("angka_tak_ditemukan:426112",)


def test_angka_sel_tanpa_penyatuan():
    assert angka_sel("1 2 3 dan 1.500.000") == {"1500000"}
