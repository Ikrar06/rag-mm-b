"""Uji adjudikasi vision untuk pasangan tabel ambigu.

Fokus pada penguraian jawaban dan penanganan kegagalan: kegagalan transient
tidak boleh membeku jadi putusan permanen, dan "model tidak menjawab" tidak
boleh terbaca sebagai "tabel berbeda".
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services import vision_cache  # noqa: E402
from backend.services.table_adjudicator import (  # noqa: E402
    PROMPT, PROMPT_SHA, Putusan, adjudikasi, parse_jawaban, susun,
)


# ─── parse_jawaban ───────────────────────────────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("putusan,harapan", [
    ("LANJUTAN", vision_cache.VERDICT_LANJUTAN),
    ("BERBEDA", vision_cache.VERDICT_BUKAN_LANJUTAN),
    ("TIDAK_JELAS", vision_cache.VERDICT_TIDAK_JELAS),
])
def test_ketiga_putusan_terurai(putusan, harapan):
    raw = f'{{"putusan": "{putusan}", "keyakinan": "tinggi", "alasan": "kolom sama"}}'
    hasil = parse_jawaban(raw)
    assert hasil.verdict == harapan and hasil.keyakinan == "tinggi"
    assert hasil.alasan == "kolom sama" and not hasil.error


@pytest.mark.unit
def test_json_dibungkus_teks_dan_pagar_kode():
    """Model kerap menambah basa-basi atau pagar kode di sekitar JSON."""
    raw = ('Berikut analisisnya:\n```json\n'
           '{"putusan": "LANJUTAN", "keyakinan": "sedang", "alasan": "tanpa header"}\n'
           '```\nSemoga membantu.')
    assert parse_jawaban(raw).verdict == vision_cache.VERDICT_LANJUTAN


@pytest.mark.unit
def test_putusan_huruf_kecil_tetap_dikenali():
    assert parse_jawaban('{"putusan": "lanjutan"}').verdict == vision_cache.VERDICT_LANJUTAN


@pytest.mark.unit
@pytest.mark.parametrize("raw", ["", "   ", None])
def test_jawaban_kosong_bukan_putusan(raw):
    h = parse_jawaban(raw)
    assert h.verdict is None and h.error


@pytest.mark.unit
@pytest.mark.parametrize("raw", [
    "tidak ada json di sini",
    '{"putusan": "MUNGKIN"}',
    '{"putusan": }',
    '["bukan", "object"]',
])
def test_jawaban_tidak_sah_jadi_error_bukan_bukan_lanjutan(raw):
    """'Tidak dapat diurai' HARUS berbeda dari 'tabel berbeda'."""
    h = parse_jawaban(raw)
    assert h.verdict is None and h.error
    assert h.verdict != vision_cache.VERDICT_BUKAN_LANJUTAN


@pytest.mark.unit
def test_alasan_dipangkas_dan_dirapatkan():
    raw = '{"putusan": "BERBEDA", "alasan": "' + ("a b  c\n" * 60) + '"}'
    h = parse_jawaban(raw)
    assert len(h.alasan) <= 200 and "\n" not in h.alasan


@pytest.mark.unit
def test_putusan_immutable_dan_properti_lanjutan():
    h = parse_jawaban('{"putusan": "LANJUTAN"}')
    assert h.lanjutan
    assert not parse_jawaban('{"putusan": "BERBEDA"}').lanjutan
    assert not Putusan().lanjutan
    with pytest.raises(Exception):
        h.verdict = "x"


@pytest.mark.unit
def test_prompt_sha_stabil_dan_cocok_dengan_prompt():
    import hashlib
    assert PROMPT_SHA == hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()


@pytest.mark.unit
def test_verdict_adjudikasi_boleh_di_cache():
    """Kalau tidak, put() menolaknya diam-diam dan cache selalu kosong."""
    for v in (vision_cache.VERDICT_LANJUTAN, vision_cache.VERDICT_BUKAN_LANJUTAN,
              vision_cache.VERDICT_TIDAK_JELAS):
        assert v in vision_cache.CACHEABLE_VERDICTS


# ─── susun ───────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_susun_menumpuk_dua_gambar():
    Image = pytest.importorskip("PIL.Image")
    import io

    def png(w, h, warna):
        b = io.BytesIO()
        Image.new("RGB", (w, h), warna).save(b, format="PNG")
        return b.getvalue()

    hasil = susun(png(200, 60, "white"), png(300, 40, "white"))
    assert hasil is not None
    img = Image.open(io.BytesIO(hasil))
    assert img.width == 300            # disamakan ke yang terlebar
    assert img.height > 60 + 40        # ada pita pemisah


@pytest.mark.integration
def test_susun_gambar_rusak_mengembalikan_none():
    pytest.importorskip("PIL.Image")
    assert susun(b"bukan png", b"juga bukan") is None


# ─── adjudikasi: jalur kegagalan ─────────────────────────────────────────────

@pytest.mark.integration
def test_render_gagal_tidak_memanggil_model(tmp_path, monkeypatch):
    """PDF tidak ada: harus error, dan model tidak boleh disentuh."""
    pytest.importorskip("fitz")
    dipanggil = []
    monkeypatch.setattr("backend.services.table_adjudicator._tanya_model",
                        lambda png: dipanggil.append(1))
    h = adjudikasi(tmp_path / "hilang.pdf", 1, [0.1, 0.1, 0.9, 0.5],
                   2, [0.1, 0.1, 0.9, 0.5])
    assert h.verdict is None and "render" in h.error
    assert not dipanggil


@pytest.mark.integration
def test_model_melempar_tidak_merambat(tmp_path, monkeypatch):
    fitz = pytest.importorskip("fitz")
    pytest.importorskip("PIL.Image")
    path = tmp_path / "a.pdf"
    doc = fitz.open()
    for _ in range(2):
        doc.new_page().insert_text((72, 100), "Tabel persyaratan")
    doc.save(str(path)); doc.close()

    def meledak(png):
        raise RuntimeError("koneksi putus")

    monkeypatch.setattr("backend.services.table_adjudicator._tanya_model", meledak)
    monkeypatch.setattr(vision_cache, "enabled", lambda: False)
    h = adjudikasi(path, 1, [0.05, 0.05, 0.95, 0.4], 2, [0.05, 0.05, 0.95, 0.4])
    assert h.verdict is None and "RuntimeError" in h.error


@pytest.mark.integration
def test_putusan_sah_diurai_dari_jalur_penuh(tmp_path, monkeypatch):
    fitz = pytest.importorskip("fitz")
    pytest.importorskip("PIL.Image")
    path = tmp_path / "a.pdf"
    doc = fitz.open()
    for _ in range(2):
        doc.new_page().insert_text((72, 100), "Tabel persyaratan")
    doc.save(str(path)); doc.close()

    monkeypatch.setattr(
        "backend.services.table_adjudicator._tanya_model",
        lambda png: '{"putusan": "LANJUTAN", "keyakinan": "tinggi", "alasan": "lebar kolom sama"}',
    )
    monkeypatch.setattr(vision_cache, "enabled", lambda: False)
    h = adjudikasi(path, 1, [0.05, 0.05, 0.95, 0.4], 2, [0.05, 0.05, 0.95, 0.4])
    assert h.lanjutan and h.alasan == "lebar kolom sama" and not h.dari_cache


# ─── keandalan alasan model ──────────────────────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("alasan,dikirim,harapan", [
    # Kasus nyata dari server: dikirim halaman 38-39, model menulis 34.
    ("Tabel di gambar kedua (halaman 34) melanjutkan yang pertama", (38, 39), (34,)),
    ("Tabel di halaman 39 melanjutkan halaman 38", (38, 39), ()),
    ("The table on page 34 continues", (38, 39), (34,)),
    ("Kolom sama dan tanpa header", (38, 39), ()),
    ("hal. 12 dan hal 99 disebut", (38, 39), (12, 99)),
    ("", (38, 39), ()),
    (None, (38, 39), ()),
])
def test_halaman_dikarang_terdeteksi(alasan, dikirim, harapan):
    from backend.services.table_adjudicator import halaman_dikarang
    assert halaman_dikarang(alasan, dikirim) == harapan


@pytest.mark.unit
def test_angka_bukan_halaman_tidak_salah_tangkap():
    """Nominal rupiah dan angka tabel tidak boleh dibaca sebagai nomor halaman."""
    from backend.services.table_adjudicator import halaman_dikarang
    assert halaman_dikarang("Nilai 1.250.000 dan 2.500.000 cocok", (38, 39)) == ()


@pytest.mark.unit
def test_timeout_cukup_untuk_pengukuran_server():
    """90 detik menjamin kegagalan: 72 dpi halaman penuh terukur 110,6 detik."""
    from backend.services.table_adjudicator import TIMEOUT_DETIK
    assert TIMEOUT_DETIK >= 600.0


@pytest.mark.unit
def test_dpi_default_yang_terukur_paling_murah():
    from backend.services.table_adjudicator import RENDER_DPI
    assert RENDER_DPI == 72


@pytest.mark.unit
def test_provenance_mencatat_dpi_dan_timeout():
    """DPI mengubah biaya DAN berpotensi mengubah putusan, jadi ia bagian
    konfigurasi eksperimen yang wajib tercatat."""
    from backend.services.table_adjudicator import RENDER_DPI, provenance
    p = provenance()
    assert p["render_dpi"] == RENDER_DPI
    assert p["timeout_detik"] >= 600.0
    assert p["variant"] == "table_continuation"
    assert p["prompt_sha256"]


@pytest.mark.integration
def test_render_memotong_ke_bbox_bukan_halaman_penuh(tmp_path):
    """Pembuktian bahwa _render TIDAK mengirim halaman penuh."""
    fitz = pytest.importorskip("fitz")
    pytest.importorskip("PIL.Image")
    import io as _io

    from PIL import Image

    from backend.services.table_adjudicator import RENDER_DPI, _render

    path = tmp_path / "a.pdf"
    doc = fitz.open()
    pg = doc.new_page()                       # A4: 595x842 pt
    for i in range(10):
        pg.insert_text((60, 520 + i * 14), f"| Uraian {i} | 1.250.000 |")
    doc.save(str(path)); doc.close()

    penuh = Image.open(_io.BytesIO(_render(path, 1, [0.0, 0.0, 1.0, 1.0])))
    potong = Image.open(_io.BytesIO(_render(path, 1, [0.08, 0.57, 0.92, 0.80])))

    assert penuh.height == pytest.approx(842 * RENDER_DPI / 72, rel=0.02)
    assert potong.height < penuh.height * 0.3
    assert potong.width * potong.height < penuh.width * penuh.height * 0.25


@pytest.mark.integration
def test_putusan_membawa_ukuran_gambar(tmp_path, monkeypatch):
    fitz = pytest.importorskip("fitz")
    pytest.importorskip("PIL.Image")
    from backend.services import vision_cache as vc
    from backend.services.table_adjudicator import adjudikasi

    path = tmp_path / "a.pdf"
    doc = fitz.open()
    for _ in range(2):
        doc.new_page().insert_text((72, 100), "Tabel")
    doc.save(str(path)); doc.close()

    monkeypatch.setattr(
        "backend.services.table_adjudicator._tanya_model",
        lambda png: '{"putusan": "LANJUTAN", "alasan": "lihat halaman 34"}',
    )
    monkeypatch.setattr(vc, "enabled", lambda: False)
    # PDF fixture hanya 2 halaman; model tetap menyebut "halaman 34".
    h = adjudikasi(path, 1, [0.1, 0.5, 0.9, 0.8], 2, [0.1, 0.1, 0.9, 0.4])
    assert h.lanjutan
    assert h.piksel_dikirim > 0
    assert h.halaman_dikarang == (34,)
