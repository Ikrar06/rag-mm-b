"""Uji klien vision produksi: pengaman ukuran, coba ulang, batas model."""

import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("PIL")
httpx = pytest.importorskip("httpx")

from backend.services import vision_io  # noqa: E402

BATAS = {"model": "qwen3-vl:8b-instruct", "ollama": "0.13.0", "patch_size": 16,
         "spatial_merge_size": 2, "sisi_min": 32, "rasio_maks": 200}
_BATAS_ASLI = vision_io.batas_model.__wrapped__


def _png(w, h):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "black").save(buf, format="PNG")
    return buf.getvalue()


def _ukuran(png):
    from PIL import Image
    return Image.open(io.BytesIO(png)).size


@pytest.fixture(autouse=True)
def batas_tetap(monkeypatch):
    monkeypatch.setattr(vision_io, "batas_model", lambda: BATAS)
    monkeypatch.setattr(vision_io, "_batas_tak_terbaca", None)
    jeda = []
    monkeypatch.setattr(vision_io.time, "sleep", jeda.append)
    return jeda


def test_siapkan_png_aman_tidak_diubah_dan_pipih_ditambal():
    aman = _png(100, 100)
    assert vision_io.siapkan_png(aman) == (aman, False)
    tambal, dipadding = vision_io.siapkan_png(_png(10, 10))
    assert dipadding and _ukuran(tambal) == (32, 32)
    # manual_p23_c03: 2087x118 ke 512 px berhenti di sisi pendek 32.
    assert _ukuran(vision_io.siapkan_png(_png(2087, 118), 512)[0]) == (566, 32)


def galat(kode):
    req = httpx.Request("POST", "http://x/api/generate")
    return httpx.HTTPStatusError("x", request=req, response=httpx.Response(kode, request=req))


def test_panggil_coba_ulang_5xx_lalu_berhasil(monkeypatch, batas_tetap):
    urutan = [galat(500), httpx.ReadTimeout("lambat"),
              {"response": "ok", "done_reason": "stop", "eval_count": 3}]

    def kirim(payload):
        assert payload["options"]["num_ctx"] == 16384 and payload["options"]["temperature"] == 0
        x = urutan.pop(0)
        if isinstance(x, Exception):
            raise x
        return x

    monkeypatch.setattr(vision_io, "_kirim", kirim)
    h = vision_io.panggil(_png(64, 64), "p", num_predict=9000, num_ctx=16384)
    assert h["response"] == "ok" and h["percobaan"] == 3 and batas_tetap == [5.0, 15.0]


def test_panggil_gagal_setelah_tiga_ulang_dan_4xx_langsung(monkeypatch, batas_tetap):
    monkeypatch.setattr(vision_io, "_kirim", lambda p: (_ for _ in ()).throw(galat(503)))
    with pytest.raises(vision_io.GagalVision, match="4 percobaan"):
        vision_io.panggil(_png(64, 64), "p", num_predict=1, num_ctx=8)
    assert batas_tetap == [5.0, 15.0, 45.0]
    monkeypatch.setattr(vision_io, "_kirim", lambda p: (_ for _ in ()).throw(galat(400)))
    with pytest.raises(vision_io.GagalVision, match="1 percobaan"):
        vision_io.panggil(_png(64, 64), "p", num_predict=1, num_ctx=8)


def test_panggil_respons_tanpa_response(monkeypatch):
    monkeypatch.setattr(vision_io, "_kirim", lambda p: {"error": "x"})
    with pytest.raises(vision_io.GagalVision, match="tanpa 'response'"):
        vision_io.panggil(_png(64, 64), "p", num_predict=1, num_ctx=8)


class _Klien:
    info: dict = {}

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, json):
        return _Jawab({"model_info": self.info})

    def get(self, url):
        return _Jawab({"version": "0.13.0"})


class _Jawab:
    def __init__(self, d):
        self.d = d

    def raise_for_status(self):
        return self

    def json(self):
        return self.d


def test_batas_model_dari_model_info(monkeypatch):
    import backend.services.vision_io as m
    monkeypatch.setattr(m, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(_Klien, "info", {"qwen3vl.vision.patch_size": 16,
                                         "qwen3vl.vision.spatial_merge_size": 2})
    monkeypatch.setattr(httpx, "Client", _Klien)
    assert _BATAS_ASLI()["sisi_min"] == 32
    monkeypatch.setattr(_Klien, "info", {"general.architecture": "qwen3vl"})
    with pytest.raises(RuntimeError, match="tidak dapat dipastikan"):
        _BATAS_ASLI()
    monkeypatch.setattr(m, "LLM_PROVIDER", "vllm")
    with pytest.raises(RuntimeError, match="hanya untuk ollama"):
        _BATAS_ASLI()


def test_amankan_melewati_sekali_bila_batas_tak_terbaca(monkeypatch):
    panggilan = []

    def gagal():
        panggilan.append(1)
        raise RuntimeError("tak terjangkau")

    monkeypatch.setattr(vision_io, "batas_model", gagal)
    kecil = _png(10, 10)
    assert vision_io.amankan(kecil) == kecil
    assert vision_io.amankan(kecil) == kecil
    assert panggilan == [1]


def test_amankan_menambal_bila_batas_terbaca():
    assert _ukuran(vision_io.amankan(_png(10, 40))) == (32, 40)
