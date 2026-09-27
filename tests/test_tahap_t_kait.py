"""Uji kait Tahap T di luar orkestrasi: gerbang indexing, deskripsi gambar,
kolom dump, dan konsistensi .env.research dengan daftar flag beku."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("llama_index.core")


def _stub_qdrant() -> None:
    """indexing mengimpor klien Qdrant di tingkat modul; gerbang yang diuji di
    sini tidak memakainya. Stub hanya dipasang bila paket aslinya tidak ada."""
    import importlib.util
    import types
    if importlib.util.find_spec("qdrant_client") is not None:
        return
    kelas = type("Stub", (), {"__init__": lambda self, *a, **k: None})
    for nama, atribut in (
        ("llama_index.vector_stores", ()),
        ("llama_index.vector_stores.qdrant", ("QdrantVectorStore",)),
        ("qdrant_client", ("QdrantClient",)),
        ("qdrant_client.models", ("Filter", "FieldCondition", "MatchValue", "Distance",
                                  "VectorParams")),
    ):
        mod = sys.modules.setdefault(nama, types.ModuleType(nama))
        for a in atribut:
            setattr(mod, a, kelas)


_stub_qdrant()

from backend import config  # noqa: E402
from backend.services import chunk_dump, indexing, vision_io  # noqa: E402
from backend.services import preprocessing as pre  # noqa: E402


@pytest.fixture
def flag_t(monkeypatch):
    for nama, nilai in (("INDEX_TABLE_TRANSCRIPTION", True), ("INDEX_IMAGE_TABLE_TRANSCRIPTION", True),
                        ("PDF_EXTRACTION_STRATEGY", "hi_res"), ("INDEX_PERSIST_IMAGES", True)):
        monkeypatch.setattr(config, nama, nilai)
    monkeypatch.setattr(vision_io, "batas_model", lambda: {
        "model": "m", "ollama": "0.13.0", "patch_size": 16, "spatial_merge_size": 2,
        "sisi_min": 32, "rasio_maks": 200})


def test_gerbang_mati_tanpa_flag(monkeypatch):
    monkeypatch.setattr(config, "INDEX_TABLE_TRANSCRIPTION", False)
    monkeypatch.setattr(config, "INDEX_IMAGE_TABLE_TRANSCRIPTION", False)
    monkeypatch.setattr(vision_io, "batas_model", lambda: pytest.fail("tidak boleh dipanggil"))
    indexing._check_table_transcription()


def test_gerbang_mencetak_batas_dan_prompt(flag_t, capsys):
    indexing._check_table_transcription()
    keluar = capsys.readouterr().out
    assert "sisi_min=32" in keluar and "image_class_v3" in keluar and "table_transcription_all" in keluar


@pytest.mark.parametrize("nama,nilai,pesan", [
    ("PDF_EXTRACTION_STRATEGY", "fast", "hi_res"),
    ("INDEX_PERSIST_IMAGES", False, "INDEX_PERSIST_IMAGES"),
    ("TABLE_TRANSCRIPTION_NUM_CTX", 100, "NUM_CTX"),
])
def test_gerbang_menolak_prasyarat(flag_t, monkeypatch, nama, nilai, pesan):
    monkeypatch.setattr(config, nama, nilai)
    with pytest.raises(ValueError, match=pesan):
        indexing._check_table_transcription()


def test_gerbang_gagal_bila_model_tak_terbaca(flag_t, monkeypatch):
    def gagal():
        raise RuntimeError("model_info tanpa vision.patch_size")
    monkeypatch.setattr(vision_io, "batas_model", gagal)
    with pytest.raises(RuntimeError, match="patch_size"):
        indexing._check_table_transcription()


def _gambar(**meta):
    return {"category": "Image", "text": "", "page": 2,
            "metadata": {"image_base64": "aGFsbw==", "bbox": [0, 0, 1, 1],
                         "image_id": "d_p2_img00", **meta}}


def test_deskripsi_meneruskan_metadata_tahap_t(monkeypatch):
    import backend.services.image_describer as idesc
    monkeypatch.setattr(pre, "PDF_DESCRIBE_IMAGES", "true")
    monkeypatch.setattr(idesc, "describe_image", lambda b: "Tangkapan layar.")
    (el,) = pre._describe_image_elements([_gambar(image_content="narasi+tabel",
                                                  transkripsi_tabel="| a | b |")])
    assert el["text"] == "Tangkapan layar."
    assert el["metadata"]["image_content"] == "narasi+tabel"
    assert el["metadata"]["transkripsi_tabel"] == "| a | b |"
    assert "image_base64" not in el["metadata"]


def test_narasi_gagal_tabel_tetap_dibawa(monkeypatch):
    import backend.services.image_describer as idesc
    monkeypatch.setattr(pre, "PDF_DESCRIBE_IMAGES", "true")
    monkeypatch.setattr(idesc, "describe_image", lambda b: None)
    (el,) = pre._describe_image_elements([_gambar(image_content="narasi+tabel",
                                                  transkripsi_tabel="| a | b |")])
    assert el["text"] == "" and el["metadata"]["transkripsi_peringatan"] == ["gagal:narasi"]
    assert pre._describe_image_elements([_gambar(image_content="narasi")]) == []


def test_dump_mencatat_kolom_tahap_t():
    class Doc:
        text = "| a | b |"
        metadata = {"chunk_id": "d_p1_c00", "element_type": "Table", "page": 1,
                    "raw_html": "<table/>", "teks_ocr": "a b", "table_source": "vision_transcription",
                    "table_origin": "image", "image_content": "tabel", "render_bbox": [0, 0, 1, 1],
                    "transkripsi_peringatan": ["rujukan=lapisan_ocr", "angka_tak_ditemukan:12"]}
    rec = chunk_dump._record(Doc())
    assert rec["text_content"] == "| a | b |" and rec["text_as_html"] == "<table/>"
    assert rec["teks_ocr"] == "a b" and rec["table_origin"] == "image"
    row = chunk_dump._row(rec)
    assert row["transkripsi_peringatan"] == "rujukan=lapisan_ocr; angka_tak_ditemukan:12"
    assert row["table_source"] == "vision_transcription" and row["image_content"] == "tabel"
    img = chunk_dump._image_record({"image_id": "d_p1_img00", "perlakuan": "buang",
                                    "klasifikasi_jenis": "cap", "memuat_tabel_data": True,
                                    "rasio_tumpang_tabel": 0.966, "image_content": None})
    assert img["perlakuan"] == "buang" and img["visual_type"] is None
    assert chunk_dump._image_row(img)["perlakuan"] == "buang"


def test_env_research_memuat_semua_flag_tahap_t():
    isi = (ROOT / ".env.research").read_text(encoding="utf-8")
    nilai = dict(l.split("=", 1) for l in isi.splitlines() if l and not l.startswith("#") and "=" in l)
    for nama in ("INDEX_TABLE_TRANSCRIPTION", "INDEX_IMAGE_TABLE_TRANSCRIPTION", "TABLE_TRANSCRIPTION_DPI",
                 "TABLE_TRANSCRIPTION_NUM_PREDICT", "TABLE_TRANSCRIPTION_NUM_CTX",
                 "TABLE_RENDER_SCAN_MARGIN", "IMAGE_CLASSIFICATION_DPI", "IMAGE_CLASSIFICATION_SIDE",
                 "IMAGE_CAP_MIN_OVERLAP"):
        diharap = config.RESEARCH_EXPECTED_FLAGS[nama]
        assert type(diharap)(nilai[nama] == "true" if isinstance(diharap, bool) else nilai[nama]) == diharap


def test_flag_default_mati(monkeypatch):
    import importlib
    for nama in ("INDEX_TABLE_TRANSCRIPTION", "INDEX_IMAGE_TABLE_TRANSCRIPTION"):
        monkeypatch.delenv(nama, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    cfg = importlib.reload(config)
    try:
        assert cfg.INDEX_TABLE_TRANSCRIPTION is False and cfg.INDEX_IMAGE_TABLE_TRANSCRIPTION is False
    finally:
        monkeypatch.undo()
        importlib.reload(config)
