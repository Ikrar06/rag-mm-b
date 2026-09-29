"""Perbandingan dump per dokumen, dan penggabungan dump run susulan."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.banding_dump import bandingkan  # noqa: E402
from lib.gabung_dump import beda_konfigurasi, gabung, manifest_gabungan  # noqa: E402


def c(cid, jenis="NarrativeText", sha=None, **kw):
    doc, hal = cid.rsplit("_p", 1)[0], int(cid.rsplit("_p", 1)[1].split("_")[0])
    return {"chunk_id": cid, "document_id": doc, "page_number": hal, "element_type": jenis,
            "text_sha": sha or cid, **kw}


V4 = [c("a_p1_c00", sha="t1"), c("a_p1_c01", "Table", text_as_html="<t>X</t>"),
      c("a_p1_c02", sha="t2"), c("b_p2_c00", sha="u1"),
      c("g_p3_c00", "ImageDescription", image_id="g_p3_img00")]
# v5: di a hal 1 tabel kecil yang di v4 dibuang kini lolos (c01), teks sesudahnya
# bergeser; b sama; g gambar kini Table asal gambar.
V5 = [c("a_p1_c00", sha="t1"), c("a_p1_c01", "Table", text_as_html="<t>BARU</t>", teks_ocr="1 2"),
      c("a_p1_c02", "Table", text_as_html="<t>X</t>", teks_ocr="panjang " * 10),
      c("a_p1_c03", sha="t2"), c("b_p2_c00", sha="u1"),
      c("g_p3_c00", "Table", image_id="g_p3_img00", table_origin="image")]


def test_bandingkan_menunjuk_dokumen_halaman_dan_sebab():
    h = bandingkan(V4, V5)
    (d,) = h["dokumen"]
    assert (d["document_id"], d["selisih"], d["per_jenis"]) == ("a", 1, {"Table": 1})
    (hal,) = d["halaman"]
    assert hal["halaman"] == 1 and hal["sebab"] == "tabel_kecil_lolos"
    assert sorted(hal["bergeser"]) == [("a_p1_c01", "a_p1_c02"), ("a_p1_c02", "a_p1_c03")]
    assert h["urutan_berubah"] == ["g hal 3: ['ImageDescription'] -> ['Table(gambar)']"]
    assert h["sebab"] == {"tabel_kecil_lolos": 1} and h["n_bergeser"] == 2
    assert h["gambar_tanpa_padanan"] == []
    dekoratif = c("b_p2_c01", "Table", image_id="b_p2_img05", table_origin="image",
                  image_content="tabel")
    assert bandingkan(V4, V5 + [dekoratif])["gambar_tanpa_padanan"] == [
        "b_p2_c01 Table(gambar) tabel"]


def test_bandingkan_gambar_hilang_dan_dokumen_satu_sisi():
    lama = V4 + [c("x_p1_c00")]
    baru = [r for r in V4 if r["document_id"] != "g"] + [c("g_p3_c01", sha="lain")]
    h = bandingkan(lama, baru)
    assert h["hanya_lama"] == ["x"] and h["dokumen"] == []
    baru2 = [r for r in V4 if r["document_id"] != "g"]
    h2 = bandingkan(V4 + [c("g_p3_c01", sha="z")], baru2 + [c("g_p3_c01", sha="z")])
    assert h2["dokumen"][0]["halaman"][0]["sebab"] == "gambar_hilang"


def test_gabung_menolak_tabrakan():
    assert len(gabung([c("a_p1_c00")], [c("b_p1_c00")], "chunk_id")) == 2
    with pytest.raises(ValueError, match="dokumen ada di kedua run"):
        gabung([c("a_p1_c00")], [c("a_p2_c00")], "chunk_id")


def test_beda_konfigurasi_abaikan_hitungan():
    m = {"chunking": {"CHUNK_SIZE": 512}, "research_flags": {"table_transcription": {
        "render_dpi": 200, "hitungan": {"x": 1}, "batas_gambar": {"ollama": "a"}}},
        "provenance": {"qdrant_collection": "v5"}}
    s = json.loads(json.dumps(m))
    s["research_flags"]["table_transcription"]["hitungan"] = {"x": 9}
    s["research_flags"]["table_transcription"]["batas_gambar"] = {"ollama": "b"}
    assert beda_konfigurasi(m, s) == []
    s["research_flags"]["table_transcription"]["render_dpi"] = 150
    s["provenance"]["qdrant_collection"] = "v4"
    assert beda_konfigurasi(m, s) == [
        "research_flags.table_transcription.render_dpi: 200 != 150",
        "provenance.qdrant_collection: 'v5' != 'v4'"]


def test_manifest_gabungan():
    mu = {"run_id": "u", "n_chunks": 10, "provenance": {"git_commit": "ea8959a"},
          "extraction_reports": {"a.pdf": {}}, "degraded_documents": []}
    ms = {"run_id": "s", "n_chunks": 2, "provenance": {"git_commit": "fix"},
          "extraction_reports": {"ukt.pdf": {}}, "degraded_documents": []}
    m = manifest_gabungan(mu, ms, 12, 3, 2, ["ukt"])
    assert m["n_chunks"] == 12 and set(m["extraction_reports"]) == {"a.pdf", "ukt.pdf"}
    runs = m["gabungan"]["runs"]
    assert [(r["peran"], r["git_commit"]) for r in runs] == [("utama", "ea8959a"), ("susulan", "fix")]
    assert runs[1]["dokumen"] == ["ukt"] and mu["n_chunks"] == 10


def test_cli_gabung(tmp_path, monkeypatch, capsys):
    import gabung_dump
    man = {"chunking": {"CHUNK_SIZE": 512}, "provenance": {"qdrant_collection": "v5"}}
    for nama, rows, imgs in (("u", [c("a_p1_c00")], [{"image_id": "a_p1_img00"}]),
                             ("s", [c("ukt_p5_c01", "Table")], [{"image_id": "ukt_p5_img00",
                                                                  "narrative_summary": "x"}])):
        d = tmp_path / nama
        d.mkdir()
        (d / "run_manifest.json").write_text(json.dumps({**man, "run_id": nama}))
        (d / "chunks.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        (d / "images.jsonl").write_text("\n".join(json.dumps(r) for r in imgs))
    argv = ["x", "--utama", str(tmp_path / "u"), "--susulan", str(tmp_path / "s"),
            "--out", str(tmp_path / "g")]
    monkeypatch.setattr(sys, "argv", argv)
    assert gabung_dump.main() == 0
    m = json.loads((tmp_path / "g" / "run_manifest.json").read_text())
    assert m["n_chunks"] == 2 and m["n_images_with_narrative"] == 1
    assert (tmp_path / "g" / "chunks_review.csv").read_text().count("\n") == 3
    assert gabung_dump.main() == 2                                  # out sudah berisi
    (tmp_path / "s" / "run_manifest.json").write_text(json.dumps(
        {**man, "chunking": {"CHUNK_SIZE": 256}}))
    monkeypatch.setattr(sys, "argv", argv[:-1] + [str(tmp_path / "g2")])
    assert gabung_dump.main() == 1
    assert "CHUNK_SIZE: 512 != 256" in capsys.readouterr().out
