"""Uji pemetaan ulang kunci berkas keputusan antar penomoran chunk_id.

Kasus nyata yang mendasarinya: perbaikan current_page memindahkan chunk teks
ke halaman yang benar, sehingga tabel rubrik_p28_c00 di v2 menjadi
rubrik_p28_c01 di v3, dan rubrik_p28_c00 di v3 kini chunk TEKS.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services.table_continuation import html_sha  # noqa: E402
from lib.keputusan_migrasi import petakan_id, posisi_tabel  # noqa: E402


def rows(spec):
    """spec: list (chunk_id, element_type, page, chunk_index, html)."""
    return [{"chunk_id": c, "element_type": e, "page_number": p, "chunk_index": i,
             "text_as_html": h} for c, e, p, i, h in spec]


V2 = rows([("rubrik_p27_c00", "Table", 27, 0, "<table>A</table>"),
           ("rubrik_p27_c01", "NarrativeText", 27, 1, None),
           ("rubrik_p28_c00", "Table", 28, 2, "<table>B</table>"),
           ("rubrik_p29_c00", "Table", 29, 3, "<table>C</table>")])
V3 = rows([("rubrik_p27_c00", "Table", 27, 0, "<table>A</table>"),
           ("rubrik_p28_c00", "NarrativeText", 28, 1, None),
           ("rubrik_p28_c01", "Table", 28, 2, "<table>B</table>"),
           ("rubrik_p29_c00", "NarrativeText", 29, 3, None),
           ("rubrik_p29_c01", "Table", 29, 4, "<table>C</table>")])


def peta(cid, lama=V2, baru=V3):
    pl, _ = posisi_tabel(lama)
    _, db = posisi_tabel(baru)
    sl = {r["chunk_id"]: html_sha(r["text_as_html"]) for r in lama}
    sb = {r["chunk_id"]: html_sha(r["text_as_html"]) for r in baru}
    return petakan_id(cid, pl, db, sl, sb)


@pytest.mark.unit
def test_id_yang_tidak_bergeser():
    h = peta("rubrik_p27_c00")
    assert h.status == "id_sama" and h.baru == "rubrik_p27_c00"


@pytest.mark.unit
def test_tabel_yang_bergeser_dipetakan_ke_tabel_bukan_ke_chunk_teks():
    h = peta("rubrik_p28_c00")
    assert h.status == "dipetakan" and h.baru == "rubrik_p28_c01"
    h = peta("rubrik_p29_c00")
    assert h.baru == "rubrik_p29_c01"


@pytest.mark.unit
def test_sidik_beda_tidak_dipetakan_otomatis():
    v3 = [dict(r) for r in V3]
    v3[2]["text_as_html"] = "<table>LAIN</table>"
    h = peta("rubrik_p28_c00", baru=v3)
    assert h.status == "sidik_beda" and h.baru is None


@pytest.mark.unit
def test_bukan_tabel_di_dump_lama_tak_ditemukan():
    assert peta("rubrik_p27_c01").status == "tak_ditemukan"


@pytest.mark.unit
def test_tabel_hilang_di_dump_baru():
    v3 = [r for r in V3 if r["chunk_id"] != "rubrik_p29_c01"]
    assert peta("rubrik_p29_c00", baru=v3).status == "tak_ditemukan"


@pytest.mark.unit
def test_html_sha_tahan_spasi_dan_kosong_berarti_tanpa_sidik():
    assert html_sha("<table> a  b </table>") == html_sha("<table>\na\tb\n</table>")
    assert html_sha("") == "" and html_sha(None) == ""


@pytest.mark.integration
def test_skrip_migrasi_menulis_berkas_baru_dan_membawa_keputusan(tmp_path, monkeypatch):
    import json, importlib.util
    lama, baru = tmp_path / "v2.jsonl", tmp_path / "v3.jsonl"
    lama.write_text("\n".join(json.dumps(r) for r in V2))
    baru.write_text("\n".join(json.dumps(r) for r in V3))
    kep = tmp_path / "k.json"
    kep.write_text(json.dumps({"pasangan": {
        "rubrik_p27_c00__rubrik_p28_c00": {"keputusan": "terima", "catatan_peninjau": "cek"},
        "rubrik_p28_c00__rubrik_p29_c00": {"keputusan": "terima"},
        "rubrik_p27_c01__rubrik_p28_c00": {"keputusan": "terima"},     # teks -> tak terpetakan
    }}))
    out = tmp_path / "k.v3.json"
    spec = importlib.util.spec_from_file_location(
        "mk", Path(__file__).resolve().parent.parent / "scripts" / "migrasi_keputusan.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    monkeypatch.setattr(sys, "argv", ["x", "--keputusan", str(kep), "--chunks-lama", str(lama),
                                      "--chunks-baru", str(baru), "--keluar", str(out)])
    assert m.main() == 0
    d = json.loads(out.read_text())
    assert set(d["pasangan"]) == {"rubrik_p27_c00__rubrik_p28_c01",
                                  "rubrik_p28_c01__rubrik_p29_c01"}
    e = d["pasangan"]["rubrik_p27_c00__rubrik_p28_c01"]
    assert e["catatan_peninjau"] == "cek" and e["keputusan"] == "terima"
    assert e["html_sha_a"] and e["html_sha_b"]
    assert list(d["tidak_terpetakan"]) == ["rubrik_p27_c01__rubrik_p28_c00"]
    assert json.loads(kep.read_text())["pasangan"]      # asli tidak ditimpa


@pytest.mark.integration
def test_skrip_menolak_menimpa_berkas_asli(tmp_path, monkeypatch):
    import json, importlib.util
    kep = tmp_path / "k.json"; kep.write_text(json.dumps({"pasangan": {}}))
    spec = importlib.util.spec_from_file_location(
        "mk2", Path(__file__).resolve().parent.parent / "scripts" / "migrasi_keputusan.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    monkeypatch.setattr(sys, "argv", ["x", "--keputusan", str(kep), "--chunks-lama", str(kep),
                                      "--chunks-baru", str(kep), "--keluar", str(kep)])
    assert m.main() == 2
