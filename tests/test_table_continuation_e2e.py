"""Uji rantai penuh: berkas keputusan -> _chunk_elements -> chunk.

Yang dibuktikan:
  1. Tanpa flag, hasil IDENTIK dengan sebelum Tahap B.
  2. Dengan flag tapi tanpa pasangan disetujui, teks identik (hanya metadata
     tautan yang bertambah).
  3. Pasangan disetujui: header diulang SETELAH prefiks section, chunk_id tetap.
  4. Header hanya diulang bila lolos kriteria — baris data tidak pernah.
  5. Kunci yang cocok tapi sidik html-nya tidak TIDAK digabung.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SECTION = "Persyaratan Ujian"
HDR_SEL = ["Persyaratan", "S1", "S2"]
HDR_MD = "| Persyaratan | S1 | S2 |\n| --- | --- | --- |"


def html(baris, th=False):
    h = ("<thead><tr>" + "".join(f"<th>{c}</th>" for c in baris[0]) + "</tr></thead>") if th else ""
    body = baris[1:] if th else baris
    return ("<table>" + h + "<tbody>" + "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in body) + "</tbody></table>")


def md(baris):
    return "\n".join("| " + " | ".join(r) + " |" for r in baris)


A_ROWS = [HDR_SEL, ["IPK minimum", "2,75", "3,00"]]
B_ROWS = [["Masa studi", "14", "8"], ["TOEFL", "450", "500"]]
C_ROWS = [["Publikasi", "0", "1"], ["Bimbingan", "6", "9"]]


def tabel_el(rows, page, th=False, isi=None):
    return {"category": "Table", "text": isi or md(rows), "page": page,
            "metadata": {"raw_html": html(rows, th=th), "table_format": "markdown",
                         "bbox": [0.1, 0.1, 0.9, 0.9]}}


def elements(a=A_ROWS, a_th=True, b=B_ROWS, c=None):
    els = [{"category": "Title", "text": SECTION, "page": 4, "metadata": {}},
           tabel_el(a, 4, th=a_th, isi=(f"{HDR_MD}\n{md(a[1:])}" if a_th else None)),
           tabel_el(b, 5)]
    if c is not None:
        els.append(tabel_el(c, 6))
    return els


def jalankan(monkeypatch, aktif, els, keputusan=None, tmp_path=None):
    import importlib
    monkeypatch.setenv("INDEX_TABLE_CONTINUATION", "true" if aktif else "false")
    monkeypatch.setenv("INDEX_STRUCTURAL_METADATA", "true")
    monkeypatch.setenv("INDEX_TABLES_AS_OWN_CHUNKS", "true")
    monkeypatch.setenv("INDEX_MIN_CHUNK_TOKENS", "0")
    if keputusan is not None:
        p = tmp_path / "k.json"
        p.write_text(json.dumps({"pasangan": keputusan}), encoding="utf-8")
        monkeypatch.setenv("TABLE_CONTINUATION_PATH", str(p))
    elif tmp_path is not None:
        monkeypatch.setenv("TABLE_CONTINUATION_PATH", str(tmp_path / "tidak_ada.json"))
    import backend.config as cfg
    importlib.reload(cfg)
    import backend.services.table_continuation as tc
    importlib.reload(tc)
    tc.reload_keputusan()
    import backend.services.preprocessing as pre
    importlib.reload(pre)
    return [c for c in pre._chunk_elements(els, "sop.pdf", document_id="sop")
            if c.get("element_type") == "Table"]


def setujui(chunks, pasangan_idx, sidik=True, **ubah):
    """Entri berkas keputusan untuk pasangan (i, j) dari daftar chunk tabel."""
    from backend.services.table_continuation import html_sha
    out = {}
    for i, j in pasangan_idx:
        a, b = chunks[i], chunks[j]
        e = {"keputusan": "terima"}
        if sidik:
            e["html_sha_a"] = html_sha(a["raw_html"])
            e["html_sha_b"] = html_sha(b["raw_html"])
        e.update(ubah)
        out[f"{a['chunk_id']}__{b['chunk_id']}"] = e
    return out


# ─── identitas tanpa flag / tanpa persetujuan ────────────────────────────────

@pytest.mark.integration
def test_flag_mati_identik(monkeypatch):
    t = jalankan(monkeypatch, False, elements())
    assert len(t) == 2
    assert all("table_group_id" not in c and "table_header_repeated" not in c for c in t)


@pytest.mark.integration
def test_flag_hidup_tanpa_berkas_teks_identik(monkeypatch, tmp_path):
    mati = jalankan(monkeypatch, False, elements())
    hidup = jalankan(monkeypatch, True, elements(), tmp_path=tmp_path)
    assert [c["text"] for c in hidup] == [c["text"] for c in mati]
    assert [c["text_sha"] for c in hidup] == [c["text_sha"] for c in mati]
    assert [c["chunk_id"] for c in hidup] == [c["chunk_id"] for c in mati]
    assert all(c["table_header_repeated"] is False for c in hidup)


# ─── pasangan disetujui ──────────────────────────────────────────────────────

@pytest.mark.integration
def test_header_diulang_setelah_prefiks_section(monkeypatch, tmp_path):
    tanpa = jalankan(monkeypatch, True, elements(), tmp_path=tmp_path)
    dengan = jalankan(monkeypatch, True, elements(), setujui(tanpa, [(0, 1)]), tmp_path)
    assert [c["chunk_id"] for c in dengan] == [c["chunk_id"] for c in tanpa]
    b = dengan[1]["text"]
    assert b.startswith(f"## {SECTION}\n\n{HDR_MD}\n| Masa studi"), b[:120]
    assert b.count(f"## {SECTION}") == 1
    assert dengan[1]["table_header_repeated"] is True
    assert dengan[1]["table_group_id"] == dengan[0]["chunk_id"] and dengan[1]["table_part"] == 1
    assert dengan[0]["text"] == tanpa[0]["text"], "potongan A tidak boleh berubah"
    assert dengan[1]["raw_html"] == tanpa[1]["raw_html"], "raw_html tidak disentuh"


@pytest.mark.integration
def test_baris_data_sebagai_kepala_tidak_diulang(monkeypatch, tmp_path):
    """921112 | BELANJA PENGADAAN BAHAN MAKANAN: tanpa <th>, baris pertama data."""
    akun = [["921112", "BELANJA PENGADAAN BAHAN MAKANAN"], ["921113", "BELANJA OBAT"]]
    lanjut = [["921114", "BELANJA PERJALANAN"], ["921115", "BELANJA JASA"]]
    els = elements(a=akun, a_th=False, b=lanjut)
    tanpa = jalankan(monkeypatch, True, els, tmp_path=tmp_path)
    dengan = jalankan(monkeypatch, True, els, setujui(tanpa, [(0, 1)]), tmp_path)
    assert "921112" not in dengan[1]["text"]
    assert dengan[1]["text"] == tanpa[1]["text"]
    assert dengan[1]["table_header_repeated"] is False
    assert dengan[1]["table_group_id"] == dengan[0]["chunk_id"], "tautan tetap diisi"


@pytest.mark.integration
def test_rantai_tiga_mewarisi_header_kepala(monkeypatch, tmp_path):
    """Header potongan KETIGA berasal dari kepala, bukan baris pertama potongan kedua."""
    els = elements(c=C_ROWS)
    tanpa = jalankan(monkeypatch, True, els, tmp_path=tmp_path)
    dengan = jalankan(monkeypatch, True, els, setujui(tanpa, [(0, 1), (1, 2)]), tmp_path)
    assert {c["table_group_id"] for c in dengan} == {tanpa[0]["chunk_id"]}
    assert [c["table_part"] for c in dengan] == [0, 1, 2]
    assert f"{HDR_MD}\n| Publikasi" in dengan[2]["text"]
    assert "| Masa studi | 14 | 8 |\n| --- " not in dengan[2]["text"]


@pytest.mark.integration
def test_rantai_putus_di_tengah(monkeypatch, tmp_path):
    els = elements(c=C_ROWS)
    tanpa = jalankan(monkeypatch, True, els, tmp_path=tmp_path)
    kep = setujui(tanpa, [(0, 1)])
    kep.update(setujui(tanpa, [(1, 2)], keputusan="tolak"))
    dengan = jalankan(monkeypatch, True, els, kep, tmp_path)
    assert dengan[2]["table_group_id"] == dengan[2]["chunk_id"] and dengan[2]["table_part"] == 0
    assert HDR_MD not in dengan[2]["text"]


@pytest.mark.integration
def test_b_sudah_diawali_header_tidak_digandakan(monkeypatch, tmp_path):
    b = [HDR_SEL, ["Masa studi", "14", "8"]]
    els = elements(b=b)
    tanpa = jalankan(monkeypatch, True, els, tmp_path=tmp_path)
    dengan = jalankan(monkeypatch, True, els, setujui(tanpa, [(0, 1)]), tmp_path)
    assert dengan[1]["text"] == tanpa[1]["text"]
    assert dengan[1]["table_header_repeated"] is False


# ─── penjaga sidik ───────────────────────────────────────────────────────────

@pytest.mark.integration
def test_kunci_cocok_sidik_beda_tidak_digabung(monkeypatch, tmp_path):
    """Kunci dari penomoran lain bisa cocok dengan pasangan tabel yang BERBEDA."""
    tanpa = jalankan(monkeypatch, True, elements(), tmp_path=tmp_path)
    kep = setujui(tanpa, [(0, 1)], html_sha_b="0000000000000000")
    dengan = jalankan(monkeypatch, True, elements(), kep, tmp_path)
    assert dengan[1]["text"] == tanpa[1]["text"]
    assert dengan[1]["table_part"] == 0


@pytest.mark.integration
def test_berkas_tanpa_sidik_tidak_digabung(monkeypatch, tmp_path):
    """Berkas keputusan lama (tanpa sidik) harus dimigrasi dulu."""
    tanpa = jalankan(monkeypatch, True, elements(), tmp_path=tmp_path)
    dengan = jalankan(monkeypatch, True, elements(), setujui(tanpa, [(0, 1)], sidik=False), tmp_path)
    assert dengan[1]["text"] == tanpa[1]["text"]
    assert dengan[1]["table_part"] == 0


@pytest.mark.integration
def test_kunci_menunjuk_chunk_yang_tidak_ada(monkeypatch, tmp_path):
    kep = {"sop_p99_c00__sop_p99_c01": {"keputusan": "terima",
                                        "html_sha_a": "x", "html_sha_b": "y"}}
    t = jalankan(monkeypatch, True, elements(c=C_ROWS), kep, tmp_path)
    assert [c["table_part"] for c in t] == [0, 0, 0]
