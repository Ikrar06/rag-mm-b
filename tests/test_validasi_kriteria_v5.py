"""validasi_kriteria_header.py atas dump v5: kriteria dinilai dari transkripsi,
dan rantai yang mati karena e-panjang dicetak beserta sel terpanjangnya."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import validasi_kriteria_header as vk  # noqa: E402
from backend.services.table_continuation import html_sha  # noqa: E402

PANJANG = "JANGKA WAKTU PENYIMPANAN ARSIP AKTIF SETELAH TAHUN ANGGARAN BERAKHIR DAN DIVERIFIKASI"


def chunk(cid, md, html="<table><tr><td>OCR</td></tr></table>", sumber="vision_transcription"):
    return {"chunk_id": cid, "element_type": "Table", "page_number": int(cid.split("_p")[1][0]),
            "text_content": f"## Bagian\n\n{md}", "text_as_html": html, "table_source": sumber}


def md(*baris):
    kepala, *data = baris
    return "\n".join(["| " + " | ".join(kepala) + " |", "|" + "---|" * len(kepala),
                      *("| " + " | ".join(b) + " |" for b in data)])


def test_v5_dinilai_dari_transkripsi_dan_e_panjang_dicetak(tmp_path, monkeypatch, capsys):
    rows = [
        chunk("ret_p1_c00", md(("No.", PANJANG), ("7", "2 tahun"))),
        chunk("ret_p2_c00", md(("", ""), ("8", "3 tahun"))),
        chunk("sop_p1_c00", md(("Persyaratan", "S1"), ("IPK", "2,75"))),
        # OCR fallback tetap dari raw_html
        chunk("sop_p2_c00", "| x |", html="<table><tr><td>TOEFL</td><td>450</td></tr></table>",
              sumber="ocr_fallback"),
    ]
    (tmp_path / "c.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    kep = {f"{a}__{b}": {"keputusan": "terima", "html_sha_a": html_sha(ha), "html_sha_b": html_sha(hb)}
           for a, b, ha, hb in [("ret_p1_c00", "ret_p2_c00", rows[0]["text_as_html"], rows[1]["text_as_html"]),
                                ("sop_p1_c00", "sop_p2_c00", rows[2]["text_as_html"], rows[3]["text_as_html"])]}
    (tmp_path / "k.json").write_text(json.dumps({"pasangan": kep}))
    monkeypatch.setattr(sys, "argv", ["x", "--keputusan", str(tmp_path / "k.json"),
                                      "--chunks-v3", str(tmp_path / "c.jsonl"),
                                      "--maks-panjang-sel", "80", "--json", str(tmp_path / "o.json")])
    assert vk.main() == 0
    keluar = capsys.readouterr().out
    assert "RANTAI MATI KARENA e-panjang (batas 80): 1" in keluar
    assert f"{len(PANJANG):>4}  ret_p1_c00" in keluar
    hasil = json.loads((tmp_path / "o.json").read_text())
    per = {r["kunci"]: r for r in hasil["hasil"]}
    assert per["ret_p1_c00__ret_p2_c00"]["A"]["aturan"] == "e-panjang/markup"
    assert per["sop_p1_c00__sop_p2_c00"]["kandidat"] == ["Persyaratan", "S1"]
    assert per["sop_p1_c00__sop_p2_c00"]["markup"] is True          # transkripsi = markup
    assert per["sop_p1_c00__sop_p2_c00"]["baris1_b"] == ["TOEFL", "450"]   # dari raw_html
    assert hasil["mati_e_panjang"][0]["sumber"] == "vision_transcription"
