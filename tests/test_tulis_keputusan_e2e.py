"""Uji jalur --tulis-keputusan --vision dari main() sampai berkas keluaran.

Ditulis setelah commit 74a2be6 menghapus `_jalankan_vision` lewat refactor
berbasis potongan baris tanpa memperbarui kedua pemanggilnya. 250 uji lulus
karena tidak satu pun menjalankan main() dengan kedua flag itu bersamaan:
seluruh uji yang ada memanggil `_saran` langsung, sehingga NameError baru
muncul saat dijalankan di server.

Pelajarannya: menguji fungsi satu per satu tidak membuktikan skripnya dapat
dijalankan. Uji di bawah sengaja melewati SELURUH jalur — argumen, pemuatan
chunk, profil PDF, adjudikasi, penulisan berkas.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

_spec = importlib.util.spec_from_file_location(
    "analisis_e2e", ROOT / "scripts" / "analisis_tabel_lintas_halaman.py")
analisis = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analisis)


HDR = ("<table><thead><tr><th>Persyaratan</th><th>S1</th><th>S2</th></tr></thead>"
       "<tbody><tr><td>IPK</td><td>2,75</td><td>3,00</td></tr></tbody></table>")
LANJUT = ("<table><tbody><tr><td>Masa studi</td><td>14</td><td>8</td></tr>"
          "<tr><td>TOEFL</td><td>450</td><td>500</td></tr></tbody></table>")


@pytest.fixture
def korpus(tmp_path):
    """chunks.jsonl + PDF berlapis teks, satu pasangan halaman berurutan."""
    fitz = pytest.importorskip("fitz")
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    doc = fitz.open()
    for _ in range(3):
        pg = doc.new_page()
        pg.insert_text((60, 100), "Persyaratan pengajuan izin ujian akhir "
                                  "bagi mahasiswa program sarjana Universitas.")
    doc.save(str(pdfs / "sop.pdf"))
    doc.close()

    baris = [
        {"chunk_id": "sop_p1_c00", "chunk_index": 0, "page": 1, "element_type": "Table",
         "file_name": "sop.pdf", "section": "Persyaratan", "raw_html": HDR,
         "bbox": [0.1, 0.55, 0.9, 0.94], "text_sha": "a1"},
        {"chunk_id": "sop_p2_c00", "chunk_index": 1, "page": 2, "element_type": "Table",
         "file_name": "sop.pdf", "section": "Persyaratan", "raw_html": LANJUT,
         "bbox": [0.1, 0.06, 0.9, 0.35], "text_sha": "a2"},
    ]
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("\n".join(json.dumps(b) for b in baris), encoding="utf-8")
    return chunks, pdfs, tmp_path


def _jalankan(argv, monkeypatch, verdict="lanjutan", alasan="kolom sama"):
    """Panggil main() dengan adjudikasi vision tiruan."""
    from backend.services import table_adjudicator as ta

    dipanggil = []

    def palsu(pdf_path, hal_a, bbox_a, hal_b, bbox_b):
        dipanggil.append((hal_a, hal_b))
        return ta.Putusan(verdict=verdict, keyakinan="tinggi", alasan=alasan,
                          piksel_dikirim=123456)

    monkeypatch.setattr(ta, "adjudikasi", palsu)
    monkeypatch.setattr(sys, "argv", ["analisis_tabel_lintas_halaman.py", *argv])
    return analisis.main(), dipanggil


@pytest.mark.integration
def test_main_dengan_tulis_keputusan_dan_vision(korpus, monkeypatch):
    """Jalur yang rusak di 74a2be6: NameError sebelum berkas sempat ditulis."""
    chunks, pdfs, tmp = korpus
    keputusan = tmp / "table_continuation.json"
    kode, dipanggil = _jalankan([
        "--chunks-jsonl", str(chunks), "--pdf-dir", str(pdfs),
        "--tulis-keputusan", str(keputusan), "--vision", "--contoh", "0",
    ], monkeypatch)

    assert kode == 0
    assert keputusan.exists(), "berkas keputusan tidak ditulis"
    assert dipanggil == [(1, 2)], "adjudikasi vision tidak dipanggil"

    d = json.loads(keputusan.read_text(encoding="utf-8"))
    (kunci, entri), = d["pasangan"].items()
    assert kunci == "sop_p1_c00__sop_p2_c00"
    assert entri["keputusan"] == "", "kolom keputusan harus lahir kosong"
    assert entri["vision"]["verdict"] == "lanjutan"
    assert entri["lapisan_teks_halaman"] == {"1": True, "2": True}
    assert entri["saran"] == "terima"


@pytest.mark.integration
def test_checkpoint_dipakai_ulang_tanpa_memanggil_model(korpus, monkeypatch):
    """Menjalankan ulang dengan path yang sama tidak mengadjudikasi ulang."""
    chunks, pdfs, tmp = korpus
    keputusan = tmp / "tc.json"
    argv = ["--chunks-jsonl", str(chunks), "--pdf-dir", str(pdfs),
            "--tulis-keputusan", str(keputusan), "--vision", "--contoh", "0"]

    _, pertama = _jalankan(argv, monkeypatch)
    assert pertama == [(1, 2)]
    assert Path(str(keputusan) + ".parsial.jsonl").exists()

    _, kedua = _jalankan(argv, monkeypatch)
    assert kedua == [], "checkpoint diabaikan — model dipanggil ulang"


@pytest.mark.integration
def test_vision_bukan_lanjutan_menolak_di_berkas(korpus, monkeypatch):
    chunks, pdfs, tmp = korpus
    keputusan = tmp / "tc2.json"
    _jalankan(["--chunks-jsonl", str(chunks), "--pdf-dir", str(pdfs),
               "--tulis-keputusan", str(keputusan), "--vision", "--contoh", "0"],
              monkeypatch, verdict="bukan_lanjutan")
    (_, entri), = json.loads(keputusan.read_text())["pasangan"].items()
    assert entri["saran"] == "tolak" and "vision" in entri["alasan_saran"]


@pytest.mark.integration
def test_ukur_adjudikasi_tidak_menulis_berkas(korpus, monkeypatch):
    chunks, pdfs, tmp = korpus
    keputusan = tmp / "tc3.json"
    kode, dipanggil = _jalankan([
        "--chunks-jsonl", str(chunks), "--pdf-dir", str(pdfs),
        "--tulis-keputusan", str(keputusan), "--ukur-adjudikasi", "1",
    ], monkeypatch)
    assert kode == 0 and dipanggil == [(1, 2)]
    assert not keputusan.exists(), "mode ukur tidak boleh menulis keputusan"


@pytest.mark.integration
def test_tanpa_vision_berkas_tetap_ditulis(korpus, monkeypatch):
    chunks, pdfs, tmp = korpus
    keputusan = tmp / "tc4.json"
    kode, dipanggil = _jalankan([
        "--chunks-jsonl", str(chunks), "--pdf-dir", str(pdfs),
        "--tulis-keputusan", str(keputusan), "--contoh", "0",
    ], monkeypatch)
    assert kode == 0 and dipanggil == []
    (_, entri), = json.loads(keputusan.read_text())["pasangan"].items()
    assert entri["vision"] is None and entri["saran"] == "terima"
