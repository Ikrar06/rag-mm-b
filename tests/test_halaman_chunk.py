"""Uji penomoran halaman dan bbox chunk teks.

Bug yang diperbaiki: `current_page` diinisialisasi 1 (truthy), sehingga
`if not current_page` tidak pernah dieksekusi dan halaman hanya berubah di
Title. Akibatnya SETIAP chunk teks dalam satu section memakai nomor halaman
judul section-nya.

Kesalahannya tidak terlihat dari pemeriksaan konsistensi: `page` di payload dan
nomor halaman di `chunk_id` sama-sama berasal dari `current_page`, jadi keduanya
selalu sepakat — sama-sama salah.

Dampak lanjutannya pada bbox: koordinat ternormalisasi hanya bermakna relatif
terhadap halaman asalnya, jadi bbox dari halaman lain yang dilekatkan ke label
halaman yang salah menunjuk tempat yang keliru.
"""

import importlib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PANJANG = ("Mahasiswa mengajukan permohonan izin ujian akhir melalui sistem "
           "akademik daring paling lambat empat belas hari kerja sebelum "
           "tanggal pelaksanaan ujian yang telah dijadwalkan panitia. ")


def _pre(monkeypatch):
    monkeypatch.setenv("INDEX_STRUCTURAL_METADATA", "true")
    monkeypatch.setenv("INDEX_MIN_CHUNK_TOKENS", "8")
    monkeypatch.setenv("INDEX_TABLE_CONTINUATION", "false")
    import backend.config as cfg
    importlib.reload(cfg)
    import backend.services.preprocessing as pre
    return importlib.reload(pre)


def par(hal, ulang=3):
    return f"HALAMAN{hal}. " + PANJANG * ulang


def asal(chunk, halaman):
    return sorted(h for h in halaman if f"HALAMAN{h}." in chunk["text"])


@pytest.mark.integration
def test_tiap_chunk_berlabel_halamannya_sendiri(monkeypatch):
    """Section merentang halaman 4-7; tiap chunk harus berlabel halamannya."""
    pre = _pre(monkeypatch)
    els = [{"category": "Title", "text": "Bab I", "page": 4,
            "metadata": {"bbox": [.1, .08, .9, .12]}}]
    for h, (y0, y1) in zip((4, 5, 6, 7), ((.80, .94), (.06, .20), (.30, .44), (.50, .64))):
        els.append({"category": "NarrativeText", "text": par(h), "page": h,
                    "metadata": {"bbox": [.1, y0, .9, y1]}})

    chunks = pre._chunk_elements(els, "sop.pdf", document_id="sop")
    for c in chunks:
        halaman_isi = asal(c, (4, 5, 6, 7))
        assert not halaman_isi or c["page"] in halaman_isi, (
            f"{c['chunk_id']} berlabel halaman {c['page']} tapi isinya dari {halaman_isi}"
        )
    # chunk_id mencerminkan halaman yang benar, bukan halaman judul section.
    assert {c["chunk_id"] for c in chunks} >= {"sop_p5_c00", "sop_p6_c00", "sop_p7_c00"}


@pytest.mark.integration
def test_page_di_payload_dan_di_chunk_id_sepakat_DAN_benar(monkeypatch):
    """Pemeriksaan konsistensi saja tidak cukup — keduanya bisa sepakat-salah."""
    pre = _pre(monkeypatch)
    els = [{"category": "Title", "text": "Bab I", "page": 2, "metadata": {}}]
    for h in (2, 3, 4):
        els.append({"category": "NarrativeText", "text": par(h), "page": h,
                    "metadata": {"bbox": [.1, .2, .9, .4]}})
    for c in pre._chunk_elements(els, "sop.pdf", document_id="sop"):
        assert f"_p{c['page']}_" in c["chunk_id"]          # sepakat
        halaman_isi = asal(c, (2, 3, 4))
        assert not halaman_isi or c["page"] in halaman_isi  # DAN benar


@pytest.mark.integration
def test_bbox_hanya_dari_halaman_chunk_itu(monkeypatch):
    """Buffer merentang dua halaman: bbox tidak boleh menggabungkan kerangka
    koordinat yang berbeda."""
    pre = _pre(monkeypatch)
    els = [
        {"category": "NarrativeText", "text": "HALAMAN4. Paragraf pendek akhir halaman.",
         "page": 4, "metadata": {"bbox": [.1, .85, .9, .94]}},
        {"category": "NarrativeText", "text": "HALAMAN5. Paragraf pendek awal halaman.",
         "page": 5, "metadata": {"bbox": [.1, .06, .9, .15]}},
    ]
    c = pre._chunk_elements(els, "sop.pdf", document_id="sop")[0]
    assert c["page"] == 4
    assert c["bbox"] == [0.1, 0.85, 0.9, 0.94]      # hanya kotak halaman 4
    assert c["page_span"] == [4, 5]                  # cakupannya tidak hilang


@pytest.mark.integration
def test_page_span_absen_untuk_chunk_satu_halaman(monkeypatch):
    """Ketiadaannya berarti page_number sudah memerikan seluruh chunk."""
    pre = _pre(monkeypatch)
    els = [{"category": "NarrativeText", "text": par(3), "page": 3,
            "metadata": {"bbox": [.1, .2, .9, .4]}}]
    assert "page_span" not in pre._chunk_elements(els, "sop.pdf", document_id="sop")[0]


@pytest.mark.integration
def test_tabel_tidak_terpengaruh(monkeypatch):
    """Cabang Table memakai page element langsung, bukan current_page."""
    monkeypatch.setenv("INDEX_TABLES_AS_OWN_CHUNKS", "true")
    pre = _pre(monkeypatch)
    els = [
        {"category": "Title", "text": "Bab I", "page": 4, "metadata": {}},
        {"category": "Table", "text": "| A | B |\n| --- | --- |\n| 1 | 2 |",
         "page": 9, "metadata": {"raw_html": "<table></table>",
                                 "table_format": "markdown", "bbox": [.1, .2, .9, .5]}},
    ]
    tabel = [c for c in pre._chunk_elements(els, "sop.pdf", document_id="sop")
             if c["element_type"] == "Table"][0]
    assert tabel["page"] == 9 and "_p9_" in tabel["chunk_id"]


@pytest.mark.integration
def test_teks_overlap_tidak_mewarisi_halaman_lama(monkeypatch):
    """Overlap berasal dari halaman sebelumnya tapi hanya jembatan konteks;
    isi chunk ditentukan element yang menyusul."""
    pre = _pre(monkeypatch)
    els = [{"category": "NarrativeText", "text": par(h, ulang=4), "page": h,
            "metadata": {"bbox": [.1, .2, .9, .6]}} for h in (5, 6)]
    chunks = pre._chunk_elements(els, "sop.pdf", document_id="sop")
    terakhir = chunks[-1]
    assert terakhir["page"] == 6 and "_p6_" in terakhir["chunk_id"]
