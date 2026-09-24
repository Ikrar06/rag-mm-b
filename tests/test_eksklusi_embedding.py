"""Setiap kunci metadata struktural WAJIB dikecualikan dari teks yang divektorkan.

Kunci di indexing._STRUCTURAL_METADATA_KEYS masuk metadata Document, dan
LlamaIndex menyertakan metadata yang tidak dikecualikan ke dalam string yang
di-embed. Sudah terjadi: page_span dan tiga kunci tautan tabel ditambahkan ke
payload tanpa ditambahkan ke EMBED_EXCLUDED_METADATA_KEYS, sehingga di v3,
2.098 chunk teks membawa "page_span: [n, m]" ke dalam embedding-nya.

Uji ini membaca sumber indexing.py, bukan mengimpornya, supaya tidak menarik
klien Qdrant.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def kunci_struktural() -> list[str]:
    src = (ROOT / "backend" / "services" / "indexing.py").read_text(encoding="utf-8")
    blok = src[src.index("_STRUCTURAL_METADATA_KEYS = ("):]
    blok = blok[: blok.index("\n)")]
    return re.findall(r'^\s+"([a-z_]+)",', blok, re.M)


@pytest.mark.unit
def test_daftar_kunci_struktural_terbaca():
    k = kunci_struktural()
    assert "chunk_id" in k and "page_span" in k, k


@pytest.mark.unit
def test_tidak_ada_kunci_struktural_yang_ikut_divektorkan():
    from backend.config import EMBED_EXCLUDED_METADATA_KEYS
    bocor = [k for k in kunci_struktural() if k not in EMBED_EXCLUDED_METADATA_KEYS]
    assert not bocor, f"ikut divektorkan: {bocor} — tambahkan ke EMBED_EXCLUDED_METADATA_KEYS"
