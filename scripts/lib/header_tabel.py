"""Pengalihan ke backend/services/header_tabel.py — implementasi kanonik.

Dipindah ke backend karena indexing kini memakainya untuk memilih baris header
yang diulang. Modul ini tetap ada supaya skrip analisis dan ujinya tidak perlu
diubah; ia menyalin SELURUH nama, termasuk yang diawali garis bawah.
"""
import sys as _sys
from pathlib import Path as _Path

_ROOT = str(_Path(__file__).resolve().parents[2])
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from backend.services import header_tabel as _asli  # noqa: E402

globals().update({k: v for k, v in vars(_asli).items() if not k.startswith("__")})
