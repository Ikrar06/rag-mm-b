"""Shim: modul pindah ke backend/services/klasifikasi_gambar.py supaya jalur
indexing dan alat ukur memakai prompt dan aturan yang sama persis."""

from backend.services.klasifikasi_gambar import (  # noqa: F401
    AMBANG_CAP, JENIS_GAMBAR, PROMPT_KLASIFIKASI, Klasifikasi, perlakuan_gambar,
    urai_klasifikasi,
)
