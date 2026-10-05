"""Tag modalitas chunk: image / table / text.

Satu aturan untuk semua tempat yang membutuhkannya — pembentukan chunk
(preprocessing), payload Qdrant, dump chunks.jsonl, dan pembuat paket eval
yang menurunkannya dari element_type untuk dump lama. Aturan yang ditulis ulang
di tiap tempat bisa menyimpang diam-diam; paket hasil rekonstruksi lalu tidak
cocok dengan payload.

Pencocokan TEPAT pada element_type:
    "ImageDescription"  -> image
    "Table"             -> table   (termasuk gambar-tabel, table_origin="image")
    selainnya           -> text    (termasuk "NarrativeText+Table": tabel kecil
                                    yang tergabung ke buffer teks bersama prosa)

Hitungan v5 (set_payload manual, 25.486 chunk): image 1.410, table 1.202
(1.160 tabel + 42 gambar-tabel), text 22.874.
"""

from __future__ import annotations

MODALITY_IMAGE = "image"
MODALITY_TABLE = "table"
MODALITY_TEXT = "text"
MODALITAS = (MODALITY_TEXT, MODALITY_TABLE, MODALITY_IMAGE)

# Field payload yang diberi payload index (keyword) di collection Qdrant.
PAYLOAD_INDEX_KEYWORD = ("modality",)


def modality_dari(element_type: str | None) -> str:
    """Modalitas dari element_type internal (pencocokan tepat)."""
    if element_type == "ImageDescription":
        return MODALITY_IMAGE
    if element_type == "Table":
        return MODALITY_TABLE
    return MODALITY_TEXT
