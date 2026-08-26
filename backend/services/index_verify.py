"""Pemeriksaan pasca-indexing atas isi Qdrant yang sebenarnya.

Berbeda dari gerbang lain di `indexing.py`, yang memeriksa NIAT (flag, laporan
ekstraksi, jangkauan model) sebelum apa pun dikirim. Modul ini memeriksa HASIL:
ia membaca kembali titik yang benar-benar tersimpan.

Alasannya konkret. `chunk_id` duplikat pernah lolos ke korpus 214 dokumen —
26.087 titik, 25.475 `chunk_id` unik, 388 duplikat di 30 dokumen — karena
`transformations=[]` tidak mematikan node parser seperti yang dikira (lihat
`node_passthrough.py`). Tidak ada satu pun pemeriksaan sebelum-kirim yang bisa
menangkapnya: dari sisi `indexing.py` daftar Document-nya benar. Yang salah
terjadi DI DALAM LlamaIndex, setelah gerbang terakhir.

`chunk_id` duplikat adalah kelas kegagalan yang mahal karena senyap. Ia tidak
menjatuhkan indexing, tidak muncul di log, dan baru terasa saat anotasi gold
sudah berjalan — `gold_chunk_ids` menunjuk ke lebih dari satu titik dengan teks
berbeda, sehingga metrik retrieval tidak lagi punya arti tunggal.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

_SCROLL_BATCH = 1_000

# Cap jumlah contoh yang dicetak di pesan galat. Duplikat bisa ribuan; laporan
# yang tidak muat di terminal tidak menolong siapa pun.
_MAX_CONTOH = 10


def _chunk_id_dari_payload(payload: dict) -> str | None:
    """Ambil chunk_id dari payload titik Qdrant.

    QdrantVectorStore menyimpan metadata node secara datar di akar payload,
    jadi `chunk_id` biasanya ada di sana. Tapi bentuk itu bergantung versi
    integrasi, dan payload lama bisa menyimpan seluruh node ter-serialisasi di
    `_node_content`. Dicoba keduanya supaya pemeriksaan ini tidak diam-diam
    melaporkan "0 duplikat" hanya karena tidak menemukan field-nya.
    """
    if not isinstance(payload, dict):
        return None

    langsung = payload.get("chunk_id")
    if isinstance(langsung, str) and langsung:
        return langsung

    isi = payload.get("_node_content")
    if isinstance(isi, str) and isi:
        try:
            meta = (json.loads(isi) or {}).get("metadata") or {}
        except (json.JSONDecodeError, AttributeError):
            return None
        nilai = meta.get("chunk_id")
        if isinstance(nilai, str) and nilai:
            return nilai
    return None


def scan_chunk_ids(client, collection_name: str) -> dict:
    """Baca seluruh titik dan hitung sebaran chunk_id.

    Mengembalikan dict: total_titik, dengan_chunk_id, tanpa_chunk_id, unik,
    duplikat (jumlah titik berlebih), dan rincian per chunk_id yang bertabrakan.
    """
    counts: Counter[str] = Counter()
    berkas: defaultdict[str, set[str]] = defaultdict(set)
    total = tanpa_id = 0

    offset = None
    while True:
        titik, offset = client.scroll(
            collection_name=collection_name,
            limit=_SCROLL_BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not titik:
            break
        for p in titik:
            total += 1
            payload = p.payload or {}
            cid = _chunk_id_dari_payload(payload)
            if cid is None:
                tanpa_id += 1
                continue
            counts[cid] += 1
            nama = payload.get("file_name")
            if isinstance(nama, str) and nama:
                berkas[cid].add(nama)
        if offset is None:
            break

    tabrakan = {cid: n for cid, n in counts.items() if n > 1}
    return {
        "total_titik": total,
        "dengan_chunk_id": total - tanpa_id,
        "tanpa_chunk_id": tanpa_id,
        "unik": len(counts),
        "duplikat": sum(n - 1 for n in tabrakan.values()),
        "chunk_id_bertabrakan": len(tabrakan),
        "dokumen_terdampak": len({f for cid in tabrakan for f in berkas[cid]}),
        "contoh": [
            {"chunk_id": cid, "titik": n, "file_name": sorted(berkas[cid]) or None}
            for cid, n in sorted(tabrakan.items(), key=lambda kv: -kv[1])[:_MAX_CONTOH]
        ],
    }


def check_unique_chunk_ids(client, collection_name: str, *, strict: bool) -> dict:
    """Tolak (strict) atau peringatkan bila ada chunk_id duplikat di Qdrant.

    `strict` diisi RESEARCH_MODE oleh pemanggil: di luar riset, duplikat tidak
    perlu menjatuhkan run produksi. Kegagalan MEMBACA Qdrant tidak pernah
    menjatuhkan run — data sudah tersimpan, dan pemeriksaan yang tidak bisa
    berjalan bukan alasan untuk membuang pekerjaan berjam-jam. Ia melaporkan
    dirinya gagal, dan itu terlihat di log.
    """
    try:
        h = scan_chunk_ids(client, collection_name)
    except Exception as e:
        logger.error(
            "chunk_id_check_gagal collection=%s error=%s — "
            "keunikan chunk_id TIDAK terverifikasi untuk run ini",
            collection_name, e, exc_info=True,
        )
        return {"status": "gagal_dibaca", "error": str(e)}

    logger.info(
        "chunk_id_check collection=%s titik=%d dengan_id=%d tanpa_id=%d unik=%d duplikat=%d",
        collection_name, h["total_titik"], h["dengan_chunk_id"],
        h["tanpa_chunk_id"], h["unik"], h["duplikat"],
    )

    # Titik tanpa chunk_id tidak bisa dianotasi sama sekali — tidak ada yang
    # dapat ditulis di gold_chunk_ids untuk menunjuknya. Diperingatkan, tidak
    # ditolak: titik sisa run lama dengan flag struktural mati bisa ada di
    # koleksi yang sama, dan itu bukan alasan membuang run yang baru selesai.
    if h["tanpa_chunk_id"]:
        logger.warning(
            "titik_tanpa_chunk_id collection=%s jumlah=%d dari=%d — titik ini "
            "tidak dapat dirujuk oleh gold_chunk_ids",
            collection_name, h["tanpa_chunk_id"], h["total_titik"],
        )

    if not h["duplikat"]:
        h["status"] = "lolos"
        return h

    h["status"] = "duplikat_ditemukan"
    rincian = "\n".join(
        f"    {c['chunk_id']}  {c['titik']} titik"
        f"{'  ' + ', '.join(c['file_name']) if c['file_name'] else ''}"
        for c in h["contoh"]
    )
    sisa = h["chunk_id_bertabrakan"] - len(h["contoh"])
    if sisa > 0:
        rincian += f"\n    ... dan {sisa} chunk_id bertabrakan lainnya"

    pesan = (
        f"chunk_id DUPLIKAT di Qdrant '{collection_name}': "
        f"{h['total_titik']} titik, {h['unik']} chunk_id unik, "
        f"{h['duplikat']} titik berlebih pada {h['chunk_id_bertabrakan']} chunk_id "
        f"di {h['dokumen_terdampak']} dokumen.\n{rincian}\n"
        f"  Artinya satu gold_chunk_id menunjuk ke lebih dari satu titik dengan "
        f"teks berbeda — metrik retrieval kehilangan arti tunggalnya.\n"
        f"  Penyebab yang sudah pernah terjadi: node parser kedua memecah chunk "
        f"dan node anak mewarisi metadata induk apa adanya. Pastikan "
        f"INDEX_DISABLE_NODE_PARSER=true DAN build_transformations() "
        f"mengembalikan [PassthroughNodeParser()], bukan daftar kosong "
        f"(lihat backend/services/node_passthrough.py)."
    )

    if strict:
        raise ValueError(pesan)
    logger.warning("chunk_id_duplikat %s", pesan)
    return h
