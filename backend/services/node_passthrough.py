"""Transformasi LlamaIndex yang meneruskan satu Document menjadi satu node.

Dibuat untuk `INDEX_DISABLE_NODE_PARSER`. Dipakai menggantikan
`transformations=[]`, yang TIDAK BEKERJA — lihat di bawah.

## Kenapa `transformations=[]` gagal diam-diam

`VectorStoreIndex.from_documents` (llama_index/core/indices/base.py:109) menulis:

    transformations = transformations or Settings.transformations

Daftar kosong bersifat *falsy* di Python, jadi `[] or Settings.transformations`
menghasilkan `Settings.transformations` — `SentenceSplitter` default. Argumen
`transformations=[]` karenanya tidak mematikan apa pun; ia hanya terlihat
mematikan. Tidak ada peringatan, tidak ada galat.

Akibatnya di korpus nyata: chunk tabel besar (yang sengaja tidak dipecah oleh
`_chunk_elements` demi RCAA) dipecah ulang oleh parser kedua menjadi beberapa
node yang **mewarisi metadata induk apa adanya** — `chunk_id`, `chunk_index`,
dan `text_sha` identik, teksnya berbeda. Itulah sumber `chunk_id` duplikat.

## Kenapa bukan sekadar melewatkan Document apa adanya

Sebuah transformasi no-op yang mengembalikan `nodes` tanpa perubahan memang
menghasilkan satu keluaran per masukan, tetapi keluarannya bertipe `Document`,
bukan `TextNode`. Qdrant menyimpan tipe itu di payload (`_node_type`), sehingga
bentuk titik berubah dari yang selama ini dipakai jalur query. `PassthroughNodeParser`
memakai `build_nodes_from_splits` — helper yang sama dengan `SentenceSplitter` —
sehingga node yang dihasilkan identik bentuknya dengan sebelumnya: `TextNode`,
relasi SOURCE ke Document induk, metadata dan daftar kunci terkecuali terwarisi.
Satu-satunya perbedaan adalah teksnya tidak dipotong.

`MetadataMode.NONE` wajib: `BaseNode.get_content()` default-nya `MetadataMode.ALL`,
yang akan menyisipkan metadata ke dalam teks node.
"""

from __future__ import annotations

from typing import Any, List, Sequence

from llama_index.core.node_parser import NodeParser
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.utils import get_tqdm_iterable

from backend.config import INDEX_DISABLE_NODE_PARSER


class PassthroughNodeParser(NodeParser):
    """Satu Document -> tepat satu TextNode, teks utuh tanpa dipotong."""

    @classmethod
    def class_name(cls) -> str:
        return "PassthroughNodeParser"

    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        out: List[BaseNode] = []
        for node in get_tqdm_iterable(nodes, show_progress, "Meneruskan node"):
            text = node.get_content(metadata_mode=MetadataMode.NONE)
            out.extend(build_nodes_from_splits([text], node, id_func=self.id_func))
        return out


def build_transformations() -> List[Any] | None:
    """Daftar transformasi yang dipakai `indexing._embed_and_store`.

    Berada di modul ini — bukan di `indexing` — supaya
    `scripts/probe_rechunk.py` dapat menguji JALUR YANG SAMA tanpa menarik
    klien Qdrant. Probe versi sebelumnya menirukan efek `transformations=[]`
    dengan "1 Document = 1 node" alih-alih menjalankannya, sehingga tidak
    pernah melihat bahwa daftar kosong itu tidak berefek.

    `None` berarti biarkan LlamaIndex memakai `Settings.transformations` —
    perilaku produksi semula.

    PERHATIAN: kalau suatu saat butuh transformasi lain (mis. ekstraktor
    metadata), TAMBAHKAN ke daftar ini — jangan hapus `PassthroughNodeParser`,
    dan jangan mengembalikan daftar kosong.
    """
    return [PassthroughNodeParser()] if INDEX_DISABLE_NODE_PARSER else None
