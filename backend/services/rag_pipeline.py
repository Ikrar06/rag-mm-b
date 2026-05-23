"""RAG pipeline service — retrieval, rerank, generation dengan history + RBAC + cache."""

import json
import logging
import random
import re
import time
from typing import Generator, List, Optional

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.postprocessor.types import BaseNodePostprocessor, NodeWithScore, QueryBundle
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

from backend.config import (
    LLM_MODEL,
    EMBED_PROVIDER,
    EMBED_MODEL,
    EMBED_BASE_URL,
    EMBED_DEVICE,
    EMBED_BATCH_SIZE,
    EMBED_TIMEOUT,
    RERANKER_PROVIDER,
    RERANKER_MODEL,
    RERANKER_BASE_URL,
    RERANKER_TOP_N,
    QDRANT_COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SIMILARITY_TOP_K,
    SCORE_THRESHOLD,
    HISTORY_TURNS,
    NEIGHBOR_EXPANSION_ENABLED,
    NEIGHBOR_EXPANSION_RADIUS,
    MAX_EXPANDED_CHUNKS,
)
from backend.services.llm_factory import get_llm
from backend.prompts.templates import (
    RAG_USER_PROMPT,
    RAG_USER_PROMPT_WITH_HISTORY,
    CHITCHAT_SYSTEM_PROMPT,
    CONDENSE_PROMPT,
)
from backend.services.indexing import get_qdrant_client
from backend.services.cache import cache_get, cache_set
from backend.services.moderation import check_moderation_v2
from backend.services.intent_classifier import classify_intent
from backend.services.output_filter import filter_output, filter_token
from backend.services import keyword_filter

logger = logging.getLogger(__name__)

# Singletons
_retriever = None
_reranker = None

# ─── Routing keyword sets ────────────────────────────────────────────────────

# L1 keyword filter (hard_block / soft_flag / safe_context) di-load dari
# backend/config/blocked_keywords.yaml lewat module keyword_filter.

# Pertanyaan tentang identitas/teknologi bot — pre-route ke chitchat
# supaya tidak masuk ke RAG yang bisa bocor nama model/vendor.
# Pattern: combination of (subject "anda/kamu") + (identity keyword).
_IDENTITY_KEYWORDS = [
    "gpt", "qwen", "claude", "gemini", "llama", "mistral", "grok", "kimi",
    "deepseek", "alibaba", "openai", "anthropic", "meta", "google", "ernie",
    "model ai", "model bahasa", "ai apa", "llm apa", "teknologi apa",
    "versi berapa", "model apa", "powered by", "dibuat oleh", "dilatih oleh",
    "siapa pembuat", "siapa kamu", "siapa anda", "kamu siapa", "anda siapa",
    "kamu robot", "kamu bot", "kamu manusia", "kamu chatbot",
]

_RAG_KEYWORDS = {
    "ukt", "krs", "khs", "ipk", "ips", "sks", "kkn",
    "cuti", "prosedur", "syarat", "biaya", "beasiswa",
    "wisuda", "skripsi", "transkrip", "ijazah",
    "registrasi", "her-registrasi", "akademik",
    "mata kuliah", "mkpk", "mkwu", "pa ", "wali",
}

_ACADEMIC_ACRONYMS = {
    "MKPK": "Mata Kuliah Penguatan Kompetensi",
    "MKWU": "Mata Kuliah Wajib Universitas",
    "KKN": "Kuliah Kerja Nyata",
    "KRS": "Kartu Rencana Studi",
    "KHS": "Kartu Hasil Studi",
    "UKT": "Uang Kuliah Tunggal",
    "IPS": "Indeks Prestasi Semester",
    "IPK": "Indeks Prestasi Kumulatif",
    "SKS": "Satuan Kredit Semester",
    "SOP": "Standar Operasional Prosedur",
    "PA": "Penasehat Akademik",
    "WD": "Wakil Dekan",
}

# ─── Canned responses ────────────────────────────────────────────────────────

_HARMFUL_RESPONSES = [
    "Maaf, saya tidak dapat membantu dengan permintaan tersebut. Saya hanya melayani pertanyaan seputar akademik Universitas Hasanuddin.",
    "Permintaan tersebut di luar cakupan saya. Saya hanya bisa membantu dengan informasi akademik UNHAS.",
    "Maaf, itu bukan topik yang bisa saya bantu. Kalau ada pertanyaan seputar akademik UNHAS, saya siap membantu.",
    "Saya tidak bisa membantu dengan hal itu. Fokus saya adalah informasi akademik Universitas Hasanuddin.",
]


def _pick(responses: list) -> str:
    return random.choice(responses)


# ─── Routing helpers ─────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return text.strip().lower().rstrip("?!.,")


def _check_keyword_filter(question: str) -> keyword_filter.KeywordFilterResult:
    """L1 keyword filter: hard_block (tolak), soft_flag (lanjut L2 dgn flag),
    safe_context downgrade (lanjut normal).

    Lihat backend/services/keyword_filter.py.
    """
    return keyword_filter.check(question)


def _is_identity_question(question: str) -> bool:
    """Cek apakah user nanya identitas/teknologi bot.

    Pre-route force ke chitchat supaya tidak masuk RAG (yang bisa bocor
    nama model/vendor walau output filter sudah redact ke placeholder).
    """
    q = _normalize(question)
    return any(kw in q for kw in _IDENTITY_KEYWORDS)


def _needs_rag_for_vision_query(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _RAG_KEYWORDS)


# ─── Post-processors ─────────────────────────────────────────────────────────

class SourceLabelPostprocessor(BaseNodePostprocessor):
    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        _ = query_bundle
        for node in nodes:
            file_name = node.metadata.get("file_name", "Dokumen UNHAS")
            clean_name = file_name.rsplit(".", 1)[0]
            parts = clean_name.rsplit("_", 1)
            if len(parts) == 2 and len(parts[1]) >= 8:
                clean_name = parts[0]
            clean_name = clean_name.replace("_", " ")
            node.node.text = f"[{clean_name}]\n{node.node.text}"
        return nodes


# ─── Query helpers ────────────────────────────────────────────────────────────

def _expand_query(question: str) -> str:
    """Expand singkatan akademik supaya retrieval lebih akurat.
    Pakai word boundary (\b) agar 'PA' tidak match substring di 'tempat', dll.
    """
    expanded = question
    q_upper = question.upper()
    for acronym, full in _ACADEMIC_ACRONYMS.items():
        if re.search(r'\b' + re.escape(acronym) + r'\b', q_upper):
            expanded += f" ({full})"
    return expanded


def _format_history(history: list[dict]) -> str:
    """Format chat history untuk injection ke prompt."""
    if not history:
        return ""
    lines = []
    for msg in history:
        role_label = "Mahasiswa" if msg["role"] == "user" else "Asisten"
        lines.append(f"{role_label}: {msg['content']}")
    return "\n".join(lines)


def _format_answer(text: str) -> str:
    text = re.sub(r'(?<=[^\n])\s+([*\-])\s+(?=\w)', r'\n\1 ', text)
    text = re.sub(r'(?<=[^\n])\s+(\d+)\.\s+(?=\w)', r'\n\1. ', text)

    lines = text.split('\n')
    result = []
    for i, line in enumerate(lines):
        is_list_item = re.match(r'^\s*([*\-]|\d+\.)\s', line)
        prev_is_list = i > 0 and re.match(r'^\s*([*\-]|\d+\.)\s', lines[i - 1])
        if is_list_item and i > 0 and not prev_is_list and lines[i - 1].strip():
            if not (result and result[-1] == ''):
                result.append('')
        result.append(line)
    return '\n'.join(result).strip()


# ─── LLM helpers ─────────────────────────────────────────────────────────────

def _count_trailing_repeats(question: str, history: list[dict]) -> int:
    """Count consecutive identical user messages at the end of history."""
    q_norm = _normalize(question)
    count = 0
    for msg in reversed(history):
        if msg["role"] != "user":
            continue
        if _normalize(msg["content"]) == q_norm:
            count += 1
        else:
            break
    return count


def _chitchat_response(question: str, history: list[dict] = None) -> str:
    """Generate chitchat response, aware of recent conversation context."""
    llm = get_llm(temperature=0.8)
    history = history or []

    history_block = ""
    if history:
        lines = []
        for msg in history[-6:]:
            role = "Mahasiswa" if msg["role"] == "user" else "Asisten"
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            lines.append(f"{role}: {content}")
        history_block = "Riwayat:\n" + "\n".join(lines) + "\n\n"

    trailing = _count_trailing_repeats(question, history)

    if trailing >= 2:
        # Bypass CHITCHAT_SYSTEM_PROMPT — model terlalu terikat ke contoh di sana
        prompt = (
            f"Kamu asisten akademik UNHAS yang ramah dan natural.\n\n"
            f"{history_block}"
            f"Mahasiswa baru saja mengirim '{question}' untuk ke-{trailing + 1} kalinya berturut-turut "
            f"tanpa pertanyaan nyata di antaranya.\n"
            f"Bereaksilah secara natural — boleh heran, bercanda ringan, atau langsung tanya apa yang "
            f"sebenarnya mereka butuhkan. JANGAN mulai dengan 'Halo! Selamat datang...' lagi.\n"
            f"Respons 1 kalimat, casual:\n"
        )
    elif trailing == 1:
        prompt = (
            f"{CHITCHAT_SYSTEM_PROMPT}\n\n"
            f"---\n"
            f"{history_block}"
            f"Pesan mahasiswa: {question}\n\n"
            f"Pengguna sudah mengirim pesan yang sama sebelumnya — JANGAN ulangi respons yang persis sama, "
            f"variasikan dan lebih singkat.\n\n"
            f"Jawaban asisten (1 kalimat):"
        )
    else:
        prompt = (
            f"{CHITCHAT_SYSTEM_PROMPT}\n\n"
            f"---\n"
            f"{history_block}"
            f"Pesan mahasiswa: {question}\n\n"
            f"Jawaban asisten (1-2 kalimat, natural, ramah, sesuaikan dengan konteks riwayat):"
        )
    return str(llm.complete(prompt))


def _low_relevance_response(question: str) -> str:
    llm = get_llm(temperature=0.5)
    prompt = (
        "Kamu adalah asisten akademik Universitas Hasanuddin yang ramah dan jujur.\n\n"
        "ATURAN KETAT — WAJIB DIIKUTI:\n"
        "1. JANGAN mengarang fakta, angka, tahun, atau nama apapun. Jika tidak tahu, katakan tidak tahu.\n"
        "2. JANGAN sebut 'referensi', 'data saya', 'dokumen', 'data yang tersedia', atau alasan teknis apapun.\n"
        "3. JANGAN menjawab pertanyaan di luar akademik UNHAS — cukup sampaikan itu bukan area yang bisa kamu bantu.\n"
        "4. Untuk pertanyaan tentang UNHAS yang tidak kamu ketahui: akui tidak tahu dan arahkan ke website resmi UNHAS atau Bagian Akademik.\n"
        "5. Jawab 1-2 kalimat, natural, santai-formal, sapa dengan 'Anda'.\n\n"
        f"Pertanyaan mahasiswa: {question}\n\n"
        "Jawaban asisten:"
    )
    return str(llm.complete(prompt))


def _trim_history_by_tokens(history: list[dict], max_tokens: int = None) -> list[dict]:
    """Trim history dari paling lama sampai estimasi token di bawah limit."""
    from backend.config import HISTORY_MAX_TOKENS
    limit = max_tokens or HISTORY_MAX_TOKENS
    total_chars = sum(len(m.get("content", "")) for m in history)
    while total_chars > limit * 4 and len(history) > 2:
        removed = history.pop(0)
        total_chars -= len(removed.get("content", ""))
    return history


_PROMPT_LEAK_TOKENS = ("[INST]", "<<SYS>>", "<|", "```", "system:", "assistant:")


def _validate_condensed(original: str, condensed: str) -> str:
    """Validasi output condensation. Fallback ke original kalau:
    - Empty / terlalu pendek
    - > 3x panjang original (LLM verbose / hallucinate)
    - Contains template artifact (prompt leak)
    """
    if not condensed or len(condensed.strip()) < 5:
        logger.info("condense_fallback reason=too_short")
        return original
    if len(condensed) > max(len(original) * 3, 200):
        logger.info("condense_fallback reason=too_long len=%d orig=%d",
                    len(condensed), len(original))
        return original
    lc = condensed.lower()
    leaked = [t for t in _PROMPT_LEAK_TOKENS if t.lower() in lc]
    if leaked:
        logger.warning("condense_fallback reason=prompt_leak tokens=%s", leaked)
        return original
    return condensed


def _condense_question(history: list[dict], question: str) -> tuple[str, bool]:
    """Return (condensed_question, is_acknowledgment).

    Output di-validasi via _validate_condensed — kalau LLM bermasalah,
    fallback ke original question.
    """
    if not history:
        return question, False

    history_str = _format_history(history)
    prompt = CONDENSE_PROMPT.format(chat_history=history_str, question=question)
    try:
        llm = get_llm()
        text = str(llm.complete(prompt)).strip()
    except Exception as e:
        logger.error("condense_llm_error error=%s", e)
        return question, False  # fallback ke original, anggap bukan ack

    # Strip common verbose prefixes small LLMs echo from the prompt
    for prefix in ["Pertanyaan standalone:", "Standalone question:", "Pertanyaan mandiri:", "Pertanyaan:"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
            break

    # Take only the first non-empty line to discard explanation text
    for line in text.split('\n'):
        line = line.strip()
        if len(line) >= 5:
            text = line
            break

    if text.startswith("<ACK>") or len(text) < 3:
        return question, True

    validated = _validate_condensed(question, text)
    logger.info(f"Condensed: '{question[:60]}' → '{validated[:120]}'")
    return validated, False


# ─── Pipeline setup ──────────────────────────────────────────────────────────

def _configure_settings():
    if EMBED_PROVIDER == "tei":
        # POC: panggil TEI service via HTTP (model di-host di container terpisah)
        from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
        Settings.embed_model = TextEmbeddingsInference(
            model_name=EMBED_MODEL,
            base_url=EMBED_BASE_URL,
            embed_batch_size=EMBED_BATCH_SIZE,
            timeout=float(EMBED_TIMEOUT),
        )
        logger.info(f"embed_provider=tei base_url={EMBED_BASE_URL} batch={EMBED_BATCH_SIZE} timeout={EMBED_TIMEOUT}s")
    else:
        # Dev: in-process via HuggingFace (CUDA atau CPU sesuai EMBED_DEVICE)
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=EMBED_MODEL,
            device=EMBED_DEVICE,
            trust_remote_code=True,
            embed_batch_size=EMBED_BATCH_SIZE,
        )
        logger.info(f"embed_provider=huggingface device={EMBED_DEVICE}")
    Settings.llm = get_llm()
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP


def _get_retriever():
    global _retriever
    if _retriever is not None:
        return _retriever

    logger.info("Initializing retriever...")
    _configure_settings()

    qdrant_client = get_qdrant_client()
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=QDRANT_COLLECTION_NAME,
    )
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    _retriever = index.as_retriever(similarity_top_k=SIMILARITY_TOP_K)
    logger.info("Retriever ready.")
    return _retriever


class _TEIRerankPostprocessor(BaseNodePostprocessor):
    """Custom reranker postprocessor yang panggil TEI /rerank endpoint via HTTP."""

    base_url: str
    top_n: int
    timeout: float = 30.0

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        if not nodes or query_bundle is None:
            return nodes

        import httpx
        try:
            resp = httpx.post(
                f"{self.base_url.rstrip('/')}/rerank",
                json={
                    "query": query_bundle.query_str,
                    "texts": [n.node.get_content() for n in nodes],
                    "raw_scores": False,
                    "return_text": False,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            results = resp.json()
        except Exception as e:
            logger.error(f"tei_rerank_error error={e} — falling back to original order")
            return nodes[: self.top_n]

        # TEI returns [{"index": i, "score": s}, ...] sudah sorted desc by score
        reranked = []
        for r in results[: self.top_n]:
            idx = r.get("index")
            if idx is None or idx >= len(nodes):
                continue
            node = nodes[idx]
            node.score = float(r.get("score", 0.0))
            reranked.append(node)
        return reranked


def _get_reranker():
    global _reranker
    if _reranker is not None:
        return _reranker
    if RERANKER_PROVIDER == "tei":
        # POC: panggil TEI rerank service via HTTP
        _reranker = _TEIRerankPostprocessor(
            base_url=RERANKER_BASE_URL,
            top_n=RERANKER_TOP_N,
        )
        logger.info(f"reranker_provider=tei base_url={RERANKER_BASE_URL}")
    else:
        # Dev: in-process via sentence-transformers
        _reranker = SentenceTransformerRerank(
            model=RERANKER_MODEL,
            top_n=RERANKER_TOP_N,
            keep_retrieval_score=False,
        )
        logger.info(f"reranker_provider=sentence_transformers model={RERANKER_MODEL}")
    return _reranker


def _expand_with_neighbors(nodes: list) -> list:
    """Production-grade: fetch chunks tetangga (di file+section yang sama) untuk setiap
    node yang ter-retrieve, supaya konteks panjang tidak terpotong.

    Strategy:
    1. Untuk setiap retrieved node, identifikasi (file_name, section, chunk_index).
    2. Fetch chunks dengan file_name & section sama, dalam range chunk_index ± RADIUS.
    3. Dedup berdasarkan (file_name, chunk_index).
    4. Cap total expanded chunks di MAX_EXPANDED_CHUNKS supaya tidak overflow LLM context.

    Score neighbor di-set 0.0 — reranker akan re-score semua di tahap berikutnya.
    """
    if not NEIGHBOR_EXPANSION_ENABLED or not nodes:
        return nodes

    from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

    # Mapping seen chunks supaya tidak duplicate fetch
    seen: set[tuple[str, int]] = set()
    for n in nodes:
        fname = n.metadata.get("file_name")
        cidx = n.metadata.get("chunk_index")
        if fname and cidx is not None:
            seen.add((fname, int(cidx)))

    client = get_qdrant_client()
    expanded_nodes = list(nodes)  # start dengan retrieved nodes

    for node in nodes:
        if len(expanded_nodes) >= MAX_EXPANDED_CHUNKS:
            break

        fname = node.metadata.get("file_name")
        section = node.metadata.get("section")
        cidx = node.metadata.get("chunk_index")
        if not fname or cidx is None:
            continue

        cidx = int(cidx)
        range_min = max(0, cidx - NEIGHBOR_EXPANSION_RADIUS)
        range_max = cidx + NEIGHBOR_EXPANSION_RADIUS

        # Build Qdrant filter: same file + same section (kalau ada) + chunk_index dalam range
        conditions = [
            FieldCondition(key="file_name", match=MatchValue(value=fname)),
            FieldCondition(key="chunk_index", range=Range(gte=range_min, lte=range_max)),
        ]
        if section:
            conditions.append(FieldCondition(key="section", match=MatchValue(value=section)))

        try:
            results, _ = client.scroll(
                collection_name=QDRANT_COLLECTION_NAME,
                scroll_filter=Filter(must=conditions),
                limit=NEIGHBOR_EXPANSION_RADIUS * 2 + 1,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            logger.warning(f"neighbor_expansion_error file={fname} error={e}")
            continue

        for point in results:
            payload = point.payload or {}
            n_cidx = payload.get("chunk_index")
            if n_cidx is None:
                continue
            key = (fname, int(n_cidx))
            if key in seen:
                continue
            seen.add(key)

            # Extract text dari payload. LlamaIndex Qdrant kadang simpan
            # text di field "text" langsung, kadang di "_node_content" sebagai JSON string.
            text = payload.get("text") or ""
            if not text:
                node_content = payload.get("_node_content")
                if isinstance(node_content, str):
                    try:
                        text = json.loads(node_content).get("text", "")
                    except (json.JSONDecodeError, AttributeError):
                        text = ""
                elif isinstance(node_content, dict):
                    text = node_content.get("text", "")
            if not text:
                continue

            # Bangun NodeWithScore manual dari payload
            from llama_index.core.schema import TextNode, NodeWithScore as NWS
            new_node = TextNode(text=text, metadata=payload)
            expanded_nodes.append(NWS(node=new_node, score=0.0))

            if len(expanded_nodes) >= MAX_EXPANDED_CHUNKS:
                break

    if len(expanded_nodes) > len(nodes):
        logger.info(
            f"neighbor_expansion retrieved={len(nodes)} expanded={len(expanded_nodes)} "
            f"radius={NEIGHBOR_EXPANSION_RADIUS}"
        )

    return expanded_nodes


def _retrieve_and_rerank(question: str, role: str = "public") -> tuple[list, float]:
    """Retrieve from Qdrant + neighbor expansion + rerank. Return (reranked_nodes, top_score).

    Pipeline:
    1. Vector search → top SIMILARITY_TOP_K chunks
    2. Neighbor expansion → tambah chunks tetangga per retrieved (radius NEIGHBOR_EXPANSION_RADIUS)
    3. Reranker → re-score semua kandidat, ambil top RERANKER_TOP_N
    4. Source labeling → tambah label nama file untuk konteks LLM

    RBAC: saat dokumen sudah punya metadata 'access_level', tambah filter di sini.
    Untuk sekarang: semua dokumen dianggap 'public' — RBAC framework siap tapi belum aktif.
    """
    # TODO POC RBAC: uncomment saat dokumen sudah diklasifikasi dengan access_level metadata
    # from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter
    # from backend.services.auth import get_role_access_levels
    # allowed = get_role_access_levels(role)
    # filters = MetadataFilters(filters=[MetadataFilter(key="access_level", value=level) for level in allowed])
    # retriever = _get_retriever()  # pass filters to retriever

    _ = role  # placeholder — aktifkan RBAC filter di sini saat docs sudah diklasifikasi

    retriever = _get_retriever()
    nodes = retriever.retrieve(question)

    if not nodes:
        return [], 0.0

    # Step 2: neighbor expansion untuk handle PDF panjang (konteks utuh)
    expanded_nodes = _expand_with_neighbors(nodes)

    # Step 3: reranker re-score semua kandidat (retrieved + neighbors)
    reranker = _get_reranker()
    labeler = SourceLabelPostprocessor()
    reranked = reranker.postprocess_nodes(expanded_nodes, query_bundle=QueryBundle(question))
    reranked = labeler.postprocess_nodes(reranked)

    top_score = float(max((n.score for n in reranked if n.score is not None), default=0.0))
    return reranked, top_score


def _build_sources(nodes: list) -> list[dict]:
    return [
        {
            "file_name": n.metadata.get("file_name", "unknown"),
            "page": n.metadata.get("page"),
            "chunk_index": n.metadata.get("chunk_index"),
            "element_type": n.metadata.get("element_type"),
            "score": round(float(n.score), 4) if n.score is not None else 0.0,
            "text_preview": n.text[:300],
        }
        for n in nodes
    ]


# ─── Main entry point ────────────────────────────────────────────────────────

def query(
    question: str,
    history: Optional[list[dict]] = None,
    role: str = "public",
) -> dict:
    """Run RAG query. Supports multi-turn (history) + RBAC (role) + cache."""
    t_start = time.time()
    history = history or []

    def _make_result(answer: str, mode: str, sources: list = None,
                     top_score: float = 0.0, condensed: str = None,
                     intent: str = None, intent_conf: float = None) -> dict:
        d = {
            "mode": mode,
            "total_time_s": round(time.time() - t_start, 2),
            "model": LLM_MODEL,
            "top_score": top_score,
        }
        if intent is not None:
            d["intent"] = intent
            d["intent_confidence"] = intent_conf
        return {
            "answer": answer,
            "sources": sources or [],
            "condensed_question": condensed,
            "debug": d,
        }

    # ── 1. Layer 1 — Keyword filter (hard_block / soft_flag / safe_context) ──
    l1 = _check_keyword_filter(question)
    if l1.is_blocked:
        return _make_result(_pick(_HARMFUL_RESPONSES), mode="blocked")

    # ── 2. Layer 2 — Moderation model (Llama Guard + circuit breaker) ────────
    # Kalau L1 menghasilkan soft-flag (mis. "narkoba" tanpa konteks akademik),
    # L2 jadi judgment kontekstual. Circuit breaker fail-open kalau Guard down —
    # query tetap diproses tapi log mencatat bypassed.
    mod = check_moderation_v2(question)
    if mod.bypassed:
        logger.warning("L2_bypassed reason=%s", mod.reason)
    if not mod.safe:
        logger.warning(
            "L2_moderation_blocked reason=%s l1_flags=%s",
            mod.reason,
            l1.matched_phrases if l1.is_flagged else [],
        )
        return _make_result(
            "Maaf, saya tidak dapat memproses permintaan tersebut.",
            mode="blocked_moderation",
        )

    # ── 3. Layer 3 — Intent classification (IndoBERT) ─────────────────────────
    # Pre-route: pertanyaan identity → force ke chitchat (bypass classifier
    # yang kadang salah classify pertanyaan model identity)
    if _is_identity_question(question):
        logger.info(f"L3_identity_override → chitchat question={question[:60]!r}")
        answer = _chitchat_response(question, history)
        return _make_result(answer, mode="chitchat",
                            intent="chitchat", intent_conf=1.0)

    intent_result = classify_intent(question)
    intent = intent_result["intent"]
    if intent_result.get("fallback"):
        logger.warning("L3_using_fallback intent=%s", intent)
    logger.info(f"L3_intent={intent} conf={intent_result['confidence']:.3f}")

    _intent = intent
    _intent_conf = intent_result["confidence"]

    if intent_result["low_confidence"]:
        return _make_result(
            "Maaf, bisa diperjelas maksud pertanyaannya?",
            mode="clarification_needed",
            intent=_intent, intent_conf=_intent_conf,
        )
    if intent == "chitchat":
        answer = filter_output(_chitchat_response(question, history))
        return _make_result(answer, mode="chitchat", intent=_intent, intent_conf=_intent_conf)
    if intent == "out_of_scope":
        return _make_result(
            "Maaf, saya hanya dapat membantu urusan akademik UNHAS. "
            "Untuk pertanyaan lain, silakan gunakan layanan yang sesuai.",
            mode="out_of_scope",
            intent=_intent, intent_conf=_intent_conf,
        )
    if intent == "get_info_private":
        from backend.services.private_api import handle_private_query
        private_result = handle_private_query(question, user_token="")
        if private_result:
            mode = private_result.pop("mode", "get_info_private")
            return {
                **private_result,
                "condensed_question": None,
                "debug": {
                    "mode": mode,
                    "total_time_s": round(time.time() - t_start, 2),
                    "model": LLM_MODEL,
                    "top_score": 0.0,
                    "intent": _intent,
                    "intent_confidence": _intent_conf,
                },
            }
        # private API tidak dikonfigurasi → fall through ke RAG

    # intent == "get_info_public" (atau private tanpa API) → lanjut ke RAG

    # ── 4. Query condensation (kalau ada history) ─────────────────────────────
    condensed = question
    if history:
        trimmed_history = _trim_history_by_tokens(list(history[-(HISTORY_TURNS * 2):]))
        condensed, is_ack = _condense_question(trimmed_history, question)
        if is_ack:
            answer = _chitchat_response(question, history)
            return _make_result(answer, mode="chitchat", condensed=condensed,
                                intent=_intent, intent_conf=_intent_conf)

        # Re-check L1 pada condensed (LLM bisa transform query jadi harmful)
        if _check_keyword_filter(condensed).is_blocked:
            return _make_result(_pick(_HARMFUL_RESPONSES), mode="blocked", condensed=condensed)

    # ── 5. Cache check (condensed first, then original as fallback) ──────────
    cached = cache_get(condensed, role=role)
    if not cached and condensed != question:
        cached = cache_get(question, role=role)
    if cached:
        logger.info(f"Cache hit for: '{condensed[:60]}'")
        return {
            **cached,
            "condensed_question": condensed,
            "debug": {**cached.get("debug", {}), "mode": "cache_hit",
                      "total_time_s": round(time.time() - t_start, 2)},
        }

    # ── 5. RAG retrieval + rerank ─────────────────────────────────────────────
    expanded = _expand_query(condensed)
    if expanded != condensed:
        logger.info(f"Query expanded: '{condensed}' → '{expanded}'")

    reranked_nodes, top_score = _retrieve_and_rerank(expanded, role=role)
    sources = _build_sources(reranked_nodes)

    # ── 6. Score threshold ────────────────────────────────────────────────────
    if top_score < SCORE_THRESHOLD or not reranked_nodes:
        logger.info(f"Top score {top_score} < {SCORE_THRESHOLD} — low-relevance fallback")
        answer = _low_relevance_response(condensed)
        result = _make_result(answer, mode="rag_low_relevance",
                              sources=sources, top_score=top_score, condensed=condensed,
                              intent=_intent, intent_conf=_intent_conf)
        result["debug"].update({
            "similarity_top_k": SIMILARITY_TOP_K,
            "reranker_top_n": RERANKER_TOP_N,
            "sources_returned": len(sources),
        })
        return result

    # ── 7. Build context + prompt ─────────────────────────────────────────────
    context_str = "\n\n".join(n.text for n in reranked_nodes)
    history_str = _format_history(history[-(HISTORY_TURNS * 2):])

    if history_str:
        prompt = RAG_USER_PROMPT_WITH_HISTORY.format(
            chat_history=history_str,
            context_str=context_str,
            query_str=question,
        )
    else:
        prompt = RAG_USER_PROMPT.format(context_str=context_str, query_str=condensed)

    # ── 8. LLM generate ──────────────────────────────────────────────────────
    llm = get_llm()
    raw_answer = str(llm.complete(prompt))
    answer = filter_output(_format_answer(raw_answer))

    logger.info(
        f"RAG done in {round(time.time() - t_start, 2)}s "
        f"— top_score={top_score:.4f}, sources={len(sources)}"
    )

    result = _make_result(answer, mode="rag",
                          sources=sources, top_score=top_score, condensed=condensed,
                          intent=_intent, intent_conf=_intent_conf)
    result["debug"].update({
        "similarity_top_k": SIMILARITY_TOP_K,
        "reranker_top_n": RERANKER_TOP_N,
        "sources_returned": len(sources),
    })

    # ── 9. Cache store ────────────────────────────────────────────────────────
    # Store by condensed (canonical standalone question)
    cache_set(condensed, answer, sources, role=role, debug=result["debug"])
    # Also store by original question so repeated same-question hits cache
    # even when condensation produces a slightly different form
    if condensed != question:
        cache_set(question, answer, sources, role=role, debug=result["debug"])

    return result


# ─── Streaming entry point ────────────────────────────────────────────────────

def query_stream(
    question: str,
    history: Optional[list[dict]] = None,
    role: str = "public",
) -> Generator[dict, None, None]:
    """Streaming RAG query. Yields event dicts:
      {'type': 'token',   'delta': str}
      {'type': 'meta',    'answer': str, 'sources': list, 'debug': dict,
                          'condensed_question': str}
    """
    t_start = time.time()
    history = history or []

    def _make_meta(answer: str, mode: str, sources: list = None,
                   top_score: float = 0.0, condensed: str = None,
                   extra: dict = None, intent: str = None, intent_conf: float = None) -> dict:
        d = {
            "mode": mode,
            "total_time_s": round(time.time() - t_start, 2),
            "model": LLM_MODEL,
            "top_score": top_score,
        }
        if intent is not None:
            d["intent"] = intent
            d["intent_confidence"] = intent_conf
        if extra:
            d.update(extra)
        return {
            "type": "meta",
            "answer": answer,
            "sources": sources or [],
            "condensed_question": condensed,
            "debug": d,
        }

    def _fake_stream(text: str, meta: dict, delay: float = 0.03):
        """Yield text word-by-word with small delay to simulate typing."""
        for word in text.split(" "):
            if word:
                yield {"type": "token", "delta": word + " "}
                time.sleep(delay)
        yield meta

    # ── 1. Layer 1 — Keyword filter (hard_block / soft_flag / safe_context) ──
    if _check_keyword_filter(question).is_blocked:
        yield from _fake_stream(_pick(_HARMFUL_RESPONSES), _make_meta("", "blocked"))
        return

    # ── 2. Layer 2 — Moderation model (Llama Guard + circuit breaker) ────────
    mod = check_moderation_v2(question)
    if mod.bypassed:
        logger.warning("L2_bypassed reason=%s", mod.reason)
    if not mod.safe:
        logger.warning("L2_moderation_blocked reason=%s", mod.reason)
        msg = "Maaf, saya tidak dapat memproses permintaan tersebut."
        yield from _fake_stream(msg, _make_meta(msg, "blocked_moderation"))
        return

    # ── 3. Layer 3 — Intent classification (IndoBERT) ─────────────────────────
    # Pre-route: pertanyaan identity → force ke chitchat
    if _is_identity_question(question):
        logger.info(f"L3_identity_override → chitchat question={question[:60]!r}")
        answer = _chitchat_response(question, history)
        yield from _fake_stream(answer, _make_meta(answer, "chitchat",
                                                    intent="chitchat", intent_conf=1.0))
        return

    intent_result = classify_intent(question)
    intent = intent_result["intent"]
    _intent = intent
    _intent_conf = intent_result["confidence"]
    logger.info(f"L3_intent={intent} conf={_intent_conf:.3f}")

    if intent_result["low_confidence"]:
        msg = "Maaf, bisa diperjelas maksud pertanyaannya?"
        yield from _fake_stream(msg, _make_meta(msg, "clarification_needed",
                                                intent=_intent, intent_conf=_intent_conf))
        return
    if intent == "chitchat":
        answer = filter_output(_chitchat_response(question, history))
        yield from _fake_stream(answer, _make_meta(answer, "chitchat",
                                                   intent=_intent, intent_conf=_intent_conf))
        return
    if intent == "out_of_scope":
        msg = ("Maaf, saya hanya dapat membantu urusan akademik UNHAS. "
               "Untuk pertanyaan lain, silakan gunakan layanan yang sesuai.")
        yield from _fake_stream(msg, _make_meta(msg, "out_of_scope",
                                                intent=_intent, intent_conf=_intent_conf))
        return
    if intent == "get_info_private":
        from backend.services.private_api import handle_private_query
        private_result = handle_private_query(question, user_token="")
        if private_result:
            answer = private_result.get("answer", "")
            mode = private_result.get("mode", "get_info_private")
            yield from _fake_stream(answer, _make_meta(answer, mode,
                                                       intent=_intent, intent_conf=_intent_conf))
            return
        # private API tidak dikonfigurasi → fall through ke RAG

    # intent == "get_info_public" (atau private tanpa API) → lanjut ke RAG

    # ── 4. Condensation ────────────────────────────────────────────────────────
    condensed = question
    if history:
        trimmed_history = _trim_history_by_tokens(list(history[-(HISTORY_TURNS * 2):]))
        condensed, is_ack = _condense_question(trimmed_history, question)
        if is_ack:
            answer = _chitchat_response(question, history)
            yield from _fake_stream(answer, _make_meta(answer, "chitchat", condensed=condensed,
                                                       intent=_intent, intent_conf=_intent_conf))
            return
        if _check_keyword_filter(condensed).is_blocked:
            yield from _fake_stream(_pick(_HARMFUL_RESPONSES),
                                    _make_meta("", "blocked", condensed=condensed))
            return

    # ── 4. Cache check ────────────────────────────────────────────────────────
    cached = cache_get(condensed, role=role)
    if not cached and condensed != question:
        cached = cache_get(question, role=role)
    if cached:
        answer = cached["answer"]
        meta = {
            "type": "meta",
            "answer": answer,
            "sources": cached.get("sources", []),
            "condensed_question": condensed,
            "debug": {**cached.get("debug", {}), "mode": "cache_hit",
                      "total_time_s": round(time.time() - t_start, 2)},
        }
        yield from _fake_stream(answer, meta, delay=0.015)  # cache: faster
        return

    # ── 5. Retrieval + rerank ─────────────────────────────────────────────────
    expanded = _expand_query(condensed)
    reranked_nodes, top_score = _retrieve_and_rerank(expanded, role=role)
    sources = _build_sources(reranked_nodes)

    # ── 6. Score threshold ────────────────────────────────────────────────────
    if top_score < SCORE_THRESHOLD or not reranked_nodes:
        answer = _low_relevance_response(condensed)
        extra = {"similarity_top_k": SIMILARITY_TOP_K,
                 "reranker_top_n": RERANKER_TOP_N,
                 "sources_returned": len(sources)}
        yield from _fake_stream(
            answer,
            _make_meta(answer, "rag_low_relevance", sources=sources,
                       top_score=top_score, condensed=condensed, extra=extra,
                       intent=_intent, intent_conf=_intent_conf),
        )
        return

    # ── 7. Build prompt ───────────────────────────────────────────────────────
    context_str = "\n\n".join(n.text for n in reranked_nodes)
    history_str = _format_history(history[-(HISTORY_TURNS * 2):])
    if history_str:
        prompt = RAG_USER_PROMPT_WITH_HISTORY.format(
            chat_history=history_str, context_str=context_str, query_str=question)
    else:
        prompt = RAG_USER_PROMPT.format(context_str=context_str, query_str=condensed)

    # ── 8. Stream LLM tokens ─────────────────────────────────────────────────
    llm = get_llm()
    parts: list[str] = []
    for token_resp in llm.stream_complete(prompt):
        delta = token_resp.delta
        if delta:
            parts.append(delta)
            yield {"type": "token", "delta": filter_token(delta)}

    full_answer = filter_output(_format_answer("".join(parts)))
    logger.info(f"Stream RAG done in {round(time.time() - t_start, 2)}s "
                f"— top_score={top_score:.4f}, sources={len(sources)}")

    # ── 9. Cache store ────────────────────────────────────────────────────────
    debug_dict = {
        "mode": "rag",
        "total_time_s": round(time.time() - t_start, 2),
        "model": LLM_MODEL,
        "top_score": top_score,
        "similarity_top_k": SIMILARITY_TOP_K,
        "reranker_top_n": RERANKER_TOP_N,
        "sources_returned": len(sources),
        "intent": _intent,
        "intent_confidence": _intent_conf,
    }
    cache_set(condensed, full_answer, sources, role=role, debug=debug_dict)
    if condensed != question:
        cache_set(question, full_answer, sources, role=role, debug=debug_dict)

    # ── 10. Meta event ────────────────────────────────────────────────────────
    yield {
        "type": "meta",
        "answer": full_answer,
        "sources": sources,
        "condensed_question": condensed,
        "debug": debug_dict,
    }
