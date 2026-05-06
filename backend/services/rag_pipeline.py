"""RAG pipeline service — retrieval, rerank, generation dengan history + RBAC + cache."""

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
    EMBED_MODEL,
    EMBED_DEVICE,
    EMBED_BATCH_SIZE,
    RERANKER_MODEL,
    RERANKER_TOP_N,
    QDRANT_COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SIMILARITY_TOP_K,
    SCORE_THRESHOLD,
    HISTORY_TURNS,
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

logger = logging.getLogger(__name__)

# Singletons
_retriever = None
_reranker = None

# ─── Routing keyword sets ────────────────────────────────────────────────────

_CHITCHAT_EXACT = {
    "halo", "hai", "hello", "hi", "hey",
    "selamat pagi", "selamat siang", "selamat sore", "selamat malam",
    "terima kasih", "makasih", "thanks", "thank you",
    "apa kabar", "bye", "dadah", "sampai jumpa",
    "ok", "oke", "baik", "mantap", "sip",
}

_IDENTITY_KEYWORDS = [
    "siapa kamu", "siapa anda", "kamu siapa", "anda siapa",
    "nama kamu", "nama anda", "namamu", "namanya",
    "siapa nama", "apa namamu",
    "kamu ini", "kamu apa", "anda ini",
    "dari mana kamu", "dari mana anda", "asalmu", "asalnya",
    "asal kamu", "asal anda",
    "siapa yang buat", "siapa yang membuat", "siapa pembuat", "siapa penciptamu",
    "siapa yang ciptakan", "siapa yang menciptakan", "dibuat oleh", "diciptakan oleh",
    "yang membuat kamu", "yang membuat anda", "kamu dibuat", "anda dibuat",
    "apa itu kamu", "apa itu anda",
    "kamu robot", "anda robot", "kamu ai", "anda ai",
    "kamu bot", "anda bot",
]

_HARMFUL_KEYWORDS = [
    "bom", "peledak", "ledak", "dinamit", "granat",
    "senjata", "pistol", "senapan", "peluru",
    "narkoba", "narkotika", "sabu", "ekstasi", "heroin",
    "racun", "bunuh", "membunuh", "pembunuhan",
    "hack", "hacking", "malware", "virus komputer", "exploit",
    "cara merakit", "cara membuat bahan", "sintesis bahan",
    "prompt injection", "abaikan instruksi", "ignore instruction",
    "forget instruction", "new instruction", "jangan ikuti",
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


def _is_chitchat(question: str) -> bool:
    return _normalize(question) in _CHITCHAT_EXACT


def _is_identity_question(question: str) -> bool:
    q = _normalize(question)
    return any(kw in q for kw in _IDENTITY_KEYWORDS)


def _is_harmful(question: str) -> bool:
    q = _normalize(question)
    return any(kw in q for kw in _HARMFUL_KEYWORDS)


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


def _condense_question(history: list[dict], question: str) -> tuple[str, bool]:
    """Return (condensed_question, is_acknowledgment)."""
    if not history:
        return question, False

    history_str = _format_history(history)
    prompt = CONDENSE_PROMPT.format(chat_history=history_str, question=question)
    llm = get_llm()
    text = str(llm.complete(prompt)).strip()

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

    logger.info(f"Condensed: '{question[:60]}' → '{text[:120]}'")

    if text.startswith("<ACK>") or len(text) < 3:
        return question, True

    return text, False


# ─── Pipeline setup ──────────────────────────────────────────────────────────

def _configure_settings():
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        device=EMBED_DEVICE,
        trust_remote_code=True,
        embed_batch_size=EMBED_BATCH_SIZE,
    )
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


def _get_reranker():
    global _reranker
    if _reranker is not None:
        return _reranker
    _reranker = SentenceTransformerRerank(
        model=RERANKER_MODEL,
        top_n=RERANKER_TOP_N,
        keep_retrieval_score=False,
    )
    return _reranker


def _retrieve_and_rerank(question: str, role: str = "public") -> tuple[list, float]:
    """Retrieve from Qdrant + rerank. Return (reranked_nodes, top_score).

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

    reranker = _get_reranker()
    labeler = SourceLabelPostprocessor()
    reranked = reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(question))
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
                     top_score: float = 0.0, condensed: str = None) -> dict:
        return {
            "answer": answer,
            "sources": sources or [],
            "condensed_question": condensed,
            "debug": {
                "mode": mode,
                "total_time_s": round(time.time() - t_start, 2),
                "model": LLM_MODEL,
                "top_score": top_score,
            },
        }

    # ── 1. Harmful check ─────────────────────────────────────────────────────
    if _is_harmful(question):
        logger.warning(f"Harmful request blocked: '{question}'")
        return _make_result(_pick(_HARMFUL_RESPONSES), mode="blocked")

    # ── 2. Chitchat / identity (always — even mid-session) ───────────────────
    if _is_identity_question(question):
        return _make_result(_chitchat_response(question, history), mode="identity")
    if _is_chitchat(question):
        return _make_result(_chitchat_response(question, history), mode="chitchat")

    # ── 3. Query condensation (kalau ada history) ─────────────────────────────
    condensed = question
    if history:
        condensed, is_ack = _condense_question(history[-(HISTORY_TURNS * 2):], question)
        if is_ack:
            answer = _chitchat_response(question, history)
            return _make_result(answer, mode="chitchat", condensed=condensed)

        # Harmful check pada condensed question
        if _is_harmful(condensed):
            return _make_result(_pick(_HARMFUL_RESPONSES), mode="blocked", condensed=condensed)

    # ── 4. Cache check (condensed first, then original as fallback) ──────────
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
                              sources=sources, top_score=top_score, condensed=condensed)
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
    answer = _format_answer(raw_answer)

    logger.info(
        f"RAG done in {round(time.time() - t_start, 2)}s "
        f"— top_score={top_score:.4f}, sources={len(sources)}"
    )

    result = _make_result(answer, mode="rag",
                          sources=sources, top_score=top_score, condensed=condensed)
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
                   extra: dict = None) -> dict:
        d = {
            "mode": mode,
            "total_time_s": round(time.time() - t_start, 2),
            "model": LLM_MODEL,
            "top_score": top_score,
        }
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

    # ── 1. Harmful ────────────────────────────────────────────────────────────
    if _is_harmful(question):
        yield from _fake_stream(_pick(_HARMFUL_RESPONSES),
                                _make_meta("", "blocked"))
        return

    # ── 2. Chitchat / identity ────────────────────────────────────────────────
    if _is_identity_question(question):
        answer = _chitchat_response(question, history)
        yield from _fake_stream(answer, _make_meta(answer, "identity"))
        return
    if _is_chitchat(question):
        answer = _chitchat_response(question, history)
        yield from _fake_stream(answer, _make_meta(answer, "chitchat"))
        return

    # ── 3. Condensation ───────────────────────────────────────────────────────
    condensed = question
    if history:
        condensed, is_ack = _condense_question(history[-(HISTORY_TURNS * 2):], question)
        if is_ack:
            answer = _chitchat_response(question, history)
            yield from _fake_stream(answer, _make_meta(answer, "chitchat", condensed=condensed))
            return
        if _is_harmful(condensed):
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
                       top_score=top_score, condensed=condensed, extra=extra),
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
            yield {"type": "token", "delta": delta}

    full_answer = _format_answer("".join(parts))
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
