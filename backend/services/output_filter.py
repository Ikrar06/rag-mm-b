"""Layer 6 — Output filter: redact sensitive info dari response LLM.

Mencegah LLM secara tidak sengaja menyebut nama model AI, tech stack internal,
pola NIM, JWT token, atau URL internal.
"""

import re
import logging

logger = logging.getLogger(__name__)

_PATTERNS = [
    # Nama model/vendor AI
    (r'\b(claude|gpt-?4?o?|gemini|llama|qwen|mistral|indobert|anthropic|openai)\b',
     "[AI system]", "ai_name"),
    # Tech stack internal
    (r'\b(qdrant|fastapi|ollama|vllm|llama[\s_]?index|langchain|redis|uvicorn|sqlalchemy)\b',
     "[internal system]", "tech_stack"),
    # Format NIM mahasiswa (huruf kapital + 2 digit + huruf + 6-7 digit)
    (r'\b[A-Z]\d{2}[A-Z]\d{6,7}\b', "[NIM]", "nim_pattern"),
    # JWT token
    (r'eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}', "[TOKEN]", "jwt_token"),
    # URL internal
    (r'https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+)[^\s]*',
     "[internal URL]", "internal_url"),
]

# Frasa internal yang kadang bocor dari context/prompt — hapus seluruh kalimat yang menyebutkannya
_LEAKED_INTERNAL_REFS = [
    r'Informasi ini[^.]*?bagian\s*"?INFORMASI RESMI UNHAS"?[^.]*?\.',
    r'(?:Berdasarkan|Menurut)\s+(?:bagian\s+)?"?INFORMASI RESMI UNHAS"?[^.]*?\.',
    r'(?:di|pada)\s+(?:bagian\s+)?"?INFORMASI RESMI UNHAS"?[^.]*?\.',
    r'(?:Informasi ini|Data ini|Sumber ini)\s+(?:dapat|bisa)\s+Anda\s+peroleh\s+dari[^.]*?\.',
    r'(?:ringkasan|data)\s+(?:yang\s+)?(?:disediakan|tersedia)\s+(?:dalam|di)\s+(?:bagian\s+)?[^.]*?\.',
]

# Prefix label dari prompt template yang kadang bocor ke jawaban (LLM mimic format)
_LEAKED_PREFIXES = [
    r'^\s*(?:jawaban|jawab|format|panduan format|format jawaban)\s*[:：]\s*',
    r'^\s*\d+[-–]?\d*\s*kalimat\s*[:：]\s*',
    r'^\s*(?:paragraf biasa|bullet|nomor|tanpa bullet)\s*[:：]\s*',
]


def _strip_leaked_prefix(text: str, session_id: str = "") -> str:
    """Hapus prefix label format yang bocor dari prompt instruction."""
    cleaned = text
    for pattern in _LEAKED_PREFIXES:
        new = re.sub(pattern, "", cleaned, count=1, flags=re.IGNORECASE)
        if new != cleaned:
            logger.warning(f"output_filter_stripped_prefix session={session_id}")
            cleaned = new
    return cleaned


def _strip_internal_refs(text: str, session_id: str = "") -> str:
    """Hapus kalimat yang menyebut sumber internal (mis. 'INFORMASI RESMI UNHAS')."""
    cleaned = text
    for pattern in _LEAKED_INTERNAL_REFS:
        new = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        if new != cleaned:
            logger.warning(f"output_filter_stripped_internal_ref session={session_id}")
            cleaned = new
    # Cleanup extra whitespace setelah penghapusan
    cleaned = re.sub(r'\s+\n', '\n', cleaned)
    cleaned = re.sub(r'\n\s*\n+', '\n\n', cleaned)
    return cleaned.strip()


def filter_output(text: str, session_id: str = "") -> str:
    """Scan dan redact informasi sensitif dari response LLM."""
    filtered = _strip_leaked_prefix(text, session_id)
    filtered = _strip_internal_refs(filtered, session_id)
    for pattern, replacement, label in _PATTERNS:
        before = filtered
        filtered = re.sub(pattern, replacement, filtered, flags=re.IGNORECASE)
        if filtered != before:
            logger.warning(f"output_filter_redacted pattern={label} session={session_id}")
    return filtered


def filter_token(delta: str) -> str:
    """Filter single token delta — dipakai saat streaming real-time."""
    return filter_output(delta)
