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


def filter_output(text: str, session_id: str = "") -> str:
    """Scan dan redact informasi sensitif dari response LLM."""
    filtered = text
    for pattern, replacement, label in _PATTERNS:
        before = filtered
        filtered = re.sub(pattern, replacement, filtered, flags=re.IGNORECASE)
        if filtered != before:
            logger.warning(f"output_filter_redacted pattern={label} session={session_id}")
    return filtered


def filter_token(delta: str) -> str:
    """Filter single token delta — dipakai saat streaming real-time."""
    return filter_output(delta)
