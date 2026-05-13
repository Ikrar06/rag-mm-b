"""Layer 2 — Content moderation via Llama Guard 3.

Dev:  MODERATION_BACKEND=passthrough  (skip, no HTTP call)
POC:  MODERATION_BACKEND=ollama       (llama-guard3:1b via local Ollama)
"""

import logging
from enum import Enum

import httpx

from backend.config import LLM_BASE_URL, MODERATION_BACKEND, MODERATION_MODEL

logger = logging.getLogger(__name__)


class ModerationBackend(str, Enum):
    OLLAMA = "ollama"
    PASSTHROUGH = "passthrough"


_BACKEND = ModerationBackend(MODERATION_BACKEND)

_PROMPT = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Task: Check if there is unsafe content in the message according to our safety policy.

<BEGIN_UNSAFE_CONTENT_CATEGORIES>
S1: Violent Crimes
S2: Non-Violent Crimes
S6: Specialized Advice (dangerous medical/legal/financial)
S9: Indiscriminate Weapons
S10: Hate Speech
S11: Suicide & Self-Harm
<END_UNSAFE_CONTENT_CATEGORIES>

<BEGIN_CONVERSATION>
User: {message}
<END_CONVERSATION>

Provide your safety assessment:
- First line must read 'safe' or 'unsafe'.
- If unsafe, second line must include violated categories.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


def check_moderation(text: str) -> tuple[bool, str]:
    """Return (is_safe, reason). Fail-closed: error → not safe."""
    if _BACKEND == ModerationBackend.PASSTHROUGH:
        return True, ""
    return _check_ollama(text)


def _check_ollama(text: str) -> tuple[bool, str]:
    prompt = _PROMPT.format(message=text[:500])
    try:
        resp = httpx.post(
            f"{LLM_BASE_URL}/api/generate",
            json={"model": MODERATION_MODEL, "prompt": prompt, "stream": False},
            timeout=8.0,
        )
        resp.raise_for_status()
        output = resp.json()["response"].strip().lower()
    except Exception as e:
        logger.error(f"moderation_error error={e}")
        return False, "moderation_unavailable"

    if output.startswith("safe"):
        return True, ""
    lines = output.split("\n")
    category = lines[1].strip() if len(lines) > 1 else "unknown"
    return False, category
