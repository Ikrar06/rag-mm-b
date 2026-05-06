"""LLM factory — return LLM instance berdasarkan LLM_PROVIDER env var.

Dev (RTX 3060):  LLM_PROVIDER=ollama  → Ollama (local)
POC (L40S):      LLM_PROVIDER=vllm    → OpenAILike pointing to vLLM endpoint
"""

from llama_index.core.llms import LLM
from backend.config import (
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_BASE_URL,
    LLM_API_KEY,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_REQUEST_TIMEOUT,
)


def get_llm(temperature: float = None) -> LLM:
    temp = temperature if temperature is not None else LLM_TEMPERATURE

    if LLM_PROVIDER == "ollama":
        from llama_index.llms.ollama import Ollama
        return Ollama(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            temperature=temp,
            request_timeout=LLM_REQUEST_TIMEOUT,
            additional_kwargs={"num_predict": LLM_MAX_TOKENS},
        )

    if LLM_PROVIDER in ("vllm", "openai"):
        from llama_index.llms.openai_like import OpenAILike
        return OpenAILike(
            model=LLM_MODEL,
            api_base=LLM_BASE_URL.rstrip("/") + "/v1",
            api_key=LLM_API_KEY,
            max_tokens=LLM_MAX_TOKENS,
            temperature=temp,
            is_chat_model=True,
            timeout=LLM_REQUEST_TIMEOUT,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r}. Pilih: ollama, vllm, openai")
