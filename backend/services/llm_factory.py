"""LLM factory — return LLM instance berdasarkan LLM_PROVIDER env var.

Dev (RTX 3060):  LLM_PROVIDER=ollama  → Ollama (local)
POC (L40S):      LLM_PROVIDER=vllm    → OpenAILike pointing to vLLM endpoint
"""

from functools import lru_cache

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


@lru_cache(maxsize=16)
def _build_llm(
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    request_timeout: int,
) -> LLM:
    if provider == "ollama":
        from llama_index.llms.ollama import Ollama

        return Ollama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            request_timeout=request_timeout,
            additional_kwargs={"num_predict": max_tokens},
        )

    if provider in ("vllm", "openai"):
        from llama_index.llms.openai_like import OpenAILike

        return OpenAILike(
            model=model,
            api_base=base_url.rstrip("/") + "/v1",
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            is_chat_model=True,
            timeout=request_timeout,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Pilih: ollama, vllm, openai")


def get_llm(temperature: float = None) -> LLM:
    temp = temperature if temperature is not None else LLM_TEMPERATURE
    return _build_llm(
        LLM_PROVIDER,
        LLM_MODEL,
        LLM_BASE_URL,
        LLM_API_KEY,
        temp,
        LLM_MAX_TOKENS,
        LLM_REQUEST_TIMEOUT,
    )
