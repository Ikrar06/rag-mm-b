"""Layer 2 — Content moderation via Llama Guard 3 dengan circuit breaker.

Dev:  MODERATION_BACKEND=passthrough  (skip, no HTTP call)
POC:  MODERATION_BACKEND=ollama       (llama-guard3:1b via local Ollama)

Circuit breaker: kalau Llama Guard down (timeout / 5xx / connection error)
melebihi failure_threshold dalam recovery window, circuit OPEN dan request
selanjutnya fail-open (lewat dengan flag bypassed=True). Setelah recovery_sec
sejak failure terakhir, masuk HALF_OPEN — satu probe request menentukan
balik CLOSED (normal) atau tetap OPEN.

Fail-open lebih aman daripada fail-closed untuk POC akademik:
- L1 keyword filter sudah catch niat jahat eksplisit
- L3 intent classifier filter query off-domain
- Fail-closed bikin SEMUA query di-block saat Llama Guard down — outage cascade
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum

import httpx

from backend.config import (
    MODERATION_BASE_URL,
    MODERATION_BACKEND,
    MODERATION_MODEL,
    MODERATION_TIMEOUT,
)

logger = logging.getLogger(__name__)


class ModerationBackend(str, Enum):
    OLLAMA = "ollama"
    PASSTHROUGH = "passthrough"


class CircuitState(str, Enum):
    CLOSED = "closed"          # normal
    OPEN = "open"              # sedang fail, fail-open semua request
    HALF_OPEN = "half_open"    # recovery probe


@dataclass
class ModerationResult:
    safe: bool
    reason: str = ""
    bypassed: bool = False     # True kalau circuit open (judgment skipped)


_BACKEND = ModerationBackend(MODERATION_BACKEND)

_PROMPT = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Task: Check if there is unsafe content in the message according to our safety policy.

<BEGIN_UNSAFE_CONTENT_CATEGORIES>
S1: Violent Crimes
S2: Non-Violent Crimes
S9: Indiscriminate Weapons
S10: Hate Speech
S11: Suicide & Self-Harm
<END_UNSAFE_CONTENT_CATEGORIES>

NOTE: Pertanyaan medis ringan ("sakit perut", "demam", "cara mengobati X"),
pertanyaan legal umum, atau pertanyaan finansial umum BUKAN unsafe content
dari sudut pandang chatbot akademik UNHAS. Itu akan di-handle oleh layer
berikutnya (intent classifier) sebagai out-of-scope dengan tone ramah —
bot tidak akan memberi advice medis/legal/finansial spesifik, jadi tidak
perlu di-block di sini.

<BEGIN_CONVERSATION>
User: {message}
<END_CONVERSATION>

Provide your safety assessment:
- First line must read 'safe' or 'unsafe'.
- If unsafe, second line must include violated categories.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


class _CircuitBreaker:
    """Thread-safe circuit breaker untuk Llama Guard.

    State transitions:
        CLOSED → OPEN       : failure_count >= threshold
        OPEN → HALF_OPEN    : elapsed since last_failure >= recovery_sec
        HALF_OPEN → CLOSED  : probe success
        HALF_OPEN → OPEN    : probe failure
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_sec: float = 30.0,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_sec = recovery_sec
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_at: float | None = None
        self._lock = threading.Lock()

    def allow_request(self) -> bool:
        """True kalau request boleh diproses (CLOSED / HALF_OPEN probe)."""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            if self.state == CircuitState.OPEN:
                if self.last_failure_at is None:
                    return True
                elapsed = time.time() - self.last_failure_at
                if elapsed >= self.recovery_sec:
                    self.state = CircuitState.HALF_OPEN
                    logger.info("moderation_circuit_half_open elapsed_sec=%.1f", elapsed)
                    return True
                return False
            # HALF_OPEN — biarkan probe lewat
            return True

    def record_success(self) -> None:
        with self._lock:
            if self.state != CircuitState.CLOSED:
                logger.info(
                    "moderation_circuit_closed prev_state=%s failures_reset=%d",
                    self.state.value,
                    self.failure_count,
                )
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_failure_at = None

    def record_failure(self) -> None:
        with self._lock:
            self.failure_count += 1
            self.last_failure_at = time.time()
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logger.warning("moderation_circuit_reopen failures=%d", self.failure_count)
            elif (
                self.state == CircuitState.CLOSED
                and self.failure_count >= self.failure_threshold
            ):
                self.state = CircuitState.OPEN
                logger.error(
                    "moderation_circuit_open failures=%d threshold=%d",
                    self.failure_count,
                    self.failure_threshold,
                )


_circuit = _CircuitBreaker(failure_threshold=5, recovery_sec=30.0)


def check_moderation(text: str) -> tuple[bool, str]:
    """Backward-compat wrapper: return (is_safe, reason).

    Untuk kode baru, prefer check_moderation_v2() yang return
    ModerationResult dengan flag bypassed.
    """
    result = check_moderation_v2(text)
    if result.bypassed:
        # Fail-open: anggap safe, tapi reason mencatat alasan bypass.
        return True, result.reason
    return result.safe, result.reason


def check_moderation_v2(text: str) -> ModerationResult:
    """Cek moderation dengan circuit breaker. Fail-open saat Llama Guard down."""
    if _BACKEND == ModerationBackend.PASSTHROUGH:
        return ModerationResult(safe=True)

    if not _circuit.allow_request():
        logger.warning("moderation_bypassed reason=circuit_open")
        return ModerationResult(safe=True, bypassed=True, reason="circuit_open")

    return _check_ollama(text)


def _check_ollama(text: str) -> ModerationResult:
    prompt = _PROMPT.format(message=text[:500])
    try:
        resp = httpx.post(
            f"{MODERATION_BASE_URL}/api/generate",
            json={"model": MODERATION_MODEL, "prompt": prompt, "stream": False},
            timeout=float(MODERATION_TIMEOUT),
        )
        resp.raise_for_status()
        output = resp.json()["response"].strip().lower()
    except (httpx.TimeoutException, httpx.HTTPError, httpx.NetworkError) as e:
        _circuit.record_failure()
        logger.error("moderation_http_error error=%s circuit=%s", e, _circuit.state.value)
        return ModerationResult(safe=True, bypassed=True, reason=f"moderation_error: {type(e).__name__}")
    except Exception as e:
        _circuit.record_failure()
        logger.error("moderation_unexpected error=%s circuit=%s", e, _circuit.state.value)
        return ModerationResult(safe=True, bypassed=True, reason=f"moderation_error: unexpected")

    _circuit.record_success()

    if output.startswith("safe"):
        return ModerationResult(safe=True)
    lines = output.split("\n")
    category = lines[1].strip() if len(lines) > 1 else "unknown"
    return ModerationResult(safe=False, reason=category)


def get_circuit_state() -> dict:
    """Untuk /api/health endpoint."""
    with _circuit._lock:
        return {
            "state": _circuit.state.value,
            "failure_count": _circuit.failure_count,
            "last_failure_at": _circuit.last_failure_at,
        }
