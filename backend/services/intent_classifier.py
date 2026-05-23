"""Layer 3 — Intent classifier menggunakan IndoBERT yang sudah ditraining.

Label: chitchat | out_of_scope | get_info_public | get_info_private
Model: models/intent_classifier (IndoBERT fine-tuned)

Resilience:
- Health-tracked: kalau model gagal load atau predict, set _healthy=False
- Fallback: saat unhealthy, return safe default "get_info_public" + fallback=True
  + confidence=0.0. RAG pipeline tetap jalan via path public RAG.
- Recovery: setelah 60 detik dari failure, retry load saat next call
"""

import logging
import threading
import time
from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from backend.config import INTENT_MODEL_PATH, INTENT_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

_MAX_LENGTH = 128
_FALLBACK_INTENT = "get_info_public"  # safe default — masuk ke RAG public path
_RECOVERY_SEC = 60.0


class _HealthTracker:
    def __init__(self) -> None:
        self.healthy = True
        self.last_failure_at: float | None = None
        self._lock = threading.Lock()

    def record_failure(self) -> None:
        with self._lock:
            self.healthy = False
            self.last_failure_at = time.time()

    def record_success(self) -> None:
        with self._lock:
            if not self.healthy:
                logger.info("intent_classifier_recovered")
            self.healthy = True
            self.last_failure_at = None

    def should_retry(self) -> bool:
        """True kalau unhealthy tapi recovery window lewat → coba load lagi."""
        with self._lock:
            if self.healthy:
                return True
            if self.last_failure_at is None:
                return True
            return (time.time() - self.last_failure_at) >= _RECOVERY_SEC

    def get_state(self) -> dict:
        with self._lock:
            return {
                "healthy": self.healthy,
                "last_failure_at": self.last_failure_at,
            }


_health = _HealthTracker()


@lru_cache(maxsize=1)
def _load_model():
    """Load model + tokenizer sekali, cache di memory selamanya."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(INTENT_MODEL_PATH).to(device)
    model.eval()
    logger.info(f"intent_classifier_loaded path={INTENT_MODEL_PATH} device={device}")
    return model, tokenizer, device


def _fallback_result(reason: str) -> dict:
    """Return safe default saat classifier unhealthy."""
    logger.warning("intent_classifier_fallback reason=%s intent=%s", reason, _FALLBACK_INTENT)
    return {
        "intent": _FALLBACK_INTENT,
        "confidence": 0.0,
        "low_confidence": False,  # explicit fallback, not low-conf → tidak ask clarification
        "all_scores": {},
        "fallback": True,
    }


def classify_intent(text: str) -> dict:
    """Classify intent of input text. Fallback ke get_info_public kalau model gagal.

    Returns:
        {
            "intent": "chitchat" | "out_of_scope" | "get_info_public" | "get_info_private",
            "confidence": float,
            "low_confidence": bool,
            "all_scores": dict,
            "fallback": bool,  # True kalau classifier unavailable
        }
    """
    if not _health.should_retry():
        return _fallback_result("circuit_open")

    try:
        model, tokenizer, device = _load_model()
    except Exception as e:
        _health.record_failure()
        _load_model.cache_clear()  # paksa reload di retry berikutnya
        logger.error("intent_model_load_failed error=%s", e)
        return _fallback_result(f"load_error: {type(e).__name__}")

    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=_MAX_LENGTH,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
    except Exception as e:
        _health.record_failure()
        logger.error("intent_predict_failed error=%s", e)
        return _fallback_result(f"predict_error: {type(e).__name__}")

    _health.record_success()

    pred_id = int(probs.argmax().item())
    confidence = float(probs[pred_id].item())

    return {
        "intent": model.config.id2label[pred_id],
        "confidence": round(confidence, 4),
        "low_confidence": confidence < INTENT_CONFIDENCE_THRESHOLD,
        "all_scores": {
            model.config.id2label[i]: round(float(probs[i].item()), 4)
            for i in range(len(model.config.id2label))
        },
        "fallback": False,
    }


def get_health() -> dict:
    """Untuk /api/health endpoint."""
    return _health.get_state()


def warm_up() -> None:
    """Trigger model load saat startup — fail soft (log saja, jangan crash)."""
    try:
        _load_model()
    except Exception as e:
        _health.record_failure()
        logger.error("intent_classifier_warmup_failed error=%s", e)
