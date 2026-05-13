"""Layer 3 — Intent classifier menggunakan IndoBERT yang sudah ditraining.

Label: chitchat | out_of_scope | get_info_public | get_info_private
Model: models/intent_classifier (IndoBERT fine-tuned)
"""

import logging
from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from backend.config import INTENT_MODEL_PATH, INTENT_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

_MAX_LENGTH = 128


@lru_cache(maxsize=1)
def _load_model():
    """Load model + tokenizer sekali, cache di memory selamanya."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(INTENT_MODEL_PATH).to(device)
    model.eval()
    logger.info(f"intent_classifier_loaded path={INTENT_MODEL_PATH} device={device}")
    return model, tokenizer, device


def classify_intent(text: str) -> dict:
    """
    Classify intent of the input text.

    Returns:
        {
            "intent": "chitchat" | "out_of_scope" | "get_info_public" | "get_info_private",
            "confidence": float,
            "low_confidence": bool,
            "all_scores": dict,
        }
    """
    model, tokenizer, device = _load_model()

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
    }
