"""Fine-tuned DistilBERT phishing classifier (inference wrapper)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import TRANSFORMER_MAX_LEN
from ..preprocessing import transformer_text


class TransformerClassifier:
    def __init__(self, model_dir: Path, device: str | None = None, max_len: int = TRANSFORMER_MAX_LEN):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        self.max_len = max_len

    def predict_proba(self, subjects, bodies, batch_size: int = 32) -> np.ndarray:
        texts = [transformer_text(s, b) for s, b in zip(subjects, bodies)]
        probs = []
        with self.torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                enc = self.tokenizer(texts[i:i + batch_size], truncation=True, max_length=self.max_len,
                                     padding=True, return_tensors="pt").to(self.device)
                with self.torch.autocast(device_type="cuda", enabled=self.device == "cuda"):
                    logits = self.model(**enc).logits
                probs.append(self.torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
        return np.concatenate(probs) if probs else np.zeros(0)
