"""Unified phishing detector: ML probability + header/link rules + human-readable reasons."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field

import joblib

from . import config
from .email_parser import ParsedEmail, from_fields, parse_raw_email
from .features import analyze_urls, keyword_hits
from .header_rules import evaluate_rules
from .models.classic import as_frame, explain_linear

log = logging.getLogger("phishguard")


@dataclass
class Verdict:
    label: str  # "phishing" | "legitimate"
    is_phishing: bool
    score: float  # final probability of phishing (0..1)
    model_score: float  # raw ML probability before rules
    confidence: str  # low | medium | high
    model_used: str
    threshold: float
    reasons: list[str] = field(default_factory=list)
    top_terms: list[dict] = field(default_factory=list)
    email: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class PhishingDetector:
    def __init__(self, model_choice: str | None = None, threshold: float | None = None):
        self.threshold = config.PHISHING_THRESHOLD if threshold is None else threshold
        choice = (model_choice or config.MODEL_CHOICE).lower()
        if choice == "auto":
            choice = "classic"
            if config.SELECTED_MODEL_FILE.exists():
                choice = json.loads(config.SELECTED_MODEL_FILE.read_text()).get("model", "classic")

        if not config.CLASSIC_MODEL_PATH.exists():
            raise FileNotFoundError(f"Classic model not found at {config.CLASSIC_MODEL_PATH}. Run training first.")
        bundle = joblib.load(config.CLASSIC_MODEL_PATH)
        self.classic = bundle["pipeline"]
        self.explainer_coef = bundle.get("explainer_coef")
        self.transformer = None

        if choice in ("transformer", "ensemble"):
            if (config.TRANSFORMER_MODEL_DIR / "config.json").exists():
                from .models.transformer import TransformerClassifier
                self.transformer = TransformerClassifier(config.TRANSFORMER_MODEL_DIR)
            else:
                log.warning("Transformer model not found; falling back to classic model.")
                choice = "classic"
        self.model_used = choice
        self._lock = threading.Lock()  # torch / sklearn objects shared across API threads

    # ---------------------------------------------------------------- scoring
    def _model_probability(self, mails: list[ParsedEmail]) -> list[float]:
        subjects = [m.subject for m in mails]
        bodies = [m.body for m in mails]
        with self._lock:
            classic = self.classic.predict_proba(as_frame(subjects, bodies))[:, 1] if self.model_used != "transformer" else None
            trans = self.transformer.predict_proba(subjects, bodies) if self.transformer is not None else None
        if self.model_used == "transformer":
            return [float(p) for p in trans]
        if self.model_used == "ensemble":
            return [float((a + b) / 2) for a, b in zip(classic, trans)]
        return [float(p) for p in classic]

    def _reasons(self, mail: ParsedEmail) -> list[str]:
        reasons = []
        u = analyze_urls(mail.urls)
        if u["ip_urls"]:
            reasons.append("Contains link(s) pointing to a raw IP address")
        if u["shorteners"]:
            reasons.append("Uses URL shortener(s) that hide the real destination: " + ", ".join(sorted(set(u["shorteners"]))[:3]))
        if u["suspicious_tld"]:
            reasons.append("Links to domains with high-risk TLDs: " + ", ".join(sorted(set(u["suspicious_tld"]))[:3]))
        if u["with_at"]:
            reasons.append("Link contains '@' which can disguise the real host")
        if u["many_subdomains"]:
            reasons.append("Link uses an unusually deep subdomain chain")
        kw = keyword_hits(f"{mail.subject} {mail.body}")
        labels = {
            "urgency": "Urgent / time-pressure language", "threat": "Threatening account language",
            "credential": "Requests credentials or account verification", "financial": "Money / payment related lure",
            "action": "Pushes you to click, download or reply", "generic_greeting": "Generic greeting instead of your name",
        }
        for cat, words in kw.items():
            if words:
                reasons.append(f"{labels[cat]}: " + ", ".join(f"'{w}'" for w in words[:4]))
        return reasons

    def predict_parsed(self, mails: list[ParsedEmail]) -> list[Verdict]:
        probs = self._model_probability(mails)
        out = []
        for mail, p in zip(mails, probs):
            hits = evaluate_rules(mail)
            score = min(1.0, max(0.0, p + sum(h.weight for h in hits)))
            is_phish = score >= self.threshold
            dist = abs(score - self.threshold)
            confidence = "high" if dist >= 0.35 else "medium" if dist >= 0.15 else "low"

            reasons = [h.detail for h in hits if h.weight > 0]
            top_terms = []
            if is_phish:
                reasons += self._reasons(mail)
                if self.explainer_coef is not None:
                    try:
                        vec = self.classic.named_steps["vec"]
                        top_terms = explain_linear(vec, self.explainer_coef, as_frame([mail.subject], [mail.body]))
                    except Exception as exc:  # explanation must never break scoring
                        log.warning("explanation failed: %s", exc)
            else:
                reasons += [h.detail for h in hits if h.weight < 0]
                if not reasons:
                    reasons = ["No significant phishing indicators found"]

            out.append(Verdict(
                label="phishing" if is_phish else "legitimate", is_phishing=is_phish,
                score=round(score, 4), model_score=round(p, 4), confidence=confidence,
                model_used=self.model_used, threshold=self.threshold, reasons=reasons,
                top_terms=top_terms, email=mail.to_dict(),
            ))
        return out

    # ---------------------------------------------------------------- public API
    def predict_raw(self, raw: bytes | str) -> Verdict:
        return self.predict_parsed([parse_raw_email(raw)])[0]

    def predict_fields(self, subject: str = "", body: str = "", sender: str = "", reply_to: str = "",
                       headers: dict | None = None) -> Verdict:
        return self.predict_parsed([from_fields(subject, body, sender, reply_to, headers)])[0]


_default: PhishingDetector | None = None


def get_detector() -> PhishingDetector:
    global _default
    if _default is None:
        _default = PhishingDetector()
    return _default
