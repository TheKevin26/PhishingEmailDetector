"""Text normalisation shared by training and inference (must stay identical in both)."""

from __future__ import annotations

import html
import re

_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_TAG_RE = re.compile(r"<[^>]{1,500}>")
_NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_MONEY_RE = re.compile(r"[$€£¥₹]\s?\d")
_WS_RE = re.compile(r"\s+")
_REPEAT_RE = re.compile(r"(.)\1{3,}")


def strip_html(text: str) -> str:
    if "<" in text and ">" in text:
        text = _TAG_RE.sub(" ", text)
    return html.unescape(text)


def normalize_text(text: str) -> str:
    """Lower-case, replace URLs/emails/numbers/money with placeholder tokens, collapse whitespace."""
    if not text:
        return ""
    text = strip_html(str(text))
    text = _URL_RE.sub(" urltoken ", text)
    text = _EMAIL_RE.sub(" emailtoken ", text)
    text = _MONEY_RE.sub(" moneytoken ", text)
    text = _NUM_RE.sub(" numtoken ", text)
    text = _REPEAT_RE.sub(r"\1\1\1", text)
    text = text.lower()
    return _WS_RE.sub(" ", text).strip()


def build_text(subject: str, body: str, max_chars: int = 20000) -> str:
    """Combine subject + body into the single model input string (raw, not normalised)."""
    subject = (subject or "").strip()
    body = (body or "").strip()[:max_chars]
    return f"{subject}\n\n{body}" if subject else body


def transformer_text(subject: str, body: str, max_chars: int = 4000) -> str:
    """Lighter cleaning for the transformer: keep casing/punctuation, mask URLs and emails."""
    body = strip_html(body or "")[:max_chars]
    text = f"Subject: {(subject or '').strip()}\n{body}"
    text = _URL_RE.sub(" [URL] ", text)
    text = _EMAIL_RE.sub(" [EMAIL] ", text)
    return _WS_RE.sub(" ", text).strip()
