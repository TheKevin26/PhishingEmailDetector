"""Hand-crafted linguistic + structural features computed from subject and body text.

These are used (1) as extra numeric inputs to the classic model and (2) to generate human-readable
"reasons" for a verdict.
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .preprocessing import strip_html

URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"'()\[\]{}]+")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
WORD_RE = re.compile(r"[A-Za-z']+")

URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly", "rebrand.ly", "cutt.ly",
    "shorturl.at", "tiny.cc", "rb.gy", "bl.ink", "s.id", "t.ly", "lnkd.in", "trib.al", "v.gd", "qrco.de",
}
SUSPICIOUS_TLDS = {
    "zip", "mov", "xyz", "top", "click", "link", "gq", "tk", "ml", "cf", "ga", "work", "support", "country",
    "kim", "men", "loan", "download", "racing", "review", "stream", "win", "bid", "ru", "cn", "su", "rest",
    "fit", "cam", "monster", "icu", "buzz", "live", "info", "biz",
}

KEYWORDS: dict[str, list[str]] = {
    "urgency": [
        "urgent", "immediately", "asap", "right away", "within 24 hours", "within 48 hours", "act now",
        "expires", "expire", "expiring", "deadline", "final notice", "last chance", "limited time",
        "as soon as possible", "today only", "don't delay", "hurry",
    ],
    "threat": [
        "suspended", "suspend", "locked", "lock", "disabled", "terminated", "unauthorized", "unusual activity",
        "suspicious activity", "security alert", "compromised", "breach", "restricted", "deactivated",
        "will be closed", "legal action", "penalty",
    ],
    "credential": [
        "verify your", "verify account", "confirm your", "update your", "validate", "login", "log in",
        "sign in", "password", "username", "credentials", "account details", "social security", "ssn",
        "pin", "security question", "reactivate", "re-activate",
    ],
    "financial": [
        "bank", "credit card", "debit card", "payment", "invoice", "refund", "wire transfer", "transfer",
        "bitcoin", "crypto", "wallet", "paypal", "billing", "tax", "irs", "transaction", "lottery",
        "inheritance", "beneficiary", "million", "funds", "claim", "prize", "winner", "won", "reward",
    ],
    "action": [
        "click here", "click below", "click the link", "follow the link", "open the attachment",
        "download", "visit", "tap here", "reply with", "call now", "kindly",
    ],
    "generic_greeting": [
        "dear customer", "dear user", "dear member", "dear client", "dear account holder", "dear sir",
        "dear friend", "dear beneficiary", "valued customer", "dear valued",
    ],
}
_KEYWORD_RES = {
    cat: re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.I)
    for cat, words in KEYWORDS.items()
}

FEATURE_NAMES = [
    "body_len_log", "subject_len", "num_words_log", "avg_word_len", "unique_word_ratio",
    "caps_ratio", "subject_caps_ratio", "exclamations", "subject_exclamation", "questions",
    "digit_ratio", "currency_symbols", "num_urls", "num_unique_domains", "has_ip_url",
    "num_shortener_urls", "num_suspicious_tld", "url_with_at", "max_url_len_log", "http_not_https",
    "url_many_subdomains", "num_emails", "has_html_tags", "subject_is_reply",
    "kw_urgency", "kw_threat", "kw_credential", "kw_financial", "kw_action", "kw_generic_greeting",
]


def _domain(url: str) -> str:
    if url.lower().startswith("www."):
        url = "http://" + url
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        host = ""
    return host.lower()


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def analyze_urls(urls: list[str]) -> dict:
    domains = [_domain(u) for u in urls]
    domains = [d for d in domains if d]
    info = {
        "num_urls": len(urls),
        "domains": sorted(set(domains)),
        "ip_urls": [u for u, d in zip(urls, [_domain(u) for u in urls]) if d and _is_ip(d)],
        "shorteners": [d for d in domains if d in URL_SHORTENERS or d.removeprefix("www.") in URL_SHORTENERS],
        "suspicious_tld": [d for d in domains if d.rsplit(".", 1)[-1] in SUSPICIOUS_TLDS],
        "with_at": [u for u in urls if "@" in u.split("//", 1)[-1].split("/", 1)[0]],
        "http_only": [u for u in urls if u.lower().startswith("http://")],
        "many_subdomains": [d for d in domains if d.count(".") >= 4],
        "max_len": max((len(u) for u in urls), default=0),
    }
    return info


def keyword_hits(text: str) -> dict[str, list[str]]:
    return {cat: sorted({m.lower() for m in rx.findall(text)}) for cat, rx in _KEYWORD_RES.items()}


def compute_features(subject: str, body: str) -> np.ndarray:
    subject = subject or ""
    raw_body = body or ""
    has_html = 1.0 if re.search(r"<\s*(html|body|a\s|div|table|p|br|font)\b", raw_body, re.I) else 0.0
    body = strip_html(raw_body)
    text = f"{subject} {body}"

    words = WORD_RE.findall(body)
    n_words = len(words)
    letters = [c for c in text if c.isalpha()]
    subj_letters = [c for c in subject if c.isalpha()]
    urls = URL_RE.findall(raw_body) + URL_RE.findall(subject)
    u = analyze_urls(urls)
    kw = keyword_hits(text)

    feats = [
        np.log1p(len(body)),
        min(len(subject), 300),
        np.log1p(n_words),
        (sum(len(w) for w in words) / n_words) if n_words else 0.0,
        (len({w.lower() for w in words}) / n_words) if n_words else 0.0,
        (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0,
        (sum(c.isupper() for c in subj_letters) / len(subj_letters)) if subj_letters else 0.0,
        min(text.count("!"), 50),
        1.0 if "!" in subject else 0.0,
        min(text.count("?"), 50),
        (sum(c.isdigit() for c in text) / len(text)) if text else 0.0,
        min(len(re.findall(r"[$€£¥₹]", text)), 50),
        min(u["num_urls"], 100),
        min(len(u["domains"]), 50),
        1.0 if u["ip_urls"] else 0.0,
        len(u["shorteners"]),
        len(u["suspicious_tld"]),
        1.0 if u["with_at"] else 0.0,
        np.log1p(u["max_len"]),
        min(len(u["http_only"]), 50),
        1.0 if u["many_subdomains"] else 0.0,
        min(len(EMAIL_RE.findall(body)), 50),
        has_html,
        1.0 if re.match(r"^\s*(re|fw|fwd)\s*:", subject, re.I) else 0.0,
    ] + [float(len(kw[c])) for c in KEYWORDS]
    return np.asarray(feats, dtype=np.float32)


class EmailFeatureExtractor(BaseEstimator, TransformerMixin):
    """sklearn transformer: DataFrame with 'subject' and 'body' columns -> dense feature matrix."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.vstack([compute_features(s, b) for s, b in zip(X["subject"], X["body"])])

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURE_NAMES, dtype=object)
