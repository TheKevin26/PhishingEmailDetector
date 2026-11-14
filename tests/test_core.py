"""Unit tests for parsing, features and rules (no trained model needed) + model tests (skipped if untrained)."""

from pathlib import Path

import pytest

from phishguard.config import CLASSIC_MODEL_PATH
from phishguard.email_parser import from_fields, parse_raw_email
from phishguard.features import FEATURE_NAMES, compute_features
from phishguard.header_rules import evaluate_rules, registered_domain
from phishguard.preprocessing import normalize_text

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def test_normalize_masks_tokens():
    out = normalize_text("Visit http://evil.com/x NOW or mail a@b.com for $500 <b>bonus</b>")
    assert "urltoken" in out and "emailtoken" in out and "moneytoken" in out
    assert "<b>" not in out and out == out.lower()


def test_feature_vector_shape_and_signals():
    f = compute_features("URGENT!!", "Dear customer verify your password at http://10.0.0.1/login bit.ly/x")
    assert f.shape == (len(FEATURE_NAMES),)
    d = dict(zip(FEATURE_NAMES, f))
    assert d["has_ip_url"] == 1 and d["kw_urgency"] >= 1 and d["kw_credential"] >= 1


def test_parse_raw_multipart():
    mail = parse_raw_email((SAMPLES / "phishing_paypal.eml").read_bytes())
    assert "suspended" in mail.subject.lower()
    assert mail.sender_address == "service@paypal-account-verify.xyz"
    assert mail.has_html and any("192.168.13.7" in u for u in mail.urls)


def test_rules_detect_spoofing():
    mail = parse_raw_email((SAMPLES / "phishing_paypal.eml").read_bytes())
    rules = {h.rule for h in evaluate_rules(mail)}
    assert {"spf_fail", "dmarc_fail", "reply_to_mismatch", "display_name_spoof", "link_text_mismatch"} <= rules


def test_rules_quiet_on_legit():
    mail = parse_raw_email((SAMPLES / "legit_meeting.eml").read_bytes())
    assert all(h.weight <= 0 for h in evaluate_rules(mail))


def test_registered_domain():
    assert registered_domain("a@mail.google.co.uk") == "google.co.uk"


def test_from_fields_html():
    m = from_fields("hi", '<p>Click <a href="http://x.ru/a">here</a></p>')
    assert m.has_html and "http://x.ru/a" in m.urls


model_required = pytest.mark.skipif(not CLASSIC_MODEL_PATH.exists(), reason="model not trained")


@model_required
def test_detector_samples():
    from phishguard.detector import get_detector
    det = get_detector()
    bad = det.predict_raw((SAMPLES / "phishing_paypal.eml").read_bytes())
    good = det.predict_raw((SAMPLES / "legit_meeting.eml").read_bytes())
    assert bad.is_phishing and bad.score > 0.8 and bad.reasons and bad.top_terms
    assert not good.is_phishing and good.score < 0.5


@model_required
def test_api():
    from fastapi.testclient import TestClient
    from phishguard.api import app
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        r = client.post("/predict", json={"subject": "Verify your account now",
                                          "body": "Your account is locked. Login at http://1.2.3.4/verify within 24 hours."})
        assert r.status_code == 200 and r.json()["label"] == "phishing"
        r = client.post("/predict/eml", files={"file": ("m.eml", (SAMPLES / "legit_meeting.eml").read_bytes())})
        assert r.json()["label"] == "legitimate"
