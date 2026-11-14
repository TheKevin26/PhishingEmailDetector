"""Rule-based checks on email headers and link structure.

The training data does not contain full headers, so these signals are applied at inference time on top of
the ML probability. Each triggered rule adds a reason and nudges the score upward (or, for strong
authentication passes, slightly downward).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from email.utils import parseaddr

import tldextract

from .email_parser import ParsedEmail

_extract = tldextract.TLDExtract(suffix_list_urls=())  # offline: use bundled public-suffix snapshot

BRANDS = [
    "paypal", "apple", "microsoft", "office365", "outlook", "amazon", "netflix", "google", "gmail", "facebook",
    "instagram", "whatsapp", "linkedin", "dropbox", "docusign", "adobe", "chase", "wells fargo",
    "bank of america", "citibank", "hsbc", "fedex", "dhl", "ups", "usps", "irs", "coinbase", "binance",
    "sbi", "hdfc", "icici", "paytm", "flipkart",
]
# Legitimate sending domains whose registered label differs from the brand word
BRAND_DOMAIN_ALIASES = {
    "office365": ("microsoft", "office", "outlook"), "outlook": ("microsoft", "office365", "office"),
    "microsoft": ("office365", "office", "outlook", "live", "microsoftonline"),
    "gmail": ("google",), "google": ("youtube", "gmail"), "apple": ("icloud", "me"),
    "facebook": ("facebookmail", "meta", "fb"), "instagram": ("facebookmail", "mail"),
    "amazon": ("amazonses", "aws"), "bank of america": ("bankofamerica", "bofa"),
    "wells fargo": ("wellsfargo",), "chase": ("chase", "jpmorgan"), "irs": ("irs",),
    "sbi": ("sbi", "onlinesbi"), "hdfc": ("hdfcbank",), "icici": ("icicibank",), "whatsapp": ("whatsapp", "facebookmail"),
}


# Well-known sending domains (registered domain). Only trusted when the message is ALSO authenticated (DMARC/DKIM
# pass aligned with From), because the From header alone can be forged. Extend with PHISHGUARD_TRUSTED_DOMAINS
# (comma separated), e.g. your own organisation's domain.
TRUSTED_DOMAINS = {
    "amazon.com", "amazon.in", "amazon.co.uk", "amazonses.com", "flipkart.com", "myntra.com", "google.com", "gmail.com",
    "youtube.com", "microsoft.com", "office.com", "outlook.com", "live.com", "apple.com", "icloud.com", "netflix.com",
    "spotify.com", "paypal.com", "hdfcbank.net", "hdfcbank.com", "sbi.co.in", "icicibank.com", "axisbank.com",
    "paytm.com", "github.com", "gitlab.com", "linkedin.com", "instagram.com", "facebookmail.com", "facebook.com",
    "swiggy.in", "zomato.com", "uber.com", "olacabs.com", "dhl.com", "fedex.com", "ups.com", "bluedart.com",
    "dropbox.com", "zoom.us", "slack.com", "docusign.net", "adobe.com", "airtel.in", "jio.com", "irctc.co.in",
    "makemytrip.com", "coursera.org", "udemy.com", "coinbase.com", "atlassian.com", "notion.so", "stripe.com",
    "twitter.com", "x.com", "whatsapp.com", "indeed.com", "naukri.com", "phonepe.com", "razorpay.com",
}
TRUSTED_DOMAINS |= {d.strip().lower() for d in os.getenv("PHISHGUARD_TRUSTED_DOMAINS", "").split(",") if d.strip()}
# Tracking / CDN hosts that legitimate bulk mail commonly links through
NEUTRAL_LINK_DOMAINS = {"sendgrid.net", "mailchimp.com", "list-manage.com", "mcusercontent.com", "amazonaws.com",
                        "cloudfront.net", "googleusercontent.com", "sparkpostmail.com", "mandrillapp.com", "hubspotlinks.com"}


@dataclass
class RuleHit:
    rule: str
    detail: str
    weight: float  # added to probability (positive = more phishy)


def registered_domain(value: str) -> str:
    value = (value or "").strip().lower()
    if "@" in value:
        value = value.rsplit("@", 1)[-1]
    value = value.strip("<>").strip()
    ext = _extract(value)
    return ".".join(p for p in (ext.domain, ext.suffix) if p)


def _url_domain(url: str) -> str:
    url = url.strip()
    if url.lower().startswith("www."):
        url = "http://" + url
    m = re.match(r"(?i)^[a-z][a-z0-9+.-]*://([^/?#:]+)", url)
    return registered_domain(m.group(1)) if m else ""


def evaluate_rules(mail: ParsedEmail) -> list[RuleHit]:
    hits: list[RuleHit] = []
    sender_dom = registered_domain(mail.sender_address)

    # 1. Authentication results (SPF / DKIM / DMARC)
    auth = f"{mail.auth_results} {mail.received_spf}".lower()
    for mech in ("spf", "dkim", "dmarc"):
        if re.search(rf"\b{mech}\s*=\s*(fail|softfail|permerror)", auth) or (
            mech == "spf" and mail.received_spf.lower().startswith(("fail", "softfail"))
        ):
            hits.append(RuleHit(f"{mech}_fail", f"{mech.upper()} authentication failed", 0.15 if mech != "dmarc" else 0.2))
    dmarc_pass = bool(re.search(r"\bdmarc\s*=\s*pass", auth))
    dkim_domains = {registered_domain(d) for d in re.findall(r"\bdkim\s*=\s*pass[^;]*?header\.(?:d|i)\s*=\s*@?([\w.-]+)", auth)}
    authenticated = bool(sender_dom) and (dmarc_pass or sender_dom in dkim_domains)
    link_doms = {_url_domain(u) for u in mail.urls} - {""}
    foreign_links = {d for d in link_doms if d != sender_dom and d not in NEUTRAL_LINK_DOMAINS
                     and d.split(".", 1)[0] != sender_dom.split(".", 1)[0]}  # amazon.in -> amazon.com is fine

    if authenticated and sender_dom in TRUSTED_DOMAINS and not foreign_links:
        hits.append(RuleHit("trusted_sender", f"Authenticated sender from well-known domain {sender_dom}, links stay on that domain", -0.45))
    elif authenticated and sender_dom in TRUSTED_DOMAINS:
        hits.append(RuleHit("trusted_sender_foreign_links", f"Authenticated sender {sender_dom} (but links point elsewhere: {', '.join(sorted(foreign_links)[:3])})", -0.15))
    elif dmarc_pass:
        hits.append(RuleHit("auth_pass", "SPF/DKIM/DMARC authentication passed", -0.05))

    # 2. Reply-To differs from From domain
    if mail.reply_to and sender_dom:
        reply_dom = registered_domain(parseaddr(mail.reply_to)[1])
        if reply_dom and reply_dom != sender_dom:
            hits.append(RuleHit("reply_to_mismatch", f"Reply-To domain ({reply_dom}) differs from sender ({sender_dom})", 0.12))

    # 3. Display name impersonates a brand that the sender domain does not belong to.
    #    The registered domain label must be the brand itself (paypal.com, paypal.co.uk) - lookalikes such as
    #    paypal-account-verify.xyz or secure-paypal.com do not count.
    name = (mail.sender_name or "").lower()
    sender_label = sender_dom.split(".", 1)[0]
    for brand in BRANDS:
        token = brand.replace(" ", "")
        allowed = {token, *BRAND_DOMAIN_ALIASES.get(brand, ())}
        if re.search(rf"\b{re.escape(brand)}\b", name) and sender_label not in allowed:
            hits.append(RuleHit("display_name_spoof", f"Display name mentions '{brand}' but sender domain is {sender_dom or 'unknown'}", 0.15))
            break

    # 4. Display name itself contains an email address with a different domain
    m = re.search(r"[\w.+-]+@([\w-]+\.[\w.-]+)", mail.sender_name or "")
    if m and registered_domain(m.group(1)) != sender_dom:
        hits.append(RuleHit("display_name_address", "Display name contains a different email address", 0.12))

    # 5. Link text shows one domain but href points to another
    mismatched = []
    for text, href in mail.html_links:
        shown = re.search(r"(?i)\b(?:https?://)?((?:[a-z0-9-]+\.)+[a-z]{2,})\b", text or "")
        if shown and href.lower().startswith(("http", "www")):
            shown_dom, real_dom = registered_domain(shown.group(1)), _url_domain(href)
            if shown_dom and real_dom and shown_dom != real_dom:
                mismatched.append(f"{shown_dom} -> {real_dom}")
    if mismatched:
        hits.append(RuleHit("link_text_mismatch", "Link text shows a different domain than it opens: " + ", ".join(mismatched[:3]), 0.2))

    # 6. Lookalike domains: a brand name inside a domain that does not belong to the brand
    #    (paypal-account-verify.xyz, secure-hdfc-login.top, amaz0n.com)
    def lookalike_of(dom: str) -> str | None:
        label = dom.split(".", 1)[0].replace("0", "o").replace("1", "l")
        for brand in BRANDS:
            token = brand.replace(" ", "")
            allowed = {token, *BRAND_DOMAIN_ALIASES.get(brand, ())}
            if len(token) >= 4 and token in label and dom.split(".", 1)[0] not in allowed:
                return brand
        return None

    sender_brand = lookalike_of(sender_dom) if sender_dom and sender_dom not in TRUSTED_DOMAINS else None
    if sender_brand:
        hits.append(RuleHit("lookalike_sender", f"Sender domain {sender_dom} imitates '{sender_brand}'", 0.2))
    fake_links = sorted({d for d in link_doms if d not in TRUSTED_DOMAINS and lookalike_of(d)})
    if fake_links:
        hits.append(RuleHit("lookalike_link", "Links to lookalike domain(s): " + ", ".join(fake_links[:3]), 0.25))

    # 7. Dangerous attachments
    risky = [a for a in mail.attachments if re.search(r"\.(exe|scr|js|jse|vbs|bat|cmd|com|pif|jar|iso|img|lnk|hta|msi|html?|zip|rar|7z|docm|xlsm)$", a or "", re.I)]
    if risky:
        hits.append(RuleHit("risky_attachment", "Potentially dangerous attachment: " + ", ".join(risky[:3]), 0.15))

    return hits
