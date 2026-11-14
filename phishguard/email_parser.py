"""Parse raw RFC 822 / MIME emails (from IMAP, Gmail API, Graph API, .eml files) into a uniform structure."""

from __future__ import annotations

import email
import email.policy
import re
from dataclasses import dataclass, field
from email.message import EmailMessage
from email.utils import getaddresses, parseaddr

from bs4 import BeautifulSoup

URL_RE = re.compile(r"""(?i)\b((?:https?://|www\.)[^\s<>"'()\[\]{}]+)""")


@dataclass
class ParsedEmail:
    subject: str = ""
    body: str = ""  # plain text (HTML converted to text)
    sender: str = ""  # full From header
    sender_name: str = ""
    sender_address: str = ""
    reply_to: str = ""
    return_path: str = ""
    to: list[str] = field(default_factory=list)
    date: str = ""
    message_id: str = ""
    urls: list[str] = field(default_factory=list)
    html_links: list[tuple[str, str]] = field(default_factory=list)  # (visible text, href)
    has_html: bool = False
    attachments: list[str] = field(default_factory=list)
    auth_results: str = ""  # Authentication-Results header (SPF/DKIM/DMARC)
    received_spf: str = ""
    headers: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "sender": self.sender,
            "reply_to": self.reply_to,
            "to": self.to,
            "date": self.date,
            "urls": self.urls[:50],
            "has_html": self.has_html,
            "attachments": self.attachments,
        }


def html_to_text(html: str) -> tuple[str, list[tuple[str, str]]]:
    """Convert HTML to readable text and collect (anchor text, href) pairs."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    links = []
    for a in soup.find_all("a", href=True):
        links.append((a.get_text(" ", strip=True), a["href"].strip()))
    text = soup.get_text(" ", strip=True)
    return text, links


def extract_urls(text: str) -> list[str]:
    return [u.rstrip(".,;:!?") for u in URL_RE.findall(text or "")]


def _decode_part(part) -> str:
    try:
        return part.get_content()
    except Exception:
        payload = part.get_payload(decode=True) or b""
        charset = part.get_content_charset() or "utf-8"
        try:
            return payload.decode(charset, errors="replace")
        except LookupError:
            return payload.decode("utf-8", errors="replace")


def parse_raw_email(raw: bytes | str) -> ParsedEmail:
    """Parse a full MIME message (bytes or str)."""
    if isinstance(raw, str):
        raw = raw.encode("utf-8", errors="replace")
    msg: EmailMessage = email.message_from_bytes(raw, policy=email.policy.default)  # type: ignore[assignment]

    plain_parts: list[str] = []
    html_parts: list[str] = []
    attachments: list[str] = []

    for part in msg.walk() if msg.is_multipart() else [msg]:
        if part.is_multipart():
            continue
        disposition = (part.get_content_disposition() or "").lower()
        filename = part.get_filename()
        ctype = part.get_content_type()
        if disposition == "attachment" or (filename and ctype not in ("text/plain", "text/html")):
            attachments.append(filename or ctype)
            continue
        if ctype == "text/plain":
            plain_parts.append(_decode_part(part))
        elif ctype == "text/html":
            html_parts.append(_decode_part(part))

    html_links: list[tuple[str, str]] = []
    html_text = ""
    if html_parts:
        html_text, html_links = html_to_text("\n".join(html_parts))

    body = "\n".join(plain_parts).strip() or html_text
    urls = extract_urls(body)
    for _, href in html_links:
        if href.lower().startswith(("http://", "https://", "www.")) and href not in urls:
            urls.append(href)

    def h(name: str) -> str:
        value = msg.get(name)
        return str(value) if value is not None else ""

    sender = h("From")
    name, addr = parseaddr(sender)
    headers = {k: str(v) for k, v in msg.items()}

    return ParsedEmail(
        subject=h("Subject"),
        body=body,
        sender=sender,
        sender_name=name,
        sender_address=addr.lower(),
        reply_to=h("Reply-To"),
        return_path=h("Return-Path"),
        to=[a for _, a in getaddresses([h("To")]) if a],
        date=h("Date"),
        message_id=h("Message-ID"),
        urls=urls,
        html_links=html_links,
        has_html=bool(html_parts),
        attachments=attachments,
        auth_results=h("Authentication-Results"),
        received_spf=h("Received-SPF"),
        headers=headers,
    )


def from_fields(subject: str = "", body: str = "", sender: str = "", reply_to: str = "",
                headers: dict[str, str] | None = None) -> ParsedEmail:
    """Build a ParsedEmail from already-separated fields (e.g. a JSON API call)."""
    headers = headers or {}
    has_html = bool(re.search(r"<\s*(html|body|a|div|table|p|br)\b", body or "", re.I))
    links: list[tuple[str, str]] = []
    if has_html:
        body, links = html_to_text(body)
    name, addr = parseaddr(sender)
    urls = extract_urls(body)
    for _, href in links:
        if href.lower().startswith(("http://", "https://", "www.")) and href not in urls:
            urls.append(href)
    lower = {k.lower(): v for k, v in headers.items()}
    return ParsedEmail(
        subject=subject or "",
        body=body or "",
        sender=sender or "",
        sender_name=name,
        sender_address=addr.lower(),
        reply_to=reply_to or lower.get("reply-to", ""),
        return_path=lower.get("return-path", ""),
        urls=urls,
        html_links=links,
        has_html=has_html,
        auth_results=lower.get("authentication-results", ""),
        received_spf=lower.get("received-spf", ""),
        headers=headers,
    )
