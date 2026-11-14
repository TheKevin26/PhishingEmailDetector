"""Generic IMAP connector - works with Gmail, Outlook.com/Office365, Yahoo, Zoho, iCloud, Fastmail, cPanel, Dovecot...

Actions per message:
  * adds IMAP keyword  "PhishGuard-Phishing"  or  "PhishGuard-Safe"
  * on Gmail (X-GM-EXT-1) also applies real Gmail labels "PhishGuard/Phishing" / "PhishGuard/Safe"
  * optional: moves phishing to a "Phishing" folder (IMAP_MOVE_PHISHING=true)
Messages are fetched with BODY.PEEK so they stay unread.
"""

from __future__ import annotations

import imaplib
import logging
import os
import re
import ssl
from collections.abc import Iterable
from datetime import datetime, timedelta

from .base import MailConnector

log = logging.getLogger("phishguard.imap")

KEYWORD_PHISH = "PhishGuard-Phishing"
KEYWORD_SAFE = "PhishGuard-Safe"
GMAIL_LABEL_PHISH = "PhishGuard/Phishing"
GMAIL_LABEL_SAFE = "PhishGuard/Safe"

KNOWN_HOSTS = {
    "gmail.com": "imap.gmail.com", "googlemail.com": "imap.gmail.com",
    "outlook.com": "outlook.office365.com", "hotmail.com": "outlook.office365.com",
    "live.com": "outlook.office365.com", "office365.com": "outlook.office365.com",
    "yahoo.com": "imap.mail.yahoo.com", "ymail.com": "imap.mail.yahoo.com",
    "zoho.com": "imap.zoho.com", "icloud.com": "imap.mail.me.com", "me.com": "imap.mail.me.com",
    "aol.com": "imap.aol.com", "gmx.com": "imap.gmx.com", "fastmail.com": "imap.fastmail.com",
    "rediffmail.com": "imap.rediffmail.com", "yandex.com": "imap.yandex.com",
}


class IMAPConnector(MailConnector):
    name = "imap"

    def __init__(self, user: str | None = None, password: str | None = None, host: str | None = None,
                 port: int | None = None, folder: str | None = None, phishing_folder: str | None = None,
                 lookback_days: int | None = None, **kw):
        self.user = user or os.environ["IMAP_USER"]
        self.password = password or os.environ["IMAP_PASSWORD"]
        domain = self.user.rsplit("@", 1)[-1].lower()
        self.host = host or os.getenv("IMAP_HOST") or KNOWN_HOSTS.get(domain) or f"imap.{domain}"
        self.port = int(port or os.getenv("IMAP_PORT", 993))
        self.folder = folder or os.getenv("IMAP_FOLDER", "INBOX")
        self.phishing_folder = phishing_folder or os.getenv("IMAP_PHISHING_FOLDER", "Phishing")
        self.lookback_days = int(lookback_days or os.getenv("IMAP_LOOKBACK_DAYS", 3))
        kw.setdefault("move_phishing", os.getenv("IMAP_MOVE_PHISHING", "false").lower() == "true")
        super().__init__(**kw)
        self.conn: imaplib.IMAP4_SSL | None = None
        self.is_gmail = False
        self.supports_move = False
        self.supports_keywords = True

    def state_key(self) -> str:
        return f"imap_{re.sub(r'[^a-z0-9]+', '_', self.user.lower())}"

    def connect(self) -> None:
        self.conn = imaplib.IMAP4_SSL(self.host, self.port, ssl_context=ssl.create_default_context())
        self.conn.login(self.user, self.password)
        caps = {c.decode().upper() if isinstance(c, bytes) else str(c).upper() for c in self.conn.capabilities}
        self.is_gmail = "X-GM-EXT-1" in caps
        self.supports_move = "MOVE" in caps
        typ, _ = self.conn.select(self.folder)
        if typ != "OK":
            raise RuntimeError(f"Cannot open folder {self.folder}")
        validity = self.conn.response("UIDVALIDITY")[1]
        self.uid_prefix = (validity[0].decode() if validity and validity[0] else "0") + ":"
        # PERMANENTFLAGS containing \* means custom keywords are allowed
        perm = self.conn.response("PERMANENTFLAGS")[1]
        if perm and perm[0] is not None:
            self.supports_keywords = b"\\*" in perm[0]
        log.info("connected to %s as %s (gmail=%s, move=%s, keywords=%s)", self.host, self.user,
                 self.is_gmail, self.supports_move, self.supports_keywords)
        if self.move_phishing:
            self.conn.create(self._quote(self.phishing_folder))  # harmless if it already exists

    @staticmethod
    def _quote(name: str) -> str:
        return '"' + name.replace("\\", "\\\\").replace('"', '\\"') + '"'

    def fetch_new(self) -> Iterable[tuple[str, bytes]]:
        assert self.conn is not None
        self.conn.noop()
        since = (datetime.now() - timedelta(days=self.lookback_days)).strftime("%d-%b-%Y")
        typ, data = self.conn.uid("SEARCH", None, f"(SINCE {since})")
        if typ != "OK":
            return
        uids = data[0].split() if data and data[0] else []
        for uid in uids:
            key = self.uid_prefix + uid.decode()
            if key in self.store:
                continue
            typ, msg = self.conn.uid("FETCH", uid, "(BODY.PEEK[])")
            if typ != "OK" or not msg or not isinstance(msg[0], tuple):
                continue
            yield key, msg[0][1]

    def apply_verdict(self, msg_id: str, verdict: dict) -> None:
        assert self.conn is not None
        uid = msg_id.split(":", 1)[1].encode()
        phish = verdict["is_phishing"]
        if phish or self.tag_safe:
            if self.supports_keywords:
                self.conn.uid("STORE", uid, "+FLAGS", f"({KEYWORD_PHISH if phish else KEYWORD_SAFE})")
            if self.is_gmail:
                label = GMAIL_LABEL_PHISH if phish else GMAIL_LABEL_SAFE
                self.conn.uid("STORE", uid, "+X-GM-LABELS", f'("{label}")')
        if phish:
            self.conn.uid("STORE", uid, "+FLAGS", r"(\Flagged)")
            if self.move_phishing:
                target = self._quote(self.phishing_folder)
                if self.supports_move:
                    self.conn.uid("MOVE", uid, target)
                else:
                    self.conn.uid("COPY", uid, target)
                    self.conn.uid("STORE", uid, "+FLAGS", r"(\Deleted)")
                    self.conn.expunge()
