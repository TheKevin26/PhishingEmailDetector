"""Gmail API connector - applies coloured Gmail labels "PhishGuard/Phishing" (red) and "PhishGuard/Safe" (green).

Setup (one time):
  1. Google Cloud Console -> create project -> enable "Gmail API".
  2. OAuth consent screen (External, add yourself as test user).
  3. Credentials -> OAuth client ID -> "Desktop app" -> download JSON as  secrets/gmail_credentials.json
  4. python -m phishguard.cli connect gmail      (browser opens once; token cached in secrets/gmail_token.json)
"""

from __future__ import annotations

import base64
import logging
import os
from collections.abc import Iterable
from pathlib import Path

from ..config import ROOT_DIR
from .base import MailConnector

log = logging.getLogger("phishguard.gmail")
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
LABELS = {
    "phishing": ("PhishGuard/Phishing", {"backgroundColor": "#cc3a21", "textColor": "#ffffff"}),
    "legitimate": ("PhishGuard/Safe", {"backgroundColor": "#16a766", "textColor": "#ffffff"}),
}


class GmailConnector(MailConnector):
    name = "gmail"

    def __init__(self, credentials_file: str | None = None, token_file: str | None = None,
                 query: str | None = None, **kw):
        self.credentials_file = Path(credentials_file or os.getenv("GMAIL_CREDENTIALS", ROOT_DIR / "secrets/gmail_credentials.json"))
        self.token_file = Path(token_file or os.getenv("GMAIL_TOKEN", ROOT_DIR / "secrets/gmail_token.json"))
        self.query = query or os.getenv("GMAIL_QUERY", "in:inbox newer_than:3d -label:PhishGuard-Phishing -label:PhishGuard-Safe")
        kw.setdefault("move_phishing", os.getenv("GMAIL_MOVE_TO_SPAM", "false").lower() == "true")
        super().__init__(**kw)
        self.service = None
        self.label_ids: dict[str, str] = {}

    def connect(self) -> None:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build

        creds = None
        if self.token_file.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_file), SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not self.credentials_file.exists():
                    raise FileNotFoundError(f"Gmail OAuth client file not found: {self.credentials_file}")
                creds = InstalledAppFlow.from_client_secrets_file(str(self.credentials_file), SCOPES).run_local_server(port=0)
            self.token_file.parent.mkdir(parents=True, exist_ok=True)
            self.token_file.write_text(creds.to_json())
        self.service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        self._ensure_labels()
        profile = self.service.users().getProfile(userId="me").execute()
        log.info("connected to Gmail as %s", profile.get("emailAddress"))

    def _ensure_labels(self) -> None:
        existing = {l["name"]: l["id"] for l in self.service.users().labels().list(userId="me").execute().get("labels", [])}
        for key, (name, color) in LABELS.items():
            if name not in existing:
                body = {"name": name, "labelListVisibility": "labelShow", "messageListVisibility": "show", "color": color}
                try:
                    existing[name] = self.service.users().labels().create(userId="me", body=body).execute()["id"]
                except Exception:  # colour palette rejected -> create without colour
                    body.pop("color")
                    existing[name] = self.service.users().labels().create(userId="me", body=body).execute()["id"]
            self.label_ids[key] = existing[name]

    def fetch_new(self) -> Iterable[tuple[str, bytes]]:
        msgs = self.service.users()
        page_token = None
        while True:
            resp = msgs.messages().list(userId="me", q=self.query, maxResults=100, pageToken=page_token).execute()
            for m in resp.get("messages", []):
                if m["id"] in self.store:
                    continue
                full = msgs.messages().get(userId="me", id=m["id"], format="raw").execute()
                yield m["id"], base64.urlsafe_b64decode(full["raw"])
            page_token = resp.get("nextPageToken")
            if not page_token:
                break

    def apply_verdict(self, msg_id: str, verdict: dict) -> None:
        phish = verdict["is_phishing"]
        if not phish and not self.tag_safe:
            return
        body = {"addLabelIds": [self.label_ids["phishing" if phish else "legitimate"]], "removeLabelIds": []}
        if phish:
            body["addLabelIds"].append("IMPORTANT")
            if self.move_phishing:
                body["addLabelIds"].append("SPAM")
                body["removeLabelIds"].append("INBOX")
        self.service.users().messages().modify(userId="me", id=msg_id, body=body).execute()
