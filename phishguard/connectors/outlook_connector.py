"""Microsoft 365 / Outlook.com connector via Microsoft Graph - sets Outlook categories "PhishGuard: Phishing" (red)
and "PhishGuard: Safe" (green), flags phishing, and can move it to Junk.

Setup (one time):
  1. https://entra.microsoft.com -> App registrations -> New registration
     (Accounts in any org directory + personal Microsoft accounts).
  2. Authentication -> "Allow public client flows" = Yes.
  3. API permissions -> Microsoft Graph -> Delegated -> Mail.ReadWrite, MailboxSettings.ReadWrite, offline_access.
  4. Put the Application (client) ID in .env as OUTLOOK_CLIENT_ID, then:
     python -m phishguard.cli connect outlook   (sign in once with the device code shown)
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from ..config import ROOT_DIR
from .base import MailConnector

log = logging.getLogger("phishguard.outlook")
GRAPH = "https://graph.microsoft.com/v1.0"
SCOPES = ["Mail.ReadWrite", "MailboxSettings.ReadWrite"]
CATEGORIES = {"phishing": ("PhishGuard: Phishing", "preset0"), "legitimate": ("PhishGuard: Safe", "preset4")}


class OutlookConnector(MailConnector):
    name = "outlook"

    def __init__(self, client_id: str | None = None, tenant: str | None = None, lookback_days: int | None = None, **kw):
        self.client_id = client_id or os.environ["OUTLOOK_CLIENT_ID"]
        self.tenant = tenant or os.getenv("OUTLOOK_TENANT", "common")
        self.lookback_days = int(lookback_days or os.getenv("OUTLOOK_LOOKBACK_DAYS", 3))
        self.cache_file = Path(os.getenv("OUTLOOK_TOKEN_CACHE", ROOT_DIR / "secrets/outlook_token.json"))
        kw.setdefault("move_phishing", os.getenv("OUTLOOK_MOVE_TO_JUNK", "false").lower() == "true")
        super().__init__(**kw)
        self.app = None
        self.cache = None

    # ------------------------------------------------------------ auth
    def _token(self) -> str:
        accounts = self.app.get_accounts()
        result = self.app.acquire_token_silent(SCOPES, account=accounts[0]) if accounts else None
        if not result:
            flow = self.app.initiate_device_flow(scopes=SCOPES)
            if "user_code" not in flow:
                raise RuntimeError(f"Device flow failed: {flow}")
            print(flow["message"], flush=True)
            result = self.app.acquire_token_by_device_flow(flow)
        if "access_token" not in result:
            raise RuntimeError(f"Auth failed: {result.get('error_description')}")
        if self.cache.has_state_changed:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_text(self.cache.serialize())
        return result["access_token"]

    def _req(self, method: str, path: str, **kw) -> requests.Response:
        url = path if path.startswith("http") else GRAPH + path
        resp = requests.request(method, url, headers={"Authorization": f"Bearer {self._token()}"}, timeout=60, **kw)
        resp.raise_for_status()
        return resp

    def connect(self) -> None:
        import msal
        self.cache = msal.SerializableTokenCache()
        if self.cache_file.exists():
            self.cache.deserialize(self.cache_file.read_text())
        self.app = msal.PublicClientApplication(self.client_id, authority=f"https://login.microsoftonline.com/{self.tenant}",
                                                token_cache=self.cache)
        me = self._req("GET", "/me").json()
        log.info("connected to Microsoft Graph as %s", me.get("userPrincipalName") or me.get("mail"))
        self._ensure_categories()

    def _ensure_categories(self) -> None:
        try:
            existing = {c["displayName"] for c in self._req("GET", "/me/outlook/masterCategories").json().get("value", [])}
            for name, color in CATEGORIES.values():
                if name not in existing:
                    self._req("POST", "/me/outlook/masterCategories", json={"displayName": name, "color": color})
        except requests.HTTPError as exc:
            log.warning("could not create colour categories (%s); categories will still be applied", exc)

    # ------------------------------------------------------------ mail
    def fetch_new(self) -> Iterable[tuple[str, bytes]]:
        since = (datetime.now(timezone.utc) - timedelta(days=self.lookback_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        url = (f"/me/mailFolders/inbox/messages?$select=id,subject,categories&$top=50"
               f"&$filter=receivedDateTime ge {since}&$orderby=receivedDateTime desc")
        while url:
            data = self._req("GET", url).json()
            for m in data.get("value", []):
                if m["id"] in self.store or any(c.startswith("PhishGuard") for c in m.get("categories", [])):
                    continue
                yield m["id"], self._req("GET", f"/me/messages/{m['id']}/$value").content
            url = data.get("@odata.nextLink")

    def apply_verdict(self, msg_id: str, verdict: dict) -> None:
        phish = verdict["is_phishing"]
        if not phish and not self.tag_safe:
            return
        patch = {"categories": [CATEGORIES["phishing" if phish else "legitimate"][0]]}
        if phish:
            patch["flag"] = {"flagStatus": "flagged"}
            patch["importance"] = "high"
        self._req("PATCH", f"/me/messages/{msg_id}", json=patch)
        if phish and self.move_phishing:
            self._req("POST", f"/me/messages/{msg_id}/move", json={"destinationId": "junkemail"})
