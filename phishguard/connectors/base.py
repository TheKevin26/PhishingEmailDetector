"""Shared connector loop: fetch new mail -> score -> tag mailbox. Provider-specific classes implement 3 methods."""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

from ..config import STATE_DIR

log = logging.getLogger("phishguard.connector")


class Scorer:
    """Scores raw emails either in-process or through a remote PhishGuard API (PHISHGUARD_API_URL)."""

    def __init__(self):
        self.api_url = os.getenv("PHISHGUARD_API_URL", "").rstrip("/")
        self.api_key = os.getenv("PHISHGUARD_API_KEY", "")
        if not self.api_url:
            from ..detector import get_detector
            self.detector = get_detector()

    def score_raw(self, raw: bytes) -> dict:
        if not self.api_url:
            return self.detector.predict_raw(raw).to_dict()
        import requests
        resp = requests.post(f"{self.api_url}/predict/raw", timeout=60,
                             headers={"X-API-Key": self.api_key} if self.api_key else {},
                             json={"raw": base64.b64encode(raw).decode(), "base64": True})
        resp.raise_for_status()
        return resp.json()


class ProcessedStore:
    """Remembers which messages were already scanned so restarts don't re-process the mailbox."""

    def __init__(self, name: str, max_items: int = 50_000):
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self.path: Path = STATE_DIR / f"{name}.json"
        self.max_items = max_items
        self.ids: list[str] = json.loads(self.path.read_text()) if self.path.exists() else []
        self._set = set(self.ids)

    def __contains__(self, msg_id: str) -> bool:
        return msg_id in self._set

    def add(self, msg_id: str) -> None:
        if msg_id not in self._set:
            self.ids.append(msg_id)
            self._set.add(msg_id)
            if len(self.ids) > self.max_items:
                drop = self.ids[: len(self.ids) - self.max_items]
                self.ids = self.ids[len(drop):]
                self._set.difference_update(drop)

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.ids))
        tmp.replace(self.path)


class MailConnector(ABC):
    name = "base"

    def __init__(self, tag_safe: bool = True, move_phishing: bool = False, dry_run: bool = False):
        self.tag_safe = tag_safe
        self.move_phishing = move_phishing
        self.dry_run = dry_run
        self.scorer = Scorer()
        self.store = ProcessedStore(self.state_key())

    def state_key(self) -> str:
        return self.name

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def fetch_new(self) -> Iterable[tuple[str, bytes]]:
        """Yield (message_id, raw RFC822 bytes) for messages not yet processed."""

    @abstractmethod
    def apply_verdict(self, msg_id: str, verdict: dict) -> None:
        """Tag / label / move the message according to the verdict."""

    def run_once(self) -> dict:
        stats = {"scanned": 0, "phishing": 0, "legitimate": 0, "errors": 0}
        for msg_id, raw in self.fetch_new():
            if msg_id in self.store:
                continue
            try:
                verdict = self.scorer.score_raw(raw)
                subject = verdict.get("email", {}).get("subject", "")[:80]
                log.info("%-10s score=%.2f  %s", verdict["label"].upper(), verdict["score"], subject)
                if not self.dry_run:
                    self.apply_verdict(msg_id, verdict)
                stats["scanned"] += 1
                stats[verdict["label"]] += 1
                self.store.add(msg_id)
            except Exception as exc:
                stats["errors"] += 1
                log.exception("failed on message %s: %s", msg_id, exc)
        self.store.save()
        return stats

    def run_forever(self, interval: int = 60) -> None:
        self.connect()
        log.info("[%s] watching mailbox every %ss (dry_run=%s)", self.name, interval, self.dry_run)
        while True:
            try:
                stats = self.run_once()
                if stats["scanned"] or stats["errors"]:
                    log.info("[%s] %s", self.name, stats)
            except Exception as exc:
                log.exception("[%s] cycle failed (%s); reconnecting", self.name, exc)
                time.sleep(5)
                try:
                    self.connect()
                except Exception:
                    log.exception("reconnect failed")
            time.sleep(interval)
