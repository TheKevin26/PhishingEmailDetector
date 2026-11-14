"""REST API - the universal integration point. Any mail system / client can call these endpoints.

Run:  python -m phishguard.cli serve   (or: uvicorn phishguard.api:app --host 0.0.0.0 --port 8000)
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import __version__
from .detector import get_detector

app = FastAPI(title="PhishGuard API", version=__version__,
              description="NLP-based phishing email detection. Returns phishing/legitimate with score and reasons.")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("PHISHGUARD_CORS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])

WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_RAW_BYTES = 10 * 1024 * 1024


def check_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("PHISHGUARD_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


class EmailIn(BaseModel):
    subject: str = ""
    body: str = Field("", description="Plain-text or HTML body")
    sender: str = Field("", description='From header, e.g. "PayPal <service@paypa1.com>"')
    reply_to: str = ""
    headers: dict[str, str] = Field(default_factory=dict, description="Optional extra headers (Authentication-Results, ...)")


class RawEmailIn(BaseModel):
    raw: str = Field(..., description="Full RFC 822 message, as text or base64")
    base64: bool = False


class BatchIn(BaseModel):
    emails: list[EmailIn] = Field(..., max_length=500)


@app.on_event("startup")
def _load_model() -> None:
    get_detector()


@app.get("/health")
def health() -> dict:
    d = get_detector()
    return {"status": "ok", "version": __version__, "model": d.model_used, "threshold": d.threshold}


@app.post("/predict", dependencies=[Depends(check_api_key)])
def predict(email: EmailIn) -> dict:
    return get_detector().predict_fields(email.subject, email.body, email.sender, email.reply_to, email.headers).to_dict()


@app.post("/predict/batch", dependencies=[Depends(check_api_key)])
def predict_batch(batch: BatchIn) -> dict:
    from .email_parser import from_fields
    mails = [from_fields(e.subject, e.body, e.sender, e.reply_to, e.headers) for e in batch.emails]
    return {"results": [v.to_dict() for v in get_detector().predict_parsed(mails)]}


@app.post("/predict/raw", dependencies=[Depends(check_api_key)])
def predict_raw(payload: RawEmailIn) -> dict:
    raw = base64.b64decode(payload.raw) if payload.base64 else payload.raw.encode("utf-8", errors="replace")
    if len(raw) > MAX_RAW_BYTES:
        raise HTTPException(413, "Email too large")
    return get_detector().predict_raw(raw).to_dict()


@app.post("/predict/eml", dependencies=[Depends(check_api_key)])
async def predict_eml(file: UploadFile = File(...)) -> dict:
    raw = await file.read(MAX_RAW_BYTES + 1)
    if len(raw) > MAX_RAW_BYTES:
        raise HTTPException(413, "Email too large")
    return get_detector().predict_raw(raw).to_dict()


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB_DIR / "index.html")
