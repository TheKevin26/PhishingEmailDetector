"""Central configuration. Values can be overridden through environment variables (or a .env file)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(os.getenv("PHISHGUARD_MODELS_DIR", ROOT_DIR / "models"))
REPORTS_DIR = ROOT_DIR / "reports"
STATE_DIR = Path(os.getenv("PHISHGUARD_STATE_DIR", ROOT_DIR / ".state"))

CLASSIC_MODEL_PATH = MODELS_DIR / "classic" / "model.joblib"
TRANSFORMER_MODEL_DIR = MODELS_DIR / "transformer"
SELECTED_MODEL_FILE = MODELS_DIR / "selected_model.json"

RANDOM_SEED = 42

# Decision threshold on the final phishing probability.
PHISHING_THRESHOLD = float(os.getenv("PHISHGUARD_THRESHOLD", "0.5"))

# Which model the detector uses: "auto" (whatever training selected), "classic", "transformer", "ensemble".
MODEL_CHOICE = os.getenv("PHISHGUARD_MODEL", "auto")

# Transformer settings
TRANSFORMER_BASE = os.getenv("PHISHGUARD_TRANSFORMER_BASE", "distilbert-base-uncased")
TRANSFORMER_MAX_LEN = int(os.getenv("PHISHGUARD_MAX_LEN", "256"))


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (avoids an extra dependency). Existing env vars win."""
    path = path or ROOT_DIR / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
