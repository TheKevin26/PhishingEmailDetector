"""Download the public dataset (Zenodo 8339691, CC BY 4.0) into data/raw/ and generate the modern email set.

  python scripts/download_data.py
"""

from __future__ import annotations

import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phishguard.config import RAW_DIR  # noqa: E402

BASE = "https://zenodo.org/records/8339691/files/{name}.csv?download=1"
FILES = ["CEAS_08", "Enron", "Ling", "Nazario", "Nigerian_Fraud", "SpamAssasin"]


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = RAW_DIR / f"{name}.csv"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"exists  {dest.name}")
            continue
        print(f"download {name}.csv ...", flush=True)
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(BASE.format(name=name), tmp)
        tmp.replace(dest)
    subprocess.run([sys.executable, str(Path(__file__).with_name("generate_modern_emails.py"))], check=True)
    print("done -> next: python scripts/prepare_data.py")


if __name__ == "__main__":
    main()
