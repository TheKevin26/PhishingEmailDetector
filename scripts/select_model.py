"""Compare classic vs transformer vs ensemble on the validation set; record the winner for deployment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phishguard.config import PROCESSED_DIR, REPORTS_DIR, SELECTED_MODEL_FILE  # noqa: E402
from scripts.metrics import compute_metrics, per_source_metrics  # noqa: E402


def main() -> None:
    val = pd.read_pickle(PROCESSED_DIR / "val.pkl")
    test = pd.read_pickle(PROCESSED_DIR / "test.pkl")
    probs = {}
    for name in ("classic", "transformer"):
        v, t = REPORTS_DIR / f"{name}_val_prob.npy", REPORTS_DIR / f"{name}_test_prob.npy"
        if v.exists() and t.exists():
            probs[name] = (np.load(v), np.load(t))
    if "classic" in probs and "transformer" in probs:
        probs["ensemble"] = tuple((probs["classic"][i] + probs["transformer"][i]) / 2 for i in range(2))

    table = {}
    for name, (pv, pt) in probs.items():
        table[name] = {"validation": compute_metrics(val["label"], pv), "test": compute_metrics(test["label"], pt)}
        table[name]["test"]["per_source"] = per_source_metrics(test, pt)

    # Primary: validation F1; tie-break on lower false-positive rate (fewer good emails flagged)
    best = max(table, key=lambda k: (table[k]["validation"]["f1"], -table[k]["validation"]["false_positive_rate"]))
    SELECTED_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    SELECTED_MODEL_FILE.write_text(json.dumps({"model": best}, indent=2))
    (REPORTS_DIR / "model_comparison.json").write_text(json.dumps({"selected": best, "models": table}, indent=2))

    print(f"{'model':12s} {'val F1':>8s} {'test acc':>9s} {'test P':>8s} {'test R':>8s} {'test F1':>8s} {'FPR':>7s} {'AUC':>7s}")
    for name, r in table.items():
        t = r["test"]
        print(f"{name:12s} {r['validation']['f1']:8.4f} {t['accuracy']:9.4f} {t['precision']:8.4f} "
              f"{t['recall']:8.4f} {t['f1']:8.4f} {t['false_positive_rate']:7.4f} {t['roc_auc']:7.4f}")
    print(f"\nselected for deployment: {best}")


if __name__ == "__main__":
    main()
