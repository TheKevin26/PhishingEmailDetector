"""Train classic ML candidates on TF-IDF + engineered features, pick the best on validation, evaluate on test."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phishguard.config import CLASSIC_MODEL_PATH, PROCESSED_DIR, RANDOM_SEED, REPORTS_DIR  # noqa: E402
from phishguard.models.classic import ColumnSubsetClassifier, build_vectorizer  # noqa: E402
from scripts.metrics import compute_metrics, per_source_metrics  # noqa: E402


def main() -> None:
    train = pd.read_pickle(PROCESSED_DIR / "train.pkl")
    val = pd.read_pickle(PROCESSED_DIR / "val.pkl")
    test = pd.read_pickle(PROCESSED_DIR / "test.pkl")
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    t0 = time.time()
    vec = build_vectorizer()
    Xtr = vec.fit_transform(train[["subject", "body"]])
    Xva = vec.transform(val[["subject", "body"]])
    print(f"vectorized: {Xtr.shape} in {time.time() - t0:.0f}s")
    ytr, yva = train["label"].values, val["label"].values

    candidates = {
        "logreg": LogisticRegression(C=8.0, max_iter=3000, solver="liblinear", random_state=RANDOM_SEED),
        "linear_svm": CalibratedClassifierCV(LinearSVC(C=0.5, random_state=RANDOM_SEED), cv=3, method="sigmoid"),
    }
    try:
        from xgboost import XGBClassifier
        candidates["xgboost"] = XGBClassifier(
            n_estimators=600, max_depth=8, learning_rate=0.1, subsample=0.9, colsample_bytree=0.5,
            tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=RANDOM_SEED,
            max_bin=64, device="cpu",  # GPU OOMs on wide sparse TF-IDF matrices with 6 GB VRAM
        )
    except ImportError:
        pass

    Xtr = Xtr.tocsr()
    results, fitted = {}, {}
    for name, clf in candidates.items():
        t0 = time.time()
        try:
            if name == "xgboost":
                # Tree model on the 20k most informative columns (by |logreg weight|)
                top = np.sort(np.argsort(-np.abs(fitted["logreg"].coef_.ravel()))[:20_000])
                clf.fit(Xtr[:, top], ytr)
                clf = ColumnSubsetClassifier(clf, top)
            else:
                clf.fit(Xtr, ytr)
        except Exception as exc:
            print(f"[skip] {name} failed: {exc}")
            continue
        prob = clf.predict_proba(Xva)[:, 1]
        m = compute_metrics(yva, prob)
        m["train_seconds"] = round(time.time() - t0, 1)
        results[name] = m
        fitted[name] = clf
        print(f"[val] {name:12s} F1={m['f1']:.4f} recall={m['recall']:.4f} precision={m['precision']:.4f} "
              f"AUC={m['roc_auc']:.4f} ({m['train_seconds']}s)")

    best = max(results, key=lambda k: (results[k]["f1"], results[k]["roc_auc"]))
    print(f"best classic model: {best}")

    best_pipe = Pipeline([("vec", vec), ("clf", fitted[best])])
    test_prob = best_pipe.predict_proba(test[["subject", "body"]])[:, 1]
    test_metrics = compute_metrics(test["label"].values, test_prob)
    test_metrics["per_source"] = per_source_metrics(test, test_prob)
    print(f"[test] {best}: " + json.dumps({k: v for k, v in test_metrics.items() if k != 'per_source'}))

    CLASSIC_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": best_pipe,
        "explainer_coef": fitted["logreg"].coef_.ravel().astype(np.float32),
        "name": best,
        "val_metrics": results[best],
        "test_metrics": test_metrics,
    }, CLASSIC_MODEL_PATH, compress=3)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(REPORTS_DIR / "classic_test_prob.npy", test_prob)
    np.save(REPORTS_DIR / "classic_val_prob.npy", fitted[best].predict_proba(Xva)[:, 1])

    (REPORTS_DIR / "classic_results.json").write_text(json.dumps(
        {"validation": results, "selected": best, "test": test_metrics}, indent=2))
    print(f"saved -> {CLASSIC_MODEL_PATH}")


if __name__ == "__main__":
    main()
