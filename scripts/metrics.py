"""Shared evaluation helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)


def compute_metrics(y_true, prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    pred = (np.asarray(prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "false_positive_rate": round(float(fp / (fp + tn)) if (fp + tn) else 0.0, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    out["roc_auc"] = round(float(roc_auc_score(y_true, prob)), 4) if len(set(y_true)) > 1 else None
    return out


def per_source_metrics(df, prob) -> dict:
    res = {}
    prob = np.asarray(prob)
    for src in sorted(df["source"].unique()):
        mask = (df["source"] == src).values
        y = df["label"].values[mask]
        pred = (prob[mask] >= 0.5).astype(int)
        res[src] = {"n": int(mask.sum()), "accuracy": round(float((pred == y).mean()), 4)}
    return res
