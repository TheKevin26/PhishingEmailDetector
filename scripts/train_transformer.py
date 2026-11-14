"""Fine-tune DistilBERT for phishing detection (GPU, mixed precision)."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phishguard.config import (PROCESSED_DIR, RANDOM_SEED, REPORTS_DIR, TRANSFORMER_BASE,  # noqa: E402
                               TRANSFORMER_MAX_LEN, TRANSFORMER_MODEL_DIR)
from phishguard.models.transformer import TransformerClassifier  # noqa: E402
from phishguard.preprocessing import transformer_text  # noqa: E402
from scripts.metrics import compute_metrics, per_source_metrics  # noqa: E402


class EmailDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.texts = [transformer_text(s, b) for s, b in zip(df["subject"], df["body"])]
        self.labels = df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=TRANSFORMER_MAX_LEN)
    args = ap.parse_args()

    torch.manual_seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} base={TRANSFORMER_BASE}")

    train = pd.read_pickle(PROCESSED_DIR / "train.pkl")
    val = pd.read_pickle(PROCESSED_DIR / "val.pkl")
    test = pd.read_pickle(PROCESSED_DIR / "test.pkl")

    tok = AutoTokenizer.from_pretrained(TRANSFORMER_BASE)
    model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_BASE, num_labels=2).to(device)

    def collate(batch):
        texts, labels = zip(*batch)
        enc = tok(list(texts), truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(labels)
        return enc

    train_dl = DataLoader(EmailDataset(train), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    best_f1, history = -1.0, []
    TRANSFORMER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        t0, running = time.time(), 0.0
        for step, batch in enumerate(train_dl, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                loss = model(**batch).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            if step % 200 == 0:
                rate = step / (time.time() - t0)
                eta = (len(train_dl) - step) / rate
                print(f"epoch {epoch + 1} step {step}/{len(train_dl)} loss={running / 200:.4f} "
                      f"eta={math.ceil(eta / 60)}min", flush=True)
                running = 0.0

        model.eval()
        model.save_pretrained(TRANSFORMER_MODEL_DIR / "_tmp")
        tok.save_pretrained(TRANSFORMER_MODEL_DIR / "_tmp")
        clf = TransformerClassifier(TRANSFORMER_MODEL_DIR / "_tmp", device=device, max_len=args.max_len)
        val_prob = clf.predict_proba(val["subject"], val["body"], batch_size=64)
        m = compute_metrics(val["label"].values, val_prob)
        history.append({"epoch": epoch + 1, **m})
        print(f"[val] epoch {epoch + 1}: F1={m['f1']:.4f} recall={m['recall']:.4f} "
              f"precision={m['precision']:.4f} AUC={m['roc_auc']:.4f}", flush=True)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            model.save_pretrained(TRANSFORMER_MODEL_DIR)
            tok.save_pretrained(TRANSFORMER_MODEL_DIR)
        del clf

    clf = TransformerClassifier(TRANSFORMER_MODEL_DIR, device=device, max_len=args.max_len)
    val_prob = clf.predict_proba(val["subject"], val["body"], batch_size=64)
    test_prob = clf.predict_proba(test["subject"], test["body"], batch_size=64)
    val_metrics = compute_metrics(val["label"].values, val_prob)
    test_metrics = compute_metrics(test["label"].values, test_prob)
    test_metrics["per_source"] = per_source_metrics(test, test_prob)
    print("[test] transformer: " + json.dumps({k: v for k, v in test_metrics.items() if k != "per_source"}))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(REPORTS_DIR / "transformer_val_prob.npy", val_prob)
    np.save(REPORTS_DIR / "transformer_test_prob.npy", test_prob)
    (REPORTS_DIR / "transformer_results.json").write_text(json.dumps(
        {"args": vars(args), "history": history, "validation": val_metrics, "test": test_metrics}, indent=2))

    import shutil
    shutil.rmtree(TRANSFORMER_MODEL_DIR / "_tmp", ignore_errors=True)
    print(f"saved -> {TRANSFORMER_MODEL_DIR}")


if __name__ == "__main__":
    main()
