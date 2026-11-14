"""Classic NLP model: TF-IDF (word + char n-grams) + hand-crafted features -> linear / boosted classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from ..features import FEATURE_NAMES, EmailFeatureExtractor
from ..preprocessing import build_text, normalize_text


def to_normalized_text(X: pd.DataFrame) -> list[str]:
    return [normalize_text(build_text(s, b)) for s, b in zip(X["subject"], X["body"])]


def build_vectorizer(word_max: int = 150_000, char_max: int = 150_000) -> ColumnTransformer:
    text_pipe_word = Pipeline([
        ("norm", FunctionTransformer(to_normalized_text, validate=False)),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=word_max,
                                  sublinear_tf=True, token_pattern=r"(?u)\b[a-z][a-z0-9_']+\b",
                                  dtype=np.float32)),
    ])
    text_pipe_char = Pipeline([
        ("norm", FunctionTransformer(to_normalized_text, validate=False)),
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=char_max,
                                  sublinear_tf=True, dtype=np.float32)),
    ])
    feats = Pipeline([("extract", EmailFeatureExtractor()), ("scale", StandardScaler())])
    return ColumnTransformer(
        [("word", text_pipe_word, ["subject", "body"]),
         ("char", text_pipe_char, ["subject", "body"]),
         ("feat", feats, ["subject", "body"])],
        sparse_threshold=1.0,
    )


class ColumnSubsetClassifier:
    """Wraps a classifier that is trained on a subset of the vectorizer's columns (e.g. XGBoost on top-k)."""

    def __init__(self, clf, columns: np.ndarray):
        self.clf = clf
        self.columns = np.asarray(columns)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        return self.clf.predict_proba(sparse.csr_matrix(X)[:, self.columns])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def as_frame(subjects, bodies) -> pd.DataFrame:
    return pd.DataFrame({"subject": list(subjects), "body": list(bodies)})


_NAMES_CACHE: dict[int, np.ndarray] = {}


def feature_names(vectorizer: ColumnTransformer) -> np.ndarray:
    """Column names in the vectorizer's output order: word__<ngram>, char__<ngram>, feat__<feature>."""
    key = id(vectorizer)
    if key not in _NAMES_CACHE:
        parts = []
        for name, trans, _ in vectorizer.transformers_:
            if name in ("word", "char"):
                vocab = trans.named_steps["tfidf"].get_feature_names_out()
                parts.append(np.char.add(f"{name}__", vocab.astype(str)))
            elif name == "feat":
                parts.append(np.char.add("feat__", np.asarray(FEATURE_NAMES)))
        _NAMES_CACHE[key] = np.concatenate(parts)
    return _NAMES_CACHE[key]


def explain_linear(vectorizer: ColumnTransformer, coef: np.ndarray, X: pd.DataFrame, top_k: int = 8) -> list[dict]:
    """Return the n-grams / features that pushed the score most towards phishing for a single email."""
    row = vectorizer.transform(X)
    row = sparse.csr_matrix(row)
    contrib = row.multiply(coef.reshape(1, -1)).tocsr()
    names = feature_names(vectorizer)
    idx, vals = contrib.indices, contrib.data
    order = np.argsort(-vals)
    out = []
    for i in order[:top_k * 3]:
        if vals[i] <= 0:
            break
        name = str(names[idx[i]])
        if name.startswith("char__"):
            continue  # char n-grams are not human readable
        kind, _, token = name.partition("__")
        if token in {"urltoken", "numtoken", "emailtoken", "moneytoken"}:
            token = {"urltoken": "<link>", "numtoken": "<number>", "emailtoken": "<email address>",
                     "moneytoken": "<money amount>"}[token]
        out.append({"token": token, "type": "feature" if kind == "feat" else "phrase", "weight": round(float(vals[i]), 3)})
        if len(out) >= top_k:
            break
    return out
