"""Metrik evaluasi (Subbab 3.4.9)."""
import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[int(np.nanargmax(f1))]) if len(thr) else 0.5


def evaluate(y: np.ndarray, p: np.ndarray, thr: float, ks=(0.05, 0.10)) -> dict:
    y = np.asarray(y)
    p = np.asarray(p, dtype="float64")
    base = float(y.mean())
    out = {"base_rate": base, "pr_auc": float(average_precision_score(y, p))}
    out["pr_auc_rel"] = out["pr_auc"] / base if base > 0 else np.nan
    out["roc_auc"] = float(roc_auc_score(y, p))
    out["brier"] = float(brier_score_loss(y, p))
    yhat = (p >= thr).astype(int)
    out["f1"] = float(f1_score(y, yhat, zero_division=0))
    out["precision"] = float(precision_score(y, yhat, zero_division=0))
    out["recall"] = float(recall_score(y, yhat, zero_division=0))
    order = np.argsort(-p, kind="mergesort")
    for k in ks:
        top = order[: max(int(len(y) * k), 1)]
        prec_k = float(y[top].mean())
        out[f"precision@{int(k*100)}"] = prec_k
        out[f"lift@{int(k*100)}"] = prec_k / base if base > 0 else np.nan
    return out
