"""Uji signifikansi dan kuantifikasi bias validasi (Subbab 3.4.10 dan 3.4.6.2)."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score


def holm_adjust(pvals: list[float]) -> list[float]:
    """Koreksi Holm-Bonferroni; mengembalikan p terkoreksi sesuai urutan masukan."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def wilcoxon_vs_ref(per_fold: pd.DataFrame, ref: str = "tabm") -> pd.DataFrame:
    """per_fold: baris = fold, kolom = model (nilai metrik per fold). Uji ref melawan tiap model lain."""
    names = [c for c in per_fold.columns if c != ref]
    rows, pvals = [], []
    for m in names:
        d = per_fold[ref] - per_fold[m]
        if np.allclose(d, 0):
            p = 1.0
            stat = 0.0
        else:
            res = wilcoxon(per_fold[ref], per_fold[m], alternative="two-sided")
            p, stat = float(res.pvalue), float(res.statistic)
        pvals.append(p)
        rows.append({"model": m, "n_fold": len(d), "mean_diff": float(d.mean()), "wilcoxon_stat": stat,
                     "p_raw": p, "wins_ref": int((d > 0).sum()), "wins_other": int((d < 0).sum())})
    adj = holm_adjust(pvals)
    for r, a in zip(rows, adj):
        r["p_holm"] = a
    return pd.DataFrame(rows)


def paired_bootstrap_delta(y_list, p_a_list, p_b_list, n_boot: int = 100, seed: int = 0) -> dict:
    """Selisih PR-AUC (A - B) rata-rata antarfold; tiap resampel meresampel baris uji pada setiap fold."""
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        per_fold = []
        for y, pa, pb in zip(y_list, p_a_list, p_b_list):
            idx = rng.integers(0, len(y), len(y))
            per_fold.append(average_precision_score(y[idx], pa[idx]) - average_precision_score(y[idx], pb[idx]))
        deltas[b] = np.mean(per_fold)
    point = float(np.mean([average_precision_score(y, pa) - average_precision_score(y, pb)
                           for y, pa, pb in zip(y_list, p_a_list, p_b_list)]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta": point, "ci_low": float(lo), "ci_high": float(hi)}


def bias_table(temporal: pd.DataFrame, random: pd.DataFrame, metric: str = "pr_auc",
               n_boot: int = 10000, seed: int = 0) -> pd.DataFrame:
    """Delta = metrik_acak - metrik_temporal per model, dengan CI bootstrap atas fold (persentil)."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in temporal.columns:
        d = (random[m] - temporal[m]).to_numpy()
        boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"model": m, "temporal": float(temporal[m].mean()), "acak": float(random[m].mean()),
                     "delta": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
                     "delta_rel_pct": float(100 * d.mean() / temporal[m].mean())})
    return pd.DataFrame(rows)
