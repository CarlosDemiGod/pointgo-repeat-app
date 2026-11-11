"""Pembagian data: rolling-origin dengan purging dan pembanding acak berstrata (Subbab 3.4.6)."""
import numpy as np
import pandas as pd


def rolling_origin(cutoffs: list, n_folds: int = 9, horizon_steps: int = 2) -> list[dict]:
    folds, n = [], len(cutoffs)
    for i in range(n_folds):
        te = n - n_folds + i                         # indeks titik potong uji
        va = te - horizon_steps                      # validasi berjarak satu horizon dari uji
        tr = list(range(0, va - horizon_steps + 1))  # latih berjarak satu horizon dari validasi
        if va < 0 or not tr:
            raise ValueError("Jumlah titik potong tidak cukup untuk rancangan fold ini.")
        folds.append(dict(fold=i + 1, train=[cutoffs[j] for j in tr], val=cutoffs[va], test=cutoffs[te]))
    return folds


def fold_masks(t0: pd.Series, fold: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tr = t0.isin(fold["train"]).to_numpy()
    va = (t0 == fold["val"]).to_numpy()
    te = (t0 == fold["test"]).to_numpy()
    return tr, va, te


def random_splits(y: np.ndarray, sizes: tuple[int, int, int], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bagi indeks baris secara acak berstrata terhadap y menjadi (latih, validasi, uji) yang saling lepas.

    sizes = (n_latih, n_validasi, n_uji). Ukuran disamakan dengan fold temporal yang bersesuaian.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    pos = rng.permutation(np.flatnonzero(y == 1))
    neg = rng.permutation(np.flatnonzero(y == 0))
    p = len(pos) / len(y)
    ip = ineg = 0
    out = {}
    for name, n in (("te", sizes[2]), ("va", sizes[1]), ("tr", sizes[0])):
        k = min(int(round(n * p)), len(pos) - ip)
        m = min(n - k, len(neg) - ineg)
        out[name] = np.concatenate([pos[ip:ip + k], neg[ineg:ineg + m]])
        ip += k
        ineg += m
    return out["tr"], out["va"], out["te"]
