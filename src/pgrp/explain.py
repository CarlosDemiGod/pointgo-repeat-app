"""Interpretasi SHAP (Subbab 3.4.11): TreeExplainer untuk model pohon, penjelas permutasi untuk TabM."""
import numpy as np
import pandas as pd


def tree_shap(model, X: pd.DataFrame, sample: int = 20000, seed: int = 0) -> tuple[np.ndarray, pd.DataFrame]:
    import shap
    Xs = X.sample(n=min(sample, len(X)), random_state=seed)
    values = shap.TreeExplainer(model).shap_values(Xs)
    if isinstance(values, list):          # beberapa versi mengembalikan [kelas0, kelas1]
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[..., 1]
    return values, Xs


def permutation_shap(predict_fn, X: pd.DataFrame, background: int = 200, sample: int = 2000, seed: int = 0):
    import shap
    Xb = X.sample(n=min(background, len(X)), random_state=seed)
    Xs = X.sample(n=min(sample, len(X)), random_state=seed + 1)
    masker = shap.maskers.Independent(Xb, max_samples=len(Xb))      # tanpa pemotongan otomatis ke 100 baris
    explainer = shap.Explainer(predict_fn, masker, algorithm="permutation")
    ex = explainer(Xs)
    return np.asarray(ex.values), Xs


def importance_table(values: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
    imp = np.abs(values).mean(axis=0)
    return (pd.DataFrame({"fitur": X.columns, "mean_abs_shap": imp})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
