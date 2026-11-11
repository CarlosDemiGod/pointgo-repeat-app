"""Menjalankan eksperimen: temporal (rolling-origin) atau acak (pembanding), semua model, banyak seed."""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import get_cutoffs
from .evaluate import best_f1_threshold, evaluate
from .models import build_model
from .preprocess import FeatureSpec
from .snapshot import load_snapshots
from .splits import fold_masks, random_splits, rolling_origin
from .tune import load_params


def _write_metrics(rows: list, path: Path) -> None:
    """Tulis metrik dengan menggabungkan isi file saat ini (aman bila proses lain menulis ke tag yang sama)."""
    new = pd.DataFrame(rows)
    if path.exists():
        try:
            old = pd.read_csv(path)
            new = pd.concat([old, new], ignore_index=True)
        except Exception:  # noqa: BLE001
            pass
    new = new.drop_duplicates(subset=["model", "fold", "seed"], keep="last")
    tmp = path.with_suffix(".tmp")
    new.to_csv(tmp, index=False)
    tmp.replace(path)


def run_experiment(cfg, scheme="temporal", unit="family", feature_set="all", models=None, seeds=None,
                   folds=None, tag=None, tuning_tag="default", overrides=None, cutoff_filter=None,
                   exclude_high_ip=False):
    models = models or cfg["models"]
    seeds = seeds if seeds is not None else cfg["seeds"]
    overrides = overrides or {}
    tag = tag or f"{scheme}_{unit}_{feature_set}"
    out_dir = cfg["work"] / "results" / tag
    pred_dir = out_dir / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "metrics.csv"

    snap = load_snapshots(cfg, unit)
    if cutoff_filter is not None:
        snap = snap[snap["t0"] >= pd.Timestamp(cutoff_filter)].reset_index(drop=True)
    if exclude_high_ip:   # sensitivitas: buang baris yang memakai IP dengan akun melebihi ambang (persentil ke-99 tingkat IP)
        thr = cfg.get("sensitivity", {}).get("ip_row_threshold", 15)
        before = len(snap)
        snap = snap[snap["n_accounts_per_ip"].fillna(0) <= thr].reset_index(drop=True)
        print(f"Tanpa IP berbagi (> {thr} akun per IP): {before - len(snap):,} baris dibuang dari {before:,}")
    cutoffs = sorted(snap["t0"].unique())
    fdefs = rolling_origin(cutoffs, cfg["folds"]["n_folds"], cfg["folds"]["horizon_steps"])
    if folds:
        fdefs = [f for f in fdefs if f["fold"] in folds]

    done = pd.read_csv(metrics_csv) if metrics_csv.exists() else pd.DataFrame()
    rows = done.to_dict("records") if len(done) else []
    spec_cols = list(snap.columns)

    for f in fdefs:
        mtr, mva, mte = fold_masks(snap["t0"], f)
        if scheme == "temporal":
            tr, va, te = snap[mtr], snap[mva], snap[mte]
        else:  # acak berstrata, ukuran disamakan dengan fold temporal
            sizes = (int(mtr.sum()), int(mva.sum()), int(mte.sum()))
            itr, iva, ite = random_splits(snap["y"].to_numpy(), sizes, seed=f["fold"])
            tr, va, te = snap.iloc[itr], snap.iloc[iva], snap.iloc[ite]
        spec = FeatureSpec(spec_cols, feature_set).fit(tr)
        y_te = te["y"].to_numpy()
        for name in models:
            # Regresi Logistik (lbfgs) deterministik: seed tidak mengubah hasil, sehingga satu pelatihan per fold cukup
            seeds_m = seeds[:1] if name == "lr" else seeds
            have = {(r["model"], r["fold"], r["seed"]) for r in rows}
            if all((name, f["fold"], s) in have for s in seeds_m) and (pred_dir / f"{name}_fold{f['fold']}.npz").exists():
                continue
            params = {**load_params(cfg, name, tuning_tag), **overrides.get(name, {})}
            probs = []
            for s in seeds_m:
                if (name, f["fold"], s) in have:
                    continue
                t0_ = time.time()
                print(f"[{tag}] fold {f['fold']} {name} seed {s}: mulai melatih pada {len(tr):,} baris ...", flush=True)
                m = build_model(name).fit(spec, tr, va, params, s, cfg)
                t_train = time.time() - t0_
                p_va = m.predict_proba(va)
                thr = best_f1_threshold(va["y"].to_numpy(), p_va)
                t1_ = time.time()
                p_te = m.predict_proba(te)
                t_inf = time.time() - t1_
                met = evaluate(y_te, p_te, thr, cfg["evaluation"]["top_k"])
                rows.append({"scheme": scheme, "tag": tag, "model": name, "fold": f["fold"], "seed": s,
                             "test_cutoff": f"{pd.Timestamp(f['test']):%Y-%m}", "n_train": len(tr), "n_val": len(va),
                             "n_test": len(te), "threshold": thr, "train_sec": t_train, "infer_sec": t_inf, **met})
                probs.append(p_te.astype("float32"))
                _write_metrics(rows, metrics_csv)
                print(f"[{tag}] fold {f['fold']} {name} seed {s}: PR-AUC={met['pr_auc']:.4f} "
                      f"(base {met['base_rate']:.4f}) ROC-AUC={met['roc_auc']:.4f} train={t_train:.0f}s", flush=True)
            if probs:
                np.savez_compressed(pred_dir / f"{name}_fold{f['fold']}.npz", y=y_te.astype("int8"),
                                    p=np.mean(probs, axis=0).astype("float32"), cid=te["cid"].to_numpy())
    return pd.DataFrame(rows)
