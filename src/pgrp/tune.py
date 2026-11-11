"""Penyetelan hiperparameter dengan Optuna (TPE), sekali pada fold 1 lalu dibekukan (Subbab 3.4.8)."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from .models import DEFAULTS, build_model

POS_WEIGHTS = [1, 2, 4, 8]


def suggest(trial, name: str) -> dict:
    s = trial.suggest_float
    c = trial.suggest_categorical
    i = trial.suggest_int
    pw = c("pos_weight", POS_WEIGHTS)
    if name == "lr":
        return dict(C=s("C", 1e-3, 1e2, log=True), pos_weight=pw)
    if name == "rf":
        # Dibatasi agar satu percobaan tidak memakan berjam-jam/RAM berlebih (kedalaman 30 dengan daun 1 sangat berat)
        return dict(max_depth=i("max_depth", 8, 24), min_samples_leaf=i("min_samples_leaf", 3, 50, log=True),
                    max_features=c("max_features", ["sqrt", 0.3, 0.5]), n_estimators=100, pos_weight=pw)
    if name == "xgb":
        return dict(learning_rate=s("learning_rate", 0.01, 0.3, log=True), max_depth=i("max_depth", 3, 10),
                    min_child_weight=s("min_child_weight", 1, 100, log=True), subsample=s("subsample", 0.5, 1.0),
                    colsample_bytree=s("colsample_bytree", 0.5, 1.0), reg_lambda=s("reg_lambda", 1e-3, 1e2, log=True),
                    reg_alpha=s("reg_alpha", 1e-3, 1e2, log=True), pos_weight=pw)
    if name == "lgbm":
        return dict(learning_rate=s("learning_rate", 0.01, 0.3, log=True), num_leaves=i("num_leaves", 15, 255),
                    min_child_samples=i("min_child_samples", 20, 500, log=True),
                    feature_fraction=s("feature_fraction", 0.5, 1.0), bagging_fraction=s("bagging_fraction", 0.5, 1.0),
                    lambda_l1=s("lambda_l1", 1e-3, 1e2, log=True), lambda_l2=s("lambda_l2", 1e-3, 1e2, log=True),
                    pos_weight=pw)
    if name == "cat":
        return dict(learning_rate=s("learning_rate", 0.01, 0.3, log=True), depth=i("depth", 4, 10),
                    l2_leaf_reg=s("l2_leaf_reg", 1, 30, log=True), random_strength=s("random_strength", 0, 5),
                    pos_weight=pw)
    if name == "tabm":
        return dict(k=32, variant=c("variant", ["tabm", "tabm-mini"]), n_blocks=i("n_blocks", 1, 4),
                    d_block=c("d_block", [64, 128, 256, 512]), dropout=s("dropout", 0.0, 0.5),
                    n_bins=i("n_bins", 8, 64), d_embedding=c("d_embedding", [8, 16, 32]),
                    lr=s("lr", 1e-4, 5e-3, log=True), weight_decay=s("weight_decay", 1e-5, 1e-2, log=True),
                    batch_size=c("batch_size", [1024, 2048]), pos_weight=pw)   # 4096 dibuang: terlalu lambat pada VRAM 6 GB
    raise KeyError(name)


def tune_model(name, spec, tr, va, cfg, n_trials, seed, timeout=None) -> dict:
    import time
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tr_fit = tr
    if name == "rf":
        n = int(cfg.get("tuning", {}).get("rf_rows", 150000))
        if len(tr) > n:
            tr_fit = tr.sample(n=n, random_state=seed)
        print(f"  RF: penyetelan memakai {len(tr_fit):,} baris latih dan 100 pohon (model akhir: 300 pohon)", flush=True)

    def objective(trial):
        params = suggest(trial, name)
        t0 = time.time()
        m = build_model(name).fit(spec, tr_fit, va, params, seed, cfg)
        val = float(average_precision_score(va["y"].to_numpy(), m.predict_proba(va)))
        trial.set_user_attr("sec", time.time() - t0)
        return val

    def progress(study, trial):
        print(f"  percobaan {trial.number + 1}/{n_trials}: PR-AUC validasi={trial.value:.4f} "
              f"({trial.user_attrs.get('sec', 0):.0f} dtk) | terbaik {study.best_value:.4f} | {trial.params}", flush=True)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[progress])
    if len(study.trials) == 0 or study.best_trial is None:
        raise RuntimeError(f"Tidak ada percobaan yang selesai untuk '{name}' (timeout terlalu pendek?).")
    best = {**DEFAULTS[name], **{k: v for k, v in study.best_params.items()}}
    if name == "rf":
        best["n_estimators"] = 300
    if name == "tabm":
        best["k"] = 32
    return {"params": best, "best_val_pr_auc": study.best_value, "n_trials": len(study.trials)}


def params_path(cfg, name: str, tag: str = "default") -> Path:
    return cfg["work"] / "tuning" / tag / f"{name}.json"


_WARNED: set = set()


def load_params(cfg, name: str, tag: str = "default") -> dict:
    p = params_path(cfg, name, tag)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))["params"]
    if (name, tag) not in _WARNED:
        _WARNED.add((name, tag))
        print(f"  [peringatan] hiperparameter hasil penyetelan untuk '{name}' tidak ditemukan; memakai nilai bawaan.")
    return dict(DEFAULTS[name])


def save_params(cfg, name: str, result: dict, tag: str = "default"):
    p = params_path(cfg, name, tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2), encoding="utf-8")
