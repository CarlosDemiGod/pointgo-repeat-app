"""Pembentukan snapshot per titik potong: populasi aktif, fitur (< t0), dan label (Subbab 3.4.3)."""
import numpy as np
import pandas as pd

from .config import get_cutoffs
from .features import build_features
from .io_utils import read_table, table_exists, write_table


def _slice_time(trx: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    """Baris dengan lo <= ts < hi. trx harus terurut menurut ts."""
    ts = trx["ts"].to_numpy()
    i = np.searchsorted(ts, np.datetime64(lo), side="left")
    j = np.searchsorted(ts, np.datetime64(hi), side="left")
    return trx.iloc[i:j]


def build_snapshot(trx: pd.DataFrame, t0: pd.Timestamp, cfg: dict) -> pd.DataFrame:
    """trx: tabel dengan kolom unit bernama 'cid' (>= 0) dan terurut menurut ts."""
    O = pd.Timedelta(days=cfg["window"]["obs_days"])
    G = pd.Timedelta(days=cfg["window"]["gap_days"])
    P = pd.Timedelta(days=cfg["window"]["pred_days"])

    hist = trx.iloc[: np.searchsorted(trx["ts"].to_numpy(), np.datetime64(t0), side="left")]   # hanya masa lalu
    ip_counts = hist.loc[hist["ip"] >= 0].groupby("ip")["cid"].nunique()                      # IP berbagi, riwayat penuh
    active = hist.loc[hist["ts"] >= t0 - O, "cid"].unique()                                   # populasi aktif
    hist_a = hist[hist["cid"].isin(active)]

    feats = build_features(hist_a, ip_counts, t0, cfg)

    fut = _slice_time(trx, t0 + G, t0 + G + P)                                                # jendela label
    positives = pd.Index(fut["cid"].unique())
    feats["y"] = feats.index.isin(positives).astype("int8")
    feats["t0"] = t0
    return feats.reset_index()


def snapshot_dir(cfg: dict, unit: str):
    return cfg["work"] / "snapshots" / unit


def prepare_unit_frame(trx: pd.DataFrame, unit: str) -> pd.DataFrame:
    col = {"family": "cid", "legacy": "cid_legacy", "buyer": "cid_buyer"}[unit]
    df = trx[trx[col] >= 0].copy()
    if col != "cid":
        df["cid"] = df[col]
    return df.sort_values("ts", kind="mergesort").reset_index(drop=True)


def build_all(trx: pd.DataFrame, cfg: dict, unit: str = "family", cutoffs=None, overwrite=False) -> pd.DataFrame:
    cutoffs = cutoffs if cutoffs is not None else get_cutoffs(cfg)
    frame = prepare_unit_frame(trx, unit)
    rows = []
    for t0 in cutoffs:
        stem = snapshot_dir(cfg, unit) / f"snap_{t0:%Y-%m}"
        if table_exists(stem) and not overwrite:
            snap = read_table(stem)
        else:
            snap = build_snapshot(frame, t0, cfg)
            write_table(snap, stem)
        rows.append({"cutoff": f"{t0:%Y-%m}", "n_aktif": len(snap), "n_positif": int(snap["y"].sum()),
                     "pct_positif": round(100 * snap["y"].mean(), 2)})
        print(f"{t0:%Y-%m}: aktif={len(snap):,} positif={int(snap['y'].sum()):,} ({100*snap['y'].mean():.2f}%)", flush=True)
    return pd.DataFrame(rows)


def load_snapshots(cfg: dict, unit: str = "family") -> pd.DataFrame:
    parts = []
    for t0 in get_cutoffs(cfg):
        stem = snapshot_dir(cfg, unit) / f"snap_{t0:%Y-%m}"
        if table_exists(stem):
            parts.append(read_table(stem))
    if not parts:
        raise FileNotFoundError(f"Tidak ada snapshot untuk unit '{unit}'. Jalankan scripts/01_build_snapshots.py dulu.")
    return pd.concat(parts, ignore_index=True)
