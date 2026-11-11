"""Inferensi untuk aplikasi: data mentah -> transaksi bersih -> snapshot -> skor terkalibrasi -> penjelasan -> uji data lampau.

Modul ini tidak bergantung pada Streamlit sehingga dapat diuji sebagai Python biasa.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from .cleaning import game_family, normalize_zone
from .snapshot import build_snapshot

REQUIRED = ["date_create", "user_id", "zone_id", "games", "product", "product_category", "price", "method", "payment_group"]
OPTIONAL = ["order_id", "game_genre", "raw_price", "ip_user"]
APP_COLS = REQUIRED + OPTIONAL
TEXT_COLS = ["order_id", "user_id", "zone_id", "games", "game_genre", "product", "product_category", "method",
             "payment_group", "ip_user"]

FEATURE_LABELS = {
    "recency_over_tenure": "Lama tak bertransaksi relatif terhadap lama jadi pelanggan",
    "recency": "Hari sejak transaksi terakhir",
    "tenure": "Hari sejak transaksi pertama",
    "freq_all": "Jumlah seluruh transaksi",
    "freq_7": "Jumlah transaksi 7 hari terakhir",
    "freq_30": "Jumlah transaksi 30 hari terakhir",
    "freq_90": "Jumlah transaksi 90 hari terakhir",
    "monetary_all": "Total belanja seluruh riwayat",
    "monetary_7": "Total belanja 7 hari terakhir",
    "monetary_30": "Total belanja 30 hari terakhir",
    "monetary_90": "Total belanja 90 hari terakhir",
    "aov_all": "Rata-rata nilai transaksi",
    "aov_30": "Rata-rata nilai transaksi 30 hari terakhir",
    "ipt_mean": "Rata-rata jeda antartransaksi (hari)",
    "ipt_std": "Keteraturan jeda antartransaksi (simpangan baku)",
    "ipt_last": "Jeda antara dua transaksi terakhir (hari)",
    "ipt_last_over_mean": "Jeda terakhir dibanding rata-rata jeda",
    "game_family": "Keluarga permainan",
    "game_genre": "Genre permainan",
    "max_price": "Nilai transaksi tertinggi",
    "n_produk": "Jumlah produk berbeda yang dibeli",
    "weekend_ratio": "Porsi transaksi akhir pekan",
    "first_trx_in_peak": "Transaksi pertama pada periode puncak Ramadan/Idulfitri",
    "ramadan_share_obs": "Porsi periode puncak pada 90 hari terakhir",
    "n_accounts_per_ip": "Jumlah akun pada alamat IP yang sama",
    "dominant_payment_group": "Kelompok pembayaran dominan",
    "n_payment_methods": "Jumlah metode pembayaran berbeda",
}


# ------------------------------------------------------------------ membaca dan membersihkan data
def read_raw(source, nrows=None) -> pd.DataFrame:
    """Baca CSV atau Parquet (path atau berkas terunggah) dan simpan hanya kolom yang dibutuhkan aplikasi."""
    name = str(getattr(source, "name", source)).lower()
    if name.endswith(".parquet"):
        df = pd.read_parquet(source)
        df = df[[c for c in APP_COLS if c in df.columns]]
        return df.head(nrows) if nrows else df
    head = pd.read_csv(source, nrows=0)
    if hasattr(source, "seek"):
        source.seek(0)
    cols = [c for c in APP_COLS if c in head.columns]
    dtype = {c: "str" for c in TEXT_COLS if c in cols}
    return pd.read_csv(source, usecols=cols, dtype=dtype, nrows=nrows)


def missing_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in REQUIRED if c not in df.columns]


def build_trx(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Tabel transaksi bersih (struktur sama dengan pipeline) beserta peta unit -> akun asli.

    Unit pelanggan = akun permainan tujuan per keluarga permainan (permainan, user_id, zone_id).
    """
    miss = missing_columns(raw)
    if miss:
        raise ValueError("Kolom wajib tidak ada: " + ", ".join(miss))
    rep = {"baris_terbaca": int(len(raw))}
    df = raw.copy()
    df["ts"] = pd.to_datetime(df["date_create"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    bad = df["ts"].isna() | df["price"].isna()
    rep["baris_tanggal_atau_harga_tidak_valid"] = int(bad.sum())
    df = df.loc[~bad]
    for c, default in (("game_genre", "Unknown"), ("raw_price", np.nan), ("ip_user", None), ("order_id", None)):
        if c not in df.columns:
            df[c] = default
    df = df.sort_values("ts", kind="mergesort")
    if df["order_id"].notna().any():
        before = len(df)
        df = df.drop_duplicates(subset="order_id", keep="first")
        rep["duplikat_dihapus"] = int(before - len(df))
    is_default = df["user_id"].isna() | df["user_id"].astype(str).str.lower().eq("default")
    rep["baris_tanpa_identitas_akun_dikeluarkan"] = int(is_default.sum())
    df = df.loc[~is_default].reset_index(drop=True)
    if df.empty:
        raise ValueError("Tidak ada transaksi yang dapat dipakai setelah pembersihan.")

    fam_map = {n: game_family(n) for n in df["games"].dropna().unique()}
    df["fam"] = df["games"].map(fam_map).fillna("TIDAK DIKETAHUI")
    zone = normalize_zone(df["zone_id"])
    uid = df["user_id"].astype(str)
    key = df["fam"] + "|" + uid + "|" + zone
    cid, _ = pd.factorize(key)
    first = ~pd.Series(cid).duplicated().to_numpy()
    idmap = pd.DataFrame({"cid": cid[first], "keluarga_permainan": df["fam"].to_numpy()[first],
                          "user_id": uid.to_numpy()[first], "zone_id": zone.to_numpy()[first]})
    trx = pd.DataFrame({
        "ts": df["ts"], "cid": cid, "ip": pd.factorize(df["ip_user"])[0],
        "fam": df["fam"].astype("category"), "genre": df["game_genre"].fillna("Unknown").astype("category"),
        "product": pd.factorize(df["product"])[0].astype(np.int32),
        "category": df["product_category"].astype("category"),
        "price": df["price"].astype("float32"),
        "cost": pd.to_numeric(df["raw_price"], errors="coerce").astype("float32"),
        "pay_group": df["payment_group"].astype("category"), "method": df["method"].astype("category"),
        "hour": df["ts"].dt.hour.astype("int8"), "dow": df["ts"].dt.dayofweek.astype("int8"),
    })
    rep.update(transaksi_bersih=int(len(trx)), akun_unik=int(len(idmap)),
               tanggal_awal=str(trx["ts"].min()), tanggal_akhir=str(trx["ts"].max()))
    return trx, idmap, rep


# -------------------------------------------------------------------------------- tanggal
def default_predict_t0(trx: pd.DataFrame) -> pd.Timestamp:
    """Prediksi dimulai sehari setelah transaksi terakhir pada data."""
    return trx["ts"].max().floor("D") + pd.Timedelta(days=1)


def labels_complete(trx: pd.DataFrame, t0, cfg: dict) -> bool:
    w = cfg["window"]
    return pd.Timestamp(t0) + pd.Timedelta(days=w["gap_days"] + w["pred_days"]) <= trx["ts"].max() + pd.Timedelta(days=1)


def latest_backtest_t0(trx: pd.DataFrame, cfg: dict) -> pd.Timestamp:
    w = cfg["window"]
    return (trx["ts"].max() + pd.Timedelta(days=1) - pd.Timedelta(days=w["gap_days"] + w["pred_days"])).floor("D")


def make_snapshot(trx: pd.DataFrame, t0, cfg: dict) -> pd.DataFrame:
    snap = build_snapshot(trx, pd.Timestamp(t0), cfg)
    if snap.empty:
        raise ValueError("Tidak ada pelanggan aktif dalam 90 hari sebelum tanggal prediksi.")
    return snap


# ------------------------------------------------------------------------------- artefak
class ArtifactBundle:
    """Model terlatih, kalibrator, dan metadata dari scripts/06_export_artifacts.py."""

    def __init__(self, directory):
        d = Path(directory)
        self.dir = d
        self.meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self.cfg = self.meta["cfg"]
        self.models, self.load_errors = {}, {}
        for name in self.meta.get("models", []):
            p = d / f"{name}.joblib"
            if not p.exists():
                self.load_errors[name] = "berkas tidak ditemukan"
                continue
            try:
                self.models[name] = joblib.load(p)
            except Exception as e:  # noqa: BLE001  (mis. pustaka belum terpasang)
                self.load_errors[name] = f"{type(e).__name__}: {e}"

    def available(self) -> list[str]:
        return list(self.models)

    def predict(self, name: str, snap: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        m = self.models[name]
        raw = np.asarray(m["model"].predict_proba(snap), dtype="float64")
        cal = m["calibrator"].predict(raw) if m.get("calibrator") is not None else raw
        return raw, np.asarray(cal, dtype="float64")

    def independent(self, t0) -> bool:
        """True bila periode uji berada setelah data validasi yang dipakai pelatihan."""
        after = self.meta.get("independent_after")
        return bool(after) and pd.Timestamp(t0) > pd.Timestamp(after)


def explain_rows(bundle: ArtifactBundle, snap_rows: pd.DataFrame, model_name: str = "lgbm", top: int = 8) -> list[pd.DataFrame]:
    """Nilai SHAP per baris (hanya model pohon). Memerlukan paket shap."""
    import shap
    m = bundle.models[model_name]["model"]
    X = m.spec.frame_trees(snap_rows)
    values = shap.TreeExplainer(m.bst).shap_values(X)
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[..., 1]
    out = []
    for i in range(len(X)):
        d = pd.DataFrame({"fitur": X.columns, "nilai": [X.iloc[i][c] for c in X.columns], "shap": values[i]})
        d["label"] = d["fitur"].map(FEATURE_LABELS).fillna(d["fitur"])
        d = d.reindex(d["shap"].abs().sort_values(ascending=False).index).head(top).reset_index(drop=True)
        out.append(d)
    return out


# ---------------------------------------------------------------------------- uji data lampau
def backtest_metrics(y, p, ks=(0.05, 0.10, 0.20)) -> dict:
    y = np.asarray(y)
    p = np.asarray(p, dtype="float64")
    base = float(y.mean())
    out = {"n": int(len(y)), "positif": int(y.sum()), "proporsi_positif": base}
    if y.min() == y.max():
        return {**out, "pr_auc": np.nan, "pr_auc_rel": np.nan, "roc_auc": np.nan}
    out["pr_auc"] = float(average_precision_score(y, p))
    out["pr_auc_rel"] = out["pr_auc"] / base if base > 0 else np.nan
    out["roc_auc"] = float(roc_auc_score(y, p))
    order = np.argsort(-p, kind="mergesort")
    for k in ks:
        top = order[: max(int(len(y) * k), 1)]
        prec = float(y[top].mean())
        out[f"presisi@{int(k * 100)}"] = prec
        out[f"lift@{int(k * 100)}"] = prec / base if base > 0 else np.nan
        out[f"jangkauan@{int(k * 100)}"] = float(y[top].sum() / max(y.sum(), 1))
    return out


def pr_curve(y, p):
    prec, rec, _ = precision_recall_curve(np.asarray(y), np.asarray(p))
    return prec, rec


def risk_groups(prob: np.ndarray) -> pd.Series:
    """Kelompok menurut persentil skor di antara pelanggan yang dinilai. Gunakan skor kontinu (mentah), bukan skor terkalibrasi."""
    pct = pd.Series(prob).rank(pct=True, method="average")
    return pd.cut(pct, bins=[0, 0.2, 0.8, 1.0], labels=["Risiko tinggi tidak kembali (20% terendah)",
                                                         "Menengah", "Peluang kembali tinggi (20% tertinggi)"],
                  include_lowest=True).astype(str)
