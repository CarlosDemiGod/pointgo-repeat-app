"""Rekayasa fitur temporal (Subbab 3.4.4). Setiap fitur hanya memakai transaksi dengan ts < t0."""
import numpy as np
import pandas as pd

CAT_COLS = ["game_family", "game_genre", "dominant_payment_group"]
META_COLS = ["cid", "t0", "y"]

CATEGORY_MAP = {
    "Diamond": "diamond",
    "Subscription / Pass": "langganan_pass",
    "Other Product": "produk_lain",
}
PRICE_BINS = [0, 1e4, 2.5e4, 5e4, 1e5, 2.5e5, np.inf]
PRICE_LABELS = ["lt10k", "10k_25k", "25k_50k", "50k_100k", "100k_250k", "ge250k"]
TIME_BINS = [0, 6, 12, 18, 24]
TIME_LABELS = ["dini_hari", "pagi", "siang", "malam"]


def peak_overlap_days(start: pd.Timestamp, end: pd.Timestamp, peaks) -> float:
    """Jumlah hari pada [start, end) yang berada di dalam periode puncak musiman."""
    total = 0.0
    for a, b in peaks:
        a, b = pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta(days=1)
        lo, hi = max(a, start), min(b, end)
        if hi > lo:
            total += (hi - lo).total_seconds() / 86400.0
    return total


def in_any_peak(ts: pd.Series, peaks) -> pd.Series:
    m = pd.Series(False, index=ts.index)
    for a, b in peaks:
        m |= (ts >= pd.Timestamp(a)) & (ts < pd.Timestamp(b) + pd.Timedelta(days=1))
    return m


def build_features(hist: pd.DataFrame, ip_counts: pd.Series, t0: pd.Timestamp, cfg: dict) -> pd.DataFrame:
    """hist: transaksi ts < t0 milik pelanggan aktif. ip_counts: akun unik per IP (riwayat penuh < t0)."""
    windows = cfg["feature_windows"]
    peaks = cfg["seasonal_peaks"]
    obs_days = cfg["window"]["obs_days"]

    df = hist.sort_values(["cid", "ts"], kind="mergesort").reset_index(drop=True)
    df["age"] = ((t0 - df["ts"]).dt.total_seconds() / 86400.0).astype("float32")
    cid = df["cid"]
    g = df.groupby("cid", sort=True)

    out = pd.DataFrame(index=g.size().index)
    out.index.name = "cid"

    # RFM dan tenure
    out["recency"] = g["age"].min()
    out["tenure"] = g["age"].max()
    out["recency_over_tenure"] = out["recency"] / (out["tenure"] + 1)
    out["freq_all"] = g.size()
    out["monetary_all"] = g["price"].sum()
    out["aov_all"] = out["monetary_all"] / out["freq_all"]
    for w in windows:
        m = df["age"] <= w
        out[f"freq_{w}"] = m.groupby(cid).sum()
        out[f"monetary_{w}"] = df["price"].where(m, 0.0).groupby(cid).sum()
        out[f"aov_{w}"] = out[f"monetary_{w}"] / out[f"freq_{w}"].replace(0, np.nan)

    # Tren
    out["freq_30_over_90"] = out["freq_30"] / (out["freq_90"] + 1)
    out["monetary_30_over_90"] = out["monetary_30"] / (out["monetary_90"] + 1)
    out["freq_7_over_30"] = out["freq_7"] / (out["freq_30"] + 1)

    # Jeda antartransaksi
    prev = g["ts"].shift(1)
    d = ((df["ts"] - prev).dt.total_seconds() / 86400.0).astype("float32")
    gd = d.groupby(cid)
    out["ipt_mean"] = gd.mean()
    out["ipt_std"] = gd.std()
    out["ipt_last"] = gd.last()
    out["ipt_last_over_mean"] = out["ipt_last"] / out["ipt_mean"].replace(0, np.nan)

    # Produk
    out["n_produk"] = g["product"].nunique()
    out["n_kategori"] = g["category"].nunique()
    cnt = df.groupby(["cid", "product"], sort=False).size()
    p = cnt / cnt.groupby(level=0).transform("sum")
    out["entropy_produk"] = (-(p * np.log(p))).groupby(level=0).sum()
    cat = df["category"].astype(str).map(CATEGORY_MAP).fillna("lainnya")
    shares = pd.crosstab(cid, cat, normalize="index")
    for name in list(CATEGORY_MAP.values()) + ["lainnya"]:
        out[f"share_kategori_{name}"] = shares[name] if name in shares.columns else 0.0
    out["max_price"] = g["price"].max()
    pb = pd.cut(df["price"], bins=PRICE_BINS, labels=PRICE_LABELS, right=False).astype(str)
    pshare = pd.crosstab(cid, pb, normalize="index")
    for lab in PRICE_LABELS:
        out[f"price_bucket_share_{lab}"] = pshare[lab] if lab in pshare.columns else 0.0

    # Pola waktu
    out["weekend_ratio"] = (df["dow"] >= 5).groupby(cid).mean()
    tp = pd.cut(df["hour"], bins=TIME_BINS, labels=TIME_LABELS, right=False).astype(str)
    tshare = pd.crosstab(cid, tp, normalize="index")
    for lab in TIME_LABELS:
        out[f"time_period_share_{lab}"] = tshare[lab] if lab in tshare.columns else 0.0
    out["hour_mean"] = g["hour"].mean()
    out["hour_std"] = g["hour"].std()

    # Pembayaran
    out["n_payment_methods"] = g["method"].nunique()
    prev_m = g["method"].shift(1)
    switched = (df["method"].astype(str) != prev_m.astype(str)) & prev_m.notna()
    out["payment_switch"] = switched.groupby(cid).sum()

    # IP berbagi: akun per IP tertinggi di antara IP yang dipakai pelanggan (riwayat penuh < t0)
    ipc = df["ip"].map(ip_counts).where(df["ip"] >= 0)
    out["n_accounts_per_ip"] = ipc.groupby(cid).max()

    # Musiman
    out["ramadan_share_obs"] = peak_overlap_days(t0 - pd.Timedelta(days=obs_days), t0, peaks) / obs_days
    first_ts = g["ts"].min()
    out["first_trx_in_peak"] = in_any_peak(first_ts, peaks).astype("int8")

    # Kategorikal (konstan per unit untuk game_family)
    out["game_family"] = g["fam"].first().astype(str)
    out["game_genre"] = g["genre"].first().astype(str)
    dom = df.groupby(["cid", "pay_group"], observed=True).size().rename("n").reset_index()
    dom = dom.sort_values(["cid", "n"], ascending=[True, False]).drop_duplicates("cid").set_index("cid")
    out["dominant_payment_group"] = dom["pay_group"].astype(str)

    num = out.select_dtypes(include="number").columns
    out[num] = out[num].astype("float32")
    return out
