"""Pembersihan data, normalisasi nama permainan, dan pembentukan unit (Subbab 3.4.1 dan 3.4.2)."""
import re

import numpy as np
import pandas as pd

from .anonymize import pseudonymize_codes

DROP_TOKENS = {"INDONESIA", "IND", "GLOBAL", "MAX", "ID"}

USECOLS = [
    "order_id", "date_create", "user_id", "zone_id", "games", "game_genre", "product",
    "product_category", "price", "raw_price", "method", "payment_group", "ip_user",
    "wa", "email_order", "username", "customer_game_key",
]


def game_family(name) -> str:
    s = re.sub(r"\(.*?\)", " ", str(name).upper())
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    out = " ".join(t for t in s.split() if t not in DROP_TOKENS) or str(name).upper().strip()
    if out.startswith("FREE FIRE"):
        return "FREE FIRE"
    if out.startswith("MOBILE LEGENDS"):
        return "MOBILE LEGENDS"
    return out


def normalize_zone(z: pd.Series) -> pd.Series:
    return z.fillna("").astype(str).str.strip().str.lstrip("|").str.lower()


def normalize_wa(s: pd.Series) -> pd.Series:
    d = s.astype("object").where(s.notna(), None)
    d = pd.Series(d, index=s.index).astype(str).str.replace(r"\D", "", regex=True)
    d = d.where(d.str.len() >= 8)
    d = d.str.replace(r"^0", "62", regex=True).str.replace(r"^8", "628", regex=True)
    return d


def buyer_key(df: pd.DataFrame) -> pd.Series:
    wa = normalize_wa(df["wa"])
    em = df["email_order"].astype("object").where(df["email_order"].notna(), None)
    em = pd.Series(em, index=df.index).str.strip().str.lower()
    un = df["username"].astype("object").where(df["username"].notna(), None)
    un = pd.Series(un, index=df.index).str.strip().str.lower()
    key = ("w:" + wa).fillna("e:" + em).fillna("u:" + un)
    return key


def clean_transactions(raw: pd.DataFrame, key: bytes) -> tuple[pd.DataFrame, dict]:
    """Ubah tabel mentah menjadi tabel transaksi bersih tanpa data pribadi.

    Keluaran memuat kode unit (cid = akun per keluarga permainan, cid_legacy = kunci lama,
    cid_buyer = pembeli, ip = alamat IP), tanpa nilai mentah pengenal.
    """
    rep = {"rows_read": int(len(raw))}
    df = raw.copy()
    df["ts"] = pd.to_datetime(df["date_create"])
    df = df.sort_values("ts", kind="mergesort").reset_index(drop=True)

    before = len(df)
    df = df.drop_duplicates(subset="order_id", keep="first")
    rep["duplicates_removed"] = int(before - len(df))

    is_default = df["user_id"].isna() | df["user_id"].astype(str).str.lower().eq("default")
    rep["default_user_removed"] = int(is_default.sum())
    df = df.loc[~is_default].reset_index(drop=True)

    fam_map = {n: game_family(n) for n in df["games"].dropna().unique()}
    df["fam"] = df["games"].map(fam_map)
    rep["game_names"] = int(df["games"].nunique())
    rep["game_families"] = int(df["fam"].nunique())

    zone = normalize_zone(df["zone_id"])
    unit_key = df["fam"] + "|" + df["user_id"].astype(str) + "|" + zone
    out = pd.DataFrame({
        "ts": df["ts"],
        "cid": pseudonymize_codes(unit_key, key),
        "cid_legacy": pseudonymize_codes(df["customer_game_key"], key),
        "cid_buyer": pseudonymize_codes(buyer_key(df), key),
        "ip": pseudonymize_codes(df["ip_user"], key),
        "fam": df["fam"].astype("category"),
        "genre": df["game_genre"].astype("category"),
        "product": pd.factorize(df["product"])[0].astype(np.int32),
        "category": df["product_category"].astype("category"),
        "price": df["price"].astype("float32"),
        "cost": df["raw_price"].astype("float32"),
        "pay_group": df["payment_group"].astype("category"),
        "method": df["method"].astype("category"),
        "hour": df["ts"].dt.hour.astype("int8"),
        "dow": df["ts"].dt.dayofweek.astype("int8"),
    })
    rep["rows_clean"] = int(len(out))
    rep["units"] = int(out["cid"].nunique())
    rep["units_legacy"] = int(out["cid_legacy"].nunique())
    rep["buyers"] = int(out.loc[out["cid_buyer"] >= 0, "cid_buyer"].nunique())
    rep["ts_min"], rep["ts_max"] = str(out["ts"].min()), str(out["ts"].max())
    return out, rep


def load_raw(csv_path, nrows=None) -> pd.DataFrame:
    return pd.read_csv(
        csv_path, usecols=USECOLS, nrows=nrows,
        dtype={c: "str" for c in ["order_id", "user_id", "zone_id", "games", "game_genre", "product",
                                   "product_category", "method", "payment_group", "ip_user", "wa",
                                   "email_order", "username", "customer_game_key"]},
    )
