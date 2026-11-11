"""Kumpulan fitur dan transformasi per keluarga model (Subbab 3.4.5). Semua yang dipelajari dari data latih (fit)."""
import numpy as np
import pandas as pd

from .features import CAT_COLS, META_COLS

RFM_COLS = ["recency", "freq_all", "monetary_all", "aov_all"]
UNK = "__unk__"
REDUNDANT = {"avg_price"}   # identik dengan aov_all pada snapshot lama; tidak dipakai
LOG_PREFIXES = ("monetary", "aov", "avg_price", "max_price")


class FeatureSpec:
    """feature_set: 'all' (RFM + temporal + produk + IP berbagi + musiman), 'rfm' (RFM dasar),
    'noip' (tanpa IP berbagi), 'noseason' (tanpa fitur musiman)."""

    def __init__(self, columns, feature_set: str = "all"):
        cols = list(columns)
        cat = [c for c in CAT_COLS if c in cols]
        num = [c for c in cols if c not in META_COLS and c not in CAT_COLS and c not in REDUNDANT]
        self.feature_set = feature_set
        self.use_ip_flag = False
        if feature_set == "rfm":
            num, cat = [c for c in RFM_COLS if c in num], []
        elif feature_set == "noip":
            num = [c for c in num if c != "n_accounts_per_ip"]
        elif feature_set == "noseason":            # tanpa fitur musiman (menguji dugaan pada fold 5/Idulfitri)
            num = [c for c in num if c not in ("ramadan_share_obs", "first_trx_in_peak")]
            self.use_ip_flag = "n_accounts_per_ip" in num
        elif feature_set == "all":
            self.use_ip_flag = "n_accounts_per_ip" in num
        else:
            raise ValueError(f"feature_set tidak dikenal: {feature_set}")
        self.base_num, self.cat_cols = num, cat
        self.ip_thr = None
        self.levels: dict[str, list[str]] = {}

    # ---- fit -------------------------------------------------------------
    def fit(self, train: pd.DataFrame) -> "FeatureSpec":
        if self.use_ip_flag:
            self.ip_thr = float(np.nanquantile(train["n_accounts_per_ip"].to_numpy(dtype="float64"), 0.99))
        for c in self.cat_cols:
            self.levels[c] = sorted(train[c].astype(str).unique().tolist())
        return self

    @property
    def num_cols(self) -> list[str]:
        return self.base_num + (["is_high_sharing_ip"] if self.use_ip_flag else [])

    # ---- transformasi ----------------------------------------------------
    def frame_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.base_num].astype("float32").copy()
        if self.use_ip_flag:
            X["is_high_sharing_ip"] = (df["n_accounts_per_ip"].to_numpy() > self.ip_thr).astype("float32")
        return X.reset_index(drop=True)

    def cat_str(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c in self.cat_cols:
            s = df[c].astype(str)
            out[c] = s.where(s.isin(self.levels[c]), UNK).to_numpy()
        return pd.DataFrame(out)

    def frame_trees(self, df: pd.DataFrame) -> pd.DataFrame:
        """Numerik + kategorikal bertipe category dengan himpunan kategori tetap (XGBoost/LightGBM)."""
        X = self.frame_numeric(df)
        C = self.cat_str(df)
        for c in self.cat_cols:
            X[c] = pd.Categorical(C[c], categories=self.levels[c] + [UNK])
        return X

    def frame_catboost(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.frame_numeric(df)
        C = self.cat_str(df)
        for c in self.cat_cols:
            X[c] = C[c].astype(str)
        return X

    def frame_linear(self, df: pd.DataFrame) -> pd.DataFrame:
        """Regresi Logistik: log1p pada fitur moneter yang menceng ke kanan."""
        X = self.frame_numeric(df)
        for c in X.columns:
            if c.startswith(LOG_PREFIXES):
                X[c] = np.log1p(X[c].clip(lower=0))
        C = self.cat_str(df)
        for c in self.cat_cols:
            X[c] = C[c]
        return X

    def encode_cat_int(self, df: pd.DataFrame) -> np.ndarray:
        """Kode ordinal (0 = tak dikenal, 1..L = kategori) untuk TabM."""
        C = self.cat_str(df)
        cols = []
        for c in self.cat_cols:
            m = {v: i + 1 for i, v in enumerate(self.levels[c])}
            cols.append(C[c].map(m).fillna(0).astype("int64").to_numpy())
        return np.stack(cols, axis=1) if cols else np.zeros((len(df), 0), dtype="int64")

    def cat_cardinalities(self) -> list[int]:
        return [len(self.levels[c]) + 1 for c in self.cat_cols]
