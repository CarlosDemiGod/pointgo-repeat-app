"""Pembungkus model: Regresi Logistik, Random Forest, XGBoost, LightGBM, CatBoost, dan TabM (Subbab 3.4.7)."""
import copy
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, QuantileTransformer, StandardScaler

from .preprocess import FeatureSpec

DEFAULTS = {
    "lr": dict(C=1.0, pos_weight=1),
    "rf": dict(max_depth=16, min_samples_leaf=10, max_features="sqrt", pos_weight=1, n_estimators=300),
    "xgb": dict(learning_rate=0.05, max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, reg_alpha=0.0, pos_weight=1),
    "lgbm": dict(learning_rate=0.05, num_leaves=63, min_child_samples=50, feature_fraction=0.8,
                 bagging_fraction=0.8, lambda_l1=0.0, lambda_l2=1.0, pos_weight=1),
    "cat": dict(learning_rate=0.08, depth=6, l2_leaf_reg=3.0, random_strength=1.0, pos_weight=1),
    "tabm": dict(variant="tabm", k=32, n_blocks=2, d_block=256, dropout=0.1, n_bins=32, d_embedding=16,
                 lr=1e-3, weight_decay=1e-4, batch_size=2048, pos_weight=1),
}


def _pw(params):
    return float(params.get("pos_weight", 1))


class BaseModel:
    name = ""

    def fit(self, spec: FeatureSpec, tr: pd.DataFrame, va: pd.DataFrame, params: dict, seed: int, cfg: dict):
        raise NotImplementedError

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------- baseline
class LRModel(BaseModel):
    name = "lr"

    def fit(self, spec, tr, va, params, seed, cfg):
        self.spec = spec
        num, cat = spec.num_cols, spec.cat_cols
        parts = [("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                                   ("sc", StandardScaler())]), num)]
        if cat:
            parts.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), cat))
        clf = LogisticRegression(C=params["C"], class_weight={0: 1.0, 1: _pw(params)}, max_iter=300,
                                 solver="lbfgs", random_state=seed)
        self.pipe = Pipeline([("pre", ColumnTransformer(parts)), ("clf", clf)])
        self.pipe.fit(spec.frame_linear(tr), tr["y"].to_numpy())
        return self

    def predict_proba(self, df):
        return self.pipe.predict_proba(self.spec.frame_linear(df))[:, 1]


class RFModel(BaseModel):
    name = "rf"

    def fit(self, spec, tr, va, params, seed, cfg):
        self.spec = spec
        n = min(len(tr), int(cfg["training"]["rf_max_train_rows"]))
        sub = tr.sample(n=n, random_state=seed) if n < len(tr) else tr
        num, cat = spec.num_cols, spec.cat_cols
        parts = [("num", SimpleImputer(strategy="median", add_indicator=True), num)]
        if cat:
            parts.append(("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat))
        clf = RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 300)), max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]), max_features=params["max_features"],
            class_weight={0: 1.0, 1: _pw(params)}, n_jobs=cfg["training"]["n_jobs"], random_state=seed)
        self.pipe = Pipeline([("pre", ColumnTransformer(parts)), ("clf", clf)])
        self.pipe.fit(self._frame(sub), sub["y"].to_numpy())
        return self

    def _frame(self, df):
        X = self.spec.frame_numeric(df)
        C = self.spec.cat_str(df)
        for c in self.spec.cat_cols:
            X[c] = C[c]
        return X

    def predict_proba(self, df):
        return self.pipe.predict_proba(self._frame(df))[:, 1]


# --------------------------------------------------------------------- GBDT
class XGBModel(BaseModel):
    name = "xgb"

    def fit(self, spec, tr, va, params, seed, cfg):
        import xgboost as xgb
        self.spec = spec
        t = cfg["training"]
        dtr = xgb.DMatrix(spec.frame_trees(tr), label=tr["y"].to_numpy(), enable_categorical=True)
        dva = xgb.DMatrix(spec.frame_trees(va), label=va["y"].to_numpy(), enable_categorical=True)
        p = dict(objective="binary:logistic", eval_metric="aucpr", tree_method="hist", seed=seed,
                 learning_rate=params["learning_rate"], max_depth=int(params["max_depth"]),
                 min_child_weight=params["min_child_weight"], subsample=params["subsample"],
                 colsample_bytree=params["colsample_bytree"], reg_lambda=params["reg_lambda"],
                 reg_alpha=params["reg_alpha"], scale_pos_weight=_pw(params))
        if t["n_jobs"] and t["n_jobs"] > 0:
            p["nthread"] = t["n_jobs"]
        use_gpu = bool(t.get("gpu", {}).get("xgb", False))
        self.device = "cpu"
        for attempt in (["cuda", "cpu"] if use_gpu else ["cpu"]):
            try:
                pp = dict(p)
                if attempt == "cuda":
                    pp["device"] = "cuda"
                self.bst = xgb.train(pp, dtr, num_boost_round=t["max_rounds"], evals=[(dva, "val")],
                                     early_stopping_rounds=t["early_stopping_rounds"], verbose_eval=False)
                self.device = attempt
                break
            except Exception as e:  # noqa: BLE001
                if attempt == "cuda":
                    print(f"  [peringatan] XGBoost GPU gagal ({type(e).__name__}); mencoba CPU.")
                else:
                    raise
        self.best_iter = self.bst.best_iteration + 1
        return self

    def predict_proba(self, df):
        import xgboost as xgb
        d = xgb.DMatrix(self.spec.frame_trees(df), enable_categorical=True)
        return self.bst.predict(d, iteration_range=(0, self.best_iter))


class LGBMModel(BaseModel):
    name = "lgbm"

    def fit(self, spec, tr, va, params, seed, cfg):
        import lightgbm as lgb
        self.spec = spec
        t = cfg["training"]
        dtr = lgb.Dataset(spec.frame_trees(tr), label=tr["y"].to_numpy(), categorical_feature=spec.cat_cols or "auto")
        dva = lgb.Dataset(spec.frame_trees(va), label=va["y"].to_numpy(), reference=dtr)
        p = dict(objective="binary", metric="average_precision", verbose=-1, seed=seed,
                 learning_rate=params["learning_rate"], num_leaves=int(params["num_leaves"]),
                 min_child_samples=int(params["min_child_samples"]), feature_fraction=params["feature_fraction"],
                 bagging_fraction=params["bagging_fraction"], bagging_freq=1, lambda_l1=params["lambda_l1"],
                 lambda_l2=params["lambda_l2"], scale_pos_weight=_pw(params), num_threads=t["n_jobs"] if t["n_jobs"] > 0 else 0)
        self.bst = lgb.train(p, dtr, num_boost_round=t["max_rounds"], valid_sets=[dva],
                             callbacks=[lgb.early_stopping(t["early_stopping_rounds"], verbose=False)])
        return self

    def predict_proba(self, df):
        return self.bst.predict(self.spec.frame_trees(df), num_iteration=self.bst.best_iteration)


class CatModel(BaseModel):
    name = "cat"

    def fit(self, spec, tr, va, params, seed, cfg):
        from catboost import CatBoostClassifier, Pool
        self.spec = spec
        t = cfg["training"]
        self.cat_cols = spec.cat_cols
        base_kw = dict(
            iterations=t["max_rounds"], learning_rate=params["learning_rate"], depth=int(params["depth"]),
            l2_leaf_reg=params["l2_leaf_reg"], random_strength=params["random_strength"],
            scale_pos_weight=_pw(params), loss_function="Logloss", eval_metric=str(t.get("cat_eval_metric", "PRAUC")),
            random_seed=seed, od_type="Iter", od_wait=t["early_stopping_rounds"], verbose=False,
            allow_writing_files=False)
        ptr = Pool(spec.frame_catboost(tr), tr["y"].to_numpy(), cat_features=self.cat_cols)
        pva = Pool(spec.frame_catboost(va), va["y"].to_numpy(), cat_features=self.cat_cols)
        use_gpu = bool(t.get("gpu", {}).get("cat", False))
        self.device = "cpu"
        for attempt in (["gpu", "cpu"] if use_gpu else ["cpu"]):
            try:
                kw = dict(base_kw)
                kw.update(task_type="GPU", devices="0") if attempt == "gpu" else kw.update(thread_count=t["n_jobs"])
                m = CatBoostClassifier(**kw)
                m.fit(ptr, eval_set=pva, use_best_model=True)
                self.device = attempt
                break
            except Exception as e:  # noqa: BLE001
                if attempt == "gpu":
                    print(f"  [peringatan] CatBoost GPU gagal ({type(e).__name__}: {str(e)[:80]}); mencoba CPU.")
                else:
                    raise
        self.model = m
        return self

    def predict_proba(self, df):
        from catboost import Pool
        return self.model.predict_proba(Pool(self.spec.frame_catboost(df), cat_features=self.cat_cols))[:, 1]


# --------------------------------------------------------------------- TabM
class TabMModel(BaseModel):
    """TabM [Gorishniy dkk., 2025] dengan piecewise-linear embeddings. Nama argumen mengikuti paket `tabm`."""
    name = "tabm"

    def _tensors(self, df, fit=False, seed=0):
        import torch
        X = self.spec.frame_numeric(df)
        if fit:
            self.num_cols_ = list(X.columns)
            self.imp = SimpleImputer(strategy="median", add_indicator=True).fit(X)
        A = self.imp.transform(X)
        n_num = len(self.num_cols_)
        num, ind = A[:, :n_num], A[:, n_num:]
        if fit:
            self.qt = QuantileTransformer(n_quantiles=min(1000, len(num)), output_distribution="normal",
                                          subsample=200_000, random_state=seed).fit(num)
        num = self.qt.transform(num).astype("float32")
        if fit:
            self.keep_ = num.std(axis=0) > 1e-6          # kolom konstan membuat compute_bins gagal
        num = num[:, self.keep_]
        cat = self.spec.encode_cat_int(df)
        if ind.shape[1]:
            cat = np.concatenate([cat, ind.astype("int64")], axis=1)
        return torch.as_tensor(num), (torch.as_tensor(cat) if cat.shape[1] else None)

    def fit(self, spec, tr, va, params, seed, cfg):
        import torch
        import torch.nn.functional as F
        from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins
        from tabm import TabM
        t = cfg["training"]
        t_fit0 = time.time()
        self.spec = spec
        torch.manual_seed(seed)
        try:
            torch.set_float32_matmul_precision("high")   # TF32 pada GPU Ampere ke atas: lebih cepat, akurasi cukup
        except Exception:  # noqa: BLE001
            pass
        np.random.seed(seed)
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_batch = int(t.get("tabm_eval_batch", 2048))   # batch besar saat evaluasi memenuhi VRAM dan sangat lambat
        xn, xc = self._tensors(tr, fit=True, seed=seed)
        yn = torch.as_tensor(tr["y"].to_numpy().astype("float32"))
        xn_v, xc_v = self._tensors(va)
        y_v = va["y"].to_numpy()

        cards = spec.cat_cardinalities()
        n_ind = (xc.shape[1] - len(cards)) if xc is not None else 0
        cards = cards + [2] * n_ind
        bins = compute_bins(xn, n_bins=int(params["n_bins"]))
        try:
            emb = PiecewiseLinearEmbeddings(bins, d_embedding=int(params["d_embedding"]), activation=False, version="B")
        except TypeError:
            emb = PiecewiseLinearEmbeddings(bins, d_embedding=int(params["d_embedding"]), activation=False)
        k = int(params["k"])
        make_kw = dict(n_num_features=xn.shape[1], cat_cardinalities=cards or None, d_out=1, k=k,
                       num_embeddings=emb, n_blocks=int(params["n_blocks"]), d_block=int(params["d_block"]),
                       dropout=float(params["dropout"]))
        if params.get("variant", "tabm") == "tabm-mini":
            make_kw["arch_type"] = "tabm-mini"
        self.model = TabM.make(**make_kw).to(self.dev)
        opt = torch.optim.AdamW(self.model.parameters(), lr=float(params["lr"]), weight_decay=float(params["weight_decay"]))
        pos_w = torch.tensor(_pw(params), device=self.dev)
        amp = str(t.get("tabm_amp", "off")).lower()
        self.use_amp = self.dev.type == "cuda" and amp in ("bf16", "fp16")
        self.amp_dtype = torch.bfloat16 if amp == "bf16" else torch.float16
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp and amp == "fp16")
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp and amp == "fp16")

        xn, yn = xn.to(self.dev), yn.to(self.dev)
        xc = xc.to(self.dev) if xc is not None else None
        bs, n = int(params["batch_size"]), len(yn)
        best, best_state, bad = -1.0, None, 0
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        print(f"      TabM: persiapan data dan model {time.time() - t_fit0:.1f} dtk | batch {bs} | {n:,} baris latih | "
              f"{n // bs + 1} langkah/epoch | perangkat {self.dev}", flush=True)
        for epoch in range(t["tabm_max_epochs"]):
            t_ep = time.time()
            self.model.train()
            perm = torch.randperm(n, device=self.dev)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                with torch.autocast(device_type=self.dev.type, dtype=self.amp_dtype, enabled=self.use_amp):
                    logits = self.model(xn[idx], xc[idx] if xc is not None else None).squeeze(-1)   # (B, k)
                loss = F.binary_cross_entropy_with_logits(
                    logits.float(), yn[idx, None].expand(-1, k), pos_weight=pos_w)              # rata-rata atas k anggota
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            if self.dev.type == "cuda":
                torch.cuda.synchronize()
            t_train = time.time() - t_ep
            t_v0 = time.time()
            pv = self._predict_tensors(xn_v, xc_v)
            score = average_precision_score(y_v, pv)
            t_val = time.time() - t_v0
            print(f"      TabM epoch {epoch + 1}/{t['tabm_max_epochs']}: PR-AUC validasi={score:.4f} "
                  f"(terbaik {max(best, score):.4f}) | latih {t_train:.1f} dtk, evaluasi {t_val:.1f} dtk", flush=True)
            if score > best:
                best, bad, best_state = score, 0, copy.deepcopy(self.model.state_dict())
            else:
                bad += 1
                if bad >= t["tabm_patience"]:
                    break
        self.model.load_state_dict(best_state)
        self.best_val = best
        del opt, xn, yn, xc                      # bebaskan memori GPU sebelum prediksi/percobaan berikutnya
        if self.dev.type == "cuda":
            torch.cuda.empty_cache()
        return self

    def _predict_tensors(self, xn, xc, batch=None):
        import torch
        batch = batch or getattr(self, "eval_batch", 2048)
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(xn), batch):
                a = xn[i:i + batch].to(self.dev)
                b = xc[i:i + batch].to(self.dev) if xc is not None else None
                with torch.autocast(device_type=self.dev.type, dtype=self.amp_dtype, enabled=self.use_amp):
                    out_ = self.model(a, b).squeeze(-1)
                p = torch.sigmoid(out_.float()).mean(dim=1)                                      # rata-rata k probabilitas
                out.append(p.cpu().numpy())
        return np.concatenate(out)

    def predict_proba(self, df):
        xn, xc = self._tensors(df)
        return self._predict_tensors(xn, xc)

    def finalize_cpu(self):
        """Pindahkan model ke CPU sebelum disimpan, agar artefak dapat dimuat di mesin tanpa GPU."""
        import torch
        self.model.cpu()
        self.dev = torch.device("cpu")
        self.use_amp = False
        return self


REGISTRY = {"lr": LRModel, "rf": RFModel, "xgb": XGBModel, "lgbm": LGBMModel, "cat": CatModel, "tabm": TabMModel}


def build_model(name: str) -> BaseModel:
    return REGISTRY[name]()
