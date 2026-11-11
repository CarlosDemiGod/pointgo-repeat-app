"""Aplikasi prediksi pembelian ulang pelanggan PointGo (prototipe skripsi).

Jalankan lokal:   streamlit run app/streamlit_app.py
Variabel lingkungan (opsional): PGRP_ARTIFACTS (folder artefak), PGRP_SHOW_ID_DEFAULT=1 (tampilkan ID akun asli secara bawaan).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from pgrp import inference as inf  # noqa: E402

st.set_page_config(page_title="Prediksi Pembelian Ulang PointGo", layout="wide")

ARTIFACT_DIR = Path(os.environ.get("PGRP_ARTIFACTS", ROOT / "app_artifacts"))
SHOW_ID_DEFAULT = os.environ.get("PGRP_SHOW_ID_DEFAULT", "0") == "1"
MODEL_NAMES = {"lgbm": "LightGBM (terkalibrasi)", "tabm": "TabM"}
BLUE, ORANGE = "#2F5D8A", "#C8692B"


@st.cache_resource(show_spinner="Memuat model ...")
def load_bundle(path: str):
    return inf.ArtifactBundle(path)


def pct(x, d=1):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%".replace(".", ",")


def num(x, d=3):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}".replace(".", ",")


# ------------------------------------------------------------------------------------ artefak
if not (ARTIFACT_DIR / "meta.json").exists():
    st.title("Prediksi Pembelian Ulang PointGo")
    st.error(
        f"Artefak model tidak ditemukan di `{ARTIFACT_DIR}`. Jalankan `python scripts/06_export_artifacts.py` "
        "lalu salin isi `work/artifacts/` ke folder `app_artifacts/`."
    )
    st.stop()

bundle = load_bundle(str(ARTIFACT_DIR))
cfg = bundle.cfg
meta = bundle.meta
models = bundle.available()

# --------------------------------------------------------------------------------- bilah samping
with st.sidebar:
    st.header("Data transaksi")
    st.caption("Data diproses di memori sesi dan tidak disimpan oleh aplikasi.")
    source_mode = st.radio("Sumber data", ["Unggah berkas", "Path lokal"])
    uploaded, local_path = None, ""
    if source_mode == "Unggah berkas":
        uploaded = st.file_uploader("Ekspor transaksi (CSV atau Parquet)", type=["csv", "parquet"])
    else:
        local_path = st.text_input("Path berkas di komputer ini", value="", help="Hanya berfungsi bila aplikasi dijalankan lokal.")
    load_clicked = st.button("Muat data")
    show_id = st.checkbox("Tampilkan ID akun asli", value=SHOW_ID_DEFAULT,
                          help="Bila dimatikan, akun ditampilkan sebagai kode (AKUN-n).")
    st.divider()
    st.caption(f"Model dilatih dengan {meta['n_train']:,} baris sampai titik potong {meta['train_cutoffs'][-1]} "
               f"(mode: {meta['mode']}).")

if load_clicked:
    try:
        source = uploaded if source_mode == "Unggah berkas" else (local_path.strip().strip('"') or None)
        if source is None or (source_mode == "Path lokal" and not Path(source).exists()):
            st.sidebar.error("Pilih berkas atau isi path yang valid.")
        else:
            with st.spinner("Membaca dan membersihkan data ..."):
                raw = inf.read_raw(source)
                trx, idmap, rep = inf.build_trx(raw)
            st.session_state["data"] = {"trx": trx, "idmap": idmap, "rep": rep,
                                        "nama": getattr(source, "name", str(source))}
            for k in ("pred", "bt", "explain"):
                st.session_state.pop(k, None)
            st.sidebar.success(f"Data dimuat: {rep['transaksi_bersih']:,} transaksi.")
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(f"Gagal memuat data: {e}")

data = st.session_state.get("data")

st.title("Prediksi Pembelian Ulang Pelanggan PointGo")
st.caption("Memperkirakan akun pelanggan aktif yang akan bertransaksi lagi dalam 30 hari, mulai 7 hari setelah tanggal prediksi.")
if bundle.load_errors:
    st.warning("Sebagian model tidak dapat dimuat: " + "; ".join(f"{MODEL_NAMES.get(k, k)} ({v})" for k, v in bundle.load_errors.items()))
if not models:
    st.error("Tidak ada model yang berhasil dimuat.")
    st.stop()

tab_data, tab_pred, tab_exp, tab_bt, tab_about = st.tabs(["Data", "Prediksi", "Penjelasan", "Uji pada data lampau", "Tentang model"])

# ----------------------------------------------------------------------------------------- Data
with tab_data:
    if data is None:
        st.info("Muat data transaksi dari bilah samping. Berkas harus memuat kolom: "
                + ", ".join(inf.REQUIRED) + ". Kolom opsional: " + ", ".join(inf.OPTIONAL) + ".")
        st.markdown("**Penting:** fitur seperti lama menjadi pelanggan memerlukan **seluruh riwayat** transaksi sampai tanggal "
                    "prediksi, bukan hanya periode terbaru.")
    else:
        rep, trx = data["rep"], data["trx"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transaksi bersih", f"{rep['transaksi_bersih']:,}")
        c2.metric("Akun unik", f"{rep['akun_unik']:,}")
        c3.metric("Tanggal awal", rep["tanggal_awal"][:10])
        c4.metric("Tanggal akhir", rep["tanggal_akhir"][:10])
        st.write({k: v for k, v in rep.items() if k.startswith(("baris", "duplikat"))})
        fam = trx.groupby("fam", observed=True).size().sort_values(ascending=False).head(10)
        st.subheader("Keluarga permainan terbanyak")
        st.dataframe(fam.rename("transaksi").reset_index().rename(columns={"fam": "keluarga permainan"}),
                     use_container_width=True, hide_index=True)
        monthly = trx.set_index("ts").resample("MS").size().rename("transaksi")
        st.subheader("Transaksi per bulan")
        st.bar_chart(monthly)

# ------------------------------------------------------------------------------------- Prediksi
with tab_pred:
    if data is None:
        st.info("Muat data terlebih dahulu.")
    else:
        trx = data["trx"]
        c1, c2, c3 = st.columns([2, 2, 1])
        model = c1.selectbox("Model", models, format_func=lambda m: MODEL_NAMES.get(m, m))
        t0 = c2.date_input("Tanggal prediksi (t0)", value=inf.default_predict_t0(trx).date(),
                           help="Bawaan: sehari setelah transaksi terakhir pada data.")
        run = c3.button("Hitung skor")
        if run:
            try:
                with st.spinner("Membentuk fitur dan menghitung skor ..."):
                    snap = inf.make_snapshot(trx, pd.Timestamp(t0), cfg)
                    raw_p, cal_p = bundle.predict(model, snap)
                st.session_state["pred"] = {"model": model, "t0": pd.Timestamp(t0), "snap": snap, "raw": raw_p, "cal": cal_p}
                st.session_state.pop("explain", None)
            except Exception as e:  # noqa: BLE001
                st.error(f"Gagal menghitung skor: {e}")
        pred = st.session_state.get("pred")
        if pred:
            w = cfg["window"]
            ws = pred["t0"] + pd.Timedelta(days=w["gap_days"])
            we = ws + pd.Timedelta(days=w["pred_days"])
            st.success(f"{len(pred['snap']):,} akun aktif dinilai dengan {MODEL_NAMES.get(pred['model'], pred['model'])}. "
                       f"Jendela yang diprediksi: {ws:%d %b %Y} sampai {we:%d %b %Y}.")
            tbl = pred["snap"][["cid", "recency", "freq_all", "freq_30", "monetary_30", "tenure"]].copy()
            tbl["skor"] = pred["cal"]
            tbl["_raw"] = pred["raw"]
            # Peringkat, persentil, dan kelompok memakai skor mentah yang kontinu; kalibrator isotonik berbentuk tangga
            # (banyak akun berskor sama) sehingga tidak cocok untuk membagi kelompok.
            tbl["persentil"] = pd.Series(pred["raw"]).rank(pct=True).to_numpy() * 100
            tbl["kelompok"] = inf.risk_groups(pred["raw"]).to_numpy()
            tbl = tbl.merge(data["idmap"], on="cid", how="left")
            tbl["akun"] = (tbl["user_id"].astype(str) + " / " + tbl["zone_id"].astype(str)) if show_id else ("AKUN-" + tbl["cid"].astype(str))
            view = pd.DataFrame({
                "akun": tbl["akun"], "permainan": tbl["keluarga_permainan"], "skor kembali (%)": (tbl["skor"] * 100).round(1),
                "persentil": tbl["persentil"].round(0).astype(int), "kelompok": tbl["kelompok"],
                "hari sejak transaksi terakhir": tbl["recency"].round(1), "jumlah transaksi": tbl["freq_all"].astype(int),
                "transaksi 30 hari": tbl["freq_30"].astype(int), "belanja 30 hari (Rp)": tbl["monetary_30"].round(0).astype(int),
                "_cid": tbl["cid"], "_raw": tbl["_raw"],
            })
            f1, f2 = st.columns(2)
            order = f1.selectbox("Urutan", ["Peluang kembali terendah dahulu (risiko tertinggi)", "Peluang kembali tertinggi dahulu"])
            k = f2.slider("Tampilkan persentase teratas dari urutan", 1, 100, 20)
            asc = order.startswith("Peluang kembali terendah")
            view = view.sort_values("_raw", ascending=asc, kind="mergesort").reset_index(drop=True)
            sel = view.head(max(int(len(view) * k / 100), 1))
            st.session_state["sel_cids"] = sel["_cid"].head(300).tolist()
            st.session_state["sel_labels"] = (sel["akun"] + " | " + sel["permainan"] + " | " + sel["skor kembali (%)"].astype(str) + "%").head(300).tolist()
            m1, m2, m3 = st.columns(3)
            m1.metric("Akun ditampilkan", f"{len(sel):,}")
            m2.metric("Rata-rata skor (terkalibrasi)", f"{pred['cal'].mean() * 100:.1f}%".replace(".", ","))
            m3.metric("Skor mentah rata-rata", f"{pred['raw'].mean() * 100:.1f}%".replace(".", ","))
            st.dataframe(sel.drop(columns=["_cid", "_raw"]).head(2000), use_container_width=True, hide_index=True)
            st.download_button("Unduh seluruh tampilan (CSV)", data=sel.drop(columns=["_cid", "_raw"]).to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"prediksi_{pred['t0']:%Y%m%d}_{pred['model']}.csv", mime="text/csv")
            fig, ax = plt.subplots(figsize=(7, 2.6))
            ax.hist(pred["cal"] * 100, bins=40, color=BLUE)
            ax.set_xlabel("Skor peluang kembali (%)")
            ax.set_ylabel("Jumlah akun")
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Skor terkalibrasi memakai data validasi satu periode; basis proporsi positif bergeser antarperiode, "
                       "sehingga **peringkat dan persentil (dari skor mentah) lebih dapat diandalkan daripada nilai peluangnya**.")

# ---------------------------------------------------------------------------------- Penjelasan
with tab_exp:
    pred = st.session_state.get("pred")
    if data is None or not pred:
        st.info("Hitung skor pada tab Prediksi terlebih dahulu.")
    elif "lgbm" not in models:
        st.warning("Penjelasan per akun memerlukan model LightGBM, yang tidak termuat.")
    else:
        st.caption("Penjelasan memakai SHAP pada LightGBM, juga ketika skor berasal dari TabM (SHAP TabM terlalu lambat).")
        labels, cids = st.session_state.get("sel_labels", []), st.session_state.get("sel_cids", [])
        if not labels:
            st.info("Tidak ada akun terpilih.")
        else:
            choice = st.selectbox("Pilih akun (300 teratas dari tampilan)", range(len(labels)), format_func=lambda i: labels[i])
            if st.button("Jelaskan akun ini"):
                try:
                    rows = pred["snap"][pred["snap"]["cid"] == cids[choice]]
                    with st.spinner("Menghitung SHAP ..."):
                        ex = inf.explain_rows(bundle, rows, "lgbm", top=10)[0]
                    st.session_state["explain"] = {"cid": cids[choice], "ex": ex}
                except ImportError:
                    st.error("Paket `shap` belum terpasang.")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Gagal menghitung penjelasan: {e}")
            exd = st.session_state.get("explain")
            if exd and exd["cid"] == cids[choice]:
                d = exd["ex"].iloc[::-1]
                fig, ax = plt.subplots(figsize=(7.5, 4))
                ax.barh(d["label"], d["shap"], color=[BLUE if v > 0 else ORANGE for v in d["shap"]])
                ax.axvline(0, color="gray", lw=0.8)
                ax.set_xlabel("Pengaruh terhadap skor (SHAP): biru menaikkan, oranye menurunkan peluang kembali")
                st.pyplot(fig)
                plt.close(fig)
                st.dataframe(exd["ex"][["label", "nilai", "shap"]].rename(columns={"label": "fitur", "shap": "SHAP"}),
                             use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------- Uji pada data lampau
with tab_bt:
    if data is None:
        st.info("Muat data terlebih dahulu.")
    else:
        trx = data["trx"]
        w = cfg["window"]
        st.write(f"Uji ini memakai periode **setelah** tanggal prediksi sebagai kunci jawaban: akun yang bertransaksi dalam "
                 f"{w['pred_days']} hari mulai {w['gap_days']} hari setelah t0 dianggap kembali. Data harus memuat periode tersebut.")
        c1, c2 = st.columns([2, 2])
        bt_t0 = c1.date_input("Tanggal prediksi (t0)", value=inf.latest_backtest_t0(trx, cfg).date(), key="bt_t0")
        bt_models = c2.multiselect("Model", models, default=models, format_func=lambda m: MODEL_NAMES.get(m, m))
        if st.button("Jalankan uji"):
            if not inf.labels_complete(trx, bt_t0, cfg):
                st.error("Data tidak memuat seluruh jendela prediksi untuk t0 ini. Pilih t0 yang lebih awal "
                         f"(paling lambat {inf.latest_backtest_t0(trx, cfg):%d %b %Y}).")
            elif not bt_models:
                st.error("Pilih sedikitnya satu model.")
            else:
                try:
                    with st.spinner("Membentuk fitur dan menghitung metrik ..."):
                        snap = inf.make_snapshot(trx, pd.Timestamp(bt_t0), cfg)
                        y = snap["y"].to_numpy()
                        res = {}
                        for m in bt_models:
                            raw_p, cal_p = bundle.predict(m, snap)
                            res[m] = {"metrics": inf.backtest_metrics(y, cal_p), "curve": inf.pr_curve(y, cal_p)}
                    st.session_state["bt"] = {"t0": pd.Timestamp(bt_t0), "res": res, "n": len(y)}
                except Exception as e:  # noqa: BLE001
                    st.error(f"Gagal menjalankan uji: {e}")
        bt = st.session_state.get("bt")
        if bt:
            if bundle.independent(bt["t0"]):
                st.success("Periode ini berada setelah data validasi model, sehingga hasil ini independen.")
            else:
                st.warning("Periode ini termasuk data latih atau validasi model, sehingga hasil terlalu optimistis dan tidak independen.")
            rows = []
            for m, r in bt["res"].items():
                x = r["metrics"]
                rows.append({"model": MODEL_NAMES.get(m, m), "akun aktif": x["n"], "proporsi kembali": pct(x["proporsi_positif"]),
                             "PR-AUC": num(x["pr_auc"]), "PR-AUC relatif": num(x["pr_auc_rel"], 2), "ROC-AUC": num(x["roc_auc"]),
                             "Lift@10%": num(x.get("lift@10"), 2), "Presisi@10%": pct(x.get("presisi@10")),
                             "Jangkauan@10%": pct(x.get("jangkauan@10"))})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("Lift@10%: berapa kali lebih sering akun 10% teratas menurut skor benar-benar kembali dibanding rata-rata. "
                       "Jangkauan@10%: porsi seluruh akun yang kembali yang tertangkap oleh 10% teratas.")
            fig, ax = plt.subplots(figsize=(6.5, 3.6))
            for i, (m, r) in enumerate(bt["res"].items()):
                prec, rec = r["curve"]
                ax.plot(rec, prec, label=MODEL_NAMES.get(m, m), color=[BLUE, ORANGE][i % 2], lw=1.8)
            base = next(iter(bt["res"].values()))["metrics"]["proporsi_positif"]
            ax.axhline(base, color="gray", ls="--", lw=1, label="tebakan acak")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Presisi")
            ax.legend(frameon=False)
            st.pyplot(fig)
            plt.close(fig)

# ------------------------------------------------------------------------------ Tentang model
with tab_about:
    st.subheader("Ringkasan model")
    st.write({"mode": meta["mode"], "dibuat": meta["dibuat"], "data latih sampai titik potong": meta["train_cutoffs"][-1],
              "validasi": meta["val_cutoff"], "uji": meta.get("test_cutoff"), "jumlah baris latih": meta["n_train"]})
    for m, x in meta.get("metrics", {}).items():
        if "pr_auc" in x:
            st.write(f"**{MODEL_NAMES.get(m, m)}** pada periode uji {meta.get('test_cutoff')}: PR-AUC {num(x['pr_auc'])}, "
                     f"ROC-AUC {num(x['roc_auc'])}, Brier mentah {num(x['brier_mentah'])}, terkalibrasi {num(x['brier_terkalibrasi'])}.")
    perf = ARTIFACT_DIR / "performance_summary.csv"
    if perf.exists():
        st.subheader("Kinerja pada penelitian (rata-rata sembilan fold temporal)")
        st.dataframe(pd.read_csv(perf), use_container_width=True, hide_index=True)
    st.subheader("Batasan")
    st.markdown(
        "- Pelanggan didefinisikan sebagai **akun permainan tujuan per keluarga permainan**, bukan individu pembeli.\n"
        "- Skor adalah peluang bertransaksi dalam jendela prediksi; **peringkat lebih dapat diandalkan daripada nilai peluang**, "
        "karena proporsi pelanggan yang kembali bergeser antarperiode (7,6% sampai 22,2% pada data penelitian).\n"
        "- Periode Ramadan dan Idulfitri sulit diprediksi karena data latih awal tidak memuat periode tersebut.\n"
        "- Model perlu dilatih ulang secara berkala dengan data baru.\n"
        "- Aplikasi ini prototipe penelitian dan tidak terintegrasi dengan sistem operasional PointGo."
    )
