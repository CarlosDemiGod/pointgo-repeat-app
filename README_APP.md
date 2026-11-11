# Aplikasi Prediksi Pembelian Ulang PointGo (prototipe)

Aplikasi Streamlit untuk memberi skor peluang pembelian ulang tiap akun pelanggan aktif dan menguji model pada data lampau.
Model: **LightGBM** (terkalibrasi) dan **TabM**, dipilih di aplikasi. Penjelasan per akun (SHAP) memakai LightGBM.

## 1. Menyiapkan artefak model (sekali, di komputer yang memiliki data penelitian)
```
conda activate gpu
cd /d "D:\path\ke\pointgo-repeat"
python scripts\06_export_artifacts.py --mode fold9 --models lgbm,tabm
xcopy work\artifacts app_artifacts /E /Y
```
- `--mode fold9`: konfigurasi fold 9 penelitian. Periode Juni dan Juli 2026 menjadi uji data lampau yang independen.
- `--mode latest`: latih pada seluruh titik potong berlabel, untuk dipakai pada data baru (tidak ada periode uji independen).
- Hasilnya `lgbm.joblib`, `tabm.joblib` (model dipindah ke CPU), `meta.json`, dan `performance_summary.csv`.

## 2. Menjalankan lokal (disarankan untuk data asli)
```
pip install -r app\requirements.txt        :: hati-hati dengan torch; lewati bila sudah terpasang
streamlit run app\streamlit_app.py
```
Di bilah samping pilih **Path lokal**, isi path ekspor transaksi (CSV atau Parquet), lalu **Muat data**.
Berkas harus memuat **seluruh riwayat** transaksi sampai tanggal prediksi (fitur seperti lama menjadi pelanggan membutuhkannya).
Kolom wajib: `date_create, user_id, zone_id, games, product, product_category, price, method, payment_group`.
Kolom opsional: `order_id, game_genre, raw_price, ip_user`. Kolom WhatsApp, surel, dan nama pengguna tidak dibaca.
Berkas 2,4 juta baris memerlukan beberapa menit dan beberapa GB memori.

## 3. Memakai aplikasi
1. **Data**: ringkasan transaksi, akun, dan periode.
2. **Prediksi**: pilih model dan tanggal prediksi (bawaan: sehari setelah transaksi terakhir). Tabel berisi skor, persentil, dan kelompok risiko; dapat diurutkan, disaring ke K% teratas, dan diunduh.
3. **Penjelasan**: pilih satu akun untuk melihat fitur yang mendorong skornya.
4. **Uji pada data lampau**: bila data memuat periode setelah tanggal prediksi, aplikasi menghitung PR-AUC, ROC-AUC, lift, dan jangkauan memakai kejadian sebenarnya sebagai kunci jawaban. Aplikasi memberi peringatan bila periode itu termasuk data latih atau validasi model.
5. **Tentang model**: informasi model dan batasan.

Catatan membaca skor: persentil dan kelompok memakai **skor mentah** (kontinu). Peluang terkalibrasi hanya nilai pendamping karena kalibrator dilatih pada satu periode validasi dan proporsi pelanggan yang kembali bergeser antarperiode.

## 4. Menjalankan online (GitHub dan Streamlit Community Cloud)
**Pertimbangkan privasi dulu.** Data yang diunggah berpindah ke server pihak ketiga. Pastikan PointGo mengizinkan, dan untuk data asli
gunakan berkas teranonimisasi (bagian 4a). Untuk data asli lengkap, jalankan lokal.

### 4a. Berkas masukan teranonimisasi
```
set PGRP_HMAC_KEY=kunci-rahasia-anda
python scripts\make_upload_file.py --csv "D:\path\ke\ekspor_transaksi.csv" --out work\upload_anonim.parquet --map-out work\peta_id.csv --sample-users 0.1
```
- `user_id`, `zone_id`, dan `ip_user` diganti kode HMAC; WhatsApp, surel, dan nama pengguna tidak ikut.
- `--sample-users 0.1` mengambil 10% akun dengan riwayat lengkap agar muat di layanan gratis (memori terbatas). Fitur jumlah akun per IP menjadi kurang akurat pada sampel, tetapi kontribusinya pada model hampir nol.
- `peta_id.csv` (kode ke ID asli) **hanya disimpan lokal**. Jangan diunggah atau dibagikan.

### 4b. Mengunggah ke GitHub
1. Buat repositori **privat** di GitHub.
2. Pastikan `.gitignore` bawaan terpasang. Berkas data (`*.csv`, `*.parquet`) dan folder `work/` tidak boleh ikut. Jangan pernah mengunggah kunci HMAC.
3. Dorong folder berikut: `src/`, `app/`, `app_artifacts/`, `.streamlit/`, `.gitignore`, `README_APP.md`.
   ```
   git init
   git add src app app_artifacts .streamlit .gitignore README_APP.md
   git commit -m "Aplikasi prediksi pembelian ulang"
   git branch -M main
   git remote add origin https://github.com/NAMA/REPO.git
   git push -u origin main
   ```
### 4c. Menerbitkan
1. Buka https://share.streamlit.io, masuk dengan GitHub, pilih **Create app**.
2. Repositori: yang tadi; cabang `main`; berkas utama `app/streamlit_app.py`; Python 3.11.
3. Setelah terbit, aplikasi mengambil dependensi dari `app/requirements.txt`. Bila pemasangan gagal karena TabM (torch), hapus empat baris terakhir berkas itu. Aplikasi tetap berjalan dengan LightGBM saja.
4. Aplikasi gratis tertidur saat tidak dipakai dan memori terbatas. Karena itu gunakan berkas sampel (bagian 4a).
5. Untuk aplikasi privat, atur akses di pengaturan aplikasi (Share).

## 5. Memperbarui model dengan data baru
Ulangi tahap 0 dan 1 pipeline dengan data baru, lalu `python scripts\06_export_artifacts.py --mode latest --models lgbm,tabm`
dan salin artefak ke `app_artifacts/`. Latih ulang secara berkala karena periode puncak dan proporsi pembelian ulang bergeser.

## 6. Batasan
- Pelanggan adalah akun permainan tujuan per keluarga permainan, bukan individu pembeli.
- Peringkat lebih dapat diandalkan daripada nilai peluang.
- Periode Ramadan dan Idulfitri lebih sulit diprediksi.
- Prototipe penelitian, tidak terintegrasi dengan sistem operasional PointGo.

## 7. Pengujian
```
python tests\test_pipeline.py
python tests\test_inference.py
```
