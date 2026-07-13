# USDIDR Hybrid ARIMA-LSTM Forecast

Dashboard Streamlit untuk model peramalan hibrida **ARIMA + LSTM** pada return
harian USDIDR, dikembangkan dari notebook riset `FP_Machine_learning_final`.

- **ARIMA** menangkap pola linier (tren & autokorelasi) pada `USDIDR_Returns`.
- **LSTM** dilatih pada residual ARIMA menggunakan fitur lag (1, 3, 5, 10 hari)
  dari OIL, GOLD, SP500, IHSG, VIX, CPI, BI rate, dan US rate.
- Prediksi akhir = prediksi ARIMA + prediksi residual LSTM.

## Struktur Proyek

```
.
├── app.py                  # Aplikasi dashboard Streamlit
├── train.py                # Skrip pelatihan offline (menghasilkan artefak model)
├── requirements.txt
├── src/
│   ├── preprocessing.py    # Pra-pemrosesan data (missing value, outlier, lag, dll)
│   └── model_utils.py      # Definisi & utilitas model hibrida ARIMA-LSTM
├── models/                 # Artefak model tersimpan (dibuat oleh train.py)
└── .streamlit/config.toml  # Konfigurasi tema Streamlit
```

## Menjalankan di Lokal

```bash
# 1. Buat virtual environment (opsional tapi disarankan)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Opsional) Latih model & simpan artefak agar dashboard cepat dibuka
python train.py

# 4. Jalankan aplikasi
streamlit run app.py
```

Jika folder `models/` belum berisi artefak, buka dashboard lalu pilih
**"Latih ulang dari data baru"** di sidebar, unggah CSV dataset, dan klik
**"Latih Model Sekarang"** untuk melatih langsung dari aplikasi (proses ini
memakan waktu karena melatih ARIMA + LSTM).

Dataset asli: [Indonesia Financial Time Series 2010–2026](https://www.kaggle.com/datasets/raphaelnazareth/indonesia-financial-time-series-dataset-2010-2026)
(kolom: `Date, OIL, GOLD, USDIDR, SP500, IHSG, VIX, CPI, BI_rate, US_rate`).

## Menyimpan Kode ke Repository GitHub

Jalankan perintah berikut di dalam folder proyek ini (ganti `USERNAME/REPO`
dengan nama repo GitHub Anda; buat dulu repo kosong di github.com jika belum ada):

```bash
git init
git add .
git commit -m "Initial commit: USDIDR hybrid ARIMA-LSTM Streamlit app"
git branch -M main
git remote add origin https://github.com/USERNAME/REPO.git
git push -u origin main
```

Jika menggunakan autentikasi token (disarankan, karena GitHub sudah tidak
menerima login password biasa untuk `git push`):

```bash
git remote set-url origin https://<TOKEN>@github.com/USERNAME/REPO.git
git push -u origin main
```

> Catatan: file model (`models/*.keras`, `*.pkl`, `*.joblib`) bisa berukuran
> besar. Jika ukurannya melebihi batas GitHub (100 MB per file), gunakan
> [Git LFS](https://git-lfs.com/) atau jangan commit folder `models/` dan
> biarkan Streamlit Cloud melatih ulang / unduh artefak dari sumber lain saat
> start-up.

## Deploy ke Streamlit Community Cloud

1. Push kode ke GitHub seperti di atas (repo publik atau privat).
2. Buka [share.streamlit.io](https://share.streamlit.io), login dengan akun GitHub.
3. Klik **"New app"**, pilih repo, branch `main`, dan file utama `app.py`.
4. Klik **Deploy**. Streamlit Cloud akan otomatis membaca `requirements.txt`.
5. Jika dataset diambil via `kagglehub`, tambahkan kredensial Kaggle
   (`KAGGLE_USERNAME`, `KAGGLE_KEY`) di menu **Settings → Secrets** aplikasi.

## Catatan Performa

- Melatih ARIMA + LSTM dari nol butuh waktu; disarankan menjalankan
  `train.py` sekali secara lokal/Colab, lalu commit folder `models/` ke repo
  supaya dashboard di Streamlit Cloud langsung memuat hasilnya (mode
  "Gunakan model tersimpan (cepat)").
- `tensorflow-cpu` dipakai di `requirements.txt` agar build lebih ringan di
  server cloud yang tidak punya GPU.
