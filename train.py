"""
Skrip pelatihan offline untuk model hibrida ARIMA-LSTM USDIDR.

Jalankan skrip ini sekali di lokal/Colab untuk menghasilkan artefak model
(models/usdidr_arima.pkl, models/usdidr_lstm.keras, models/usdidr_scaler_X.joblib,
models/usdidr_scaler_y.joblib). Artefak ini kemudian dimuat langsung oleh app.py
di Streamlit Cloud, sehingga aplikasi tidak perlu melatih ulang model setiap kali
dibuka (proses training ARIMA+LSTM cukup berat untuk dijalankan on-the-fly).

Cara pakai:
    python train.py
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))

import numpy as np
import tensorflow as tf

# Mengunci random seed agar hasil training (termasuk R², RMSE, MAE) konsisten
# setiap kali skrip ini dijalankan ulang, selama data input juga tidak berubah.
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

from src.preprocessing import (
    load_raw_dataframe_from_kaggle,
    run_full_pipeline,
    time_based_split,
)
from src.model_utils import (
    fit_arima,
    extract_arima_predictions,
    train_lstm_on_residuals,
    predict_hybrid,
    evaluate,
    save_artifacts,
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
ARTIFACT_PREFIX = os.path.join(MODELS_DIR, "usdidr")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("1/5 - Mengunduh & memuat data mentah dari Kaggle...")
    df_raw = load_raw_dataframe_from_kaggle()

    print("2/5 - Menjalankan pipeline pra-pemrosesan...")
    df_processed = run_full_pipeline(df_raw)
    train_df, test_df = time_based_split(df_processed, train_ratio=0.8)
    print(f"   Data latih: {len(train_df)} baris | Data uji: {len(test_df)} baris")

    print("3/5 - Melatih model ARIMA...")
    arima_result, order = fit_arima(train_df["USDIDR_Returns"])
    print(f"   Parameter ARIMA terbaik: {order}")
    train_df, test_df = extract_arima_predictions(arima_result, train_df, test_df)

    print("4/5 - Melatih LSTM pada residual ARIMA...")
    lstm_model, scaler_X, scaler_y, X_test_seq, history = train_lstm_on_residuals(
        train_df, test_df, epochs=150, patience=20
    )

    print("5/5 - Evaluasi & menyimpan artefak model...")
    hasil_df = predict_hybrid(lstm_model, scaler_y, X_test_seq, test_df)
    metrics_arima = evaluate(hasil_df["Actual"], hasil_df["ARIMA_Pred"])
    metrics_hybrid = evaluate(hasil_df["Actual"], hasil_df["Hybrid_Pred"])

    print(f"   ARIMA              -> {metrics_arima}")
    print(f"   Hybrid ARIMA-LSTM  -> {metrics_hybrid}")

    save_artifacts(ARTIFACT_PREFIX, arima_result, order, lstm_model, scaler_X, scaler_y)

    # Simpan juga data hasil prediksi & test_df untuk ditampilkan langsung di dashboard
    hasil_df.to_csv(os.path.join(MODELS_DIR, "usdidr_hasil_prediksi.csv"), index=False)
    test_df.to_csv(os.path.join(MODELS_DIR, "usdidr_test_df.csv"), index=False)
    train_df.to_csv(os.path.join(MODELS_DIR, "usdidr_train_df.csv"), index=False)

    print(f"\nSelesai. Artefak disimpan di: {MODELS_DIR}")


if __name__ == "__main__":
    main()
