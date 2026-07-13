"""
Dashboard Streamlit: Model Hibrida ARIMA-LSTM untuk USDIDR

Cara pakai artefak model:
- Jika folder models/ berisi artefak hasil `train.py`, dashboard akan langsung
  memuat dan menampilkan hasilnya (cepat, cocok untuk Streamlit Cloud).
- Jika artefak belum ada, unggah CSV dataset lalu klik "Latih Model Sekarang"
  untuk melatih ARIMA + LSTM langsung di aplikasi (proses ini bisa memakan
  waktu beberapa menit tergantung ukuran data & epoch).
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

sys.path.append(os.path.dirname(__file__))

from src.preprocessing import run_full_pipeline, time_based_split, TARGET_COL
from src.model_utils import (
    fit_arima,
    extract_arima_predictions,
    train_lstm_on_residuals,
    predict_hybrid,
    evaluate,
    load_artifacts,
    WINDOW_SIZE,
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
ARTIFACT_PREFIX = os.path.join(MODELS_DIR, "usdidr")

st.set_page_config(
    page_title="USDIDR Hybrid ARIMA-LSTM Forecast",
    page_icon="\U0001F4C8",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def _load_saved_artifacts():
    arima_result, order, lstm_model, scaler_X, scaler_y = load_artifacts(
        ARTIFACT_PREFIX
    )
    return arima_result, order, lstm_model, scaler_X, scaler_y


def artifacts_available() -> bool:
    required = [
        f"{ARTIFACT_PREFIX}_arima.pkl",
        f"{ARTIFACT_PREFIX}_lstm.keras",
        f"{ARTIFACT_PREFIX}_scaler_X.joblib",
        f"{ARTIFACT_PREFIX}_scaler_y.joblib",
    ]
    return all(os.path.exists(p) for p in required)


def precomputed_results_available() -> bool:
    return os.path.exists(os.path.join(MODELS_DIR, "usdidr_hasil_prediksi.csv"))


def plot_forecast(hasil_df: pd.DataFrame, r2_arima: float, r2_hybrid: float):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(hasil_df["Date"], hasil_df["Actual"], label="Aktual", color="black")
    ax.plot(
        hasil_df["Date"],
        hasil_df["ARIMA_Pred"],
        label="ARIMA",
        linestyle="--",
    )
    ax.plot(
        hasil_df["Date"],
        hasil_df["Hybrid_Pred"],
        label="Hibrida ARIMA-LSTM",
        linestyle="-.",
    )
    ax.text(
        0.02,
        0.98,
        f"ARIMA R\u00b2 : {r2_arima:.4f}\nHybrid R\u00b2 : {r2_hybrid:.4f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.85),
    )
    ax.set_title("Perbandingan Prediksi USDIDR_Returns")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Returns (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def show_metrics(metrics_arima, metrics_hybrid):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ARIMA")
        st.metric("RMSE", f"{metrics_arima['rmse']:.4f}")
        st.metric("MAE", f"{metrics_arima['mae']:.4f}")
        st.metric("R\u00b2", f"{metrics_arima['r2']:.4f}")
    with col2:
        st.markdown("#### Hibrida ARIMA-LSTM")
        st.metric("RMSE", f"{metrics_hybrid['rmse']:.4f}")
        st.metric("MAE", f"{metrics_hybrid['mae']:.4f}")
        st.metric("R\u00b2", f"{metrics_hybrid['r2']:.4f}")


def run_with_saved_artifacts():
    with st.spinner("Memuat model tersimpan..."):
        arima_result, order, lstm_model, scaler_X, scaler_y = _load_saved_artifacts()
        test_df = pd.read_csv(os.path.join(MODELS_DIR, "usdidr_test_df.csv"))
        test_df["Date"] = pd.to_datetime(test_df["Date"])

        from src.preprocessing import LAG_FEATURE_COLUMNS
        from src.model_utils import create_sequences

        X_test_scaled = scaler_X.transform(test_df[LAG_FEATURE_COLUMNS])
        y_test_scaled = scaler_y.transform(test_df[["Residual"]])
        X_test_seq, _ = create_sequences(X_test_scaled, y_test_scaled, WINDOW_SIZE)

        hasil_df = predict_hybrid(lstm_model, scaler_y, X_test_seq, test_df)

    st.success(f"Model dimuat. Parameter ARIMA: {order}")

    metrics_arima = evaluate(hasil_df["Actual"], hasil_df["ARIMA_Pred"])
    metrics_hybrid = evaluate(hasil_df["Actual"], hasil_df["Hybrid_Pred"])

    st.subheader("Perbandingan Prediksi vs Aktual")
    fig = plot_forecast(hasil_df, metrics_arima["r2"], metrics_hybrid["r2"])
    st.pyplot(fig)

    st.subheader("Metrik Evaluasi")
    show_metrics(metrics_arima, metrics_hybrid)

    with st.expander("Lihat tabel hasil prediksi"):
        st.dataframe(hasil_df, use_container_width=True)


def run_live_training(df_raw: pd.DataFrame, epochs: int, patience: int):
    progress = st.progress(0, text="Menjalankan pra-pemrosesan data...")

    df_processed = run_full_pipeline(df_raw)
    train_df, test_df = time_based_split(df_processed, train_ratio=0.8)
    progress.progress(20, text="Melatih model ARIMA...")

    arima_result, order = fit_arima(train_df[TARGET_COL])
    train_df, test_df = extract_arima_predictions(arima_result, train_df, test_df)
    progress.progress(45, text="Melatih LSTM pada residual ARIMA (mungkin beberapa menit)...")

    lstm_model, scaler_X, scaler_y, X_test_seq, history = train_lstm_on_residuals(
        train_df, test_df, epochs=epochs, patience=patience
    )
    progress.progress(85, text="Menghitung prediksi & evaluasi...")

    hasil_df = predict_hybrid(lstm_model, scaler_y, X_test_seq, test_df)
    metrics_arima = evaluate(hasil_df["Actual"], hasil_df["ARIMA_Pred"])
    metrics_hybrid = evaluate(hasil_df["Actual"], hasil_df["Hybrid_Pred"])
    progress.progress(100, text="Selesai.")
    progress.empty()

    st.success(f"Pelatihan selesai. Parameter ARIMA terbaik: {order}")

    st.subheader("Proses Pelatihan LSTM")
    fig_loss, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(history.history["loss"], label="Loss Latih")
    ax.plot(history.history["val_loss"], label="Loss Validasi")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig_loss)

    st.subheader("Perbandingan Prediksi vs Aktual")
    fig = plot_forecast(hasil_df, metrics_arima["r2"], metrics_hybrid["r2"])
    st.pyplot(fig)

    st.subheader("Metrik Evaluasi")
    show_metrics(metrics_arima, metrics_hybrid)

    with st.expander("Lihat tabel hasil prediksi"):
        st.dataframe(hasil_df, use_container_width=True)

    if st.button("\U0001F4BE Simpan model ini sebagai artefak default"):
        os.makedirs(MODELS_DIR, exist_ok=True)
        from src.model_utils import save_artifacts

        save_artifacts(
            ARTIFACT_PREFIX, arima_result, order, lstm_model, scaler_X, scaler_y
        )
        test_df.to_csv(
            os.path.join(MODELS_DIR, "usdidr_test_df.csv"), index=False
        )
        st.cache_resource.clear()
        st.success("Artefak disimpan ke folder models/. Muat ulang halaman untuk memakainya.")


def main():
    st.title("\U0001F4C8 Prediksi USDIDR: Model Hibrida ARIMA-LSTM")
    st.markdown(
        """
Dashboard ini menampilkan model peramalan **return harian USDIDR** dengan
pendekatan hibrida: **ARIMA** menangkap pola linier, lalu **LSTM** dilatih
pada residual ARIMA untuk menangkap pola non-linier menggunakan fitur lag
dari OIL, GOLD, SP500, IHSG, VIX, CPI, BI rate, dan US rate.
"""
    )

    st.sidebar.header("Pengaturan")
    mode = st.sidebar.radio(
        "Sumber hasil",
        ["Gunakan model tersimpan (cepat)", "Latih ulang dari data baru"],
        index=0 if artifacts_available() else 1,
    )

    if mode == "Gunakan model tersimpan (cepat)":
        if artifacts_available():
            run_with_saved_artifacts()
        else:
            st.warning(
                "Belum ada artefak model tersimpan di folder `models/`. "
                "Jalankan `python train.py` secara lokal terlebih dahulu, "
                "commit hasilnya ke repo, atau pilih 'Latih ulang dari data baru' "
                "di sidebar untuk melatih langsung di aplikasi ini."
            )
    else:
        st.markdown(
            "Unggah CSV dengan kolom `Date, OIL, GOLD, USDIDR, SP500, IHSG, VIX, "
            "CPI, BI_rate, US_rate` (format yang sama dengan dataset "
            "*Indonesia Financial Time Series 2010-2026*)."
        )
        uploaded = st.file_uploader("Unggah dataset CSV", type=["csv"])
        epochs = st.sidebar.slider("Epoch LSTM", 10, 300, 100, step=10)
        patience = st.sidebar.slider("EarlyStopping patience", 5, 50, 20, step=5)

        if uploaded is not None:
            df_raw = pd.read_csv(uploaded)
            st.write("Pratinjau data:", df_raw.head())
            if st.button("\U0001F680 Latih Model Sekarang"):
                run_live_training(df_raw, epochs=epochs, patience=patience)
        else:
            st.info("Unggah file CSV untuk memulai pelatihan.")

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Dibangun dari notebook riset ARIMA-LSTM hibrida untuk USDIDR. "
        "RMSE/MAE/R\u00b2 dihitung pada skala return harian (%)."
    )


if __name__ == "__main__":
    main()
