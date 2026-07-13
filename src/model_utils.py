"""
Modul model hibrida ARIMA-LSTM untuk USDIDR_Returns.

Alur:
1. ARIMA menangkap pola linier pada USDIDR_Returns.
2. Residual ARIMA (aktual - prediksi ARIMA) dimodelkan oleh LSTM
   menggunakan fitur lag ekonomi & pasar.
3. Prediksi akhir = prediksi ARIMA + prediksi residual LSTM.
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .preprocessing import LAG_FEATURE_COLUMNS, TARGET_COL

WINDOW_SIZE = 20
RESIDUAL_CLIP = 2.0


def fit_arima(train_series: pd.Series):
    import pmdarima as pm
    from statsmodels.tsa.arima.model import ARIMA

    auto_model = pm.auto_arima(
        train_series, seasonal=False, stepwise=True, suppress_warnings=True
    )
    order = auto_model.order
    arima_model = ARIMA(train_series, order=order)
    arima_result = arima_model.fit()
    return arima_result, order


def extract_arima_predictions(arima_result, train_df, test_df):
    arima_pred_train = arima_result.predict(start=0, end=len(train_df) - 1)
    residual_train = train_df[TARGET_COL].values - arima_pred_train.values

    arima_pred_test = arima_result.forecast(steps=len(test_df))
    residual_test = test_df[TARGET_COL].values - arima_pred_test.values

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["ARIMA_Pred"] = arima_pred_train.values
    train_df["Residual"] = residual_train
    test_df["ARIMA_Pred"] = arima_pred_test.values
    test_df["Residual"] = residual_test
    return train_df, test_df


def create_sequences(X: np.ndarray, y: np.ndarray, window_size: int = WINDOW_SIZE):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i : i + window_size])
        ys.append(y[i + window_size])
    return np.array(Xs), np.array(ys)


def build_lstm(input_shape):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential(
        [
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def train_lstm_on_residuals(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 20,
):
    from tensorflow.keras.callbacks import EarlyStopping

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(train_df[LAG_FEATURE_COLUMNS])
    y_train_scaled = scaler_y.fit_transform(train_df[["Residual"]])
    X_test_scaled = scaler_X.transform(test_df[LAG_FEATURE_COLUMNS])
    y_test_scaled = scaler_y.transform(test_df[["Residual"]])

    X_train_seq, y_train_seq = create_sequences(
        X_train_scaled, y_train_scaled, window_size
    )
    X_test_seq, y_test_seq = create_sequences(
        X_test_scaled, y_test_scaled, window_size
    )

    model = build_lstm((X_train_seq.shape[1], X_train_seq.shape[2]))
    early_stop = EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    history = model.fit(
        X_train_seq,
        y_train_seq,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0,
    )

    return model, scaler_X, scaler_y, X_test_seq, history


def predict_hybrid(model, scaler_y, X_test_seq, test_df, window_size=WINDOW_SIZE):
    residual_pred_scaled = model.predict(X_test_seq, verbose=0)
    residual_pred = scaler_y.inverse_transform(residual_pred_scaled).flatten()
    residual_pred = np.clip(residual_pred, -RESIDUAL_CLIP, RESIDUAL_CLIP)

    arima_pred_aligned = test_df["ARIMA_Pred"].values[window_size:]
    actual_aligned = test_df[TARGET_COL].values[window_size:]
    date_aligned = test_df["Date"].values[window_size:]

    hybrid_pred = arima_pred_aligned + residual_pred

    hasil_df = pd.DataFrame(
        {
            "Date": date_aligned,
            "Actual": actual_aligned,
            "ARIMA_Pred": arima_pred_aligned,
            "Hybrid_Pred": hybrid_pred,
        }
    )
    return hasil_df


def evaluate(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def save_artifacts(path_prefix, arima_result, order, lstm_model, scaler_X, scaler_y):
    import pickle

    with open(f"{path_prefix}_arima.pkl", "wb") as f:
        pickle.dump({"result": arima_result, "order": order}, f)
    lstm_model.save(f"{path_prefix}_lstm.keras")
    joblib.dump(scaler_X, f"{path_prefix}_scaler_X.joblib")
    joblib.dump(scaler_y, f"{path_prefix}_scaler_y.joblib")


def load_artifacts(path_prefix):
    import pickle
    from tensorflow.keras.models import load_model

    with open(f"{path_prefix}_arima.pkl", "rb") as f:
        arima_data = pickle.load(f)
    lstm_model = load_model(f"{path_prefix}_lstm.keras")
    scaler_X = joblib.load(f"{path_prefix}_scaler_X.joblib")
    scaler_y = joblib.load(f"{path_prefix}_scaler_y.joblib")
    return arima_data["result"], arima_data["order"], lstm_model, scaler_X, scaler_y
