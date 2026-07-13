"""
Modul pra-pemrosesan data USDIDR.
Logika di file ini mengikuti alur notebook riset (FP_Machine_learning_final):
1. Penanganan missing value (median imputation)
2. Deteksi & koreksi outlier USDIDR (return harian > 20%)
3. Sinkronisasi frekuensi data bulanan (CPI, BI_rate, US_rate) -> harian
4. Transformasi ke return (%) untuk stasioneritas
5. Rekayasa fitur lag (1, 3, 5, 10 hari)
"""

import numpy as np
import pandas as pd

MONTHLY_INDICATORS = ["CPI", "BI_rate", "US_rate"]
RETURN_COLUMNS = ["OIL", "GOLD", "USDIDR", "SP500", "IHSG", "VIX"]
LAGS = [1, 3, 5, 10]
OUTLIER_THRESHOLD = 20  # persen, batas wajar return harian USDIDR

LAG_FEATURE_COLUMNS = (
    [f"USDIDR_Returns_Lag_{l}" for l in LAGS]
    + [f"OIL_Returns_Lag_{l}" for l in LAGS]
    + [f"GOLD_Returns_Lag_{l}" for l in LAGS]
    + [f"SP500_Returns_Lag_{l}" for l in LAGS]
    + [f"IHSG_Returns_Lag_{l}" for l in LAGS]
    + [f"VIX_Returns_Lag_{l}" for l in LAGS]
    + [f"CPI_Lag_{l}" for l in LAGS]
    + [f"BI_rate_Lag_{l}" for l in LAGS]
    + [f"US_rate_Lag_{l}" for l in LAGS]
)

TARGET_COL = "USDIDR_Returns"


def load_raw_dataframe_from_kaggle() -> pd.DataFrame:
    """Unduh dataset dari Kaggle (butuh kredensial Kaggle di lingkungan deploy)."""
    import kagglehub
    import os

    path = kagglehub.dataset_download(
        "raphaelnazareth/indonesia-financial-time-series-dataset-2010-2026"
    )
    files = os.listdir(path)
    csv_file = next(f for f in files if f.lower().endswith(".csv"))
    return pd.read_csv(os.path.join(path, csv_file))


def clean_missing_and_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Isi missing value dengan median dan koreksi outlier USDIDR via interpolasi linier."""
    df1 = df.copy()
    df1["Date"] = pd.to_datetime(df1["Date"], errors="coerce")
    df1 = df1.sort_values(by="Date").reset_index(drop=True)

    for col in ["OIL", "GOLD", "SP500", "IHSG", "VIX", "USDIDR"]:
        if col in df1.columns:
            df1[col] = df1[col].fillna(df1[col].median())

    returns_check = df1["USDIDR"].pct_change() * 100
    outlier_idx = returns_check[returns_check.abs() > OUTLIER_THRESHOLD].index
    df1.loc[outlier_idx, "USDIDR"] = np.nan
    df1["USDIDR"] = df1["USDIDR"].interpolate(method="linear")

    return df1


def sync_monthly_frequency(df1: pd.DataFrame) -> pd.DataFrame:
    """Sinkronkan indikator bulanan (CPI, BI_rate, US_rate) agar konsisten per bulan."""
    df1 = df1.copy()
    for col in MONTHLY_INDICATORS:
        if col not in df1.columns:
            continue
        df1["YearMonth"] = df1["Date"].dt.to_period("M")
        df1[col] = df1.groupby("YearMonth")[col].transform("first")
    if "YearMonth" in df1.columns:
        df1 = df1.drop(columns=["YearMonth"])
    return df1


def compute_returns(df1: pd.DataFrame) -> pd.DataFrame:
    """Ubah level harga menjadi return harian (%) untuk stasioneritas."""
    df_returns = df1.copy()
    for col in RETURN_COLUMNS:
        if col in df_returns.columns:
            df_returns[f"{col}_Returns"] = df_returns[col].pct_change() * 100

    keep_cols = ["Date"] + [
        f"{col}_Returns" for col in RETURN_COLUMNS if col in df1.columns
    ]
    df_returns = df_returns[keep_cols].dropna().reset_index(drop=True)
    return df_returns


def add_lag_features(df_returns: pd.DataFrame, df1: pd.DataFrame) -> pd.DataFrame:
    """Gabungkan return dengan indikator makro dan buat fitur lag."""
    extra_cols = [c for c in MONTHLY_INDICATORS if c in df1.columns]
    df_processed = pd.merge(
        df_returns, df1[["Date"] + extra_cols], on="Date", how="left"
    )
    df_processed = df_processed.sort_values(by="Date").reset_index(drop=True)

    for col in extra_cols:
        for l in LAGS:
            df_processed[f"{col}_Lag_{l}"] = df_processed[col].shift(l)

    market_cols = [
        c
        for c in [
            "OIL_Returns",
            "GOLD_Returns",
            "SP500_Returns",
            "IHSG_Returns",
            "VIX_Returns",
        ]
        if c in df_processed.columns
    ]
    for col in market_cols:
        for l in LAGS:
            df_processed[f"{col}_Lag_{l}"] = df_processed[col].shift(l)

    for l in LAGS:
        df_processed[f"USDIDR_Returns_Lag_{l}"] = df_processed[TARGET_COL].shift(l)

    df_processed = df_processed.dropna().reset_index(drop=True)
    return df_processed


def run_full_pipeline(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Jalankan seluruh alur pra-pemrosesan dari data mentah hingga siap dimodelkan."""
    df1 = clean_missing_and_outliers(df_raw)
    df1 = sync_monthly_frequency(df1)
    df_returns = compute_returns(df1)
    df_processed = add_lag_features(df_returns, df1)
    return df_processed


def time_based_split(df_processed: pd.DataFrame, train_ratio: float = 0.8):
    train_size = int(len(df_processed) * train_ratio)
    train_df = df_processed.iloc[:train_size].reset_index(drop=True)
    test_df = df_processed.iloc[train_size:].reset_index(drop=True)
    return train_df, test_df
