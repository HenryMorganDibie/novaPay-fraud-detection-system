# src/features.py
import pandas as pd
from sklearn.impute import SimpleImputer

# ----------------------
# Features expected by the model
# ----------------------
NUMERIC_FEATURES = [
    "amount_src", "amount_usd", "fee", "account_age_days",
    "txn_hour", "txn_day_of_week", "txn_day_of_month",
    "txn_velocity_1h", "txn_velocity_24h",
    "mean_amount_prev_3d", "txn_count_prev_3d",
    "exchange_rate_src_to_dest", "risk_score_internal",
    "device_trust_score", "ip_risk_score",
    "chargeback_history_count", "velocity_ratio"
]

CATEGORICAL_FEATURES = [
    "channel", "kyc_tier", "device_type",
    "home_country", "source_currency", "dest_currency",
    "ip_country", "ip_missing", "device_trust_missing",
    "new_device", "is_weekend", "is_high_amount_ratio",
    "kyc_missing", "corridor_risk", "location_mismatch",
    "ip_address"
]

# ----------------------
# Preprocessing function
# ----------------------
def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess input DataFrame for CatBoost."""
    
    df = df.copy()
    
    # ----------------------
    # Datetime features
    # ----------------------
    if "txn_timestamp" in df.columns:
        dt = pd.to_datetime(df["txn_timestamp"], errors='coerce')
        df["txn_hour"] = dt.dt.hour.fillna(0).astype(int)
        df["txn_day_of_week"] = dt.dt.dayofweek.fillna(0).astype(int)
        df["txn_day_of_month"] = dt.dt.day.fillna(1).astype(int)
        df.drop(columns=["txn_timestamp"], inplace=True)
    else:
        df["txn_hour"] = 0
        df["txn_day_of_week"] = 0
        df["txn_day_of_month"] = 1
    
    df["is_weekend"] = (df["txn_day_of_week"].isin([5,6])).astype(int)
    
    # ----------------------
    # Add missing numeric columns with 0
    # ----------------------
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        # Force numeric type and replace non-convertible values
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    # ----------------------
    # Add missing categorical columns with "unknown"
    # ----------------------
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = "unknown"
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype(str)
    
    # ----------------------
    # Impute numeric features
    # ----------------------
    imputer = SimpleImputer(strategy="median")
    df[NUMERIC_FEATURES] = imputer.fit_transform(df[NUMERIC_FEATURES])
    
    return df
