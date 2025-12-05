import pandas as pd
from sklearn.impute import SimpleImputer
from datetime import datetime, timezone

# Global median must be passed from the app that loads the history
GLOBAL_MEDIAN_AMOUNT = 100.0 # Placeholder/safe default value if not provided

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
# Core Preprocessing 
# ----------------------
def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess input DataFrame for CatBoost, handling types and missing values."""
    
    df = df.copy()
    
    # Handle timestamp features if present (using 'timestamp' column name for consistency)
    if "timestamp" in df.columns:
        dt = pd.to_datetime(df["timestamp"], errors='coerce')
        df["txn_hour"] = dt.dt.hour.fillna(0).astype(int)
        df["txn_day_of_week"] = dt.dt.dayofweek.fillna(0).astype(int)
        df["txn_day_of_month"] = dt.dt.day.fillna(1).astype(int)
        df.drop(columns=["timestamp"], errors='ignore', inplace=True) 
    else:
        df["txn_hour"] = 0
        df["txn_day_of_week"] = 0
        df["txn_day_of_month"] = 1
    
    df["is_weekend"] = (df["txn_day_of_week"].isin([5, 6])).astype(int)
    
    # Impute and clean numeric features
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0.0 # Add missing numeric features with 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    # Impute and clean categorical features
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = "unknown" # Add missing categorical features with "unknown"
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype(str)
    
    return df

# ----------------------
# Real-Time Feature Generation 
# ----------------------
def create_realtime_features(data_point: dict, history_df: pd.DataFrame, model_feature_names: list, global_median: float) -> pd.DataFrame:
    """
    Generates all features for a single transaction, including the real-time 
    look-up for behavioral velocity features.
    """
    df_new = pd.DataFrame([data_point])
    df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], utc=True)
    
    current_timestamp = df_new['timestamp'].iloc[0] 
    current_amount_usd = df_new['amount_usd'].iloc[0]
    customer_id = df_new['customer_id'].iloc[0]
    
    # --- 1. Behavioral/Velocity Features (Accurate Look-back) ---
    customer_history = history_df[history_df['customer_id'] == customer_id]
    prior_txns = customer_history[customer_history.index < current_timestamp]

    if not prior_txns.empty:
        cutoff_date = current_timestamp - pd.Timedelta(days=3)
        recent_txns = prior_txns[prior_txns.index >= cutoff_date]
        
        txn_count = len(recent_txns)
        mean_amount = recent_txns['amount_usd'].mean()
        
        if pd.isna(mean_amount) or mean_amount == 0:
            mean_amount = prior_txns['amount_usd'].median()
    else:
        txn_count = 0
        mean_amount = global_median 
    
    mean_amount = mean_amount if not pd.isna(mean_amount) else global_median
        
    df_new['txn_count_prev_3d'] = txn_count
    df_new['mean_amount_prev_3d'] = mean_amount
    
    # Calculate velocity ratio and high-amount ratio
    df_new['velocity_ratio'] = data_point.get('txn_velocity_1h', 0) / (data_point.get('txn_velocity_24h', 0) + 1)
    df_new['is_high_amount_ratio'] = int((current_amount_usd / (mean_amount + 1e-6)) > 2.0)
    
    # --- 2. Missingness flags ---
    df_new['ip_missing'] = int(pd.isna(data_point.get('ip_address')))
    df_new['kyc_missing'] = int(data_point.get('kyc_tier') == 'Unknown')
    df_new['device_trust_missing'] = int(pd.isna(data_point.get('device_trust_score')))
    
    # --- 3. Run General Preprocessing ---
    df_processed = preprocess_features(df_new)
    
    # --- 4. Final Feature Alignment ---
    df_processed.drop(columns=['customer_id'], errors='ignore', inplace=True)
    
    final_df = pd.DataFrame(index=[0])
    for col in model_feature_names:
        if col in df_processed.columns:
            final_df[col] = df_processed[col].iloc[0]
        else:
            if col in NUMERIC_FEATURES:
                final_df[col] = 0.0
            else:
                final_df[col] = 'unknown'

    return final_df[model_feature_names]