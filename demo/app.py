import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timezone

# --- Load Assets ---
# Load the trained CatBoost model
MODEL_PATH = 'models/CatBoost (Tuned)_fraud_model.joblib'
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model file not found. Ensure '{MODEL_PATH}' exists.")
    st.stop()
    
# Load the ORIGINAL CSV dataset (nova_pay_transactions.csv) to act as 'history' 
# for accurate behavioral feature calculation, as the pickle file was missing 'timestamp'.
try:
    history_df = pd.read_csv('data/nova_pay_transactions.csv')
    
    # Perform initial required cleaning/typing on the history data
    history_df['timestamp'] = pd.to_datetime(
        history_df['timestamp'], 
        utc=True,
        errors='coerce',
        format='mixed' 
    )
    
    history_df['amount_usd'] = pd.to_numeric(history_df['amount_usd'], errors='coerce')

    # Select only the columns needed for rolling calculations
    # Added 'timestamp' to the subset to drop rows where date parsing failed (NaT)
    history_df = history_df[['customer_id', 'timestamp', 'amount_usd']].dropna(subset=['amount_usd', 'timestamp'])
    
    # Set the timestamp as the index and sort for rolling calculation efficiency
    history_df = history_df.set_index('timestamp').sort_values(by=['customer_id', 'timestamp'])
    st.sidebar.success("Historical data loaded successfully.")

except FileNotFoundError:
    st.error("Historical data file (data/nova_pay_transactions.csv) not found. Cannot calculate behavioral features.")
    st.stop()
    
# Calculate a global median to use as a fallback for new customers
GLOBAL_MEDIAN_AMOUNT = history_df['amount_usd'].median()


# --- Feature Engineering Helper ---
def create_demo_features(data_point, history_df):
    """Generates all necessary features for the single input transaction."""
    df_new = pd.DataFrame([data_point])
    
    # Ensure timestamp is a proper datetime object
    df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], utc=True)
    
    # 1. Time Features
    df_new['txn_hour'] = df_new['timestamp'].dt.hour
    df_new['txn_day_of_week'] = df_new['timestamp'].dt.dayofweek 
    df_new['is_weekend'] = df_new['txn_day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df_new['txn_day_of_month'] = df_new['timestamp'].dt.day
    
    current_timestamp = df_new['timestamp'].iloc[0] 
    current_amount_usd = df_new['amount_usd'].iloc[0]
    customer_id = df_new['customer_id'].iloc[0]
    
    # 2. Behavioral/Velocity Features (Accurate Look-back)
    
    customer_history = history_df[history_df['customer_id'] == customer_id]
    
    # Filter history to only transactions *before* the current one
    prior_txns = customer_history[customer_history.index < current_timestamp]

    if not prior_txns.empty:
        # Filter to the 3-day window before the current transaction
        cutoff_date = current_timestamp - pd.Timedelta(days=3)
        recent_txns = prior_txns[prior_txns.index >= cutoff_date]
        
        # Count and Mean Amount
        txn_count = len(recent_txns)
        mean_amount = recent_txns['amount_usd'].mean()
        
        # Handle the case where the customer exists but has no transactions in the 3-day window
        if pd.isna(mean_amount) or mean_amount == 0:
            # Fallback to customer's overall median or global median if no prior txns
            mean_amount = prior_txns['amount_usd'].median() if not prior_txns.empty else GLOBAL_MEDIAN_AMOUNT
            
    else:
        # Truly New customer (no history) fallback
        txn_count = 0
        mean_amount = GLOBAL_MEDIAN_AMOUNT 
        
    df_new['txn_count_prev_3d'] = txn_count
    df_new['mean_amount_prev_3d'] = mean_amount
    
    # Calculate ratio *after* mean_amount is determined
    # Use a small epsilon (+1e-6) to prevent division by zero
    df_new['is_high_amount_ratio'] = int((current_amount_usd / (mean_amount + 1e-6)) > 2.0)
    
    # Other derived features (using known values from input for simplicity)
    # Ensure no division by zero for velocity ratio
    df_new['velocity_ratio'] = data_point['txn_velocity_1h'] / (data_point['txn_velocity_24h'] + 1)
    
    # 3. Missingness flags (based on input values)
    df_new['ip_missing'] = int(pd.isna(data_point.get('ip_address')))
    df_new['kyc_missing'] = int(data_point.get('kyc_tier') == 'Unknown')
    df_new['device_trust_missing'] = int(pd.isna(data_point.get('device_trust_score')))
    
    # --- OPTIMIZED DROP ---
    # Drop columns not used by the model before final selection
    df_new.drop(columns=['timestamp', 'customer_id'], errors='ignore', inplace=True) 
    
    # Reorder/select columns to match the trained CatBoost feature order
    final_features = model.feature_names_
    
    # Fill any missing columns with safe defaults for the model prediction
    for col in final_features:
        if col not in df_new.columns:
            if col in ['home_country', 'source_currency', 'dest_currency', 'channel', 'ip_country', 'kyc_tier', 'new_device', 'location_mismatch']:
                 df_new[col] = 'Unknown'
            else:
                 df_new[col] = 0.0

    return df_new[final_features]


# --- Streamlit App Layout ---

st.title("💸 NovaPay Fraud Detection Prototype")
st.markdown("Enter transaction details below to get a real-time fraud risk score from the CatBoost model.")

# Define input fields (matching the important features from the model)
st.sidebar.header("Transaction Details")

# Transaction IDs and Time (needed for feature calculation, but not model input)
customer_id = st.sidebar.text_input("Customer ID", value='402cccc9-28de-45b3-9af7-cc5302aa1f93')
timestamp_str = st.sidebar.text_input("Timestamp (UTC)", value=datetime.now(timezone.utc).isoformat())

# Financial Details
amount_usd = st.sidebar.number_input("Amount (USD)", value=100.0, min_value=0.0)
fee = st.sidebar.number_input("Fee", value=2.0)
exchange_rate_src_to_dest = st.sidebar.number_input("Exchange Rate (Src to Dest)", value=1.0)
amount_src = st.sidebar.number_input("Amount (Source Currency)", value=100.0)
source_currency = st.sidebar.selectbox("Source Currency", ['USD', 'CAD', 'MXN', 'EUR'])
dest_currency = st.sidebar.selectbox("Destination Currency", ['USD', 'CAD', 'MXN', 'EUR', 'CNY'])

# Behavioral/Device Risk Scores
ip_risk_score = st.sidebar.number_input("IP Risk Score", value=0.35, min_value=0.0, max_value=1.0)
risk_score_internal = st.sidebar.number_input("Internal Risk Score", value=0.24, min_value=0.0, max_value=1.0)
device_trust_score = st.sidebar.number_input("Device Trust Score", value=0.68, min_value=0.0, max_value=1.0)

# Velocity (Input as given, used to derive velocity_ratio)
txn_velocity_1h = st.sidebar.number_input("Txn Velocity (1hr)", value=0)
txn_velocity_24h = st.sidebar.number_input("Txn Velocity (24hr)", value=0)

# Categorical Features
channel = st.sidebar.selectbox("Channel", ['mobile', 'web', 'ATM'])
kyc_tier = st.sidebar.selectbox("KYC Tier", ['standard', 'enhanced', 'basic', 'Unknown'])
home_country = st.sidebar.selectbox("Home Country", ['US', 'CA', 'MX'])


if st.sidebar.button("Predict Fraud Risk"):
    
    # 1. Create Raw Input Dictionary
    raw_input = {
        'customer_id': customer_id,
        'timestamp': datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')), # Convert to datetime object
        'home_country': home_country,
        'source_currency': source_currency,
        'dest_currency': dest_currency,
        'channel': channel,
        'amount_src': amount_src,
        'amount_usd': amount_usd,
        'fee': fee,
        'exchange_rate_src_to_dest': exchange_rate_src_to_dest,
        'new_device': False, 
        'ip_address': '1.1.1.1', 
        'ip_country': 'US', 
        'location_mismatch': False, 
        'ip_risk_score': ip_risk_score,
        'kyc_tier': kyc_tier,
        'account_age_days': 500, 
        'device_trust_score': device_trust_score,
        'chargeback_history_count': 0,
        'risk_score_internal': risk_score_internal,
        'txn_velocity_1h': txn_velocity_1h,
        'txn_velocity_24h': txn_velocity_24h,
        'corridor_risk': 0.05, 
    }
    
    # 2. Engineer Features
    with st.spinner('Engineering features...'):
        X_predict = create_demo_features(raw_input, history_df)
    
    # 3. Predict
    with st.spinner('Running CatBoost model prediction...'):
        # CatBoost returns probability of class 1 (fraud) by default
        fraud_proba = model.predict_proba(X_predict)[0][1]
        
    # 4. Display Results
    st.header("Prediction Result")
    
    # Use the optimal threshold found during tuning (0.6705)
    optimal_threshold = 0.6705 
    
    if fraud_proba >= optimal_threshold:
        st.error(f"⚠️ **HIGH RISK OF FRAUD** - BLOCK TRANSACTION")
        st.metric(label="Fraud Probability", value=f"{fraud_proba*100:.2f}%")
        st.markdown(f"*(Score $\ge$ {optimal_threshold} is classified as Fraud)*")
    else:
        st.success("✅ **LOW RISK** - APPROVE TRANSACTION")
        st.metric(label="Fraud Probability", value=f"{fraud_proba*100:.2f}%")
        st.markdown(f"*(Score < {optimal_threshold} is classified as Non-Fraud)*")
        
    st.markdown("---")
    st.subheader("Engineered Features Used in Prediction")
    st.dataframe(X_predict.T)