import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timezone
import os

import sys
import os.path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import create_realtime_features 

# --- CONFIGURATION ---
MODEL_PATH = 'models/CatBoost (Tuned)_fraud_model.joblib'
HISTORY_DATA_PATH = 'data/nova_pay_transactions.csv'
OPTIMAL_THRESHOLD = 0.6705 

# --- 1. Load Assets (Done once on app startup) ---

@st.cache_resource
def load_model_and_data():
    """Loads the model and historical data for feature look-up."""
    model_obj = None
    history_df = None
    
    try:
        # Paths are relative to the project root, which is correct
        model_obj = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"Error: Model file not found. Ensure '{MODEL_PATH}' exists.")
        return None, None, None
    
    try:
        history_df = pd.read_csv(HISTORY_DATA_PATH)
        
        # Initial required cleaning/typing on the history data
        history_df['timestamp'] = pd.to_datetime(
            history_df['timestamp'], 
            utc=True,
            errors='coerce',
            format='mixed' 
        )
        history_df['amount_usd'] = pd.to_numeric(history_df['amount_usd'], errors='coerce')

        # Select columns, drop NaT/NaNs, set index for efficient lookup
        history_df = history_df[['customer_id', 'timestamp', 'amount_usd']].dropna(subset=['amount_usd', 'timestamp'])
        history_df = history_df.set_index('timestamp').sort_values(by=['customer_id', 'timestamp'])
        
        # Calculate a global median to use as a fallback for new customers
        global_median = history_df['amount_usd'].median()
        
        st.sidebar.success("Historical data and model loaded successfully.")
        return model_obj, history_df, global_median

    except FileNotFoundError:
        st.error(f"Error: Historical data file '{HISTORY_DATA_PATH}' not found. Behavioral features cannot be calculated.")
        return model_obj, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading: {e}")
        return model_obj, None, None

model, history_df, GLOBAL_MEDIAN_AMOUNT = load_model_and_data()
if model is None or history_df is None or GLOBAL_MEDIAN_AMOUNT is None:
    st.stop()


# --- Streamlit App Layout ---

st.title("💸 NovaPay Fraud Detection Prototype")
st.markdown("Enter transaction details below to get a real-time fraud risk score from the CatBoost model.")
st.markdown("---")

# Define input fields in the sidebar
st.sidebar.header("Transaction Details")

# Transaction IDs and Time (needed for feature calculation)
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
    
    try:
        # 1. Create Raw Input Dictionary (includes all fields the feature function needs)
        raw_input = {
            'customer_id': customer_id,
            'timestamp': datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')), 
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
    except ValueError:
        st.error("Invalid timestamp format. Please use ISO format (e.g., 2024-01-01T12:00:00+00:00).")
        st.stop()


    # 2. Engineer Features (Function call uses the imported logic)
    with st.spinner('Engineering behavioral and static features...'):
        X_predict = create_realtime_features(
            raw_input, 
            history_df, 
            model.feature_names_, 
            GLOBAL_MEDIAN_AMOUNT
        )
    
    # 3. Predict
    with st.spinner('Running CatBoost model prediction...'):
        fraud_proba = model.predict_proba(X_predict)[0][1]
        
    # 4. Display Results
    st.header("Prediction Result")
    
    if fraud_proba >= OPTIMAL_THRESHOLD:
        st.error(f"⚠️ **HIGH RISK OF FRAUD** - BLOCK TRANSACTION")
        st.metric(label="Fraud Probability", value=f"{fraud_proba*100:.2f}%")
        st.markdown(f"*(Score $\ge$ {OPTIMAL_THRESHOLD} is classified as Fraud)*")
    else:
        st.success("✅ **LOW RISK** - APPROVE TRANSACTION")
        st.metric(label="Fraud Probability", value=f"{fraud_proba*100:.2f}%")
        st.markdown(f"*(Score < {OPTIMAL_THRESHOLD} is classified as Non-Fraud)*")
        
    st.markdown("---")
    st.subheader("Engineered Features Used in Prediction (For Review)")
    st.dataframe(X_predict.T)
    
    # 5. Feature Explanation 
    st.subheader("Key Velocity Feature Output")
    st.markdown(f"""
        - **Transaction Count (Last 3d):** `{X_predict['txn_count_prev_3d'].iloc[0]:.0f}`
        - **Mean Amount (Last 3d):** `${X_predict['mean_amount_prev_3d'].iloc[0]:.2f}`
        - **High Amount Ratio Flag:** `{'YES' if X_predict['is_high_amount_ratio'].iloc[0] == 1 else 'NO'}`
    """)