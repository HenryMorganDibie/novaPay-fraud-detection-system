# Feature Engineering Documentation

## 1. Time-Based Features
| Feature | Description | Reason |
|---------|-------------|--------|
| txn_day_of_month | Day of month extracted from transaction timestamp | Fraud patterns vary by month day |
| txn_day_of_week | Day of week extracted from timestamp (0=Mon) | Detect weekday/weekend fraud trends |
| txn_hour | Hour extracted from timestamp (0-23) | Fraud often occurs at unusual hours |

## 2. Customer Behavior Features
| Feature | Description | Reason |
|---------|-------------|--------|
| mean_amount_prev_3d | Average amount over last 3 days | Detect unusual spending spikes |
| txn_count_prev_3d | Number of transactions in last 3 days | High frequency may indicate fraud |
| account_age_days | Days since account creation | New accounts more vulnerable |

## 3. Transaction/Device Features
| Feature | Description | Reason |
|---------|-------------|--------|
| device_trust_score | Device trustworthiness score | New/untrusted devices often indicate fraud |
| ip_risk_score | Risk score from IP | Higher risk IPs indicate potential fraud |
| channel | Transaction channel (web, mobile, POS) | Fraud rate varies by channel |

## 4. Amount/Financial Features
| Feature | Description | Reason |
|---------|-------------|--------|
| amount_src | Transaction amount in source currency | Outlier amounts may indicate fraud |
| amount_usd | Standardized transaction amount | Compare across currencies |
| fee | Transaction fee | Large/irregular fees may indicate fraud |
| exchange_rate_src_to_dest | Used exchange rate | Detect suspicious conversions |

## 5. KYC Features
| Feature | Description | Reason |
|---------|-------------|--------|
| kyc_tier | Verification tier | Lower verification levels are riskier |

---

**Notes:**
- Features were selected to capture **temporal patterns, anomalous behavior, and trust indicators**.
- These engineered features improve model performance for tree-based algorithms like CatBoost and XGBoost.