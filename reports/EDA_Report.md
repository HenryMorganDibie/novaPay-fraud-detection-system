# EDA Report - Fraud Detection

## 1. Dataset Overview
- Rows: 10,200
- Columns: 36
- Target variable: `is_fraud` (1.94% fraud rate)

## 2. Missing Values
| Feature | Missing % |
|---------|-----------|
| device_id | 0.3% |
| kyc_tier | 5% |

## 3. Target Distribution
- Fraud: 2%
- Non-Fraud: 98%
*The dataset is highly imbalanced.*

## 4. Key Feature Insights
### Transaction Amounts
- Fraudulent transactions tend to have higher mean amounts.
- Non-fraud transactions are clustered at lower amounts.

### Time Features
- Fraudulent transactions more likely to occur during late night hours (22:00–04:00).

### Categorical Features
- Certain channels (`web` and `mobile`) have higher fraud rates.
- Lower KYC tiers are overrepresented in fraudulent transactions.

## 5. Visualizations
- Histogram of transaction amounts by fraud status
- Boxplots for `txn_hour` vs fraud
- Fraud distribution across channels

## 6. Summary
- Time-based features, customer aggregates, and device/IP trust scores are critical for fraud detection.