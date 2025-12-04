# 📘 Fraud Detection Modeling Report

# Model Development, Evaluation, Selection & Insights

Following the exploratory data analysis and feature engineering stages, the next step in building the **NovaPay Fraud Detection System** was model development. Because fraud is rare, high-stakes, and unpredictable, the modeling strategy was designed to handle:

* Extreme class imbalance
* High-cardinality categorical features
* Non-linear transaction behaviours
* Noisy real-world financial patterns

This document explains the full modeling process.

# 🔍 1. Problem Framing

The objective is to classify each transaction as:

* **0 = Legitimate**
* **1 = Fraudulent**

Fraud accounts for **<2%** of all transactions, so accuracy is **not** meaningful.
The model was evaluated using:

* **AUC-ROC** — overall separability
* **AUC-PR** — best metric for rare classes
* **Fraud F1-Score**
* **Threshold tuning** to balance false positives & false negatives

# 🧱 2. Feature Structure for Modeling

After feature engineering, the dataset contained:

## ✔️ Numerical Features

Examples:

* `amount_src`, `amount_usd`, `fee`
* `account_age_days`
* `txn_hour`, `txn_dayofweek`
* Rolling device/customer statistics
* Risk score aggregations

## ✔️ Categorical Features

Examples:

* `channel`
* `kyc_tier`
* `device_type`
* `country`
* `payment_method`
* `is_blacklisted`, `high_risk_country`

Handling these properly was crucial for performance.

# 🤖 3. Models Tested

Three ensemble tree-based models were selected:

## 3.1 XGBoost

**Strengths:**

* Great for tabular data
* Strong non-linear modeling

**Weaknesses:**

* Requires one-hot encoding
* Sensitive to extreme imbalance

## 3.2 LightGBM

**Strengths:**

* Very fast
* Supports categorical features via integer encoding

**Weaknesses:**

* Slightly unstable with highly imbalanced data
* Performance fluctuates on noisy data

## 3.3 CatBoost (Final Selected Model)

**Strengths:**

* Best native handling of categorical features
* Excellent robustness with imbalanced data
* Stable performance with noisy real-world data
* Reduces overfitting via ordered boosting

**CatBoost proved to be the best match for fraud data.**

# ⚙️ 4. Preprocessing Strategy

Different models required different pipelines:

## For XGBoost & LightGBM

* Missing values → median/mode
* Categorical features → one-hot encoding
* Numerical features → unchanged
* One-hot encoding expanded dataset → **hundreds of columns**

## For CatBoost

* No one-hot encoding
* Categorical features passed as strings
* CatBoost internally performs:

  * Target statistics
  * Ordered boosting
  * Regularized encoding

This improved model stability dramatically.

# 🧪 5. Training Strategy

* **80/20 stratified split** — preserves fraud distribution.
* **Class imbalance handling** — all models used:

```
scale_pos_weight = non_fraud_count / fraud_count
```

This ensures more focus on fraud cases.

# 🎛️ 6. Hyperparameter Tuning

## XGBoost (Randomized Search)

* Trees: 300
* Depth: 6
* Learning Rate: 0.05
* Subsample: 0.7
* Class Weight: tuned

## LightGBM

* Trees: 300
* Max Depth: 6
* Feature & Row Sampling: 0.7
* `is_unbalance = True`

## CatBoost (RandomizedSearchCV)

* Depth: {5, 6, 8}
* Learning Rate: {0.01, 0.03, 0.05, 0.1}
* L2 Regularization: {1, 3, 5, 10}
* 10 randomized iterations
* Optimized on **Fraud F1-score**

CatBoost consistently outperformed the other models.

# 📊 7. Evaluation Metrics

|               Metric | Meaning                         | Why Important                    |
| -------------------: | ------------------------------- | -------------------------------- |
|          **AUC-ROC** | Distinguishes fraud vs normal   | General performance indicator    |
|           **AUC-PR** | Precision-Recall balance        | Best for rare fraud cases        |
|   **Fraud F1-Score** | Balance of precision & recall   | Operationally important          |
| **Threshold Tuning** | Converts probabilities → labels | Reduces unnecessary false alerts |

# 🏆 8. Model Results

## 8.1 XGBoost

* **AUC-ROC:** 0.7148
* **AUC-PR:** 0.0848
* **Fraud F1:** 0.16

Good learning ability but struggled because of sparse one-hot encoded features.

## 8.2 LightGBM

* **AUC-ROC:** 0.6814
* **AUC-PR:** 0.0723
* **Fraud F1:** 0.12

Fast, but underperformed on extreme imbalance.

## 8.3 CatBoost (Tuned) — ⭐ Best Model

* Best metrics across all categories
* Most stable and consistent
* Handles categorical variables effectively
* Lower false positives at the same recall
* Fast inference for real-time systems

CatBoost excels in handling:

* Rare categories
* Non-linear relationships
* Interaction-heavy features
* Complex behavioural patterns

# 🧠 9. Why CatBoost Was Selected

CatBoost won because it:

* Handles categorical features without encoding
* Works extremely well with imbalanced datasets
* Reduces overfitting using ordered boosting
* Produces stable and interpretable results
* Provides predictable real-time performance

**CatBoost is the most production-ready algorithm for a fraud scoring system.**

# 🚀 10. Final Model for Deployment

**Selected Model:** `CatBoostClassifier` (Tuned)

Saved as:

```
catboost_fraud_model.cbm
```

## Next Steps

* Build the inference pipeline
* Integrate into API / Streamlit demo
* Implement model monitoring & drift detection
* Document threshold strategy for fraud operations
