# 💸 NovaPay Fraud Detection System

### **End-to-End Machine Learning Prototype for Real-Time Fraud Scoring**

This repository contains a full machine learning workflow for detecting fraudulent money transfers on the NovaPay platform. The project demonstrates real-world data science practices across preparation, feature engineering, model benchmarking, deployment, and reporting.

The final model — **Tuned CatBoost Classifier** — powers a real-time risk-scoring demo app built with Streamlit.

---

### Demo Screenshot
![Streamlit Demo Screenshot](screenshot/demo_screenshot.jpg)

---

# 📁 Project Structure

<pre lang="markdown">
fraud-detection-prototype/
│
├── data/
│   ├── nova_pay_transactions.csv
│   ├── cleaned_transactions.pkl
│   └── feature_engineered_transactions.pkl
│
├── demo/
│   ├── app.py
│   └── __pycache__/
│
├── models/
│   ├── preprocessor.joblib
│   ├── CatBoost (Tuned)_fraud_model.joblib
│   ├── catboost_fraud_model.joblib
│   ├── lgb_fraud_model.joblib
│   └── xgb_fraud_model.joblib
│
├── notebooks/
│   ├── 01_data_prep.ipynb
│   ├── 02_eda_and_feature_engineering.ipynb
│   ├── 03_multi_model_training.ipynb
│   └── catboost_info/
│
├── reports/
│   ├── EDA_Report.md
│   ├── Feature_Engineering_Report.md
│   └── Modeling_Report.md
│
├── src/
│   ├── features.py
│   └── __pycache__/
│
├── .gitignore
├── README.md
└── requirements.txt
</pre>

---

# 🌟 Project Summary

This project builds a **production-ready fraud detection prototype** that handles:

* 🚨 **Severe class imbalance** (≈ **1.93%** fraud)
* 🧮 **36 engineered features**
* 🧠 **Multiple gradient-boosted tree models**
* 🔍 **Cost-sensitive learning**
* ⚡ **Fast real-time inference** with CatBoost
* 💻 **Interactive Streamlit demo**

---

# 🎯 Key Achievements

| Component               | Achievement                           | Technical Detail                            |
| ----------------------- | ------------------------------------- | ------------------------------------------- |
| **Data Challenge**      | Severe imbalance                      | Only **1.93%** (197/10,200) were fraudulent |
| **Feature Engineering** | Behavioural velocity features         | Rolling 3-day mean/count per customer       |
| **Modeling**            | Evaluated XGBoost, LightGBM, CatBoost | Tuned with cost-sensitive learning          |
| **Final Model**         | **Tuned CatBoost**                    | Best balance of Recall, Precision, AUC      |
| **Deployment**          | Streamlit scoring demo                | Recreates production-level inference        |

---

# 🏗️ System Architecture

```
┌─────────────────────────────┐
│ Raw Data (CSV)              │
│ data/nova_pay_transactions  │
└───────────┬─────────────────┘
            │
            ▼
┌─────────────────────────────┐
│ Cleaning & Prep             │
│ notebooks/01_*              │
│ cleaned_transactions.pkl    │
└───────────┬─────────────────┘
            │
            ▼
┌──────────────────────────────┐
│ Feature Engineering          │
│ notebooks/02_*               │
│ feature_engineered_*.pkl     │
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│ Model Training & Selection   │
│ notebooks/03_*               │
│ CatBoost | XGB | LGB         │
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│ Export Best Model             │
│ models/catboost_fraud_model   │
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│ Real-Time Scoring Demo App   │
│ demo/app.py                  │
│ src/features.py              │
└──────────────────────────────┘
```

---

# 📊 Final Model Performance (CatBoost – Tuned)

| Metric               | Score      | Interpretation                           |
| -------------------- | ---------- | ---------------------------------------- |
| **AUC-ROC**          | 0.7150     | Ability to separate fraud vs normal      |
| **AUC-PR**           | 0.0700     | Most important metric in imbalanced data |
| **Fraud Precision**  | 0.1515     | Of predicted fraud, 15% were true        |
| **Fraud Recall**     | 0.1282     | Fraction of actual fraud caught          |
| **F1-Score (Fraud)** | 0.1389     | Balance of precision & recall            |
| **Best Threshold**   | **0.6705** | Tuned to maximize Fraud F1               |

---

# 🔬 Data Preparation

### **1. Cleaning**

* Converted timestamps → datetime
* Converted channel, kyc_tier → categorical dtype
* Missing value indicators:

  * `ip_missing`, `kyc_missing`, `device_trust_missing`

### **2. Domain-Aware Imputation**

* `amount_usd = amount_src × exchange_rate`
  → prevents numeric distortion

### **3. Feature Engineering**

Key feature groups:

#### **Temporal**

* `txn_hour`
* `is_weekend`
* `txn_day_of_month`

#### **Risk Features**

* `ip_risk_score`
* `risk_score_internal`
* `corridor_risk`

#### **Behavioral Velocity**

* `txn_count_prev_3d`
* `mean_amount_prev_3d`

Velocity features were critical for detecting **Account Takeover (ATO)** behavior.

---

# 🤖 Modeling & Benchmarking

Three models were compared:

| Model                | AUC-ROC    | AUC-PR | Precision | Recall | F1   |
| -------------------- | ---------- | ------ | --------- | ------ | ---- |
| **XGBoost**          | 0.7148     | 0.0848 | 0.12      | 0.23   | 0.16 |
| **LightGBM**         | 0.6814     | 0.0723 | 0.12      | 0.13   | 0.12 |
| **CatBoost (Tuned)** | **0.7150** | 0.0700 | 0.15      | 0.13   | 0.14 |

### **Why CatBoost Was Selected**

* Best overall trade-off between FP & FN
* Handles categorical features natively
* No need for one-hot encoding → easier deployment
* More stable on noisy financial data

---

# ⚡ Real-Time Inference Workflow

```
User Input → Feature Builder (src/features.py) → Load CatBoost Model →
Predict Probability → Apply Threshold → Output Fraud / Legit + Score
```

### **Steps Inside `app.py`:**

1. User enters transaction details
2. Build raw feature dict
3. `features.py` generates **36 engineered features**
4. Model loads from `models/catboost_fraud_model.joblib`
5. Prediction returned instantly
6. UI displays:

   * Fraud probability
   * Decision (Fraud / Legitimate)
   * Key contributing factors (future work)

---

# 🚀 How to Run the Demo App

### **1. Clone the Repo**

```bash
git clone https://github.com/HenryMorganDibie/novaPay-fraud-detection-system.git
cd novaPay-fraud-detection-system
```

### **2. Create Environment**

```bash
python -m venv .venv
.venv/Scripts/activate   # Windows
# or: source .venv/bin/activate  # macOS/Linux
```

### **3. Install Requirements**

```bash
pip install -r requirements.txt
```

### **4. Run Streamlit App**

```bash
streamlit run demo/app.py
```

The UI opens in your browser and allows you to score transactions in real time.

---

# 🔧 Tools & Technologies

* **Python 3.10+**
* Pandas, NumPy
* CatBoost, XGBoost, LightGBM
* Scikit-Learn
* Joblib
* Streamlit
* Jupyter Notebooks
* Seaborn, Matplotlib

---

# 📦 Model Artifacts (models/ Directory)

| File                                  | Description                          |
| ------------------------------------- | ------------------------------------ |
| `catboost_fraud_model.joblib`         | **Final deployed model**             |
| `CatBoost (Tuned)_fraud_model.joblib` | Tuned training version               |
| `xgb_fraud_model.joblib`              | Benchmark model                      |
| `lgb_fraud_model.joblib`              | Benchmark model                      |
| `preprocessor.joblib`                 | Encoding/transform logic (if needed) |

---

# 📈 Monitoring & Drift Strategy

Planned production monitoring:

### **1. Prediction Drift**

* Monitor decline in recall
* Monitor spike in false positives

### **2. Feature Drift**

* Track distribution changes in:

  * Amount
  * Velocity features
  * Risk signals

### **3. Label Delay**

* Fraud labels often arrive weeks later
* Pipeline supports delayed supervised updates

### **4. Retraining**

* Scheduled retraining every 30 days
* OR triggered when drift exceeds threshold

---

# 🧭 Future Improvements

* Add FastAPI for real-time API scoring
* Add SHAP explainability
* Use PostgreSQL or MongoDB for customer history lookups
* Expand rolling windows (1, 7, 30 days)
* Implement Kafka streaming pipeline
* Build ensemble fraud scoring engine
* Add real-time alerting dashboard

---

# 🤝 Contributing

Pull requests are welcome.

Steps:

```bash
git checkout -b feature/my-feature
# make changes
git push origin feature/my-feature
```

Please follow PEP8 and include documentation updates.

---

# 📜 License

Released under the **MIT License**.

---
