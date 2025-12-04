import joblib
from pathlib import Path

MODEL_PATH = Path("models/CatBoost (Tuned)_fraud_model.joblib")

loaded = joblib.load(MODEL_PATH)

print("Type of loaded object:", type(loaded))
print("Content of loaded object:", loaded)
