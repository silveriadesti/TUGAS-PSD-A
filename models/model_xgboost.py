# models/model_xgboost.py

import xgboost as xgb
import joblib
import os
import sys

# Tambahkan path root ke sys agar bisa import modul preprocessing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_preprocessing_xgb import load_and_preprocess_data

def train_xgb_model(X_train, y_train, model_path):
    print("[INFO] Training model...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    print(f"[INFO] Saving model to {model_path}")
    joblib.dump(model, model_path)

    if os.path.exists(model_path):
        print(f"[SUCCESS] Model saved to {model_path}")
    else:
        print("[ERROR] Failed to save model!")

if __name__ == "__main__":
    file_path = os.path.join("data", "amazon.csv")
    model_path = os.path.join("models", "xgb_model.pkl")

    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(file_path)
    train_xgb_model(X_train, y_train, model_path)
