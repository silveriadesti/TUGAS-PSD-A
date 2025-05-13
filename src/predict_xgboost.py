# src/predict_xgboost.py

import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from data_preprocessing_xgb import load_and_preprocess_data

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    print("Root Mean Squared Error (RMSE):", round(rmse, 4))
    print("R² Score:", round(r2, 4))
    print("Mean Absolute Percentage Error (MAPE):", round(mape, 2), "%")

def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.values, label='Actual Price', color='blue')
    plt.plot(y_pred, label='Predicted Price', color='orange')
    plt.title('Actual vs Predicted Stock Prices (XGBoost)')
    plt.xlabel('Time Index')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # Load data
    file_path = os.path.join("data", "amazon.csv")
    model_path = os.path.join("models", "xgb_model.pkl")

    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(file_path)

    # Load trained model
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}. Jalankan model_xgboost.py dulu.")
        return

    model = joblib.load(model_path)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    evaluate_model(y_test, y_pred)

    # Prediksi hari terakhir
    print("\nHarga aktual hari terakhir:", round(y_test.values[-1], 2))
    print("Harga prediksi hari terakhir:", round(y_pred[-1], 2))

    # Plot
    plot_predictions(y_test, y_pred)

if __name__ == "__main__":
    main()
