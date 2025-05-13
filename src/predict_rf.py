import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from data_preprocessing_lstm import load_data, preprocess_data
from models.model_rf import build_rf_model
import matplotlib.pyplot as plt

def train_and_predict_rf(file_path, lag=60, test_size=0.2, n_estimators=100, use_features=None):
    """
    Melatih Random Forest, memprediksi harga saham, mengevaluasi, dan menampilkan hasil.

    Args:
        file_path (str): Path ke file CSV.
        lag (int): Window lag.
        test_size (float): Rasio data untuk testing.
        n_estimators (int): Jumlah pohon dalam Random Forest.
        use_features (list): Fitur yang digunakan. Default = ['Close'].
    """
    # 1. Load dan Preprocess
    df = load_data(file_path)
    X_train_lstm, X_test_lstm, y_train, y_test, scaler, X_train_rf, X_test_rf = preprocess_data(
        df, lag=lag, test_size=test_size, use_features=use_features
    )

    # 2. Build dan Train Model
    model = build_rf_model(n_estimators=n_estimators)
    model.fit(X_train_rf, y_train)

    # 3. Predict dan reshape
    y_pred_scaled = model.predict(X_test_rf).reshape(-1, 1)

    # 4. Inverse Scaling
    dummy_pred = np.zeros((len(y_pred_scaled), X_test_rf.shape[1] + 1))
    dummy_pred[:, 0] = y_pred_scaled.ravel()
    y_pred = scaler.inverse_transform(dummy_pred)[:, 0]

    dummy_actual = np.zeros((len(y_test), X_test_rf.shape[1] + 1))
    dummy_actual[:, 0] = y_test
    y_test_original = scaler.inverse_transform(dummy_actual)[:, 0]

    # 5. Evaluation
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
    print(f"Random Forest RMSE: {rmse:.4f}")

    # 6. Plotting
    test_index = df.index[-len(y_test):]  # Ambil indeks waktu dari data asli
    plt.figure(figsize=(12, 6))
    plt.plot(test_index, y_test_original, label='Actual')
    plt.plot(test_index, y_pred, label='Predicted')
    plt.title('Random Forest - Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return y_test_original, y_pred

if __name__ == '__main__':
    file_path = 'data/amazon.csv'
    train_and_predict_rf(file_path, lag=60, test_size=0.2, n_estimators=100, use_features=['Close'])