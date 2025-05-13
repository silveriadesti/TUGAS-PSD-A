import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error

from data_preprocessing_lstm import load_data
from models.model_lstm import create_dataset, build_lstm_model

def load_and_preprocess_data(file_path, test_size=0.2):
    df = load_data(file_path)
    data = df[['Close']].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    train_size = int(len(data_scaled) * (1 - test_size))
    train_data = data_scaled[:train_size]
    test_data = data_scaled[train_size:]

    return train_data, test_data, scaler

if __name__ == '__main__':
    file_path = 'data/amazon.csv'
    time_step = 60

    # Load and prepare data
    train_data, test_data, scaler = load_and_preprocess_data(file_path)
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Reshape for LSTM input
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Train model
    model = build_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    # Predict
    predicted_stock_price = model.predict(X_test)

    # Inverse transform predictions and actual values
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, predicted_stock_price))
    r2 = r2_score(y_test, predicted_stock_price)
    mape = mean_absolute_percentage_error(y_test, predicted_stock_price)

    print(f"LSTM RMSE: {rmse:.4f}")
    print(f"LSTM R² Score: {r2:.4f}")
    print(f"LSTM MAPE: {mape:.2f}%")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, color='blue', label='Actual Price')
    plt.plot(predicted_stock_price, color='red', label='Predicted Price')
    plt.title('LSTM Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
