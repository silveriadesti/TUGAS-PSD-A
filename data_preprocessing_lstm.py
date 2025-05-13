import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    return df

def preprocess_data_lstm(df, target_column='Close', lag=60, test_size=0.2, use_features=None):
    if use_features is None:
        use_features = ['Close']
    data = df[use_features].copy()

    for feature in use_features:
        for i in range(1, lag + 1):
            data[f'{feature}_lag_{i}'] = data[feature].shift(i)
    data.dropna(inplace=True)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    num_features_original = len(use_features)
    X = scaled_data[:, num_features_original:]
    y = scaled_data[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    X_train_lstm = X_train.reshape((X_train.shape[0], lag, num_features_original))
    X_test_lstm = X_test.reshape((X_test.shape[0], lag, num_features_original))

    return X_train_lstm, X_test_lstm, y_train, y_test, scaler
