# data_preprocessing_xgb.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Pastikan tanggal dalam urutan yang benar
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # Fitur yang digunakan untuk prediksi
    features = ['Open', 'High', 'Low', 'Volume']
    target = 'Close'

    X = df[features]
    y = df[target]

    # Normalisasi fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )

    return X_train, X_test, y_train, y_test, scaler
