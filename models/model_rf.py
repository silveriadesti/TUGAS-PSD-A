from sklearn.ensemble import RandomForestRegressor

def build_rf_model(n_estimators=100, random_state=0):
    """
    Membangun model Random Forest Regressor.

    Args:
        n_estimators (int): Jumlah pohon dalam forest.
        random_state (int): Seed random untuk reproduktibilitas.

    Returns:
        RandomForestRegressor: Model yang belum dilatih.
    """
    return RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

if __name__ == '__main__':
    model = build_rf_model()
    print(model)
