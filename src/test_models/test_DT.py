import joblib
import os
import yaml
import json
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dvclive import Live

def load_data(test_path: str, target_column: str):
    """Загружает тестовые данные и возвращает X_test, y_test."""
    data = pd.read_csv(test_path)
    X_test = data.drop(columns=[target_column])
    y_test = data[target_column]
    return X_test, y_test

def compute_metrics(y_true, y_pred):
    """Вычисляет метрики регрессии."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }

def main():
    # Загрузка конфигурации (у вас parameters.yaml)
    with open('config/parameters.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Загрузка тестовых данных
    X_test, y_test = load_data(
        test_path=config["data"]["test_path"],
        target_column=config["data"]["target"]   # у вас ключ "target", а не "target_column"
    )

    # Загрузка модели
    model_path = os.path.join(config['models']['models_path'], "DecisionTree.pkl")
    model = joblib.load(model_path)

    # Предсказание и обратное преобразование (expm1, т.к. обучали на log1p)
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    # Метрики
    metrics = compute_metrics(y_test.values, y_pred)

    # Логирование с DVCLive
    with Live(dir="dvclive/DecisionTree", save_dvc_exp=True) as live:
        for metric, value in metrics.items():
            live.log_metric(f"test/{metric}", value)
if __name__ == "__main__":
    main()