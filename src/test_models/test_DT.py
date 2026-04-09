import joblib
import os
import yaml
import json
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dvclive import Live


def work_with_dataset(config):
    data_train = pd.read_csv(config["data"]["train_path"])
    data_val = pd.read_csv(config["data"]["val_path"])
    data_test = pd.read_csv(config["data"]["test_path"])

    X_train = data_train.drop(columns=config["data"]["target"])
    y_train = data_train[config["data"]["target"]]
    X_val = data_val.drop(columns=config["data"]["target"])
    y_val = data_val[config["data"]["target"]]
    X_test = data_test.drop(columns=config["data"]["target"])
    y_test = data_test[config["data"]["target"]]

    return X_train, y_train, X_val, y_val, X_test, y_test


def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def main():
    with open('config/parameters.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Загрузка данных
    X_train, y_train, X_val, y_val, X_test, y_test = work_with_dataset(config)

    # Загрузка модели
    model_path = os.path.join(config['models']['models_path'], "DecisionTree.pkl")
    model = joblib.load(model_path)

    # Предсказание и обратное преобразование
    y_train_pred_log = model.predict(X_train)
    y_train_pred = np.expm1(y_train_pred_log)

    y_val_pred_log = model.predict(X_val)
    y_val_pred = np.expm1(y_val_pred_log)

    y_test_pred_log = model.predict(X_test)
    y_test_pred = np.expm1(y_test_pred_log)

    # Метрики
    train_metrics = compute_metrics(y_train.values, y_train_pred)
    val_metrics   = compute_metrics(y_val.values,   y_val_pred)
    test_metrics  = compute_metrics(y_test.values,  y_test_pred)

    all_metrics = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics
    }

    # Сохранение метрик в JSON
    reports_dir = config["reports"].get("metrics_path", "reports/")
    os.makedirs(reports_dir, exist_ok=True)
    metrics_file = os.path.join(reports_dir, "DecisionTree_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4, ensure_ascii=False)

    # Логирование с DVCLive
    with Live(dir="dvclive/DecisionTree", save_dvc_exp=True) as live:
        for subset, metrics in all_metrics.items():
            for metric_name, value in metrics.items():
                live.log_metric(f"{subset}/{metric_name}", value)


if __name__ == "__main__":
    main()