import os, yaml, json
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
import matplotlib.pyplot as plt

def work_with_dataset(config):
    data_train = pd.read_csv(config["data"]["train_path"])

    x_train = data_train.drop(columns=config["data"]["target"])
    y_train = data_train[config["data"]["target"]] 

    y_train_log = np.log1p(y_train)

    return x_train, y_train_log

def main():
    with open('config/parameters.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    x_train, y_train_log = work_with_dataset(config)

    params = config['models']['XGBoost']
    
    model = XGBRegressor(**params)

    model.fit(x_train, y_train_log)

    importance = model.feature_importances_
    os.makedirs(config["reports"]["images_path"], exist_ok=True)
    plt.bar(range(len(importance)), importance)

    plt.savefig(config["reports"]["images_path"] + "XGBoost_feature_importance.png")
    plt.close()

    os.makedirs(config["models"]["models_path"], exist_ok=True)
    model.save_model(config["models"]["models_path"] + "XGBoost.json")

if __name__ == "__main__":
    main()