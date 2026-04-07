import joblib, os, yaml, json
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def main():
    # 
    with open('config/parameters.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Load data
    data_train = pd.read_csv(config["data"]["train_path"])

    x_train = data_train.drop(columns=config["data"]["target"])
    y_train = data_train[config["data"]["target"]]

    y_train_log = np.log1p(y_train)

    # Load model params
    params = config['models']['LinearRegression']
    
    model = LinearRegression(**params)
    model.fit(x_train, y_train_log)

    # Log coefficients
    coef_dict = {f"coef_{i}": float(v) for i, v in enumerate(model.coef_)}

    with open(config["reports"]["coeffs_path"] + "LinearRegression_coeff.json", "w") as f:
        json.dump(coef_dict, f, indent=4)
    
    features = list(coef_dict.keys())
    coefficients = list(coef_dict.values())

    plt.figure(figsize=(8, 5))
    plt.bar(features, coefficients, color='skyblue')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title("Коэффициенты линейной модели")
    plt.ylabel("Значение коэффициента")
    plt.xlabel("Признаки")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(config["reports"]["images_path"] + "LinearRegression_coefficients.png")
    plt.close()

    os.makedirs(config['models']['models_path'], exist_ok=True)
    joblib.dump(model, config['models']['models_path'] + "LinearRegression.pkl")


if __name__ == "__main__":
    main()