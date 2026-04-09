import joblib, os, yaml
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor, plot_tree
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

    params = config['models']['DecisionTree']
    
    model = DecisionTreeRegressor(**params)

    model.fit(x_train, y_train_log)

    os.makedirs(config["reports"]["images_path"], exist_ok=True)
    plt.figure(figsize=(48, 24))
    
    plot_tree(model, max_depth=2, filled=True)
    plt.savefig(config["reports"]["images_path"] + "DecisionTree_structure.png")
    plt.close()

    os.makedirs(config['models']['models_path'], exist_ok=True)
    joblib.dump(model, config['models']['models_path'] + "DecisionTree.pkl")

if __name__ == "__main__":
    main()