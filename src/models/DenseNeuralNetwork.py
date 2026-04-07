import joblib
import os
import yaml
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout

def main():
    # 1. Загрузка конфигурации
    with open('config/parameters.yaml', 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)

    # 2. Загрузка данных
    data_train = pd.read_csv(config["data"]["train_path"])
    target_col = config["data"]["target"]
    x = data_train.drop(columns=[target_col])
    y = data_train[target_col]
    y_train_log = np.log1p(y)  # логарифмирование, если нужно

    # 3. Разделение на train/validation
    val_split = config["models"]["DenseNeuralNetwork"]["dataset_params"]["validation_split"]
    random_state = config["models"]["DenseNeuralNetwork"]["dataset_params"]["random_state"]
    x_train, x_val, y_train, y_val = train_test_split(
        x, y_train_log,
        test_size=val_split,
        random_state=random_state
    )

    # 4. Параметры модели из конфига (уже оптимизированные)
    model_params = config["models"]["DenseNeuralNetwork"]["model_params"]
    dropout_rate = model_params.get("dropout_rate", 0.2)
    n_layers = model_params.get("n_layers", 3)
    preout_neurons = model_params.get("preout_neurons", 64)
    learning_rate = model_params.get("learning_rate", 0.001)
    activation = model_params.get("activation", 'relu')

    # 5. Функция создания модели (без Optuna, использует параметры из config)
    def create_model(input_dim):
        model = Sequential()
        model.add(Input(shape=(input_dim,)))
        for i in range(n_layers):
            units = (n_layers - i) * preout_neurons
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))
            model.add(Dense(units, activation=activation))
        model.add(Dense(1))  # выходной слой для регрессии
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=[
                tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                tf.keras.metrics.MeanAbsoluteError(name="mae"),
                tf.keras.metrics.R2Score(name='r2_score'),
            ]
        )
        return model

    # Создаём модель с правильной размерностью входа
    model = create_model(x_train.shape[1])

    # 6. Подготовка каталогов и TensorBoard
    os.makedirs("logs/fit/", exist_ok=True)
    log_dir = f"logs/fit/{datetime.datetime.now().strftime('%H-%M-%S_%d-%m-%Y')}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq="epoch",
        profile_batch=0
    )

    # 7. Обучение
    train_params = config["models"]["DenseNeuralNetwork"]["train_params"]
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        callbacks=[tensorboard_callback],
        epochs=train_params["epochs"],
        verbose=train_params["verbose"]
    )

    # 8. Сохранение графиков обучения
    images_path = config["reports"]["images_path"]
    os.makedirs(images_path, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig(os.path.join(images_path, "DenseNeuralNetwork_loss_curve.png"))
    plt.close()

    # 9. Гистограммы весов Dense слоёв
    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    for idx, layer in enumerate(dense_layers[:n_layers]):
        weights = layer.kernel.numpy().flatten()
        plt.figure(figsize=(12, 6))
        plt.hist(weights, bins=100)
        plt.title(f"Dense layer {idx+1} Weight Distribution")
        plt.xlabel("Weight value")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(images_path, f"DenseNeuralNetwork_weights_layer_{idx+1}.png"))
        plt.close()

    # 10. Визуализация архитектуры модели
    tf.keras.utils.plot_model(
        model,
        to_file=os.path.join(images_path, "DenseNeuralNetwork_model_graph.png"),
        show_shapes=True,
        show_layer_names=True,
        dpi=600
    )

    # 11. Сохранение модели
    models_path = config["models"]["models_path"]
    os.makedirs(models_path, exist_ok=True)
    model.save(os.path.join(models_path, "DenseNeuralNetwork.keras"))

if __name__ == "__main__":
    main()