import os
import yaml
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout


def work_with_dataset(config):
    train_df = pd.read_csv(config["data"]["train_path"])
    val_df = pd.read_csv(config["data"]["val_path"])

    X_train = train_df.drop(columns=config["data"]["target"])
    y_train = train_df[config["data"]["target"]]
    X_val = val_df.drop(columns=config["data"]["target"])
    y_val = val_df[config["data"]["target"]]

    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    return X_train, y_train_log, X_val, y_val_log


def create_model(input_dim, model_params):
    n_layers = model_params.get("n_layers", 3)
    preout_neurons = model_params.get("preout_neurons", 64)
    dropout_rate = model_params.get("dropout_rate", 0.2)
    activation = model_params.get("activation", "relu")
    learning_rate = model_params.get("learning_rate", 0.001)

    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    for i in range(n_layers):
        units = (n_layers - i) * preout_neurons
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
        model.add(Dense(units, activation=activation))

    model.add(Dense(1))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.R2Score(name="r2_score"),
        ]
    )
    return model


def log_weights_histograms(epoch, model, writer):
    with writer.as_default():
        for layer in model.layers:
            if hasattr(layer, 'kernel'):
                tf.summary.histogram(f"weights/{layer.name}", layer.kernel, step=epoch)
                if layer.bias is not None:
                    tf.summary.histogram(f"biases/{layer.name}", layer.bias, step=epoch)
        writer.flush()


def main():
    with open('config/parameters.yaml', 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)

    X_train, y_train, X_val, y_val = work_with_dataset(config)

    model_params = config["models"]["DenseNeuralNetwork"]["model_params"]
    train_params = config["models"]["DenseNeuralNetwork"]["train_params"]

    model = create_model(X_train.shape[1], model_params)

    log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime('%H-%M-%S_%d-%m-%Y'))
    os.makedirs(log_dir, exist_ok=True)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,       
        write_graph=True,
        write_images=False,
        update_freq="epoch",
        profile_batch=0
    )

    writer = tf.summary.create_file_writer(log_dir)

    # Колбэк для вызова функции логирования гистограмм
    hist_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch: log_weights_histograms(epoch, model, writer)
    )

    tf.summary.trace_on(graph=True, profiler=False)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        callbacks=[tensorboard_callback, hist_callback],
        epochs=train_params["epochs"],
        verbose=train_params.get("verbose", 1),
    )

    # Экспорт графа в TensorBoard
    with tf.summary.create_file_writer(log_dir).as_default():
        tf.summary.trace_export(
            name="DNN_graph_trace",
            step=0,
            profiler_outdir=log_dir
        )

    # Сохранение статичных графиков
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

    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    for idx, layer in enumerate(dense_layers[:-1]):
        weights = layer.kernel.numpy().flatten()
        plt.figure(figsize=(12, 6))
        plt.hist(weights, bins=100)
        plt.title(f"Dense layer {idx+1} Weight Distribution (final epoch)")
        plt.xlabel("Weight value")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(images_path, f"DenseNeuralNetwork_weights_layer_{idx+1}.png"))
        plt.close()

    tf.keras.utils.plot_model(
        model,
        to_file=os.path.join(images_path, "DenseNeuralNetwork_model_graph.png"),
        show_shapes=True,
        show_layer_names=True,
        dpi=600,
    )

    models_path = config["models"]["models_path"]
    os.makedirs(models_path, exist_ok=True)
    model.save(os.path.join(models_path, "DenseNeuralNetwork.keras"))

    print("TensorBoard: tensorboard --logdir logs/fit")


if __name__ == "__main__":
    main()