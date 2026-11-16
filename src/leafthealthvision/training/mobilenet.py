import os
import sqlite3
import tempfile

import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from codecarbon import EmissionsTracker
from mlflow.tracking import MlflowClient
from tensorflow.keras import applications, layers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from leafthealthvision.config.settings import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    IMAGE_RESCALE,
    IMAGE_TARGET_SIZE,
    METRICS,
    MLFLOW_ARTIFACT_LOCATION,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    OPTIMIZER,
    ROOT,
)
from leafthealthvision.data import load_datasets

# --- MLflow: seguimiento local de experimentos (SQLite -> mlflow.db) ---
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# El experimento ya creado puede apuntar a una ruta inaccesible (p.ej. /content,
# creada en otro entorno). mlflow solo borra experiments "suavemente", así que no
# se puede borrar y volver a crear con el mismo nombre; relocalizamos el store
# directamente cuando la ubicación difiere de la configurada.
if MLFLOW_ARTIFACT_LOCATION:
    client = MlflowClient()
    existing = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if existing is None:
        client.create_experiment(
            MLFLOW_EXPERIMENT_NAME, artifact_location=MLFLOW_ARTIFACT_LOCATION
        )
    else:
        if existing.lifecycle_stage == "deleted":
            client.restore_experiment(existing.experiment_id)
        if existing.artifact_location != MLFLOW_ARTIFACT_LOCATION:
            db_path = MLFLOW_TRACKING_URI.split(":///", 1)[1]
            if not os.path.isabs(db_path):
                db_path = os.path.join(ROOT, db_path)
            with sqlite3.connect(db_path) as con:
                con.execute(
                    "UPDATE experiments SET artifact_location = ? WHERE name = ?",
                    (MLFLOW_ARTIFACT_LOCATION, MLFLOW_EXPERIMENT_NAME),
                )

mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


class MlflowEpochLogger(tf.keras.callbacks.Callback):
    """Registra las métricas de cada época en MLflow."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=epoch)


train_ds, validation_ds, test_ds, class_names = load_datasets()

img_height, img_width = IMAGE_TARGET_SIZE

num_classes = len(class_names)

# --- Transfer learning: base pre-entrenado MobileNetV2 (guía TF) ---
base_model = applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights="imagenet",
)

# Congelar la base: solo se entrena la cabeza nueva
base_model.trainable = False

inputs = Input(shape=(img_height, img_width, 3))
x = layers.Rescaling(IMAGE_RESCALE)(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(num_classes)(x)
model = Model(inputs, outputs)

model.compile(
    optimizer=OPTIMIZER,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=METRICS,
)


model.summary()


epochs = 10
early_stop = tf.keras.callbacks.EarlyStopping(
    patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True
)

# Parámetros del experimento
params = {
    "model": "mobilenetv2",
    "base_model_weights": "imagenet",
    "base_trainable": False,
    "epochs": epochs,
    "batch_size": BATCH_SIZE,
    "img_height": img_height,
    "img_width": img_width,
    "image_rescale": IMAGE_RESCALE,
    "optimizer": OPTIMIZER,
    "loss": "SparseCategoricalCrossentropy(from_logits=True)",
    "metrics": ",".join(METRICS),
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "num_classes": num_classes,
}

# Huella de carbono del entrenamiento (kg CO2eq)
tracker = EmissionsTracker(
    save_to_file=True, output_dir="results", output_file="emissions"
)

with mlflow.start_run(run_name="mobilenetv2_transfer_learning"):
    mlflow.log_params(params)

    tracker.start()
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epochs,
        callbacks=[early_stop, MlflowEpochLogger()],
    )

    emisiones_kg = tracker.stop() or 0.0
    energia_kwh = getattr(getattr(tracker, "_total_energy", None), "kWh", None)

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")

    # Guardar y registrar las curvas como artefacto de MLflow
    with tempfile.TemporaryDirectory() as tmpdir:
        curves_path = os.path.join(tmpdir, "training_curves.png")
        plt.savefig(curves_path, bbox_inches="tight")
        mlflow.log_artifact(curves_path)

    plt.show()

    final_metrics = {
        "final_accuracy": float(acc[-1]),
        "final_val_accuracy": float(val_acc[-1]),
        "final_loss": float(loss[-1]),
        "final_val_loss": float(val_loss[-1]),
        "carbon_emissions_kg": float(emisiones_kg),
    }
    if energia_kwh is not None:
        final_metrics["energy_consumed_kwh"] = float(energia_kwh)
    mlflow.log_metrics(final_metrics)

    try:
        mlflow.tensorflow.log_model(model, name="model")
    except Exception as exc:  # noqa: BLE001
        print(f"No se pudo registrar el modelo en MLflow: {exc}")

    print(f"Huella de carbono del entrenamiento: {emisiones_kg:.6f} kg CO2eq")
