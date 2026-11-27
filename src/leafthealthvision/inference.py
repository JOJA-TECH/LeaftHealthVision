"""Inferencia con modelos entrenados y mapa de calor Grad-CAM de la clase detectada."""

import argparse
import csv
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import tensorflow as tf
from codecarbon import EmissionsTracker
from mlflow.tracking import MlflowClient

from leafthealthvision.config.settings import (
    IMAGE_TARGET_SIZE,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    PLANTVILLAGE,
    PLANTVILLAGE_CLASSES,
    ROOT,
)

COST_LOG = ROOT / "results" / "inference_cost.csv"
COST_FIELDS = [
    "timestamp",
    "image",
    "model",
    "class",
    "confidence",
    "latency_s",
    "energy_kwh",
    "emissions_kg",
]


def _log_cost(image_path, model, class_name, confidence, latency_s, energy_kwh, emissions_kg):
    COST_LOG.parent.mkdir(parents=True, exist_ok=True)
    new_file = not COST_LOG.exists()
    with COST_LOG.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COST_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "image": str(image_path),
                "model": getattr(model, "name", type(model).__name__),
                "class": class_name,
                "confidence": confidence,
                "latency_s": latency_s,
                "energy_kwh": energy_kwh,
                "emissions_kg": emissions_kg,
            }
        )


def _load_keras(path):
    try:
        return tf.keras.models.load_model(path, compile=False)
    except TypeError:
        # Modelos de versiones anteriores de Keras incluyen quantization_config, que la actual rechaza.
        with zipfile.ZipFile(path) as archive:
            config = json.loads(archive.read("config.json"))

        def drop_quantization(value):
            if isinstance(value, dict):
                value.pop("quantization_config", None)
                for item in value.values():
                    drop_quantization(item)
            elif isinstance(value, list):
                for item in value:
                    drop_quantization(item)

        drop_quantization(config)
        config.pop("compile_config", None)
        model = tf.keras.models.model_from_json(json.dumps(config))
        model.load_weights(path)
        return model


def _latest_model_uri():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"No existe el experimento {MLFLOW_EXPERIMENT_NAME}")

    client = MlflowClient()
    logged_models = client.search_logged_models([experiment.experiment_id])
    if logged_models:
        latest = max(logged_models, key=lambda model: model.creation_timestamp)
        local = ROOT / "mlruns" / str(experiment.experiment_id) / "models" / latest.model_id / "artifacts"
        # Los modelos de Colab apuntan a /content; usar la copia local cuando exista.
        return str(local) if local.is_dir() else latest.model_uri

    runs = mlflow.search_runs(
        [experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise ValueError("No hay modelos entrenados en MLflow")
    return f"runs:/{runs.iloc[0]['run_id']}/model"


def load_model(model_uri=None):
    """Carga el último modelo del experimento MLflow o uno dado (URI o ruta .keras)."""
    if model_uri is None:
        model_uri = _latest_model_uri()

    path = Path(model_uri)
    if path.is_dir() and not (path / "model.keras").exists():
        path = path / "data" / "model.keras"
    if path.exists():
        return _load_keras(path)
    return mlflow.tensorflow.load_model(model_uri)


def _class_names(model):
    num_classes = model.output_shape[-1]
    color_dir = ROOT / PLANTVILLAGE["local_path"] / PLANTVILLAGE["image_subfolder"]
    names = sorted(p.name for p in color_dir.iterdir() if p.is_dir()) if color_dir.is_dir() else []
    if len(names) != num_classes:
        names = list(PLANTVILLAGE_CLASSES)
    return names if len(names) == num_classes else [str(i) for i in range(num_classes)]


def grad_cam(model, image_batch, class_index):
    """Mapa de calor [0, 1] del tamaño de entrada para la clase indicada."""
    conv_layers = []

    def collect(current):
        for layer in current.layers:
            if isinstance(layer, tf.keras.Model):
                collect(layer)
            elif len(layer.output.shape) == 4:
                conv_layers.append(layer)

    collect(model)
    layer = conv_layers[-1]

    # Keras 3 no expone las activaciones intermedias de modelos anidados.
    captured = {}
    original_call = layer.call

    def capture(*args, **kwargs):
        output = original_call(*args, **kwargs)
        captured["features"] = output
        return output

    layer.call = capture
    try:
        with tf.GradientTape() as tape:
            predictions = model(image_batch, training=False)
            features = captured["features"]
            score = tf.reduce_sum(tf.cast(predictions[:, class_index], tf.float32))
    finally:
        layer.call = original_call

    gradients = tape.gradient(score, features)
    weights = tf.reduce_mean(tf.cast(gradients, tf.float32), axis=(1, 2))
    cam = tf.reduce_sum(tf.cast(features, tf.float32) * weights[:, None, None, :], axis=-1)
    cam = tf.nn.relu(cam)[0]
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return tf.image.resize(cam[..., None], IMAGE_TARGET_SIZE)[..., 0].numpy()


def predict(image_path, model, class_names=None):
    """Devuelve ``(clase, confianza, mapa_de_calor)`` y registra el costo de la inferencia."""
    tracker = EmissionsTracker(save_to_file=False, log_level="ERROR")
    tracker.start()
    started = time.perf_counter()
    try:
        image = tf.keras.utils.load_img(image_path, target_size=IMAGE_TARGET_SIZE)
        image_batch = np.expand_dims(np.asarray(image), 0)
        scores = tf.nn.softmax(tf.cast(model(image_batch, training=False), tf.float32))[0].numpy()
        index = int(np.argmax(scores))
        names = list(class_names) if class_names is not None else _class_names(model)
        heatmap = grad_cam(model, image_batch, index)
        class_name, confidence = names[index], float(scores[index])
    finally:
        latency_s = time.perf_counter() - started
        emissions_kg = tracker.stop() or 0.0

    energy_kwh = getattr(getattr(tracker, "_total_energy", None), "kWh", None)
    _log_cost(image_path, model, class_name, confidence, latency_s, energy_kwh, emissions_kg)
    return class_name, confidence, heatmap


def show_prediction(image_path, class_name, confidence, heatmap):
    """Muestra la imagen original y la zona detectada superpuesta."""
    image = tf.keras.utils.load_img(image_path)
    fig, (ax_original, ax_heatmap) = plt.subplots(1, 2, figsize=(10, 5))
    ax_original.imshow(image)
    ax_original.set_title("Original")
    ax_heatmap.imshow(image)
    ax_heatmap.imshow(heatmap, cmap="jet", alpha=0.5)
    ax_heatmap.set_title(f"{class_name} ({confidence:.2%})")
    for ax in (ax_original, ax_heatmap):
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Inferencia con mapa de calor Grad-CAM")
    parser.add_argument("image", help="ruta de la imagen a clasificar")
    parser.add_argument("--model", help="URI de MLflow o ruta a un modelo .keras")
    args = parser.parse_args()

    model = load_model(args.model)
    class_name, confidence, heatmap = predict(args.image, model)
    print(f"{class_name}: {confidence:.2%}")
    print(f"Costo guardado en {COST_LOG}")
    show_prediction(args.image, class_name, confidence, heatmap)
