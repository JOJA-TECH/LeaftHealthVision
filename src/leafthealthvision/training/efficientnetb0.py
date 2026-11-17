import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suprime avisos informativos de TensorFlow (oneDNN, absl)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # forzar CPU (sin depender de GPU/cuDNN)

import numpy as np
import platform
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf

import mlflow
from codecarbon import EmissionsTracker

from tensorflow import keras
from tensorflow.keras import layers, models, applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from leafthealthvision.data import load_datasets
from leafthealthvision.config.settings import EARLY_STOPPING_PATIENCE, IMAGE_TARGET_SIZE, OPTIMIZER, METRICS


def _system_specs() -> dict:
    """Especificaciones del equipo para dejar como evidencia del run."""
    specs = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "cpu_logical_cores": os.cpu_count(),
        "cpu_name": platform.processor() or "desconocido",
        "gpus_visible_to_tf": [d.name for d in tf.config.list_physical_devices("GPU")],
    }
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        specs["gpu_nvidia_smi"] = out.splitlines()[0] if out else None
    except Exception:  # noqa: BLE001
        specs["gpu_nvidia_smi"] = None
    return specs


train_ds, validation_ds, test_ds, class_names = load_datasets()

img_height, img_width = IMAGE_TARGET_SIZE
num_classes = len(class_names)

# --- Transfer learning: base pre-entrenado EfficientNetB0 ---
base_model = applications.EfficientNetB0(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet',
)

# Congelar la base: solo se entrena la cabeza nueva
base_model.trainable = False

inputs = Input(shape=(img_height, img_width, 3))
# A diferencia de MobileNetV2, EfficientNetB0 ya normaliza internamente:
# no se añade layers.Rescaling(IMAGE_RESCALE).
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(num_classes)(x)
model = Model(inputs, outputs)

model.compile(optimizer=OPTIMIZER,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=METRICS)

model.summary()


epochs = int(os.environ.get("EFFICIENTNETB0_EPOCHS", "10"))
steps_per_epoch = os.environ.get("EFFICIENTNETB0_STEPS_PER_EPOCH")
validation_steps = os.environ.get("EFFICIENTNETB0_VALIDATION_STEPS")

early_stop = tf.keras.callbacks.EarlyStopping(
    patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True
)

specs = _system_specs()
print("=== Especificaciones del equipo (evidencia) ===")
for key, value in specs.items():
    print(f"{key}: {value}")

tracker = EmissionsTracker(
    project_name="leafthealthvision-efficientnetb0",
    output_dir=".",
    gpu_ids=[],  # la MX330 en WDDM no expone power/utilization vía NVML
)
tracker.start()
t0 = time.perf_counter()

fit_kwargs = dict(
    x=train_ds,
    validation_data=validation_ds,
    epochs=epochs,
    callbacks=[early_stop],
)
if steps_per_epoch:
    fit_kwargs["steps_per_epoch"] = int(steps_per_epoch)
if validation_steps:
    fit_kwargs["validation_steps"] = int(validation_steps)

history = model.fit(**fit_kwargs)

elapsed_seconds = time.perf_counter() - t0
emissions_kg_co2 = tracker.stop()

print("=== Evidencia de consumo e huella de carbono ===")
print(f"tiempo_entrenamiento_segundos: {elapsed_seconds:.2f}")
print(f"emisiones_co2_kg: {emissions_kg_co2:.6f}")
print(f"epocas_ejecutadas: {epochs}")
print(f"clases: {num_classes}")

experiments_dir = Path(__file__).resolve().parents[1] / "experiments" / time.strftime("%Y%m%d_%H%M%S")
experiments_dir.mkdir(parents=True, exist_ok=True)

evidence = {
    "system_specs": specs,
    "tiempo_entrenamiento_segundos": round(elapsed_seconds, 2),
    "emisiones_co2_kg": round(emissions_kg_co2, 6),
    "epocas": epochs,
    "clases": num_classes,
    "architecture": "EfficientNetB0",
    "extractor_frozen": True,
}
evidence_path = experiments_dir / "evidence_run.json"
evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plot_path = experiments_dir / "training_history.png"
plt.savefig(plot_path, dpi=150)
print(f"Grafico guardado en: {plot_path}")
print(f"Evidencia guardada en: {evidence_path}")

# --- MLflow: dejar evidencia de la corrida ---
mlflow.set_tracking_uri("file://./mlruns")
mlflow.set_experiment("leafthealthvision-efficientnetb0")

with mlflow.start_run(run_name="efficientnetb0_baseline"):
    mlflow.log_params(
        {
            "architecture": "EfficientNetB0",
            "weights": "imagenet",
            "include_preprocessing": True,
            "extractor_frozen": True,
            "variant": "color",
            "resolution": f"{IMAGE_TARGET_SIZE[0]}x{IMAGE_TARGET_SIZE[1]}",
            "channels": "RGB",
            "optimizer": OPTIMIZER,
            "loss": "SparseCategoricalCrossentropy(from_logits=True)",
            "metrics": METRICS,
            "seed": 42,
            "num_classes": num_classes,
            "epochs": epochs,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "augmentation": "ninguna (solo cabeza entrenable)",
        }
    )
    mlflow.log_metrics(
        {
            "tiempo_entrenamiento_segundos": round(elapsed_seconds, 2),
            "emisiones_co2_kg": round(emissions_kg_co2, 6),
            "epocas_ejecutadas": float(epochs),
            "clases": float(num_classes),
            "train_accuracy_final": float(acc[-1]) if acc else None,
            "val_accuracy_final": float(val_acc[-1]) if val_acc else None,
            "train_loss_final": float(loss[-1]) if loss else None,
            "val_loss_final": float(val_loss[-1]) if val_loss else None,
        }
    )
    mlflow.log_artifacts(str(experiments_dir), artifact_path="experiment")
    print(f"MLflow run guardado en: {mlflow.get_tracking_uri()}")

# --- Guardar modelo entrenado, clases y evaluación en test ---
model_path = experiments_dir / "model.keras"
model.save(model_path)

test_metrics = model.evaluate(test_ds, return_dict=True)
print("Métricas de test:", test_metrics)

np.save(experiments_dir / "class_names.npy", np.array(class_names))

evidence.update({
    "test_metrics": test_metrics,
    "model_path": str(model_path),
    "class_names_path": str(experiments_dir / "class_names.npy"),
})
evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")

mlflow.log_metrics({f"test_{k}": float(v) for k, v in test_metrics.items()})
mlflow.log_artifacts(str(experiments_dir), artifact_path="experiment")

print(f"Modelo guardado en: {model_path}")
print(f"Clases guardadas en: {experiments_dir / 'class_names.npy'}")
print("Entrenamiento completo.")
