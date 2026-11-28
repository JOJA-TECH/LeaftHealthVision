"""Entrenamiento MobileNetV2 ImageNet (baseline congelado) sobre PlantVillage.

Carga la partición canónica desde manifiestos (`data/splits/{train,val,test}.csv`),
calcula pesos de clase EXCLUSIVAMENTE con train, aplica el preprocesamiento oficial
de MobileNetV2 (`preprocess_input`, [0,255] -> [-1,1]) como una capa Keras dentro
del modelo (una sola vez: entrenamiento == inferencia, sin doble aplicación), y
deja el test como evaluación final bloqueada.

Si los manifiestos acordados no están disponibles, levanta un error claro y NO
entrena ni vuelve a dividir por lotes (bloqueo deliberado).
"""

import argparse
import csv
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import tensorflow as tf
from codecarbon import EmissionsTracker
from keras import ops
from keras.saving import register_keras_serializable
from mlflow.tracking import MlflowClient
from tensorflow.keras import applications, layers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from leafthealthvision.config.settings import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EXPERIMENTS_DIR,
    FINE_TUNE_BASE_TRAINABLE,
    FINE_TUNE_ENABLED,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_FROZEN_LAYERS,
    FINE_TUNE_LEARNING_RATE,
    IMAGE_TARGET_SIZE,
    LEARNING_RATE,
    MANIFEST_DIR,
    METRICS,
    MLFLOW_ARTIFACT_LOCATION,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    OPTIMIZER,
    ROOT,
    SEED,
)
from leafthealthvision.data import load_datasets_from_manifests, set_deterministic_seed

# --- MLflow: seguimiento local de experimentos (SQLite -> mlflow.db) ---
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

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
            with __import__("sqlite3").connect(db_path) as con:
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


@register_keras_serializable(
    "leafthealthvision", name="MobileNetV2Preprocess"
)
class MobileNetV2Preprocess(layers.Layer):
    """Capa Keras que aplica el preprocesamiento oficial MobileNetV2: [0,255] -> [-1,1].

    `mobilenet_v2.preprocess_input` (modo TF) equivale numéricamente a
    ``x / 127.5 - 1`` (0 -> -1, 255 -> 1). Se implementa con operaciones de
    tensor Keras para que trace simbólicamente y se serialice al guardar el
    modelo. Va dentro del modelo una sola vez: entrenamiento e inferencia
    aplican el mismo preprocesado, sin doble aplicación.
    """

    def call(self, inputs):
        # Equivalente numérico a mobilenet_v2.preprocess_input (modo TF):
        # x / 127.5 - 1  (0 -> -1, 255 -> 1). keras.ops para trazar simbólicamente.
        return ops.divide(inputs, 127.5) - 1.0


def build_model(img_height, img_width, num_classes, base_trainable=False,
                frozen_layers=None, weights="imagenet"):
    """Construye MobileNetV2 con preprocesamiento interno y cabeza nueva.

    `preprocess_input` va dentro del modelo (capa Lambda), así el artefacto
    guardado recibe imágenes [0,255] y produce el mismo preprocesado en
    entrenamiento e inferencia.
    """
    base_model = applications.MobileNetV2(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        weights=weights,
    )
    base_model.trainable = base_trainable

    # Fine-tuning parcial opcional: descongelar desde una capa en adelante.
    if base_trainable and frozen_layers:
        for layer in base_model.layers:
            layer.trainable = layer.name in frozen_layers or _is_after(
                base_model.layers, layer, frozen_layers
            )

    inputs = Input(shape=(img_height, img_width, 3))
    x = MobileNetV2Preprocess(name="mobilenetv2_preprocess")(inputs)
    # training=True solo cuando la base es entrenable (fine-tuning); False en baseline congelado.
    x = base_model(x, training=base_trainable)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes)(x)  # logits (from_logits=True)
    model = Model(inputs, outputs)
    return model


def _is_after(all_layers, layer, frozen_layers):
    """Marca como entrenable las capas posteriores a la última de frozen_layers."""
    start = False
    for l in all_layers:
        if l.name == frozen_layers[-1]:
            start = True
        if start and l is not layer:
            return True
    return False


def _calcular_umbral_confianza(model, val_ds, class_names, percentil: float = 20.0) -> float:
    """Calcula un umbral de baja confianza SOLO a partir del conjunto de validación.

    Toma la probabilidad máxima de las predicciones correctas en validación y
    devuelve el percentil indicado (por defecto 20). Umbrales por debajo de este
    valor se marcan como "baja confianza" en el prototipo. Nunca se usa el test
    para fijar este umbral.
    """
    import numpy as np

    confianzas_aciertos = []
    for images, labels in val_ds:
        logits = model.predict(images, verbose=0)
        probs = tf.nn.softmax(logits).numpy()
        pred = np.argmax(probs, axis=1)
        conf = np.max(probs, axis=1)
        correct = pred == labels.numpy()
        confianzas_aciertos.extend(conf[correct].tolist())

    if not confianzas_aciertos:
        return 0.5
    return float(np.percentile(confianzas_aciertos, percentil))


def _save_artifacts(run_dir, variant, seed, model, history, eval_metrics,
                    confusion_matrix, class_report, class_names,
                    manifest_info, class_weights, commit_sha, versions, hardware,
                    umbral_confianza: float | None = None):
    """Guarda modelo, configuración efectiva, métricas, matriz, reporte e historial."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Modelo Keras
    model.save(run_dir / "model.keras")

    # Configuración efectiva
    config = {
        "variant": variant,
        "seed": seed,
        "model": "mobilenetv2",
        "base_model_weights": "imagenet",
        "base_trainable": bool(FINE_TUNE_ENABLED and FINE_TUNE_BASE_TRAINABLE),
        "fine_tuned": bool(FINE_TUNE_ENABLED),
        "epochs": FINE_TUNE_EPOCHS if FINE_TUNE_ENABLED else 10,
        "batch_size": BATCH_SIZE,
        "img_height": IMAGE_TARGET_SIZE[0],
        "img_width": IMAGE_TARGET_SIZE[1],
        "preprocess": "mobilenet_v2.preprocess_input ([0,255]->[-1,1], una vez dentro del modelo)",
        "image_rescale": None,  # no se usa layers.Rescaling para MobileNetV2
        "optimizer": OPTIMIZER,
        "learning_rate": FINE_TUNE_LEARNING_RATE if FINE_TUNE_ENABLED else LEARNING_RATE,
        "loss": "SparseCategoricalCrossentropy(from_logits=True)",
        "metrics": ",".join(METRICS),
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "class_weight_mode": "train_only",
        "class_weights": {str(k): float(v) for k, v in class_weights.items()},
        "num_classes": len(class_names),
        "class_names": list(class_names),
        "manifest_dir": str(MANIFEST_DIR),
        "manifest_sha256": manifest_info.get("manifest_sha256"),
        "split_sizes": manifest_info.get("splits"),
        "commit": commit_sha,
        "versions": versions,
        "hardware": hardware,
        "umbral_confianza_validacion": (
            float(umbral_confianza) if umbral_confianza is not None else None
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with (run_dir / "config_effective.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # metrics.json
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2, ensure_ascii=False)

    # Matriz de confusión: CSV + imagen
    import numpy as np
    cm = np.asarray(confusion_matrix)
    csv_path = run_dir / "confusion_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + list(class_names))
        for i, row in enumerate(cm):
            writer.writerow([class_names[i]] + [int(v) for v in row])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Matriz de confusión (test)")
        plt.colorbar()
        ticks = np.arange(len(class_names))
        plt.xticks(ticks, class_names, rotation=90, fontsize=6)
        plt.yticks(ticks, class_names, fontsize=6)
        plt.ylabel("Clase real")
        plt.xlabel("Clase predicha")
        plt.tight_layout()
        plt.savefig(run_dir / "confusion_matrix.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        print(f"No se pudo generar la imagen de la matriz de confusión: {exc}")

    # Reporte por clase (texto + JSON)
    with (run_dir / "classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(class_report)
    with (run_dir / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(class_report if isinstance(class_report, dict) else {}, f, indent=2, ensure_ascii=False)

    # Historial de entrenamiento
    if history is not None:
        hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        with (run_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(hist, f, indent=2, ensure_ascii=False)

    print(f"Artefactos guardados en: {run_dir}")


def _evaluate_test(model, test_ds, class_names):
    """Evaluación final bloqueada en test: accuracy, macro-F1, matriz, reporte."""
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    y_true, y_pred = [], []
    for images, labels in test_ds:
        logits = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(logits, axis=1).tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    labels = list(range(len(class_names)))
    accuracy = float(accuracy_score(y_true, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    precision_macro, recall_macro, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report_dict = classification_report(
        y_true, y_pred, labels=labels, target_names=list(class_names),
        output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, labels=labels, target_names=list(class_names), zero_division=0
    )

    eval_metrics = {
        "test_accuracy": accuracy,
        "test_balanced_accuracy": balanced_acc,
        "test_macro_f1": macro_f1,
        "test_weighted_f1": weighted_f1,
        "test_macro_precision": float(precision_macro),
        "test_macro_recall": float(recall_macro),
        "test_weighted_precision": float(precision_weighted),
        "test_weighted_recall": float(recall_weighted),
        "test_n_samples": len(y_true),
        "test_num_classes": len(class_names),
    }
    return eval_metrics, cm, report_text, report_dict


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "desconocido"


def _versions_and_hardware() -> tuple[dict, dict]:
    import tensorflow as tf
    versions = {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": getattr(tf, "keras", None) and tf.keras.__version__,
    }
    try:
        import mlflow as _mlflow
        versions["mlflow"] = _mlflow.__version__
    except Exception:  # noqa: BLE001
        versions["mlflow"] = None
    hardware = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cuda_available": bool(tf.config.list_physical_devices("GPU")),
    }
    return versions, hardware


def main(variant: str = "frozen", seed: int = SEED, quick: bool = False,
         epochs_override: int | None = None):
    """Entrena MobileNetV2 y guarda artefactos. Bloquea si faltan manifiestos.

    Si ``quick`` es True (modo testeo), se usan pocas épocas y un subconjunto
    pequeño de los splits para que la corrida complete en 5-10 min en CPU.
    Nunca se usa este modo para resultados finales.
    """
    set_deterministic_seed(seed)

    try:
        train_ds, val_ds, test_ds, class_names, class_weights, manifest_info = (
            load_datasets_from_manifests(seed=seed, batch_size=BATCH_SIZE)
        )
    except FileNotFoundError as exc:
        print(f"[BLOQUEO] {exc}")
        print("No se ejecuta model.fit ni evaluación. Esperando manifiestos acordados.")
        return

    # --- Modo rápido para testeo: subset pequeño y pocas épocas ---
    if quick:
        train_ds = train_ds.take(20)   # ~640 imágenes de train
        val_ds = val_ds.take(5)        # ~160 imágenes de val
        test_ds = test_ds.take(5)      # ~160 imágenes de test
        epochs = epochs_override or 2
        print("[MODO RÁPIDO] Testeo end-to-end: 20/5/5 batches, "
              f"{epochs} época(s). No usar para resultados finales.")
    else:
        epochs = epochs_override or (FINE_TUNE_EPOCHS if FINE_TUNE_ENABLED else 10)

    img_height, img_width = IMAGE_TARGET_SIZE
    num_classes = len(class_names)

    base_trainable = bool(FINE_TUNE_ENABLED and FINE_TUNE_BASE_TRAINABLE)
    model = build_model(
        img_height, img_width, num_classes,
        base_trainable=base_trainable,
        frozen_layers=FINE_TUNE_FROZEN_LAYERS,
    )

    model.compile(
        optimizer=OPTIMIZER,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=METRICS,
    )
    model.summary()

    lr = FINE_TUNE_LEARNING_RATE if FINE_TUNE_ENABLED else LEARNING_RATE
    # Aplicar learning rate si el optimizador es string 'adam'.
    if OPTIMIZER == "adam":
        model.optimizer.learning_rate = lr

    early_stop = tf.keras.callbacks.EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True
    )

    versions, hardware = _versions_and_hardware()
    commit_sha = _git_commit()

    run_dir = EXPERIMENTS_DIR / "mobilenetv2" / variant / f"seed_{seed}"
    (run_dir / "results").mkdir(parents=True, exist_ok=True)

    tracker = EmissionsTracker(
        save_to_file=True, output_dir=str(run_dir / "results"), output_file="emissions"
    )

    with mlflow.start_run(run_name=f"mobilenetv2_{variant}_seed{seed}"):
        params = {
            "model": "mobilenetv2",
            "variant": variant,
            "base_model_weights": "imagenet",
            "base_trainable": base_trainable,
            "fine_tuned": bool(FINE_TUNE_ENABLED),
            "epochs": epochs,
            "batch_size": BATCH_SIZE,
            "img_height": img_height,
            "img_width": img_width,
            "preprocess": "mobilenet_v2.preprocess_input ([0,255]->[-1,1])",
            "optimizer": OPTIMIZER,
            "learning_rate": float(lr),
            "loss": "SparseCategoricalCrossentropy(from_logits=True)",
            "metrics": ",".join(METRICS),
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "class_weight_mode": "train_only",
            "num_classes": num_classes,
            "seed": seed,
            "commit": commit_sha,
        }
        mlflow.log_params(params)
        mlflow.log_params(manifest_info.get("split_sizes", {}))
        mlflow.log_param("manifest_sha256_train", manifest_info.get("manifest_sha256", {}).get("train"))

        tracker.start()
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=[early_stop, MlflowEpochLogger()],
        )
        emisiones_kg = tracker.stop() or 0.0

        # Curvas de entrenamiento
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        acc = history.history.get("accuracy", [])
        val_acc = history.history.get("val_accuracy", [])
        loss = history.history.get("loss", [])
        val_loss = history.history.get("val_loss", [])
        epochs_range = range(len(acc))
        fig = plt.figure(figsize=(8, 8))
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
        curves_path = run_dir / "training_curves.png"
        plt.savefig(curves_path, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(curves_path))

        # Evaluación final bloqueada en test
        eval_metrics, cm, report_text, _report_dict = _evaluate_test(
            model, test_ds, class_names
        )
        eval_metrics["carbon_emissions_kg"] = float(emisiones_kg)

        # Umbral de baja confianza definido SOLO con validación (nunca con test).
        umbral_confianza = _calcular_umbral_confianza(model, val_ds, class_names)
        eval_metrics["umbral_confianza_validacion"] = float(umbral_confianza)

        mlflow.log_metrics(eval_metrics)
        mlflow.log_param("umbral_confianza_validacion", float(umbral_confianza))

        # Guardar artefactos locales ANTES de subirlos a MLflow.
        _save_artifacts(
            run_dir, variant, seed, model, history, eval_metrics, cm, report_text,
            class_names, manifest_info, class_weights, commit_sha, versions, hardware,
            umbral_confianza=umbral_confianza,
        )

        mlflow.log_artifact(str(run_dir / "classification_report.txt"))
        mlflow.log_artifact(str(run_dir / "confusion_matrix.csv"))
        mlflow.log_artifact(str(run_dir / "confusion_matrix.png"))

        try:
            mlflow.tensorflow.log_model(model, name="model")
        except Exception as exc:  # noqa: BLE001
            print(f"No se pudo registrar el modelo en MLflow: {exc}")

        print(f"Huella de carbono del entrenamiento: {emisiones_kg:.6f} kg CO2eq")
        print(f"Test accuracy: {eval_metrics['test_accuracy']:.4f}")
        print(f"Test macro-F1: {eval_metrics['test_macro_f1']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrenamiento MobileNetV2 (baseline/fine-tuned) sobre PlantVillage."
    )
    parser.add_argument("--variant", default="frozen",
                        help="variante: 'frozen' (baseline) o 'finetuned'")
    parser.add_argument("--seed", type=int, default=SEED, help="semilla determinista")
    parser.add_argument("--quick", action="store_true",
                        help="modo testeo rápido: subset pequeño y pocas épocas (5-10 min en CPU). No usar para resultados finales.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="sobrescribir número de épocas")
    args = parser.parse_args()
    main(variant=args.variant, seed=args.seed, quick=args.quick,
         epochs_override=args.epochs)
