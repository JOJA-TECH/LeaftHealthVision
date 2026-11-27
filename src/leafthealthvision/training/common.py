"""Piezas compartidas entre los scripts de entrenamiento (baseline y fine-tuning)."""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import subprocess
from collections import Counter
from pathlib import Path

import mlflow
import numpy as np
import sklearn
import tensorflow as tf
import yaml

from leafthealthvision.evaluation.error_analysis import save_misclassifications
from leafthealthvision.evaluation.metrics import save_evaluation_artifacts
from leafthealthvision.plantvillage import (
    audit_manifests,
    load_splits,
    make_dataset,
    sha256_file,
    validate_manifest_splits,
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("dataset", {}).get("variant") != "color":
        raise ValueError("Este pipeline fue solicitado para PlantVillage raw/color.")
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    # Reproducibilidad práctica; puede reducir rendimiento en algunas GPU.
    tf.config.experimental.enable_op_determinism()


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def system_specs() -> dict:
    """Especificaciones del equipo y versiones que afectan la reproducibilidad."""
    specs = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": tf.keras.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "mlflow": mlflow.__version__,
        "cpu_logical_cores": os.cpu_count(),
        "cpu_name": platform.processor() or "desconocido",
        "gpus_visible_to_tf": [device.name for device in tf.config.list_physical_devices("GPU")],
    }
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        specs["gpu_nvidia_smi"] = out.splitlines()[0] if out else None
    except Exception:  # noqa: BLE001
        specs["gpu_nvidia_smi"] = None
    return specs


def checkpoint_sha256(path: Path) -> str | None:
    return sha256_file(path) if path.exists() else None


def resolve_output_dir(config: dict, root: Path, stage: str, seed: int) -> Path:
    """Permite aislar corridas sintéticas sin pisar los artefactos reales."""
    configured = config.get("output_dir")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else root / path
    return root / "experiments" / "efficientnetb0" / stage / f"seed_{seed}"


def prepare_output_dir(path: Path) -> None:
    """No sobrescribe una corrida existente; evita mezclar artefactos de dos ejecuciones."""
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"{path} ya contiene artefactos. Usa otro output_dir o elimina esa corrida de forma explícita."
        )
    path.mkdir(parents=True, exist_ok=True)


def save_split_evidence(output_dir: Path, manifest_dir: Path, splits: dict) -> dict:
    """Guarda hashes, conteos reales, auditoría y filas no resueltas del split."""
    audit = audit_manifests(manifest_dir)
    audit["resolved_counts"] = {name: len(records.paths) for name, records in splits.items()}
    audit["unresolved_counts"] = {name: len(records.unresolved_image_ids) for name, records in splits.items()}
    (output_dir / "split_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit


def save_class_weights(
    output_dir: Path,
    labels: list[int],
    weights: dict[int, float],
    class_names: list[str],
) -> list[dict]:
    """Deja evidencia de que los pesos salen solo de train."""
    counts = Counter(labels)
    rows = [
        {
            "class_id": class_id,
            "class_name": class_names[class_id],
            "train_count": counts[class_id],
            "class_weight": weights[class_id],
        }
        for class_id in range(len(class_names))
    ]
    with (output_dir / "class_weights.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def save_training_history(output_dir: Path, history) -> dict[str, list[float]]:
    """Guarda el historial y las curvas de entrenamiento como artefactos."""
    data = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    (output_dir / "training_history.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    if "loss" in data and "val_loss" in data:
        import matplotlib.pyplot as plt

        epochs = range(1, len(data["loss"]) + 1)
        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs, data["loss"], label="train")
        axes[0].plot(epochs, data["val_loss"], label="val")
        axes[0].set(title="Pérdida", xlabel="Época", ylabel="loss")
        axes[0].legend()
        if "accuracy" in data and "val_accuracy" in data:
            axes[1].plot(epochs, data["accuracy"], label="train")
            axes[1].plot(epochs, data["val_accuracy"], label="val")
            axes[1].set(title="Accuracy", xlabel="Época", ylabel="accuracy")
            axes[1].legend()
        else:
            axes[1].axis("off")
        figure.tight_layout()
        figure.savefig(output_dir / "training_curves.png", dpi=150)
        plt.close(figure)
    return data


class MlflowEpochLogger(tf.keras.callbacks.Callback):
    """Registra las métricas de validación/train de cada época en el run activo."""

    def on_epoch_end(self, epoch, logs=None):
        if mlflow.active_run() is None:
            return
        values = {
            key: float(value)
            for key, value in (logs or {}).items()
            if isinstance(value, (int, float, np.floating))
        }
        if values:
            mlflow.log_metrics(values, step=epoch)


def build_datasets(config: dict, seed: int) -> tuple[dict, dict, list[str]]:
    """Manifiestos -> tf.data.Dataset para train/val/test, sin re-dividir el dataset."""
    root = project_root()
    dataset = config["dataset"]
    manifest_dir = root / dataset["manifest_dir"]
    validate_manifest_splits(manifest_dir)
    splits = load_splits(manifest_dir, root / dataset["color_dir"])
    image_size = tuple(config["model"]["image_size"])
    batch_size = int(config["training"]["batch_size"])
    datasets = {
        name: make_dataset(records, image_size, batch_size, name == "train", seed)
        for name, records in splits.items()
    }
    return splits, datasets, splits["train"].class_names


def save_unresolved(output_dir: Path, splits: dict) -> list[dict]:
    """Persiste las filas del manifiesto sin equivalente en color; nunca se omiten en silencio."""
    unresolved = [
        {"split": split_name, "manifest_image": image_id}
        for split_name, records in splits.items()
        for image_id in records.unresolved_image_ids
    ]
    if unresolved:
        with (output_dir / "unresolved_color_manifest_rows.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["split", "manifest_image"])
            writer.writeheader()
            writer.writerows(unresolved)
    return unresolved


def training_callbacks(output_dir: Path, patience: int) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(output_dir / "best_model.keras", monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.CSVLogger(output_dir / "keras_training_log.csv"),
        tf.keras.callbacks.TerminateOnNaN(),
        MlflowEpochLogger(),
    ]


def optional_fit_kwargs(config: dict) -> dict:
    """Limitadores opcionales de pasos por época; útiles en hardware limitado y pruebas."""
    training = config["training"]
    return {key: int(training[key]) for key in ("steps_per_epoch", "validation_steps") if training.get(key)}


def stop_emissions(tracker) -> float:
    """La medición de carbono nunca debe romper una corrida ya terminada."""
    try:
        return float(tracker.stop() or 0.0)
    except Exception:  # noqa: BLE001
        return 0.0


def evaluate_and_save(output_dir: Path, splits: dict, probabilities, class_names: list[str]) -> dict[str, float]:
    """Métricas multiclase, matriz de confusión y ejemplos mal clasificados sobre el test del manifiesto."""
    true_labels = np.asarray(splits["test"].labels)
    metrics = save_evaluation_artifacts(output_dir, true_labels, probabilities, class_names)
    save_misclassifications(output_dir, splits["test"].paths, true_labels, probabilities, class_names)
    return metrics
