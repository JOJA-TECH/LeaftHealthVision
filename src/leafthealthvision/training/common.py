"""Piezas compartidas entre los scripts de entrenamiento (baseline y fine-tuning)."""

from __future__ import annotations

import csv
import random
import subprocess
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

from leafthealthvision.evaluation.error_analysis import save_misclassifications
from leafthealthvision.evaluation.metrics import save_evaluation_artifacts
from leafthealthvision.plantvillage import load_splits, make_dataset


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


def build_datasets(config: dict, seed: int) -> tuple[dict, dict, list[str]]:
    """Manifiestos -> tf.data.Dataset para train/val/test, sin re-dividir el dataset."""
    root = project_root()
    dataset = config["dataset"]
    splits = load_splits(root / dataset["manifest_dir"], root / dataset["color_dir"])
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
