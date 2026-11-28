"""Métricas y artefactos de evaluación multiclase."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_predictions(
    y_true: np.ndarray, y_probabilities: np.ndarray, class_names: list[str]
) -> tuple[dict[str, float], np.ndarray, dict]:
    """Calcula accuracy y promedios macro y weighted sin ocultar el desbalance."""
    if len(y_true) != len(y_probabilities):
        raise ValueError("y_true y y_probabilities deben tener el mismo número de filas")
    if y_probabilities.ndim != 2 or y_probabilities.shape[1] != len(class_names):
        raise ValueError("y_probabilities debe tener forma (n, n_clases)")
    y_pred = y_probabilities.argmax(axis=1)
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    metrics = {
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "n_test": int(len(y_true)),
        "precision_macro": float(macro[0]),
        "recall_macro": float(macro[1]),
        "f1_macro": float(macro[2]),
        "precision_weighted": float(weighted[0]),
        "recall_weighted": float(weighted[1]),
        "f1_weighted": float(weighted[2]),
    }
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    return metrics, confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names))), report


def save_evaluation_artifacts(
    output_dir: Path,
    y_true: np.ndarray,
    y_probabilities: np.ndarray,
    class_names: list[str],
) -> dict[str, float]:
    """Guarda métricas, reporte y matrices de confusión CSV/PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, matrix, report = evaluate_predictions(y_true, y_probabilities, class_names)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output_dir / "classification_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "precision", "recall", "f1-score", "support"])
        for label, values in report.items():
            if isinstance(values, dict):
                writer.writerow([label, values.get("precision"), values.get("recall"), values.get("f1-score"), values.get("support")])
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *class_names])
        writer.writerows([[name, *row] for name, row in zip(class_names, matrix)])
    figure, axis = plt.subplots(figsize=(16, 14))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)), xlabel="Predicción", ylabel="Clase real")
    axis.set_xticklabels(class_names, rotation=90, fontsize=6)
    axis.set_yticklabels(class_names, fontsize=6)
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(figure)
    return metrics
