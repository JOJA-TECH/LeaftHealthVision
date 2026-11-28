"""Artefactos para inspeccionar errores; no infiere causas ni sustituye la revisión humana."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REVIEW_COLUMNS = [
    "path",
    "true_class",
    "predicted_class",
    "confidence",
    "true_probability",
    "tipo_error",
    "banda_confianza",
    "review_status",
    "human_hypothesis",
    "reviewed_by",
    "review_notes",
]


def _relationship(true_class: str, predicted_class: str) -> str:
    """Tipología objetiva de la confusión; no afirma ninguna causa."""
    true_species = true_class.split("___", 1)[0]
    predicted_species = predicted_class.split("___", 1)[0]
    if true_species != predicted_species:
        return "especies_distintas"
    true_healthy = "healthy" in true_class.lower()
    predicted_healthy = "healthy" in predicted_class.lower()
    if true_healthy != predicted_healthy:
        return "sano_vs_enfermo"
    return "misma_especie"


def _confidence_band(confidence: float) -> str:
    if confidence >= 0.90:
        return "alta"
    if confidence >= 0.50:
        return "media"
    return "baja"


def _review_row(
    path: str,
    true_class: str,
    predicted_class: str,
    confidence: float,
    true_probability: float,
) -> dict:
    return {
        "path": path,
        "true_class": true_class,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "true_probability": true_probability,
        "tipo_error": _relationship(true_class, predicted_class),
        "banda_confianza": _confidence_band(confidence),
        "review_status": "pendiente",
        "human_hypothesis": "",
        "reviewed_by": "",
        "review_notes": "",
    }


def _write_review_guide(output_dir: Path) -> None:
    (output_dir / "error_review_guide.md").write_text(
        "# Revisión de errores\n\n"
        "He guardado en `misclassifications.csv` todos los errores del test y en "
        "`error_review_queue.csv` la cola ordenada por confianza descendente.\n\n"
        "Para cada imagen revisada, completo `human_hypothesis`, `reviewed_by` y "
        "`review_notes`, y cambio `review_status` a `revisado`. Reviso al menos 10 "
        "errores reales antes de entregar el trabajo.\n",
        encoding="utf-8",
    )


def save_misclassifications(
    output_dir: Path,
    paths: list[str],
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    max_images: int = 36,
    max_review: int = 20,
) -> None:
    """Guarda todos los errores, una cola de revisión humana y una galería."""
    predictions = probabilities.argmax(axis=1)
    incorrect = np.flatnonzero(predictions != y_true)
    rows = [
        _review_row(
            paths[index],
            class_names[y_true[index]],
            class_names[predictions[index]],
            float(probabilities[index, predictions[index]]),
            float(probabilities[index, y_true[index]]),
        )
        for index in incorrect
    ]
    with (output_dir / "misclassifications.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    queue = sorted(rows, key=lambda row: row["confidence"], reverse=True)[:max_review]
    with (output_dir / "error_review_queue.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(queue)
    _write_review_guide(output_dir)
    if not len(incorrect):
        return
    selected = incorrect[:max_images]
    columns = 6
    rows_count = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(rows_count, columns, figsize=(18, 3 * rows_count), squeeze=False)
    for axis, index in zip(axes.flat, selected):
        with Image.open(paths[index]) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(
            f"real: {class_names[y_true[index]]}\n"
            f"pred: {class_names[predictions[index]]} ({probabilities[index, predictions[index]]:.2f})",
            fontsize=7,
        )
        axis.axis("off")
    for axis in axes.flat[len(selected):]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "misclassifications.png", dpi=180)
    plt.close(figure)
