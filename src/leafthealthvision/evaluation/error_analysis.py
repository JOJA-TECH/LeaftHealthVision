"""Artefactos para inspeccionar errores; no infiere sus causas."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def save_misclassifications(
    output_dir: Path,
    paths: list[str],
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    max_images: int = 36,
) -> None:
    """Guarda todas las rutas erróneas y una cuadrícula de ejemplos para revisión humana."""
    predictions = probabilities.argmax(axis=1)
    incorrect = np.flatnonzero(predictions != y_true)
    with (output_dir / "misclassifications.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "true_class", "predicted_class", "confidence"])
        for index in incorrect:
            writer.writerow([paths[index], class_names[y_true[index]], class_names[predictions[index]], float(probabilities[index, predictions[index]])])
    if not len(incorrect):
        return
    selected = incorrect[:max_images]
    columns = 6
    rows = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(18, 3 * rows), squeeze=False)
    for axis, index in zip(axes.flat, selected):
        with Image.open(paths[index]) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(f"real: {class_names[y_true[index]]}\npred: {class_names[predictions[index]]} ({probabilities[index, predictions[index]]:.2f})", fontsize=7)
        axis.axis("off")
    for axis in axes.flat[len(selected):]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "misclassifications.png", dpi=180)
    plt.close(figure)
