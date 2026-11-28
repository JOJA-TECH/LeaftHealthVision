"""Evaluación adicional en PlantDoc para clases compatibles con PlantVillage.

PlantDoc se usa SOLO como evaluación externa de cambio de dominio, nunca para
entrenar ni para ajustar hiperparámetros. Las métricas se calculan únicamente
sobre las clases que admitan un mapeo justificable a las clases de PlantVillage;
el resto se reporta como fuera de alcance.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from leafthealthvision.config.settings import IMAGE_TARGET_SIZE
from leafthealthvision.plantvillage import load_splits


def _clases_plantvillage(manifest_dir: Path) -> list[str]:
    """Devuelve los nombres de clase de PlantVillage desde los manifiestos."""
    root = Path(__file__).resolve().parents[3]
    splits = load_splits(manifest_dir, root / "data" / "plantvillage" / "raw/color")
    return splits["train"].class_names


def cargar_mapeo_plantdoc(mapeo_path: Path, class_names: list[str]) -> dict[str, int]:
    """Carga el mapeo etiqueta_PlantDoc -> indice_clase_PlantVillage.

    Formato CSV: etiqueta_plantdoc,clase_plantvillage
    Debe ser proporcionado y verificado por el equipo (no se infiere
    automáticamente para evitar mapeos incorrectos).
    """
    mapeo_path = Path(mapeo_path)
    if not mapeo_path.exists():
        raise FileNotFoundError(
            f"No existe el archivo de mapeo PlantDoc: {mapeo_path}. "
            "El equipo debe proporcionar plantdoc_a_plantvillage.csv con las "
            "correspondencias verificadas antes de evaluar en PlantDoc."
        )
    class_to_index = {n: i for i, n in enumerate(class_names)}
    mapeo = {}
    with mapeo_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            etiqueta = row["etiqueta_plantdoc"].strip()
            clase = row["clase_plantvillage"].strip()
            if clase not in class_to_index:
                raise ValueError(
                    f"Clase PlantVillage desconocida en mapeo: {clase}"
                )
            mapeo[etiqueta] = class_to_index[clase]
    return mapeo


def _listar_imagenes_plantdoc(plantdoc_dir: Path, mapeo: dict[str, int]):
    """Lista (ruta, indice_clase_plantvillage) para imágenes cuya etiqueta esté mapeada.

    Asume estructura plantdoc_dir/<etiqueta>/<imagen>. El equipo debe adaptar
    si la estructura de PlantDoc difiere.
    """
    plantdoc_dir = Path(plantdoc_dir)
    if not plantdoc_dir.exists():
        raise FileNotFoundError(
            f"No existe el directorio PlantDoc: {plantdoc_dir}. "
            "Descarga PlantDoc y verifica su licencia antes de evaluar."
        )

    items = []
    for etiqueta_dir in sorted(plantdoc_dir.iterdir()):
        if not etiqueta_dir.is_dir():
            continue
        etiqueta = etiqueta_dir.name
        if etiqueta not in mapeo:
            continue  # clase fuera de alcance: no se evalúa
        idx_clase = mapeo[etiqueta]
        for img_path in etiqueta_dir.iterdir():
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                items.append((img_path, idx_clase))
    return items


def evaluar_en_plantdoc(
    model,
    manifest_dir: Path,
    plantdoc_dir: Path,
    mapeo_path: Path,
    run_dir: Path,
    class_names: list[str] | None = None,
    batch_size: int = 32,
) -> dict:
    """Evalúa el modelo en PlantDoc solo para clases mapeadas.

    Devuelve un diccionario de métricas sobre clases compatibles y guarda
    artefactos en run_dir/plantdoc/. Bloquea si faltan PlantDoc o el mapeo.
    """
    root = Path(__file__).resolve().parents[3]
    class_names = class_names or _clases_plantvillage(root / "data" / "splits")
    run_dir = Path(run_dir)
    plantdoc_out = run_dir / "plantdoc"
    plantdoc_out.mkdir(parents=True, exist_ok=True)

    mapeo = cargar_mapeo_plantdoc(mapeo_path, class_names)
    items = _listar_imagenes_plantdoc(plantdoc_dir, mapeo)

    if not items:
        res = {
            "plantdoc_n_clases_compatibles": len(set(mapeo.values())),
            "plantdoc_n_imagenes_evaluadas": 0,
            "plantdoc_nota": "No hay imágenes evaluables con el mapeo proporcionado.",
        }
        with (plantdoc_out / "plantdoc_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print("PlantDoc: 0 imágenes evaluables (verificar mapeo y directorio).")
        return res

    y_true, y_pred, confs, archivos = [], [], [], []
    for img_path, idx_clase in items:
        image = tf.keras.utils.load_img(str(img_path), target_size=IMAGE_TARGET_SIZE)
        batch = np.expand_dims(np.asarray(image), 0).astype(np.float32)
        logits = model.predict(batch, verbose=0)
        probs = tf.nn.softmax(logits).numpy()[0]
        pred = int(np.argmax(probs))
        y_true.append(idx_clase)
        y_pred.append(pred)
        confs.append(float(np.max(probs)))
        archivos.append(str(img_path))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    clases_presentes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    nombres_clases = [class_names[i] for i in clases_presentes]

    res = {
        "plantdoc_n_clases_compatibles": len(set(mapeo.values())),
        "plantdoc_n_clases_evaluadas": len(clases_presentes),
        "plantdoc_n_imagenes_evaluadas": len(items),
        "plantdoc_accuracy": float(accuracy_score(y_true, y_pred)),
        "plantdoc_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "plantdoc_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "plantdoc_weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "plantdoc_clases_evaluadas": nombres_clases,
    }

    # Reporte por clase (sobre clases presentes en PlantDoc evaluables)
    report_dict = classification_report(
        y_true, y_pred, target_names=nombres_clases, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, target_names=nombres_clases, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=clases_presentes)

    with (plantdoc_out / "plantdoc_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    with (plantdoc_out / "plantdoc_classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(report_text)
    with (plantdoc_out / "plantdoc_classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)

    import csv as _csv
    with (plantdoc_out / "plantdoc_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow([""] + nombres_clases)
        for i, row in enumerate(cm):
            w.writerow([nombres_clases[i]] + [int(v) for v in row])

    print(
        f"PlantDoc evaluado: {len(items)} imágenes, {len(clases_presentes)} clases, "
        f"macro-F1={res['plantdoc_macro_f1']:.4f}"
    )
    return res
