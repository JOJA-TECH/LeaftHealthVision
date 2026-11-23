"""Pipeline reproducible de PlantVillage basado en manifiestos de particiones.

Los manifiestos se generaron sobre la variante ``segmented``. Esta utilidad
permite entrenar con ``color`` resolviendo el nombre equivalente de cada
imagen, sin volver a dividir el dataset.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SplitRecords:
    """Rutas, etiquetas y metadatos de una partición."""

    paths: list[str]
    labels: list[int]
    class_names: list[str]
    image_ids: list[str]
    unresolved_image_ids: list[str]


def normalized_image_name(filename: str) -> str:
    """Normaliza el sufijo de ``segmented`` conservando el UUID identificador.

    El texto posterior a ``___`` se repite dentro de varias categorías, por lo
    que quitar el UUID haría ambiguo el emparejamiento con ``color``.
    """
    stem = Path(filename).stem.lower()
    return stem.removesuffix("_final_masked")


def image_core_name(filename: str) -> str:
    """Nombre sin UUID, usado solo como alternativa cuando color no lo tiene."""
    normalized = normalized_image_name(filename)
    return normalized.split("___", 1)[-1]


def _color_index(color_dir: Path) -> tuple[dict[tuple[str, str], Path], dict[tuple[str, str], list[Path]]]:
    """Indexa las imágenes color por (categoría, nombre normalizado)."""
    index: dict[tuple[str, str], Path] = {}
    core_index: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for class_dir in sorted(path for path in color_dir.iterdir() if path.is_dir()):
        for image_path in class_dir.iterdir():
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            normalized = normalized_image_name(image_path.name)
            key = (class_dir.name, normalized)
            index[key] = image_path
            core_index[(class_dir.name, image_core_name(image_path.name))].append(image_path)
    return index, core_index


def load_manifest_split(manifest_path: Path, color_dir: Path) -> SplitRecords:
    """Carga un manifiesto y resuelve cada imagen a ``raw/color``.

    Una fila sin equivalente en color queda registrada en ``unresolved_image_ids``.
    El entrenamiento debe persistir ese artefacto; nunca se omite de forma
    silenciosa ni se sustituye por una imagen distinta.
    """
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"archivo", "categoria", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifiesto inválido: {manifest_path}")

    index, core_index = _color_index(color_dir)
    class_names = sorted({row["categoria"] for row in rows})
    class_to_id = {name: index for index, name in enumerate(class_names)}

    paths: list[str] = []
    labels: list[int] = []
    image_ids: list[str] = []
    unresolved: list[str] = []
    for row in rows:
        normalized = normalized_image_name(row["archivo"])
        key = (row["categoria"], normalized)
        path = index.get(key)
        if path is None:
            candidates = core_index.get((row["categoria"], image_core_name(row["archivo"])), [])
            if len(candidates) == 1:
                path = candidates[0]
        if path is None:
            unresolved.append(f"{key[0]}/{row['archivo']}")
            continue
        paths.append(str(path))
        labels.append(class_to_id[row["categoria"]])
        image_ids.append(row["archivo"])

    return SplitRecords(paths, labels, class_names, image_ids, unresolved)


def load_splits(manifest_dir: Path, color_dir: Path) -> dict[str, SplitRecords]:
    """Carga train/val/test y comprueba que preservan el mismo espacio de clases."""
    splits = {
        name: load_manifest_split(manifest_dir / f"{name}.csv", color_dir)
        for name in ("train", "val", "test")
    }
    expected = splits["train"].class_names
    for name, records in splits.items():
        if records.class_names != expected:
            raise ValueError(f"El manifiesto {name} no tiene las mismas clases que train")
    return splits


def class_weights(labels: Iterable[int], num_classes: int) -> dict[int, float]:
    """Calcula N/(K*n_clase) exclusivamente con el conjunto de entrenamiento."""
    counts = Counter(labels)
    missing = set(range(num_classes)) - set(counts)
    if missing:
        raise ValueError(f"Clases sin ejemplos en train: {sorted(missing)}")
    total = sum(counts.values())
    return {class_id: total / (num_classes * count) for class_id, count in counts.items()}


def _decode_image(tf: object, path: object, label: object) -> tuple[object, object]:
    image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    # EfficientNetB0(include_preprocessing=True) espera float32 en [0, 255].
    return tf.cast(image, tf.float32), label


def _resize(tf: object, image: object, label: object, image_size: tuple[int, int]) -> tuple[object, object]:
    return tf.image.resize(image, image_size, antialias=True), label


def make_dataset(
    records: SplitRecords,
    image_size: tuple[int, int],
    batch_size: int,
    training: bool,
    seed: int,
) -> object:
    """Construye un ``tf.data.Dataset`` sin reescalado adicional."""
    import tensorflow as tf

    dataset = tf.data.Dataset.from_tensor_slices((records.paths, records.labels))
    if training:
        dataset = dataset.shuffle(min(len(records.paths), 10_000), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda path, label: _decode_image(tf, path, label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.map(
        lambda image, label: _resize(tf, image, label, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
