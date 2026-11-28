"""Carga y prepara los datasets de PlantVillage para entrenamiento, validación y prueba.

Ofrece dos caminos:

1. ``load_datasets()`` — partición por lotes (comportamiento histórico, se conserva
   para interoperabilidad con la rama `feature/efficentnetb0`). NO usa manifiestos.
2. ``load_datasets_from_manifests()`` — partición canónica compartida por manifiestos
   CSV (`data/splits/{train,val,test}.csv`). Es el camino que deben usar
   MobileNetV2 y EfficientNetB0 para ser comparables en Ruta F.

Los manifiestos no se versionan (datos voluminosos, `data/` en `.gitignore`). Si no
están disponibles, `load_datasets_from_manifests()` levanta un error claro y el
entrenamiento no arranca: nunca se vuelve a dividir por lotes en su lugar.
"""

import hashlib
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf

from leafthealthvision.config.settings import (
    BATCH_SIZE,
    IMAGE_TARGET_SIZE,
    MANIFEST_COLUMNS,
    MANIFEST_DIR,
    PLANTVILLAGE_CLASSES,
    ROOT,
    SEED,
)

AUTOTUNE = tf.data.AUTOTUNE

# Ruta histórica de imágenes a color (usada por load_datasets() y por inference.py).
DATA_DIR = ROOT / "data" / "plantvillage" / "raw/color"

# Índice de clase canónico: orden alfabético de PLANTVILLAGE_CLASSES.
CLASS_TO_INDEX = {name: i for i, name in enumerate(PLANTVILLAGE_CLASSES)}


def set_deterministic_seed(seed: int = SEED) -> None:
    """Fija semillas de Python, NumPy y TensorFlow para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _load_manifest(manifest_path: Path) -> list[dict]:
    """Lee un manifiesto CSV y valida columnas obligatorias.

    Formato esperado (cabecera): archivo,categoria,split
    Columnas opcionales admitidas: hoja, hash
    """
    import csv

    if not manifest_path.exists():
        raise FileNotFoundError(f"No existe el manifiesto: {manifest_path}")

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in MANIFEST_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"Manifiesto {manifest_path} le falta columnas {missing}; "
                f"tiene: {reader.fieldnames}"
            )
        rows = list(reader)
    return rows


def _verify_split_integrity(rows_by_split: dict[str, list[dict]]) -> None:
    """Comprueba que no comparta hojas ni hashes entre splits y que las clases
    coincidan con el orden canónico. Levanta si hay fuga o inconsistencia."""
    seen_hashes: dict[str, str] = {}   # hash -> split donde apareció
    seen_leaf_file: dict[str, str] = {}  # nombre hoja (sin UUID) -> split

    for split, rows in rows_by_split.items():
        categories = {r["categoria"] for r in rows}
        unknown = categories - set(PLANTVILLAGE_CLASSES)
        if unknown:
            raise ValueError(
                f"Split {split} contiene clases no canónicas: {sorted(unknown)}"
            )

        for row in rows:
            file_hash = row.get("hash")
            if file_hash:
                if file_hash in seen_hashes and seen_hashes[file_hash] != split:
                    raise ValueError(
                        f"Hash duplicado entre splits ({seen_hashes[file_hash]} / {split}): {file_hash}"
                    )
                seen_hashes[file_hash] = split

            archivo = row.get("archivo", "")
            # Nombre de hoja: todo lo antes del último bloque tipo UUID.
            hoja = _leaf_name(archivo)
            if hoja:
                if hoja in seen_leaf_file and seen_leaf_file[hoja] != split:
                    raise ValueError(
                        f"Hoja compartida entre splits ({seen_leaf_file[hoja]} / {split}): {hoja}"
                    )
                seen_leaf_file[hoja] = split


def _leaf_name(archivo: str) -> str:
    """Devuelve el nombre de hoja asociado a un archivo (sin UUID ni extensión).

    PlantVillage nombra las imágenes como <Clase>___<enfermedad>_<UUID>[_final_masked].<ext>.
    La hoja queda implícita en el grupo de archivos que comparten el prefijo de clase
    y enfermedad cuando no hay UUID explícito. Si no hay patrón reconocible, devuelve "".
    """
    name = Path(archivo).stem
    # Quitar sufijo _final_masked si lo hubiera.
    name = name.removesuffix("_final_masked")
    partes = name.split("___")
    if len(partes) >= 2:
        # Quedarse con la parte de clase+enfermedad, sin el posible UUID final.
        core = partes[1]
        # Un bloque UUID tiene 32 hex (o 36 con guiones); si el último segmento lo es, lo quitamos.
        tokens = core.split("_")
        if tokens and (len(tokens[-1]) in (32, 36)) and all(
            c in "0123456789abcdefABCDEF-" for c in tokens[-1]
        ):
            tokens = tokens[:-1]
        return f"{partes[0]}___{'_'.join(tokens)}"
    return ""


def _resolve_image_path(image_dir: Path, categoria: str, archivo: str) -> Path:
    """Resuelve la ruta de una imagen a partir del manifiesto.

    Intenta image_dir/categoria/archivo y, si falla, image_dir/archivo (por si el
    manifiesto ya lleva rutas relativas).
    """
    candidate = image_dir / categoria / archivo
    if candidate.exists():
        return candidate
    candidate = image_dir / archivo
    if candidate.exists():
        return candidate
    return candidate  # se reportará como faltante al cargar


def _build_dataset(
    rows: list[dict],
    image_dir: Path,
    batch_size: int,
    seed: int,
    shuffle: bool,
    repeat: bool = False,
) -> tf.data.Dataset:
    """Construye un tf.data.Dataset de (imagen_float_[0,255], label_int) desde filas de manifiesto.

    La imagen se lee desde disco y se devuelve en [0,255] (float32). El
    preprocesamiento a [-1,1] (MobileNetV2 preprocess_input) se aplica dentro del
    modelo, nunca aquí, para evitar doble aplicación y garantizar
    entrenamiento == inferencia.
    """
    set_deterministic_seed(seed)

    paths = [
        str(_resolve_image_path(image_dir, r["categoria"], r["archivo"])) for r in rows
    ]
    labels = [CLASS_TO_INDEX[r["categoria"]] for r in rows]

    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int32))
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    def _load_and_resize(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, IMAGE_TARGET_SIZE)
        image = tf.cast(image, tf.float32)  # [0,255]
        return image, label

    ds = ds.map(_load_and_resize, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 4096), seed=seed, reshuffle_each_iteration=False)
    ds = ds.batch(batch_size)
    if repeat:
        ds = ds.repeat()
    ds = ds.prefetch(AUTOTUNE)
    return ds


def load_datasets_from_manifests(
    manifest_dir: Path | None = None,
    image_dir: Path | None = None,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    class_weight_mode: str = "train_only",
):
    """Carga train/val/test desde manifiestos CSV con integridad verificada.

    Returns:
        (train_ds, val_ds, test_ds, class_names, class_weights, manifest_info)
        donde class_weights es un dict {indice_clase: peso} calculado EXCLUSIVAMENTE
        con train y manifest_info contiene conteos, hashes de manifiestos y tamaños
        reales de split.

    Raises:
        FileNotFoundError: si faltan los manifiestos (bloqueo deliberado: no se
            entrena ni se vuelve a dividir por lotes).
    """
    set_deterministic_seed(seed)

    manifest_dir = Path(manifest_dir or MANIFEST_DIR)
    image_dir = Path(image_dir or DATA_DIR)

    # Rutas de manifiestos relativas al directorio indicado (permite apuntar a
    # manifiestos de prueba o a la ubicación acordada por el equipo).
    manifest_paths = {s: manifest_dir / f"{s}.csv" for s in ("train", "val", "test")}

    missing = [str(p) for p in manifest_paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Manifiestos canónicos no disponibles. No se puede entrenar "
            f"MobileNetV2 sin partición acordada. Faltan: {missing}. "
            "No se volverá a dividir por lotes. Entregue el bloqueo al equipo y "
            "acuerde la fuente de verdad (conteos 43.486/5.398/5.401 vs "
            "43.490/5.398/5.396)."
        )

    rows_by_split = {}
    manifest_hashes = {}
    for split, path in manifest_paths.items():
        rows = _load_manifest(path)
        # Filtrar filas de este split (por si un CSV trae varios splits mezclados).
        rows = [r for r in rows if r.get("split", split) == split]
        rows_by_split[split] = rows
        manifest_hashes[split] = _sha256_file(path)

    _verify_split_integrity(rows_by_split)

    class_names = list(PLANTVILLAGE_CLASSES)

    # Conteo de clases SOLO en train para los pesos.
    train_labels = [CLASS_TO_INDEX[r["categoria"]] for r in rows_by_split["train"]]
    counter = Counter(train_labels)
    num_classes = len(class_names)
    total = sum(counter.values())
    class_weights = {}
    for i in range(num_classes):
        n_clase = counter.get(i, 0)
        class_weights[i] = (total / (num_classes * n_clase)) if n_clase else 0.0

    train_ds = _build_dataset(rows_by_split["train"], image_dir, batch_size, seed, shuffle=True)
    val_ds = _build_dataset(rows_by_split["val"], image_dir, batch_size, seed, shuffle=False)
    test_ds = _build_dataset(rows_by_split["test"], image_dir, batch_size, seed, shuffle=False)

    manifest_info = {
        "manifest_dir": str(manifest_dir),
        "image_dir": str(image_dir),
        "manifest_files": {s: str(p) for s, p in manifest_paths.items()},
        "manifest_sha256": manifest_hashes,
        "splits": {
            s: {"n_rows": len(rows_by_split[s]), "n_classes": len({r["categoria"] for r in rows_by_split[s]})}
            for s in ("train", "val", "test")
        },
        "class_weights": class_weights,
        "num_classes": num_classes,
        "class_names": class_names,
        "seed": seed,
    }
    return train_ds, val_ds, test_ds, class_names, class_weights, manifest_info


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Carga histórica por lotes (conservada; NO usa manifiestos).
# ---------------------------------------------------------------------------

data_dir = DATA_DIR


def load_datasets():
    """Carga los datasets de PlantVillage desde disco (partición por lotes).

    Mantiene el comportamiento histórico para interoperabilidad con
    `feature/efficentnetb0`. No es la partición canónica por manifiestos.
    """
    full_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        label_mode="int",
        image_size=IMAGE_TARGET_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )
    class_names = full_dataset.class_names
    full_dataset = full_dataset.shuffle(
        buffer_size=32, seed=SEED, reshuffle_each_iteration=False
    )

    n_batches = int(full_dataset.cardinality().numpy())
    n_train = int(n_batches * 0.8)
    n_val = int(n_batches * 0.1)

    train_dataset = full_dataset.take(n_train)
    rest = full_dataset.skip(n_train)
    validation_dataset = rest.take(n_val)
    test_dataset = rest.skip(n_val)

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    return (train_dataset, validation_dataset, test_dataset, class_names)
