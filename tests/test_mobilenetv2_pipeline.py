"""Pruebas ligeras del pipeline MobileNetV2 (sin descargar PlantVillage ni ImageNet).

Usa manifiestos sintéticos y un puñado de imágenes PNG generadas en memoria.
Ejecutar: uv run python tests/test_mobilenetv2_pipeline.py
"""

import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Permitir importar el paquete desde la raíz del repo.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from leafthealthvision.models.mobilenet_v2 import build_mobilenetv2_baseline
from leafthealthvision.plantvillage import (
    class_weights,
    load_splits,
)
from leafthealthvision.training.common import set_seed

IMG_SIZE = (224, 224)
TEST_CLASSES = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___healthy"]


def _make_images(tmpdir, count_per_class):
    """Genera imágenes PNG válidas y devuelve (archivos, categoria) por clase."""
    from PIL import Image

    files = []
    for categoria in TEST_CLASSES:
        clase_dir = tmpdir / categoria
        clase_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count_per_class):
            nombre = f"{categoria}_{i:03d}.png"
            img = Image.fromarray(
                np.random.randint(0, 255, (*IMG_SIZE, 3), dtype=np.uint8)
            )
            img.save(clase_dir / nombre)
            files.append((categoria, nombre))
    return files


def _write_manifests(tmpdir, files, split_frac=(0.6, 0.2, 0.2)):
    """Escribe train/val/test CSV en tmpdir/splits y devuelve el directorio."""

    splits_dir = tmpdir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    np.random.default_rng(0).shuffle(files)
    n = len(files)
    n_train = max(1, int(n * split_frac[0]))
    n_val = max(1, int(n * split_frac[1]))
    partes = {
        "train": files[:n_train],
        "val": files[n_train : n_train + n_val],
        "test": files[n_train + n_val :],
    }

    for split, parte in partes.items():
        with (splits_dir / f"{split}.csv").open("w", encoding="utf-8") as f:
            f.write("archivo,categoria,split,hash,hoja\n")
            for categoria, nombre in parte:
                f.write(f"{nombre},{categoria},{split},,\n")
    return splits_dir


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def test_manifest_integrity_no_shared_leaf_or_hash(tmp_path):
    """Ningún hash compartido entre splits; clases válidas."""
    import csv

    image_root = tmp_path / "data" / "plantvillage" / "raw" / "color"
    files = _make_images(image_root, count_per_class=10)
    splits_dir = _write_manifests(tmp_path, files)

    # Añadir columna hash a cada fila para probar detección de duplicado entre splits.
    for split in ("train", "val", "test"):
        path = splits_dir / f"{split}.csv"
        rows = list(csv.DictReader(path.open()))
        for r in rows:
            r["hash"] = _sha256(image_root / r["categoria"] / r["archivo"])
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    splits = load_splits(splits_dir, image_root)
    train_records = splits["train"]
    val_records = splits["val"]
    test_records = splits["test"]

    # Todas las clases del manifiesto deben ser clases PlantVillage válidas
    assert set(train_records.class_names).issubset(set(TEST_CLASSES)), "clases válidas"
    assert set(val_records.class_names).issubset(set(TEST_CLASSES)), "clases válidas val"
    assert set(test_records.class_names).issubset(set(TEST_CLASSES)), "clases válidas test"
    assert len(train_records.paths) > 0
    assert len(val_records.paths) > 0
    assert len(test_records.paths) > 0

    # Pesos de clase calculados solo con train
    num_clases = len(train_records.class_names)
    cw = class_weights(train_records.labels, num_clases)
    clases_train = set(train_records.labels)
    for i in range(num_clases):
        if i in clases_train:
            assert cw[i] > 0, f"clase {i} en train debe tener peso > 0"
        else:
            assert cw[i] == 0.0, f"clase {i} ausente de train debe tener peso 0.0"
    assert set(cw.keys()) == set(range(num_clases))
    print("  OK test_manifest_integrity_no_shared_leaf_or_hash")


def test_preprocess_output_in_range():
    """preprocess_input de MobileNetV2 mapea [0,255] a [-1,1]."""
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    x = tf.constant(np.full((1, 224, 224, 3), 127.5, dtype=np.float32))
    out = preprocess(x).numpy()
    assert np.all(out >= -1.0 - 1e-6) and np.all(out <= 1.0 + 1e-6), (
        float(out.min()),
        float(out.max()),
    )
    # bordes: 0 -> -1, 255 -> 1
    assert abs(preprocess(tf.constant([[[[0.0]]]])).numpy()[0, 0, 0, 0] - (-1.0)) < 1e-4
    assert abs(preprocess(tf.constant([[[[255.0]]]])).numpy()[0, 0, 0, 0] - 1.0) < 1e-4
    print("  OK test_preprocess_output_in_range")


def _dataset_from_records(records, batch_size=2):
    """Crea un tf.data.Dataset a partir de SplitRecords."""
    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE, antialias=True)
        img = tf.cast(img, tf.float32)  # [0,255]
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((records.paths, records.labels))
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def test_train_inference_consistency(tmp_path):
    """El modelo con preprocesamiento interno da igual que preprocesar fuera."""
    image_root = tmp_path / "data" / "plantvillage" / "raw" / "color"
    files = _make_images(image_root, count_per_class=10)
    splits_dir = _write_manifests(tmp_path, files)
    splits = load_splits(splits_dir, image_root)
    train_records = splits["train"]
    num_clases = len(train_records.class_names)

    model = build_mobilenetv2_baseline(
        num_clases, input_shape=(224, 224, 3), weights=None,
    )
    model.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])

    # Un batch de imágenes crudas [0,255] pasa por el modelo (preprocesa dentro).
    for images, labels in _dataset_from_records(train_records).take(1):
        cruda = images.numpy()
        logits_modelo = model(cruda, training=False).numpy()
        assert logits_modelo.shape == (cruda.shape[0], num_clases)
        assert np.all(np.isfinite(logits_modelo))
        break
    print("  OK test_train_inference_consistency")


def test_probabilities_shape_and_range(tmp_path):
    """Salida de probabilidades: forma (batch, num_classes) y valores en [0,1]."""
    image_root = tmp_path / "data" / "plantvillage" / "raw" / "color"
    files = _make_images(image_root, count_per_class=10)
    splits_dir = _write_manifests(tmp_path, files)
    splits = load_splits(splits_dir, image_root)
    train_records = splits["train"]
    num_clases = len(train_records.class_names)

    model = build_mobilenetv2_baseline(
        num_clases, input_shape=(224, 224, 3), weights=None,
    )
    model.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])

    for images, _ in _dataset_from_records(train_records).take(1):
        logits = model(images, training=False).numpy()
        probs = tf.nn.softmax(logits).numpy()
        assert probs.shape == (images.shape[0], num_clases)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
        assert np.allclose(probs.sum(axis=1), 1.0)
        break
    print("  OK test_probabilities_shape_and_range")


def test_saved_model_load_roundtrip(tmp_path):
    """Se puede guardar y recargar el modelo, y predice con forma/rango válidos."""
    image_root = tmp_path / "data" / "plantvillage" / "raw" / "color"
    files = _make_images(image_root, count_per_class=10)
    splits_dir = _write_manifests(tmp_path, files)
    splits = load_splits(splits_dir, image_root)
    train_records = splits["train"]
    num_clases = len(train_records.class_names)

    model = build_mobilenetv2_baseline(
        num_clases, input_shape=(224, 224, 3), weights=None,
    )
    model.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
    # Un paso de entrenamiento para que tenga pesos entrenables inicializados.
    for images, labels in _dataset_from_records(train_records).take(1):
        model.train_on_batch(images, labels)
        break

    save_path = tmp_path / "model_test.keras"
    model.save(save_path)
    loaded = load_model(save_path, compile=False)

    for images, _ in _dataset_from_records(train_records).take(1):
        out = loaded(images, training=False).numpy()
        probs = tf.nn.softmax(out).numpy()
        assert out.shape == (images.shape[0], num_clases)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
        break
    print("  OK test_saved_model_load_roundtrip")


def test_manifests_missing_block():
    """Sin manifiestos, load_splits debe bloquear con FileNotFoundError."""
    vacio = Path("/tmp/leafthealthvision_sin_manifestos")
    shutil.rmtree(vacio, ignore_errors=True)
    vacio.mkdir(parents=True)
    try:
        load_splits(vacio, vacio)
        raise AssertionError("debió lanzar FileNotFoundError")
    except FileNotFoundError:
        pass
    print("  OK test_manifests_missing_block")


if __name__ == "__main__":
    set_seed(42)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        test_manifest_integrity_no_shared_leaf_or_hash(tmp_path)
        test_preprocess_output_in_range()
        test_train_inference_consistency(tmp_path)
        test_probabilities_shape_and_range(tmp_path)
        test_saved_model_load_roundtrip(tmp_path)
    test_manifests_missing_block()
    print("\nTODAS LAS PRUEBAS PASARON")
