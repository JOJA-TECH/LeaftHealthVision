"""Carga y prepara los datasets de PlantVillage para entrenamiento, validación y prueba."""

import tensorflow as tf
from .config.settings import BATCH_SIZE, IMAGE_TARGET_SIZE, PLANTVILLAGE, ROOT, SEED

AUTOTUNE = tf.data.AUTOTUNE

data_dir = ROOT / PLANTVILLAGE["local_path"] / PLANTVILLAGE["image_subfolder"]


def load_datasets():
    """Carga los datasets de PlantVillage desde disco.
    Returns:
    tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
        ``(train_dataset, validation_dataset, test_dataset, class_names)``
        listos para consumir en ``model.fit``.
    """
    full_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        label_mode="int",
        image_size=IMAGE_TARGET_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )
    class_names = full_dataset.class_names
    # cache() carga todo el dataset en RAM (~100 GB descomprimido para 163k
    # imágenes) y agota la memoria con GPU integrada; se lee desde disco en su
    # lugar. El buffer de shuffle se reduce a batches que quepan en memoria.
    full_dataset = full_dataset.shuffle(
        buffer_size=32, seed=SEED, reshuffle_each_iteration=False
    )

    n_batches = int(full_dataset.cardinality().numpy())
    n_train = int(n_batches * PLANTVILLAGE["train_split"])
    n_val = int(n_batches * PLANTVILLAGE["val_split"])

    train_dataset = full_dataset.take(n_train)
    rest = full_dataset.skip(n_train)
    validation_dataset = rest.take(n_val)
    test_dataset = rest.skip(n_val)

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    return (train_dataset, validation_dataset, test_dataset, class_names)
