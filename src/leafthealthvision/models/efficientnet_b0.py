"""EfficientNetB0 baseline congelado; no contiene fine-tuning."""

from __future__ import annotations

import tensorflow as tf


def build_efficientnetb0_baseline(
    num_classes: int,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.2,
) -> tf.keras.Model:
    """Crea una cabeza softmax sobre EfficientNetB0 ImageNet totalmente congelada."""
    inputs = tf.keras.Input(shape=input_shape, name="image")
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.10),
        ],
        name="train_augmentation",
    )
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    # En Keras 3, EfficientNet incluye el preprocesamiento (rescale + normalización)
    # dentro del modelo y espera float32 en [0, 255]; no existe include_preprocessing.
    base_model.trainable = False

    x = augmentation(inputs)
    # Mantiene BatchNormalization en inferencia incluso durante model.fit.
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="classification_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    model = tf.keras.Model(inputs, outputs, name="efficientnetb0_baseline")

    frozen_variables = {id(variable) for variable in base_model.variables}
    leaked = [variable.name for variable in model.trainable_variables if id(variable) in frozen_variables]
    if leaked:
        raise RuntimeError(f"El extractor no quedó congelado: {leaked[:3]}")
    return model


def unfreeze_top_layers(model: tf.keras.Model, num_layers: int) -> dict:
    """Descongela las últimas ``num_layers`` capas del extractor dejando BatchNorm congelada.

    Devuelve conteos para dejarlos como evidencia en MLflow.
    """
    base_model = model.get_layer("efficientnetb0")
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    open_bn = [layer.name for layer in base_model.layers if isinstance(layer, tf.keras.layers.BatchNormalization) and layer.trainable]
    if open_bn:
        raise RuntimeError(f"BatchNorm quedó entrenable: {open_bn[:3]}")
    trainable = [layer.name for layer in base_model.layers if layer.trainable]
    return {"unfrozen_layers": len(trainable), "batchnorm_frozen": True}
