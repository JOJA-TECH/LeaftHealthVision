"""EfficientNetB0 baseline congelado y utilidades de fine-tuning."""

from __future__ import annotations

import tensorflow as tf


def _iter_layers(layer: tf.keras.layers.Layer):
    """Recorre capas anidadas para no dejar BatchNorm entrenable por descuido."""
    yield layer
    if isinstance(layer, tf.keras.Model):
        for sublayer in layer.layers:
            yield from _iter_layers(sublayer)


def build_efficientnetb0_baseline(
    num_classes: int,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.2,
    weights: str | None = "imagenet",
) -> tf.keras.Model:
    """Crea una cabeza softmax sobre EfficientNetB0 totalmente congelada.

    ``weights=None`` solo se usa en pruebas sintéticas; el pipeline real exige
    ``imagenet``. Keras 3 incluye el preprocesamiento dentro del modelo y espera
    float32 en ``[0, 255]``.
    """
    if weights not in (None, "imagenet"):
        raise ValueError(f"Pesos no soportados para EfficientNetB0: {weights!r}")
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
        weights=weights,
        input_shape=input_shape,
    )
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

    Devuelve conteos y nombres para dejarlos como evidencia en MLflow y metadata.
    """
    if num_layers < 1:
        raise ValueError("unfreeze_last_n debe ser >= 1")
    base_model = model.get_layer("efficientnetb0")
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    for layer in base_model.layers[-num_layers:]:
        layer.trainable = True

    batchnorm_layers = [layer for layer in _iter_layers(base_model) if isinstance(layer, tf.keras.layers.BatchNormalization)]
    for layer in batchnorm_layers:
        layer.trainable = False

    batchnorm_variables = {id(variable) for layer in batchnorm_layers for variable in layer.variables}
    leaked = [variable.name for variable in model.trainable_variables if id(variable) in batchnorm_variables]
    if leaked:
        raise RuntimeError(f"BatchNorm quedó entrenable: {leaked[:3]}")

    trainable = [layer.name for layer in base_model.layers if layer.trainable and layer.weights]
    if not trainable:
        raise RuntimeError("El fine-tuning no dejó ninguna capa entrenable en el extractor")
    return {
        "unfrozen_layers": len(trainable),
        "unfrozen_layer_names": trainable,
        "batchnorm_frozen": True,
        "batchnorm_layers": len(batchnorm_layers),
    }
