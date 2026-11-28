"""MobileNetV2 baseline congelado y utilidades de fine-tuning."""

from __future__ import annotations

import tensorflow as tf
from keras.saving import register_keras_serializable


def _iter_layers(layer: tf.keras.layers.Layer):
    """Recorre capas anidadas para no dejar BatchNorm entrenable por descuido."""
    yield layer
    if isinstance(layer, tf.keras.Model):
        for sublayer in layer.layers:
            yield from _iter_layers(sublayer)


@register_keras_serializable(
    "leafthealthvision", name="MobileNetV2Preprocess"
)
class MobileNetV2Preprocess(tf.keras.layers.Layer):
    """Capa Keras que aplica el preprocesamiento oficial MobileNetV2:
    [0,255] -> [-1,1], una sola vez dentro del modelo.

    Es serializable para que el modelo guardado se pueda recargar sin
    problemas de safe_mode en Keras 3.
    """

    def call(self, inputs):
        # Equivalente numérico a mobilenet_v2.preprocess_input (modo TF):
        # x / 127.5 - 1  (0 -> -1, 255 -> 1).
        return (inputs / 127.5) - 1.0


def build_mobilenetv2_baseline(
    num_classes: int,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.2,
    weights: str | None = "imagenet",
) -> tf.keras.Model:
    """Crea una cabeza softmax sobre MobileNetV2 totalmente congelada.

    ``weights=None`` solo se usa en pruebas sintéticas; el pipeline real exige
    ``imagenet``. Keras 3 incluye el preprocesamiento dentro del modelo y espera
    float32 en ``[0, 255]``. El preprocesamiento oficial de MobileNetV2
    (``[0,255]`` -> ``[-1,1]``) se aplica como capa interna una sola vez,
    garantizando que entrenamiento e inferencia sean idénticos.
    """
    if weights not in (None, "imagenet"):
        raise ValueError(f"Pesos no soportados para MobileNetV2: {weights!r}")
    inputs = tf.keras.Input(shape=input_shape, name="image")

    # Preprocesamiento oficial MobileNetV2: [0,255] -> [-1,1], aplicado una
    # sola vez dentro del modelo. No combinar con layers.Rescaling(1/255).
    x = MobileNetV2Preprocess(name="mobilenetv2_preprocess")(inputs)

    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
    )
    base_model.trainable = False

    # training=False mantiene BatchNormalization en inferencia durante model.fit.
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="classification_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_baseline")

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
    base_model = model.get_layer("mobilenetv2")
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    for layer in base_model.layers[-num_layers:]:
        layer.trainable = True

    batchnorm_layers = [
        layer for layer in _iter_layers(base_model) if isinstance(layer, tf.keras.layers.BatchNormalization)
    ]
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
