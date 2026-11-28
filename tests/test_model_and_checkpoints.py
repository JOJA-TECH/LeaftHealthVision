"""Pruebas del contrato EfficientNetB0, BatchNorm y checkpoints."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from leafthealthvision.models.efficientnet_b0 import build_efficientnetb0_baseline, unfreeze_top_layers
from leafthealthvision.training.common import checkpoint_sha256, save_training_history, training_callbacks


def _all_layers(layer):
    yield layer
    if isinstance(layer, tf.keras.Model):
        for sublayer in layer.layers:
            yield from _all_layers(sublayer)


class EfficientNetContractTest(unittest.TestCase):
    def setUp(self):
        self.model = build_efficientnetb0_baseline(
            num_classes=3,
            input_shape=(32, 32, 3),
            dropout_rate=0.2,
            weights=None,
        )
        self.base = self.model.get_layer("efficientnetb0")

    def test_extractor_congelado_y_salida_correcta(self):
        self.assertFalse(self.base.trainable)
        base_ids = {id(variable) for variable in self.base.variables}
        self.assertFalse(any(id(variable) in base_ids for variable in self.model.trainable_variables))
        self.assertEqual(self.model.output_shape, (None, 3))

    def test_finetune_descongela_pero_mantiene_batchnorm_congelada(self):
        unfrozen = unfreeze_top_layers(self.model, 20)
        self.assertTrue(unfrozen["batchnorm_frozen"])
        self.assertGreaterEqual(unfrozen["unfrozen_layers"], 1)
        self.assertTrue(unfrozen["unfrozen_layer_names"])
        batchnorm = [layer for layer in _all_layers(self.base) if isinstance(layer, tf.keras.layers.BatchNormalization)]
        self.assertTrue(batchnorm)
        self.assertTrue(all(not layer.trainable for layer in batchnorm))
        bn_ids = {id(variable) for layer in batchnorm for variable in layer.variables}
        leaked = [variable.name for variable in self.model.trainable_variables if id(variable) in bn_ids]
        self.assertEqual(leaked, [])

    def test_unfreeze_invalido_falla(self):
        with self.assertRaises(ValueError):
            unfreeze_top_layers(self.model, 0)


class CheckpointTest(unittest.TestCase):
    def test_callbacks_guardan_mejor_checkpoint_e_historial(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            model = tf.keras.Sequential([tf.keras.layers.Input((2,)), tf.keras.layers.Dense(1)])
            model.compile(optimizer="sgd", loss="mse")
            x = np.ones((8, 2), dtype=np.float32)
            y = np.zeros((8, 1), dtype=np.float32)
            history = model.fit(
                x,
                y,
                epochs=2,
                validation_data=(x, y),
                callbacks=training_callbacks(output, patience=1),
                verbose=0,
            )
            self.assertTrue((output / "best_model.keras").exists())
            self.assertTrue((output / "keras_training_log.csv").exists())
            self.assertEqual(len(checkpoint_sha256(output / "best_model.keras")), 64)
            saved = save_training_history(output, history)
            self.assertEqual(saved["loss"], [float(value) for value in history.history["loss"]])
            self.assertTrue((output / "training_history.json").exists())
            self.assertTrue((output / "training_curves.png").exists())
            loaded = tf.keras.models.load_model(output / "best_model.keras")
            np.testing.assert_allclose(model.predict(x, verbose=0), loaded.predict(x, verbose=0), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
