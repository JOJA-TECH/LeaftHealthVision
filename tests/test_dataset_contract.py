"""Contrato de preprocesamiento: EfficientNetB0 recibe float32 en [0, 255]."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from leafthealthvision.plantvillage import SplitRecords, make_dataset


class DatasetContractTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.paths = []
        for index, color in enumerate([(255, 0, 0), (0, 255, 0)]):
            path = self.root / f"{index}.png"
            Image.new("RGB", (4, 4), color).save(path)
            self.paths.append(str(path))

    def _records(self):
        return SplitRecords(
            paths=self.paths,
            labels=[0, 1],
            class_names=["rojo", "verde"],
            image_ids=["0.png", "1.png"],
            unresolved_image_ids=[],
        )

    def test_sin_rescaling_externo_y_forma_esperada(self):
        dataset = make_dataset(self._records(), image_size=(8, 8), batch_size=2, training=False, seed=42)
        images, labels = next(iter(dataset))
        self.assertEqual(images.dtype, tf.float32)
        self.assertEqual(tuple(images.shape), (2, 8, 8, 3))
        self.assertEqual(tuple(labels.shape), (2,))
        self.assertGreater(float(tf.reduce_max(images)), 1.0)
        self.assertGreaterEqual(float(tf.reduce_min(images)), 0.0)
        self.assertLessEqual(float(tf.reduce_max(images)), 255.0)
        # 255 debe seguir siendo 255, no 1.0.
        self.assertAlmostEqual(float(tf.reduce_max(images)), 255.0, places=3)

    def test_clases_y_etiquetas_coherentes(self):
        dataset = make_dataset(self._records(), image_size=(8, 8), batch_size=2, training=False, seed=42)
        _, labels = next(iter(dataset))
        np.testing.assert_array_equal(labels.numpy(), np.array([0, 1]))


if __name__ == "__main__":
    unittest.main()
