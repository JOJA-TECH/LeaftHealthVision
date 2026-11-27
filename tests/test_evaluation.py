"""Pruebas de métricas, artefactos de evaluación y cola de revisión de errores."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from leafthealthvision.evaluation.error_analysis import save_misclassifications
from leafthealthvision.evaluation.metrics import evaluate_predictions, save_evaluation_artifacts


class MetricsTest(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([0, 1, 2, 0])
        self.probabilities = np.array(
            [
                [0.9, 0.05, 0.05],
                [0.1, 0.8, 0.1],
                [0.1, 0.7, 0.2],  # error: 2 -> 1
                [0.7, 0.2, 0.1],
            ]
        )
        self.class_names = ["Tomato___healthy", "Tomato___Late_blight", "Tomato___Early_blight"]

    def test_accuracy_balanced_y_matriz(self):
        metrics, matrix, report = evaluate_predictions(self.y_true, self.probabilities, self.class_names)
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["n_test"], 4)
        self.assertEqual(matrix.shape, (3, 3))
        self.assertEqual(matrix.sum(), 4)
        self.assertIn("balanced_accuracy", metrics)
        self.assertIn("Tomato___healthy", report)

    def test_artefactos_de_evaluacion(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            metrics = save_evaluation_artifacts(output, self.y_true, self.probabilities, self.class_names)
            self.assertAlmostEqual(metrics["accuracy"], 0.75)
            for name in ("metrics.json", "classification_report.csv", "confusion_matrix.csv", "confusion_matrix.png"):
                self.assertTrue((output / name).exists(), name)
            self.assertEqual(json.loads((output / "metrics.json").read_text())["accuracy"], metrics["accuracy"])

    def test_cola_de_errores_sin_hipotesis_inventada(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            paths = []
            for index in range(4):
                path = output / f"{index}.png"
                Image.new("RGB", (4, 4), (index * 40, 0, 0)).save(path)
                paths.append(str(path))
            save_misclassifications(output, paths, self.y_true, self.probabilities, self.class_names)
            with (output / "misclassifications.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["true_class"], "Tomato___Early_blight")
            self.assertEqual(row["predicted_class"], "Tomato___Late_blight")
            self.assertEqual(row["tipo_error"], "misma_especie")
            self.assertEqual(row["review_status"], "pendiente")
            self.assertEqual(row["human_hypothesis"], "")
            self.assertEqual(row["reviewed_by"], "")
            self.assertTrue((output / "error_review_queue.csv").exists())
            self.assertTrue((output / "error_review_guide.md").exists())

    def test_forma_invalida_falla(self):
        with self.assertRaises(ValueError):
            evaluate_predictions(np.array([0, 1]), np.ones((2, 2)), ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
