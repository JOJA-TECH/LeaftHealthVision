"""Pruebas mínimas de la resolución segmented->color y de los pesos de clase."""

import csv
import tempfile
import unittest
from pathlib import Path

from leafthealthvision.plantvillage import (
    class_weights,
    image_core_name,
    load_manifest_split,
    normalized_image_name,
)

CATEGORIA = "Tomato___healthy"


def _fake_image(directory: Path, name: str) -> None:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")  # a la resolución solo le importa la ruta


def _write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["archivo", "categoria", "split"])
        writer.writeheader()
        writer.writerows(rows)


class NormalizationTest(unittest.TestCase):
    def test_preserva_uuid_y_quita_sufijo(self):
        self.assertEqual(normalized_image_name("AbC-1___Lab 09_final_masked.jpg"), "abc-1___lab 09")
        self.assertEqual(image_core_name("AbC-1___Lab 09_final_masked.jpg"), "lab 09")


class ManifestResolutionTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        color = root / "color" / CATEGORIA
        # Emparejamiento exacto por (categoría, UUID + nombre).
        _fake_image(color, "uuid1___lab 01_final_masked.jpg")
        # Fallback por nombre núcleo único (UUID distinto en el manifiesto).
        _fake_image(color, "uuid2___lab 02_final_masked.jpg")
        # Núcleo ambiguo: dos imágenes comparten nombre sin UUID.
        _fake_image(color, "uuid3___lab 03_final_masked.jpg")
        _fake_image(color, "uuid4___lab 03_final_masked.jpg")

        self.manifest = root / "train.csv"
        _write_manifest(
            self.manifest,
            [
                {"archivo": "uuid1___lab 01_final_masked.jpg", "categoria": CATEGORIA, "split": "train"},
                {"archivo": "otro___lab 02_final_masked.jpg", "categoria": CATEGORIA, "split": "train"},
                {"archivo": "otro___lab 03_final_masked.jpg", "categoria": CATEGORIA, "split": "train"},
                {"archivo": "nadie___lab 99_final_masked.jpg", "categoria": CATEGORIA, "split": "train"},
            ],
        )
        self.records = load_manifest_split(self.manifest, root / "color")

    def test_match_exacto_y_fallback_unico(self):
        self.assertEqual(len(self.records.paths), 2)
        self.assertTrue(self.records.paths[0].endswith("uuid1___lab 01_final_masked.jpg"))
        self.assertTrue(self.records.paths[1].endswith("uuid2___lab 02_final_masked.jpg"))
        self.assertEqual(self.records.labels, [0, 0])

    def test_ambiguo_y_ausente_quedan_resueltos_como_unresolved(self):
        self.assertEqual(len(self.records.unresolved_image_ids), 2)
        self.assertIn(f"{CATEGORIA}/otro___lab 03_final_masked.jpg", self.records.unresolved_image_ids)
        self.assertIn(f"{CATEGORIA}/nadie___lab 99_final_masked.jpg", self.records.unresolved_image_ids)

    def test_clases_ordenadas_y_etiquetas_coherentes(self):
        self.assertEqual(self.records.class_names, [CATEGORIA])


class ClassWeightsTest(unittest.TestCase):
    def test_formula_n_sobre_k_por_nclase(self):
        weights = class_weights([0, 0, 1, 2], 3)
        self.assertAlmostEqual(weights[0], 4 / (3 * 2))
        self.assertAlmostEqual(weights[1], 4 / 3)
        self.assertAlmostEqual(weights[2], 4 / 3)

    def test_clase_vacía_falla(self):
        with self.assertRaises(ValueError):
            class_weights([0, 0], 3)


if __name__ == "__main__":
    unittest.main()
