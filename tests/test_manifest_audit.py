"""Pruebas de auditoría de manifiestos: duplicados, clases y fuga por hoja."""

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from leafthealthvision.plantvillage import (
    audit_manifests,
    manifest_hashes,
    validate_manifest_splits,
)

FIELDS = ["archivo", "categoria", "split", "hoja", "origen"]


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class ManifestAuditTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.splits = self.root / "splits"
        self.splits.mkdir()

    def _base(self):
        _write(
            self.splits / "train.csv",
            [
                {"archivo": "a.jpg", "categoria": "A___healthy", "split": "train", "hoja": "A.csv|1", "origen": "hoja"},
                {"archivo": "b.jpg", "categoria": "A___healthy", "split": "train", "hoja": "A.csv|2", "origen": "hoja"},
            ],
        )
        _write(
            self.splits / "val.csv",
            [{"archivo": "c.jpg", "categoria": "A___healthy", "split": "val", "hoja": "A.csv|3", "origen": "hoja"}],
        )
        _write(
            self.splits / "test.csv",
            [{"archivo": "d.jpg", "categoria": "A___healthy", "split": "test", "hoja": "A.csv|4", "origen": "hoja"}],
        )

    def test_split_limpio_pasa_y_guarda_hashes(self):
        self._base()
        audit = validate_manifest_splits(self.splits)
        self.assertEqual(audit["counts"], {"train": 2, "val": 1, "test": 1})
        self.assertTrue(audit["same_classes"])
        self.assertEqual(audit["leaf_groups_shared_across_splits"], 0)
        self.assertEqual(audit["duplicate_manifest_rows"], 0)
        hashes = manifest_hashes(self.splits)
        self.assertEqual(set(hashes), {"train", "val", "test"})
        for value in hashes.values():
            self.assertEqual(len(value), 64)
            int(value, 16)
        self.assertEqual(hashlib.sha256((self.splits / "train.csv").read_bytes()).hexdigest(), hashes["train"])

    def test_hoja_compartida_entre_splits_falla(self):
        self._base()
        _write(
            self.splits / "val.csv",
            [{"archivo": "c.jpg", "categoria": "A___healthy", "split": "val", "hoja": "A.csv|1", "origen": "hoja"}],
        )
        audit = audit_manifests(self.splits)
        self.assertEqual(audit["leaf_groups_shared_across_splits"], 1)
        with self.assertRaises(ValueError):
            validate_manifest_splits(self.splits)

    def test_archivo_duplicado_entre_splits_falla(self):
        self._base()
        _write(
            self.splits / "test.csv",
            [{"archivo": "a.jpg", "categoria": "A___healthy", "split": "test", "hoja": "A.csv|4", "origen": "hoja"}],
        )
        audit = audit_manifests(self.splits)
        self.assertEqual(audit["duplicate_manifest_rows"], 1)
        with self.assertRaises(ValueError):
            validate_manifest_splits(self.splits)

    def test_clases_distintas_fallan(self):
        self._base()
        _write(
            self.splits / "test.csv",
            [{"archivo": "d.jpg", "categoria": "B___healthy", "split": "test", "hoja": "B.csv|4", "origen": "hoja"}],
        )
        with self.assertRaises(ValueError):
            validate_manifest_splits(self.splits)

    def test_split_incorrecto_falla(self):
        self._base()
        _write(
            self.splits / "val.csv",
            [{"archivo": "c.jpg", "categoria": "A___healthy", "split": "train", "hoja": "A.csv|3", "origen": "hoja"}],
        )
        with self.assertRaises(ValueError):
            validate_manifest_splits(self.splits)


if __name__ == "__main__":
    unittest.main()
