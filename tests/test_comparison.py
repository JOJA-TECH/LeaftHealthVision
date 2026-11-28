"""Pruebas de la normalización de artefactos usada por la app comparativa."""

import json
import tempfile
import unittest
from pathlib import Path

from leafthealthvision.evaluation.comparison import (
    comparability_notes,
    discover_runs,
    estimate_cost,
    load_run,
)

EFF_METRICS = {
    "accuracy": 0.7392512972572276,
    "f1_macro": 0.6614989513025004,
    "n_test": 5396,
}
MOB_METRICS = {
    "test_accuracy": 0.65625,
    "test_macro_f1": 0.5150258405798571,
    "test_n_samples": 160,
    "carbon_emissions_kg": 4.067980273860779e-05,
}


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


class ComparisonTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.eff = self.root / "efficientnetb0" / "baseline"
        self.mob = self.root / "mobilenetv2" / "frozen"
        self.eff.mkdir(parents=True)
        self.mob.mkdir(parents=True)
        _write_json(self.eff / "metrics.json", EFF_METRICS)
        _write_json(self.mob / "metrics.json", MOB_METRICS)

    def tearDown(self):
        self.tmp.cleanup()

    def test_normaliza_esquemas_distintos(self):
        eff = load_run(self.eff, self.root)
        mob = load_run(self.mob, self.root)
        self.assertEqual(eff.model, "EfficientNetB0")
        self.assertEqual(mob.model, "MobileNetV2")
        self.assertAlmostEqual(eff.metrics["accuracy"], EFF_METRICS["accuracy"])
        self.assertAlmostEqual(mob.metrics["accuracy"], MOB_METRICS["test_accuracy"])
        self.assertAlmostEqual(mob.metrics["f1_macro"], MOB_METRICS["test_macro_f1"])
        self.assertEqual(eff.n_test, 5396)
        self.assertEqual(mob.n_test, 160)
        self.assertAlmostEqual(mob.emissions_kg, MOB_METRICS["carbon_emissions_kg"])

    def test_lee_emisiones_y_metadata_sin_inventar_valores(self):
        (self.eff / "metadata.json").write_text(
            json.dumps(
                {
                    "stage": "baseline",
                    "tiempo_entrenamiento_segundos": 165.17,
                    "emisiones_co2_kg": 0.0004121509974395147,
                    "epocas_ejecutadas": 2,
                    "split_sizes": {"train": 43490, "test": 5396},
                    "split_audit": {"sha256": {"test": "abc123"}},
                }
            ),
            encoding="utf-8",
        )
        (self.eff / "emissions.csv").write_text(
            "duration,emissions,energy_consumed,cpu_energy,ram_energy,gpu_energy\n"
            "165.17,0.000412,0.001588,0.000671,0.000917,0.0\n",
            encoding="utf-8",
        )
        report = load_run(self.eff, self.root)
        self.assertAlmostEqual(report.energy_kwh, 0.001588)
        self.assertAlmostEqual(report.training_seconds, 165.17)
        self.assertEqual(report.epochs, 2)
        self.assertEqual(report.split_test_sha256, "abc123")
        self.assertAlmostEqual(estimate_cost(report.energy_kwh, 150.0), 0.2382)

        incomplete = load_run(self.mob, self.root)
        self.assertIsNone(incomplete.energy_kwh)
        self.assertIsNone(incomplete.training_seconds)

    def test_detecta_protocolos_distintos(self):
        notes = comparability_notes(load_run(self.eff, self.root), load_run(self.mob, self.root))
        self.assertTrue(any("160" in note for note in notes))
        self.assertEqual(len(discover_runs(self.root)), 2)

    def test_descubre_corridas_y_soporta_directorio_vacio(self):
        self.assertEqual(len(discover_runs(self.root)), 2)
        self.assertEqual(discover_runs(self.root / "no_existe"), [])

    def test_carga_fuentes_reales(self):
        experiments = Path(__file__).resolve().parents[1] / "experiments"
        runs = discover_runs(experiments)
        self.assertEqual(len(runs), 3)

        colab = next(r for r in runs if r.model == "EfficientNetB0" and r.stage == "baseline")
        self.assertAlmostEqual(colab.metrics["accuracy"], 0.9429206819866568)
        self.assertAlmostEqual(colab.metrics["balanced_accuracy"], 0.9444201854644911)
        self.assertEqual(colab.n_test, 5396)
        self.assertEqual(colab.epochs, 10)
        self.assertAlmostEqual(colab.energy_kwh, 0.007399908214394835)
        self.assertEqual(len(colab.class_names), 38)
        self.assertEqual(len(colab.history["accuracy"]), 10)
        self.assertTrue(colab.model_path.is_file())
        self.assertTrue(colab.artifacts)

        mobile = next(r for r in runs if r.model == "MobileNetV2")
        self.assertAlmostEqual(mobile.metrics["accuracy"], 0.9433094054849991)
        self.assertAlmostEqual(mobile.metrics["f1_macro"], 0.9336965819678212)
        self.assertEqual(mobile.n_test, 5433)
        self.assertEqual(mobile.epochs, 10)
        self.assertAlmostEqual(mobile.energy_kwh, 0.01223513204984261)
        self.assertEqual(len(mobile.history["accuracy"]), 10)
        self.assertTrue(mobile.model_path.is_file())
        self.assertTrue(mobile.artifacts)

        notas = comparability_notes(colab, mobile)
        self.assertTrue(any("muestras distintas" in nota for nota in notas))
        self.assertTrue(any("hardware distinto" in nota for nota in notas))


if __name__ == "__main__":
    unittest.main()
