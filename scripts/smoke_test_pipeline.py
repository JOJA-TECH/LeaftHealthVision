"""Corrida sintética mínima del pipeline baseline + fine-tuning.

No usa PlantVillage ni escribe en experiments/: valida el cableado completo
(manifiestos, tf.data, modelo, callbacks, métricas, artefactos y MLflow) en CPU.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leafthealthvision.training.efficientnetb0 import run as run_baseline  # noqa: E402
from leafthealthvision.training.finetune import run as run_finetune  # noqa: E402

CLASSES = ["Tomato___healthy", "Tomato___Late_blight", "Tomato___Early_blight"]
SPLIT_SIZES = {"train": 6, "val": 2, "test": 2}


def _write_manifests(splits_dir: Path) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split, count in SPLIT_SIZES.items():
        rows = []
        for class_index, class_name in enumerate(CLASSES):
            for index in range(count):
                rows.append(
                    {
                        "archivo": f"{class_name}_{split}_{index}.png",
                        "categoria": class_name,
                        "split": split,
                        "hoja": f"{class_name}.csv|{split}_{index}",
                        "origen": "hoja",
                    }
                )
        with (splits_dir / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            import csv

            writer = csv.DictWriter(handle, fieldnames=["archivo", "categoria", "split", "hoja", "origen"])
            writer.writeheader()
            writer.writerows(rows)


def _write_images(color_dir: Path) -> None:
    rng = np.random.default_rng(7)
    for class_index, class_name in enumerate(CLASSES):
        class_dir = color_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for split, count in SPLIT_SIZES.items():
            for index in range(count):
                array = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
                array[:, :, class_index % 3] = 200
                image = Image.fromarray(array, mode="RGB")
                ImageDraw.Draw(image).rectangle((2, 2, 29, 29), outline=(255, 255, 255))
                image.save(class_dir / f"{class_name}_{split}_{index}.png")


def _config(smoke_root: Path, splits_dir: Path, color_dir: Path, output_dir: Path, finetune: bool) -> dict:
    config = {
        "dataset": {
            "source": "synthetic-smoke-test",
            "variant": "color",
            "color_dir": str(color_dir),
            "manifest_dir": str(splits_dir),
        },
        "model": {"image_size": [32, 32], "dropout_rate": 0.2, "weights": None},
        "training": {
            "seed": 42,
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 1e-3,
            "early_stopping_patience": 1,
            "steps_per_epoch": 1,
            "validation_steps": 1,
        },
        "mlflow": {
            "tracking_uri": str(smoke_root / "mlruns"),
            "experiment_name": "leafthealthvision_smoke",
        },
        "output_dir": str(output_dir),
    }
    if finetune:
        config["fine_tuning"] = {
            "init_from": str(smoke_root / "baseline" / "best_model.keras"),
            "unfreeze_last_n": 20,
            "learning_rate": 1e-5,
        }
    return config


def main() -> None:
    smoke_root = Path(tempfile.mkdtemp(prefix="leafthealthvision_smoke_"))
    splits_dir = smoke_root / "splits"
    color_dir = smoke_root / "color"
    _write_manifests(splits_dir)
    _write_images(color_dir)

    baseline_config = _config(smoke_root, splits_dir, color_dir, smoke_root / "baseline", finetune=False)
    baseline_path = smoke_root / "baseline.yaml"
    baseline_path.write_text(yaml.safe_dump(baseline_config, sort_keys=False), encoding="utf-8")
    baseline_dir = run_baseline(baseline_path)

    finetune_config = _config(smoke_root, splits_dir, color_dir, smoke_root / "finetune", finetune=True)
    finetune_path = smoke_root / "finetune.yaml"
    finetune_path.write_text(yaml.safe_dump(finetune_config, sort_keys=False), encoding="utf-8")
    finetune_dir = run_finetune(finetune_path)

    required = [
        "config.yaml",
        "model.keras",
        "best_model.keras",
        "metrics.json",
        "classification_report.csv",
        "confusion_matrix.csv",
        "confusion_matrix.png",
        "misclassifications.csv",
        "error_review_queue.csv",
        "training_history.json",
        "training_curves.png",
        "keras_training_log.csv",
        "split_audit.json",
        "class_weights.csv",
        "metadata.json",
    ]
    for directory in (baseline_dir, finetune_dir):
        missing = [name for name in required if not (directory / name).exists()]
        if missing:
            raise AssertionError(f"Faltan artefactos en {directory}: {missing}")

    baseline_metadata = json.loads((baseline_dir / "metadata.json").read_text(encoding="utf-8"))
    finetune_metadata = json.loads((finetune_dir / "metadata.json").read_text(encoding="utf-8"))
    assert baseline_metadata["checkpoint_selection"]["test_used_for_selection"] is False
    assert finetune_metadata["checkpoint_selection"]["test_used_for_selection"] is False
    assert finetune_metadata["unfrozen"]["batchnorm_frozen"] is True
    assert baseline_metadata["split_audit"]["sha256"] == finetune_metadata["split_audit"]["sha256"]
    assert baseline_metadata["split_sizes"] == finetune_metadata["split_sizes"]

    print("SMOKE OK")
    print("directorio:", smoke_root)
    print("baseline:", baseline_dir)
    print("finetune:", finetune_dir)
    print("baseline metrics:", baseline_metadata["metrics"])
    print("finetune metrics:", finetune_metadata["metrics"])
    print("unfrozen:", finetune_metadata["unfrozen"])


if __name__ == "__main__":
    main()
