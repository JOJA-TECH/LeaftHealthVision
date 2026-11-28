"""Entrenamiento reproducible del baseline congelado de EfficientNetB0 para PlantVillage color.

Uso: ``python -m leafthealthvision.training.efficientnetb0 --config configs/efficientnetb0_baseline.yaml``
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import tensorflow as tf
from codecarbon import EmissionsTracker

from leafthealthvision.models.efficientnet_b0 import build_efficientnetb0_baseline
from leafthealthvision.plantvillage import class_weights, sha256_file
from leafthealthvision.training.common import (
    build_datasets,
    checkpoint_sha256,
    evaluate_and_save,
    git_commit,
    load_config,
    optional_fit_kwargs,
    prepare_output_dir,
    project_root,
    resolve_output_dir,
    save_class_weights,
    save_split_evidence,
    save_training_history,
    save_unresolved,
    set_seed,
    stop_emissions,
    system_specs,
    training_callbacks,
)
from leafthealthvision.tracking.mlflow_tracking import configure_mlflow, log_directory


def run(config_path: Path) -> Path:
    root = project_root()
    config = load_config(config_path)
    training = config["training"]
    seed = int(training["seed"])
    set_seed(seed)

    splits, datasets, class_names = build_datasets(config, seed)
    image_size = tuple(config["model"]["image_size"])

    model = build_efficientnetb0_baseline(
        num_classes=len(class_names),
        input_shape=(*image_size, 3),
        dropout_rate=float(config["model"]["dropout_rate"]),
        weights=config["model"].get("weights", "imagenet"),
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(training["learning_rate"])),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    output_dir = resolve_output_dir(config, root, "baseline", seed)
    prepare_output_dir(output_dir)
    shutil.copy2(config_path, output_dir / "config.yaml")
    manifest_dir = root / config["dataset"]["manifest_dir"]
    split_audit = save_split_evidence(output_dir, manifest_dir, splits)
    unresolved = save_unresolved(output_dir, splits)
    callbacks = training_callbacks(output_dir, int(training["early_stopping_patience"]))

    tracking = config["mlflow"]
    configure_mlflow(tracking["tracking_uri"], tracking["experiment_name"])
    weights = class_weights(splits["train"].labels, len(class_names))
    weight_rows = save_class_weights(output_dir, splits["train"].labels, weights, class_names)

    # La MX330 en WDDM no expone power/utilization vía NVML; se mide solo CPU.
    tracker = EmissionsTracker(
        project_name="leafthealthvision-efficientnetb0-baseline",
        output_dir=str(output_dir),
        gpu_ids=[],
    )
    tracker.start()
    t0 = time.perf_counter()
    emissions_kg_co2 = None
    try:
        with mlflow.start_run(run_name="efficientnetb0_baseline") as active_run:
            params = {
                "architecture": "EfficientNetB0",
                "weights": str(config["model"].get("weights", "imagenet")),
                "include_preprocessing": True,
                "extractor_frozen": True,
                "variant": config["dataset"]["variant"],
                "resolution": f"{image_size[0]}x{image_size[1]}",
                "channels": "RGB",
                "dropout_rate": float(config["model"]["dropout_rate"]),
                "batch_size": int(training["batch_size"]),
                "max_epochs": int(training["epochs"]),
                "early_stopping_patience": int(training["early_stopping_patience"]),
                "learning_rate": float(training["learning_rate"]),
                "optimizer": "Adam",
                "loss": "SparseCategoricalCrossentropy",
                "seed": seed,
                "num_classes": len(class_names),
                "augmentation": "horizontal_flip,rotation_0.05,zoom_0.10",
                "split_manifests": str(Path(config["dataset"]["manifest_dir"])),
                "split_train_sha256": split_audit["sha256"]["train"],
                "split_val_sha256": split_audit["sha256"]["val"],
                "split_test_sha256": split_audit["sha256"]["test"],
                "dataset_source": config["dataset"]["source"],
                "checkpoint_selection": "val_loss",
            }
            params.update({key: int(value) for key, value in optional_fit_kwargs(config).items()})
            mlflow.set_tags(
                {
                    "stage": "baseline",
                    "seed": str(seed),
                    "git_commit": str(git_commit(root)),
                    "dataset_variant": config["dataset"]["variant"],
                }
            )
            mlflow.log_params(params)

            history = model.fit(
                datasets["train"],
                validation_data=datasets["val"],
                epochs=int(training["epochs"]),
                callbacks=callbacks,
                class_weight=weights,
                verbose=2,
                **optional_fit_kwargs(config),
            )
            elapsed_seconds = time.perf_counter() - t0
            emissions_kg_co2 = stop_emissions(tracker)
            epochs_run = len(history.history["loss"])
            save_training_history(output_dir, history)

            model_path = output_dir / "model.keras"
            best_model_path = output_dir / "best_model.keras"
            model.save(model_path)
            probabilities = model.predict(datasets["test"], verbose=1)
            metrics = evaluate_and_save(output_dir, splits, probabilities, class_names)

            mlflow.log_metrics(
                {
                    "tiempo_entrenamiento_segundos": round(elapsed_seconds, 2),
                    "emisiones_co2_kg": emissions_kg_co2,
                    "epocas_ejecutadas": float(epochs_run),
                    "best_val_loss": float(min(history.history["val_loss"])),
                    **metrics,
                }
            )
            metadata = {
                "stage": "baseline",
                "executed_at_utc": datetime.now(timezone.utc).isoformat(),
                "seed": seed,
                "mlflow_run_id": active_run.info.run_id,
                "git_commit": git_commit(root),
                "config_path": str(config_path),
                "config_sha256": sha256_file(config_path),
                "dataset_source": config["dataset"]["source"],
                "dataset_variant": config["dataset"]["variant"],
                "weights": config["model"].get("weights", "imagenet"),
                "dropout_rate": float(config["model"]["dropout_rate"]),
                "early_stopping_patience": int(training["early_stopping_patience"]),
                "manifest_dir": str(Path(config["dataset"]["manifest_dir"])),
                "split_audit": split_audit,
                "split_sizes": {name: len(records.paths) for name, records in splits.items()},
                "manifest_counts": split_audit["counts"],
                "resolved_counts": split_audit["resolved_counts"],
                "unresolved_counts": split_audit["unresolved_counts"],
                "unresolved_color_manifest_rows": unresolved,
                "class_names": class_names,
                "class_weights": {class_names[index]: weights[index] for index in weights},
                "class_weight_rows": weight_rows,
                "checkpoint_selection": {
                    "monitor": "val_loss",
                    "restore_best_weights": True,
                    "test_used_for_selection": False,
                },
                "model": {"path": str(model_path), "sha256": checkpoint_sha256(model_path)},
                "best_model": {"path": str(best_model_path), "sha256": checkpoint_sha256(best_model_path)},
                "training_history": "training_history.json",
                "metrics": metrics,
                "system_specs": system_specs(),
                "tiempo_entrenamiento_segundos": round(elapsed_seconds, 2),
                "emisiones_co2_kg": emissions_kg_co2,
                "epocas_ejecutadas": epochs_run,
            }
            (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
            log_directory(output_dir)
    finally:
        if emissions_kg_co2 is None:
            stop_emissions(tracker)  # parada de emergencia si el entrenamiento falló
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/efficientnetb0_baseline.yaml"))
    args = parser.parse_args()
    destination = run(args.config.resolve())
    print(f"Experimento guardado en {destination}")


if __name__ == "__main__":
    main()
