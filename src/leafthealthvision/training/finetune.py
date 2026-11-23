"""Fine-tuning de EfficientNetB0 a partir del checkpoint del baseline congelado.

Descongela las últimas N capas del extractor (BatchNorm siempre congelada) y
ajusta con la misma partición, semilla y métricas del baseline para que la
comparación sea válida.

Uso: ``python -m leafthealthvision.training.finetune --config configs/efficientnetb0_finetune.yaml``
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

from leafthealthvision.models.efficientnet_b0 import build_efficientnetb0_baseline, unfreeze_top_layers
from leafthealthvision.plantvillage import class_weights
from leafthealthvision.training.common import (
    build_datasets,
    evaluate_and_save,
    git_commit,
    load_config,
    optional_fit_kwargs,
    project_root,
    save_unresolved,
    set_seed,
    stop_emissions,
    training_callbacks,
)
from leafthealthvision.tracking.mlflow_tracking import configure_mlflow, log_directory


def run(config_path: Path) -> Path:
    root = project_root()
    config = load_config(config_path)
    training = config["training"]
    fine = config["fine_tuning"]
    seed = int(training["seed"])
    set_seed(seed)

    splits, datasets, class_names = build_datasets(config, seed)
    image_size = tuple(config["model"]["image_size"])

    init_from = root / fine["init_from"]
    if not init_from.exists():
        raise FileNotFoundError(
            f"No existe el checkpoint del baseline: {init_from}. "
            "Ejecuta antes el baseline congelado (training.efficientnetb0)."
        )
    # Mismas formas y dropout que el baseline; load_weights falla si el config no coincide.
    model = build_efficientnetb0_baseline(
        num_classes=len(class_names),
        input_shape=(*image_size, 3),
        dropout_rate=float(config["model"]["dropout_rate"]),
    )
    model.load_weights(init_from)
    unfrozen = unfreeze_top_layers(model, int(fine["unfreeze_last_n"]))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(fine["learning_rate"])),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    output_dir = root / "experiments" / "efficientnetb0" / "finetune" / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    unresolved = save_unresolved(output_dir, splits)
    callbacks = training_callbacks(output_dir, int(training["early_stopping_patience"]))

    tracking = config["mlflow"]
    configure_mlflow(tracking["tracking_uri"], tracking["experiment_name"])
    weights = class_weights(splits["train"].labels, len(class_names))

    # La MX330 en WDDM no expone power/utilization vía NVML; se mide solo CPU.
    tracker = EmissionsTracker(
        project_name="leafthealthvision-efficientnetb0-finetune",
        output_dir=str(output_dir),
        gpu_ids=[],
    )
    tracker.start()
    t0 = time.perf_counter()
    emissions_kg_co2 = None
    try:
        with mlflow.start_run(run_name="efficientnetb0_finetune") as active_run:
            params = {
                "architecture": "EfficientNetB0",
                "weights": "imagenet",
                "include_preprocessing": True,
                "extractor_frozen": False,
                "init_from": str(fine["init_from"]),
                "unfreeze_last_n": int(fine["unfreeze_last_n"]),
                "unfrozen_layers": unfrozen["unfrozen_layers"],
                "batchnorm_frozen": unfrozen["batchnorm_frozen"],
                "learning_rate_finetune": float(fine["learning_rate"]),
                "variant": "color",
                "resolution": f"{image_size[0]}x{image_size[1]}",
                "channels": "RGB",
                "batch_size": int(training["batch_size"]),
                "max_epochs": int(training["epochs"]),
                "optimizer": "Adam",
                "loss": "SparseCategoricalCrossentropy",
                "seed": seed,
                "num_classes": len(class_names),
                "augmentation": "horizontal_flip,rotation_0.05,zoom_0.10",
                "split_manifests": str(Path(config["dataset"]["manifest_dir"])),
                "dataset_source": config["dataset"]["source"],
            }
            params.update({key: int(value) for key, value in optional_fit_kwargs(config).items()})
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

            model.save(output_dir / "model.keras")
            probabilities = model.predict(datasets["test"], verbose=1)
            metrics = evaluate_and_save(output_dir, splits, probabilities, class_names)

            mlflow.log_metrics(
                {
                    "tiempo_entrenamiento_segundos": round(elapsed_seconds, 2),
                    "emisiones_co2_kg": emissions_kg_co2,
                    "epocas_ejecutadas": float(epochs_run),
                    **metrics,
                }
            )
            metadata = {
                "executed_at_utc": datetime.now(timezone.utc).isoformat(),
                "seed": seed,
                "mlflow_run_id": active_run.info.run_id,
                "git_commit": git_commit(root),
                "init_from": str(fine["init_from"]),
                "unfreeze_last_n": int(fine["unfreeze_last_n"]),
                "learning_rate_finetune": float(fine["learning_rate"]),
                "dataset_source": config["dataset"]["source"],
                "dataset_variant": "color",
                "manifest_dir": str(Path(config["dataset"]["manifest_dir"])),
                "split_sizes": {name: len(records.paths) for name, records in splits.items()},
                "unresolved_color_manifest_rows": unresolved,
                "class_names": class_names,
                "tiempo_entrenamiento_segundos": round(elapsed_seconds, 2),
                "emisiones_co2_kg": emissions_kg_co2,
                "epocas_ejecutadas": epochs_run,
            }
            (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            log_directory(output_dir)
    finally:
        if emissions_kg_co2 is None:
            stop_emissions(tracker)  # parada de emergencia si el entrenamiento falló
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/efficientnetb0_finetune.yaml"))
    args = parser.parse_args()
    destination = run(args.config.resolve())
    print(f"Experimento guardado en {destination}")


if __name__ == "__main__":
    main()
