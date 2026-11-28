"""Entrenamiento reproducible del baseline congelado de MobileNetV2 para PlantVillage.

Uso: ``python -m leafthealthvision.training.mobilenetv2 --config configs/mobilenetv2_baseline.yaml``
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import mlflow
import tensorflow as tf
from codecarbon import EmissionsTracker

from leafthealthvision.models.mobilenet_v2 import (
    build_mobilenetv2_baseline,
)
from leafthealthvision.plantvillage import class_weights, sha256_file
from leafthealthvision.tracking.mlflow_tracking import configure_mlflow
from leafthealthvision.training.common import (
    build_datasets,
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
    training_callbacks,
)


def run(config_path: Path) -> Path:
    root = project_root()
    config = load_config(config_path)
    training = config["training"]
    seed = int(training["seed"])
    set_seed(seed)

    splits, datasets, class_names = build_datasets(config, seed)
    image_size = tuple(config["model"]["image_size"])

    model = build_mobilenetv2_baseline(
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
    save_unresolved(output_dir, splits)
    callbacks = training_callbacks(output_dir, int(training["early_stopping_patience"]))

    tracking = config["mlflow"]
    configure_mlflow(tracking["tracking_uri"], tracking["experiment_name"])
    weights = class_weights(splits["train"].labels, len(class_names))
    save_class_weights(output_dir, splits["train"].labels, weights, class_names)

    # La MX330 en WDDM no expone power/utilization vía NVML; se mide solo CPU.
    tracker = EmissionsTracker(
        project_name="leafthealthvision-mobilenetv2-baseline",
        output_dir=str(output_dir),
        gpu_ids=[],
    )

    with mlflow.start_run(run_name=f"mobilenetv2_baseline_seed{seed}"):
        mlflow.log_params(
            {
                "model": "mobilenetv2",
                "variant": "baseline",
                "base_model_weights": "imagenet",
                "base_trainable": False,
                "epochs": int(training["epochs"]),
                "batch_size": int(training["batch_size"]),
                "img_height": int(image_size[0]),
                "img_width": int(image_size[1]),
                "preprocess": "mobilenet_v2.preprocess_input ([0,255]->[-1,1])",
                "optimizer": "adam",
                "learning_rate": float(training["learning_rate"]),
                "loss": "SparseCategoricalCrossentropy(from_logits=False)",
                "metrics": "accuracy",
                "early_stopping_patience": int(training["early_stopping_patience"]),
                "class_weight_mode": "train_only",
                "num_classes": len(class_names),
                "seed": seed,
                "commit": git_commit(root),
            }
        )
        mlflow.log_params(split_audit)
        mlflow.log_params(
            {
                "manifest_sha256_train": sha256_file(manifest_dir / "train.csv"),
                "manifest_sha256_val": sha256_file(manifest_dir / "val.csv"),
                "manifest_sha256_test": sha256_file(manifest_dir / "test.csv"),
            }
        )

        tracker.start()
        history = model.fit(
            datasets["train"],
            validation_data=datasets["val"],
            epochs=int(training["epochs"]),
            class_weight=weights,
            callbacks=callbacks,
            **optional_fit_kwargs(config),
        )
        emisiones_kg = stop_emissions(tracker)

        # Curvas de entrenamiento
        save_training_history(output_dir, history)

        # Evaluación final bloqueada en test
        probabilities = model.predict(datasets["test"], verbose=1)
        eval_metrics = evaluate_and_save(output_dir, splits, probabilities, class_names)
        eval_metrics["carbon_emissions_kg"] = float(emisiones_kg)
        mlflow.log_metrics(eval_metrics)

        # Guardar modelo final
        model.save(output_dir / "model.keras")
        model.save(output_dir / "baseline_model.keras")

        mlflow.log_artifact(str(output_dir / "model.keras"), artifact_path="model")
        mlflow.log_artifact(str(output_dir / "metrics.json"), artifact_path="evaluation")
        mlflow.log_artifact(str(output_dir / "classification_report.csv"), artifact_path="evaluation")
        mlflow.log_artifact(str(output_dir / "confusion_matrix.csv"), artifact_path="evaluation")
        mlflow.log_artifact(str(output_dir / "confusion_matrix.png"), artifact_path="evaluation")
        mlflow.log_artifact(str(output_dir / "training_history.json"), artifact_path="training")
        mlflow.log_artifact(str(output_dir / "training_curves.png"), artifact_path="training")
        mlflow.log_artifact(str(output_dir / "class_weights.csv"), artifact_path="evaluation")
        mlflow.log_artifact(str(output_dir / "split_audit.json"), artifact_path="evaluation")
        mlflow.log_artifact(str(config_path), artifact_path="config")

        print(f"Huella de carbono del entrenamiento: {emisiones_kg:.6f} kg CO2eq")
        print(f"Test accuracy: {eval_metrics['accuracy']:.4f}")
        print(f"Test macro-F1: {eval_metrics['f1_macro']:.4f}")
        print(f"Artefactos guardados en: {output_dir}")

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrenamiento MobileNetV2 (baseline congelado) sobre PlantVillage."
    )
    parser.add_argument("--config", required=True, help="ruta al archivo YAML de configuración")
    args = parser.parse_args()
    run(Path(args.config))
