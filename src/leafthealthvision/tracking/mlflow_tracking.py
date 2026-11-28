"""Funciones pequeñas para reutilizar el tracking local de MLflow."""

from __future__ import annotations

import os
from pathlib import Path

import mlflow


def configure_mlflow(tracking_uri: str, experiment_name: str) -> None:
    # MLflow 3 exige este opt-in incluso para el backend de archivos usado en local.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_directory(directory: Path) -> None:
    """Registra todos los artefactos de una ejecución real."""
    mlflow.log_artifacts(str(directory), artifact_path="experiment")
