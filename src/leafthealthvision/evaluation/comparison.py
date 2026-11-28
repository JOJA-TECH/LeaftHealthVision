"""Normalización de artefactos de corridas locales para la comparación en Streamlit.

Los scripts históricos guardan el mismo concepto con nombres distintos
(``accuracy`` vs ``test_accuracy``, ``emisiones_co2_kg`` vs ``carbon_emissions_kg``).
Esta capa lee lo que existe y deja ``None`` lo que falta: nunca inventa valores.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

METRIC_ALIASES = {
    "accuracy": ("accuracy", "test_accuracy"),
    "balanced_accuracy": ("balanced_accuracy", "test_balanced_accuracy"),
    "precision_macro": ("precision_macro", "test_macro_precision"),
    "recall_macro": ("recall_macro", "test_macro_recall"),
    "f1_macro": ("f1_macro", "test_macro_f1"),
    "precision_weighted": ("precision_weighted", "test_weighted_precision"),
    "recall_weighted": ("recall_weighted", "test_weighted_recall"),
    "f1_weighted": ("f1_weighted", "test_weighted_f1"),
    "n_test": ("n_test", "test_n_samples"),
}

METRIC_LABELS = {
    "accuracy": "Accuracy (test)",
    "balanced_accuracy": "Balanced accuracy",
    "precision_macro": "Precision macro",
    "recall_macro": "Recall macro",
    "f1_macro": "F1 macro",
    "precision_weighted": "Precision weighted",
    "recall_weighted": "Recall weighted",
    "f1_weighted": "F1 weighted",
    "n_test": "Imágenes evaluadas",
}


@dataclass
class RunReport:
    """Datos comparables de una corrida, con su procedencia y sus artefactos."""

    path: Path
    name: str
    model: str = "desconocido"
    stage: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    training_seconds: float | None = None
    emissions_kg: float | None = None
    energy_kwh: float | None = None
    energy_breakdown: dict[str, float] = field(default_factory=dict)
    epochs: int | None = None
    split_sizes: dict[str, int] = field(default_factory=dict)
    split_test_sha256: str | None = None
    hardware: str = ""
    provenance: dict[str, str] = field(default_factory=dict)
    class_names: list[str] = field(default_factory=list)
    history: dict[str, list[float]] = field(default_factory=dict)
    model_path: Path | None = None
    artifacts: dict[str, Path] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.model} · {self.stage or self.name}" if self.model != "desconocido" else self.name

    @property
    def n_test(self) -> int | None:
        value = self.metrics.get("n_test")
        return int(value) if value is not None else None


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8-sig") as handle:
            content = json.load(handle)
    except (OSError, ValueError):
        return {}
    return content if isinstance(content, dict) else {}


def _emissions_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def _read_emissions(path: Path) -> dict[str, str]:
    """Devuelve la última fila de CodeCarbon (CSV con o sin extensión)."""
    rows = _emissions_rows(path)
    return rows[-1] if rows else {}


def _parse_classification_report(text: str) -> tuple[int | None, list[str]]:
    """Extrae el soporte total y las clases de un reporte de scikit-learn."""
    total = None
    classes = []
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0] == "accuracy":
            total = _as_int(parts[-1])
            continue
        if len(parts) < 5 or parts[0] in ("macro", "weighted"):
            continue
        try:
            [float(value) for value in parts[-4:-1]]
            int(float(parts[-1]))
        except ValueError:
            continue
        classes.append(" ".join(parts[:-4]))
    return total, classes


def _hardware(row: dict[str, str]) -> str:
    return " · ".join(
        value
        for value in (
            row.get("cpu_model"),
            row.get("gpu_model"),
            " / ".join(filter(None, (row.get("country_name"), row.get("region")))),
        )
        if value
    )


def _energy_breakdown(row: dict[str, str]) -> dict[str, float]:
    return {
        key: value
        for key in ("cpu_energy", "gpu_energy", "ram_energy")
        if (value := _as_float(row.get(key))) is not None
    }


def _read_mlflow(db_path: Path) -> dict:
    """Lee métricas finales, parámetros, nombre y series de un SQLite de MLflow."""
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        latest: dict[str, dict[str, float]] = {}
        for run_id, key, value in connection.execute("SELECT run_uuid,key,value FROM latest_metrics"):
            latest.setdefault(run_id, {})[key] = value
        names = dict(connection.execute("SELECT run_uuid,value FROM tags WHERE key = 'mlflow.runName'"))
        params: dict[str, dict[str, str]] = {}
        for run_id, key, value in connection.execute("SELECT run_uuid,key,value FROM params"):
            params.setdefault(run_id, {})[key] = value
        steps: dict[str, int] = {}
        series: dict[str, dict[str, list[tuple[int, float]]]] = {}
        for run_id, key, step, value in connection.execute(
            "SELECT run_uuid,key,step,value FROM metrics "
            "WHERE key IN ('accuracy','loss','val_accuracy','val_loss') ORDER BY run_uuid,key,step"
        ):
            steps[run_id] = max(steps.get(run_id, -1), step)
            series.setdefault(run_id, {}).setdefault(key, []).append((step, value))
    finally:
        connection.close()
    return {"latest": latest, "names": names, "params": params, "steps": steps, "series": series}


def _as_float(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_int(value) -> int | None:
    number = _as_float(value)
    return int(number) if number is not None else None


def _first(sources, keys: tuple[str, ...]):
    if isinstance(sources, dict):
        sources = (sources,)
    for mapping in sources:
        for key in keys:
            if mapping.get(key) is not None:
                return mapping[key]
    return None


def _infer_model(path: Path, value=None) -> str:
    text = str(value or "")
    names = {part.lower() for part in path.parts}
    if "efficientnet" in text.lower() or any("efficientnetb0" in name for name in names):
        return "EfficientNetB0"
    if "mobilenet" in text.lower() or any("mobilenetv2" in name for name in names):
        return "MobileNetV2"
    return "desconocido"


def load_run(run_dir: Path, experiments_dir: Path | None = None) -> RunReport:
    """Lee una corrida y devuelve un informe normalizado."""
    run_dir = Path(run_dir)
    metrics_raw = _read_json(run_dir / "metrics.json")
    metadata = _read_json(run_dir / "metadata.json")
    config = _read_json(run_dir / "config_effective.json")
    history = _read_json(run_dir / "training_history.json")
    if not history:
        history = _read_json(run_dir / "history.json")
    provenance = _read_json(run_dir / "provenance.json")
    emissions = _read_emissions(run_dir / "emissions.csv")
    if not emissions:
        emissions = _read_emissions(run_dir / "results" / "emissions")

    metrics: dict[str, float] = {}
    for canonical, aliases in METRIC_ALIASES.items():
        value = _as_float(_first(metrics_raw, aliases))
        if value is not None:
            metrics[canonical] = value

    split_sizes: dict[str, int] = {}
    raw_sizes = _first((metadata, config), ("split_sizes", "split_sizes")) or {}
    for split, value in raw_sizes.items():
        count = value.get("n_rows") if isinstance(value, dict) else value
        parsed = _as_int(count)
        if parsed is not None:
            split_sizes[split] = parsed

    split_sha = None
    audit = metadata.get("split_audit") or {}
    if isinstance(audit, dict):
        split_sha = (audit.get("sha256") or {}).get("test")
    split_sha = split_sha or (config.get("manifest_sha256") or {}).get("test")

    hardware = ""
    specs = metadata.get("system_specs") or {}
    if isinstance(specs, dict):
        hardware = str(specs.get("cpu_name") or "")
    if not hardware and isinstance(config.get("hardware"), dict):
        hw = config["hardware"]
        hardware = f"{hw.get('platform', '')} · {hw.get('cpu_count', '')} CPUs"

    epochs = _as_int(metadata.get("epocas_ejecutadas"))
    if epochs is None and isinstance(history.get("accuracy"), list):
        epochs = len(history["accuracy"])

    energy_breakdown = {}
    for key in ("cpu_energy", "ram_energy", "gpu_energy"):
        value = _as_float(emissions.get(key))
        if value is not None:
            energy_breakdown[key] = value

    artifacts: dict[str, Path] = {}
    wanted = {
        "Matriz de confusión": ("confusion_matrix.png",),
        "Curvas de entrenamiento": ("training_curves.png",),
        "Reporte de clasificación": ("classification_report.csv", "classification_report.txt"),
        "Errores revisados": ("misclassifications.csv",),
    }
    for label, filenames in wanted.items():
        for filename in filenames:
            candidate = run_dir / filename
            if candidate.is_file():
                artifacts[label] = candidate
                break

    model_path = None
    for filename in ("model.keras", "best_model.keras"):
        candidate = run_dir / filename
        if candidate.is_file():
            model_path = candidate
            break

    name = str(run_dir.relative_to(experiments_dir)) if experiments_dir else run_dir.name
    return RunReport(
        path=run_dir,
        name=name,
        model=_infer_model(run_dir, _first((metadata, config), ("model", "model"))),
        stage=str(_first((metadata, config), ("stage", "variant")) or ""),
        metrics=metrics,
        training_seconds=_as_float(metadata.get("tiempo_entrenamiento_segundos")) or _as_float(emissions.get("duration")),
        emissions_kg=_as_float(_first((metadata, metrics_raw), ("emisiones_co2_kg", "carbon_emissions_kg")))
        or _as_float(emissions.get("emissions")),
        energy_kwh=_as_float(emissions.get("energy_consumed")),
        energy_breakdown=energy_breakdown,
        epochs=epochs,
        split_sizes=split_sizes,
        split_test_sha256=split_sha,
        hardware=hardware,
        provenance={str(k): str(v) for k, v in provenance.items()},
        class_names=list(metadata.get("class_names") or config.get("class_names") or []),
        history={key: value for key, value in history.items() if isinstance(value, list)},
        model_path=model_path,
        artifacts=artifacts,
    )


def load_colab_runs(colab_dir: Path, experiments_dir: Path | None = None) -> list[RunReport]:
    """Lee las corridas de Colab desde el SQLite de MLflow y sus artefactos."""
    colab_dir = Path(colab_dir)
    db_path = colab_dir / "mlflow.db"
    if not db_path.is_file():
        return []
    data = _read_mlflow(db_path)
    emissions = _emissions_rows(colab_dir / "emissions.csv")
    artifacts_dir = colab_dir / "leafthealthvision" / "artifacts"
    by_name = {name: run_id for run_id, name in data["names"].items()}
    runs: list[RunReport] = []

    for stage in ("baseline", "finetune"):
        prefix = f"{stage}_"
        metrics = {}
        for mapping in data["latest"].values():
            for key, value in mapping.items():
                canonical = key[len(prefix):] if key.startswith(prefix) else ""
                if canonical in METRIC_ALIASES:
                    metrics[canonical] = float(value)
        if "recall_macro" in metrics:
            metrics["balanced_accuracy"] = metrics["recall_macro"]

        report_name = f"classification_report_{stage}.txt"
        report_path = artifacts_dir / report_name
        n_test = None
        class_names: list[str] = []
        if report_path.is_file():
            n_test, class_names = _parse_classification_report(
                report_path.read_text(encoding="utf-8", errors="replace")
            )
            if n_test is not None:
                metrics["n_test"] = n_test

        carbon = next((row for row in emissions if stage in row.get("project_name", "").lower()), {})
        run_id = by_name.get(f"colab_{stage}")
        params = data["params"].get(run_id, {})
        last_step = data["steps"].get(run_id)
        epochs = last_step + 1 if last_step is not None else _as_int(params.get("max_epochs"))
        hardware = _hardware(carbon)

        artifacts = {}
        for label, filename in (
            ("Reporte de clasificación", report_name),
            ("Matriz de confusión", f"confusion_matrix_{stage}.png"),
            ("Errores revisados", f"misclassifications_{stage}.csv"),
        ):
            candidate = artifacts_dir / filename
            if candidate.is_file():
                artifacts[label] = candidate

        model_path = None
        filenames = [f"{stage}_best.keras", f"{stage}_model.keras"]
        if stage == "finetune":
            filenames += ["finetuned_best.keras", "finetuned_model.keras"]
        for filename in filenames:
            candidate = artifacts_dir / filename
            if candidate.is_file():
                model_path = candidate
                break

        base = str(colab_dir.relative_to(experiments_dir)) if experiments_dir else colab_dir.name
        runs.append(
            RunReport(
                path=colab_dir,
                name=f"{base}/{stage}",
                model=_infer_model(colab_dir, params.get("architecture")),
                stage=stage,
                metrics=metrics,
                training_seconds=_as_float(carbon.get("duration")),
                emissions_kg=_as_float(carbon.get("emissions")),
                energy_kwh=_as_float(carbon.get("energy_consumed")),
                energy_breakdown=_energy_breakdown(carbon),
                epochs=epochs,
                split_sizes={"test": n_test} if n_test is not None else {},
                hardware=hardware,
                provenance={
                    "origen": "Colab (notebook remoto)",
                    "fuente_métricas": f"MLflow latest_metrics (prefijo {prefix})",
                    "fuente_emisiones": "emissions.csv",
                    "run_mlflow": f"colab_{stage}",
                },
                class_names=class_names,
                history={
                    key: [value for _, value in points]
                    for key, points in data["series"].get(run_id, {}).items()
                },
                model_path=model_path,
                artifacts=artifacts,
            )
        )
    return runs


def load_colab_artifact_runs(colab_dir: Path, experiments_dir: Path | None = None) -> list[RunReport]:
    """Lee corridas de Colab guardadas como artefactos sueltos, sin SQLite de MLflow."""
    colab_dir = Path(colab_dir)
    artifacts_dir = colab_dir / "leafthealthvision" / "artifacts"
    carbon = next(iter(_emissions_rows(colab_dir / "emissions.csv")), {})
    base = str(colab_dir.relative_to(experiments_dir)) if experiments_dir else colab_dir.name
    runs: list[RunReport] = []

    for metrics_path in sorted(artifacts_dir.glob("metrics_*.json")):
        stage = metrics_path.stem.removeprefix("metrics_")
        metrics_raw = _read_json(metrics_path)
        config = _read_json(artifacts_dir / f"config_effective_{stage}.json")
        history = _read_json(artifacts_dir / f"history_{stage}.json")
        metrics = {}
        for canonical, aliases in METRIC_ALIASES.items():
            value = _as_float(_first(metrics_raw, aliases))
            if value is not None:
                metrics[canonical] = value

        emissions_kg = _as_float(_first(metrics_raw, ("carbon_emissions_kg", "emisiones_co2_kg")))
        if emissions_kg is None:
            emissions_kg = _as_float(carbon.get("emissions"))

        report_path = artifacts_dir / f"classification_report_{stage}.txt"
        class_names: list[str] = []
        if report_path.is_file():
            _, class_names = _parse_classification_report(
                report_path.read_text(encoding="utf-8", errors="replace")
            )

        artifacts = {}
        for label, filename in (
            ("Reporte de clasificación", f"classification_report_{stage}.txt"),
            ("Matriz de confusión", f"confusion_matrix_{stage}.png"),
            ("Errores revisados", f"misclassifications_{stage}.csv"),
        ):
            candidate = artifacts_dir / filename
            if candidate.is_file():
                artifacts[label] = candidate

        model_path = None
        for filename in (f"{stage}_best.keras", f"{stage}_model.keras"):
            candidate = artifacts_dir / filename
            if candidate.is_file():
                model_path = candidate
                break

        runs.append(
            RunReport(
                path=colab_dir,
                name=f"{base}/{stage}",
                model=_infer_model(colab_dir, config.get("model")),
                stage=str(config.get("variant") or stage),
                metrics=metrics,
                training_seconds=_as_float(carbon.get("duration")),
                emissions_kg=emissions_kg,
                energy_kwh=_as_float(carbon.get("energy_consumed")),
                energy_breakdown=_energy_breakdown(carbon),
                epochs=_as_int(config.get("epochs"))
                or len(history.get("accuracy", []))
                or None,
                split_sizes={"test": int(metrics["n_test"])} if "n_test" in metrics else {},
                hardware=_hardware(carbon),
                provenance={
                    "origen": "Colab (notebook remoto)",
                    "fuente_métricas": metrics_path.name,
                    "fuente_emisiones": "emissions.csv",
                    "variante": stage,
                },
                class_names=class_names,
                history={key: value for key, value in history.items() if isinstance(value, list)},
                model_path=model_path,
                artifacts=artifacts,
            )
        )
    return runs


def discover_runs(experiments_dir: Path) -> list[RunReport]:
    """Descubre las corridas de las dos fuentes reales; si faltan, cualquier ``metrics.json``."""
    root = Path(experiments_dir)
    if not root.is_dir():
        return []
    runs = load_colab_runs(root / "efficientnetb0" / "colab_results", root)
    runs += load_colab_artifact_runs(root / "mobilenetv2" / "colab_results", root)
    if runs:
        return runs
    return [load_run(path.parent, root) for path in sorted(root.rglob("metrics.json"))]


def comparability_notes(first: RunReport, second: RunReport) -> list[str]:
    """Señala diferencias de protocolo que invalidan una comparación directa."""
    notes = []
    if first.n_test and second.n_test and first.n_test != second.n_test:
        notes.append(
            f"Test evaluado con muestras distintas: {first.n_test} vs {second.n_test}."
        )
    if (
        first.split_test_sha256
        and second.split_test_sha256
        and first.split_test_sha256 != second.split_test_sha256
    ):
        notes.append("Los manifiestos de test tienen hashes distintos.")
    if first.epochs and second.epochs and first.epochs != second.epochs:
        notes.append(f"Épocas ejecutadas distintas: {first.epochs} vs {second.epochs}.")
    if first.hardware and second.hardware and first.hardware != second.hardware:
        notes.append("Las corridas se ejecutaron en hardware distinto.")
    origins = {
        run.provenance.get("origen", "") for run in (first, second)
    }
    if len(origins) > 1 and any(origins):
        notes.append("Las corridas tienen procedencias distintas; revisa sus notas.")
    return notes


def estimate_cost(energy_kwh: float | None, price_per_kwh: float) -> float | None:
    """Coste eléctrico estimado; ``price_per_kwh`` es un supuesto del usuario."""
    if energy_kwh is None:
        return None
    return energy_kwh * price_per_kwh
