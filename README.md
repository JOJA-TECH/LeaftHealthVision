# LeaftHealthVision

Clasificación de enfermedades de hojas con PlantVillage usando TensorFlow/Keras.
Pipeline de EfficientNetB0 por manifiestos (CRISP-ML(Q)): **baseline congelado**
(`efficientnetb0_baseline`) y **fine-tuning** (`efficientnetb0_finetune`) que parte
del checkpoint del baseline.

## Instalación y datos

```bash
uv sync
uv run python scripts/download_plantvillage.py
```

Los manifiestos `data/splits/{train,val,test}.csv` están versionados en git y son la
partición canónica: **43 490 / 5 398 / 5 396** imágenes (54 284 únicas, 38 clases),
generadas desde `raw/color`, deduplicadas por SHA-256 y agrupadas por hoja física.
El EDA histórico sobre `raw/segmented` produjo 43 486 / 5 398 / 5 401 (54 285 únicas);
esas cifras no se usan para entrenar y no deben sobrescribir los CSV actuales.
La procedencia y la decisión están en `docs/particion.md`. Los hashes están en
`data/splits/SHA256SUMS`; `python scripts/audit_splits.py` regenera
`data/splits/partition_audit.json` y verifica duplicados, clases y fuga por hoja.

## Estado actual

El pipeline está implementado y validado con una corrida sintética. Las corridas reales
baseline/fine-tuning sobre PlantVillage **no se han ejecutado en este equipo** (CPU, sin
GPU utilizable por TensorFlow): `experiments/efficientnetb0/` está vacío. Los valores
del notebook 02 son de una ejecución Colab y no se copian como resultados locales.
Los pendientes y comandos están en `docs/experimentos.md`.

## Ejecutar el baseline congelado

```bash
uv run python -m leafthealthvision.training.efficientnetb0 --config configs/efficientnetb0_baseline.yaml
```

Entrena la cabeza sobre EfficientNetB0 ImageNet congelada con `raw/color`, pesos por
clase calculados solo en train, aumento de datos y semilla 42. Artefactos en
`experiments/efficientnetb0/baseline/seed_42/`: checkpoints, métricas, matriz de
confusión, cola de errores para revisión humana, historial/curvas, `split_audit.json`,
`class_weights.csv` y `metadata.json` (hashes de config, manifiestos y checkpoints,
specs del equipo, emisiones). El test se evalúa una sola vez al final; el checkpoint
se elige por `val_loss`. Run `efficientnetb0_baseline` en MLflow con métricas por época.

## Ejecutar el fine-tuning

```bash
uv run python -m leafthealthvision.training.finetune --config configs/efficientnetb0_finetune.yaml
```

Requiere haber corrido el baseline: carga
`experiments/efficientnetb0/baseline/seed_42/best_model.keras`, descongela las
últimas 20 capas (BatchNorm congelada recursivamente) y ajusta con lr 1e-5 sobre la
misma partición y semilla. `model.weights: null` porque los pesos vienen del
checkpoint. Artefactos en `experiments/efficientnetb0/finetune/seed_42/`, incluidos
`init_from_sha256` y el detalle de capas descongeladas, y run
`efficientnetb0_finetune` en MLflow.

## Consultar experimentos

```bash
MLFLOW_ALLOW_FILE_STORE=true uv run mlflow ui --backend-store-uri ./mlruns
```

Nota: `configure_mlflow` activa `MLFLOW_ALLOW_FILE_STORE` automáticamente; MLflow 3
lo exige incluso para tracking directo con backend de archivos.

## Pruebas

```bash
uv run python -m unittest discover -s tests
```

Cubren la resolución de nombres `segmented`→`color`, pesos de clase, auditoría de
manifiestos (duplicados, clases y fuga por hoja), contrato `[0, 255]` de
EfficientNetB0, congelación/BatchNorm, checkpoints, métricas y cola de errores. Si una
fila del manifiesto no tiene imagen equivalente en `color` no se sustituye: queda en
`unresolved_color_manifest_rows.csv` dentro del experimento.

Corrida sintética end-to-end (no usa PlantVillage ni escribe en `experiments/`):

```bash
uv run python scripts/smoke_test_pipeline.py
```

Valida baseline + fine-tuning, callbacks, artefactos y MLflow en CPU con imágenes
sintéticas. No sustituye una corrida real.

## Notebooks

- `notebooks/01_eda_plantvillage_crispmlq.ipynb`: EDA histórico sobre `segmented`
  (43 486 / 5 398 / 5 401). No es la partición canónica y no debe sobrescribir
  `data/splits/`.
- `notebooks/02_entrenamiento_efficientnetb0_manifiestos.ipynb`: ejecución Colab con
  batch 620, 10 épocas de baseline, 5 de fine-tuning y `mixed_float16`; usa los
  manifiestos subidos o los genera desde `color`. Sus métricas guardadas son de esa
  corrida, no artefactos locales, y no son directamente comparables con las configs
  locales (batch 32, hasta 50 épocas).
- `notebooks/efficientnetb0_colab.ipynb`: experimento exploratorio previo (split al
  vuelo sin agrupación por hoja; sus métricas no son comparables con el pipeline).
