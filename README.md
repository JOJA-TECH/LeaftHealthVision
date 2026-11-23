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

Los manifiestos de partición `data/splits/{train,val,test}.csv` están versionados en
git (43 486 / 5 398 / 5 401 imágenes, agrupados por hoja física para evitar fuga).
Se regeneran con `notebooks/01_eda_plantvillage_crispmlq.ipynb` si hace falta.

## Ejecutar el baseline congelado

```bash
uv run python -m leafthealthvision.training.efficientnetb0 --config configs/efficientnetb0_baseline.yaml
```

Entrena la cabeza sobre EfficientNetB0 ImageNet congelada con `raw/color`, pesos por
clase, aumento de datos y semilla 42. Artefactos en
`experiments/efficientnetb0/baseline/seed_42/` (métricas, matriz de confusión,
errores, historial, metadata con specs del equipo y huella de carbono) y run
`efficientnetb0_baseline` en MLflow.

## Ejecutar el fine-tuning

```bash
uv run python -m leafthealthvision.training.finetune --config configs/efficientnetb0_finetune.yaml
```

Requiere haber corrido el baseline: carga
`experiments/efficientnetb0/baseline/seed_42/best_model.keras`, descongela las
últimas 20 capas (BatchNorm siempre congelada) y ajusta con lr 1e-5 sobre la misma
partición y semilla. Artefactos en `experiments/efficientnetb0/finetune/seed_42/` y
run `efficientnetb0_finetune` en MLflow.

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

Cubren la resolución de nombres `segmented`→`color` y los pesos de clase. Si una fila
del manifiesto no tiene imagen equivalente en `color` no se sustituye: queda en
`unresolved_color_manifest_rows.csv` dentro del experimento.

## Notebooks

- `notebooks/01_eda_plantvillage_crispmlq.ipynb`: EDA y generación de manifiestos.
- `notebooks/02_entrenamiento_efficientnetb0_manifiestos.ipynb`: réplica completa en
  Colab (baseline + fine-tuning + evaluación + errores) con la misma metodología;
  sube `data/splits/*.csv` cuando se pidan para usar la partición acordada.
- `notebooks/efficientnetb0_colab.ipynb`: experimento exploratorio previo (split al
  vuelo sin agrupación por hoja; sus métricas no son comparables con el pipeline).
