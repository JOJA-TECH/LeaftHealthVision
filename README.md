# LeaftHealthVision

Clasificación de enfermedades de hojas con PlantVillage usando TensorFlow/Keras.
El baseline `efficientnetb0_baseline` usa transferencia de aprendizaje con
EfficientNetB0 ImageNet completamente congelada; no realiza fine-tuning.

## Ejecutar EfficientNetB0 Baseline

Desde la raíz del repositorio:

```bash
uv sync
uv run python scripts/download_plantvillage.py
uv run python -m leafthealthvision.training.baseline --config configs/efficientnetb0_baseline.yaml
MLFLOW_ALLOW_FILE_STORE=true uv run mlflow ui --backend-store-uri ./mlruns
```

El entrenamiento usa `raw/color` y los manifiestos reproducibles existentes
en `data/splits/`, con semilla 42. Sus artefactos se escriben en
`experiments/efficientnetb0/baseline/seed_42/` y se registran en MLflow.

Antes de ejecutar, comprueba que el directorio `data/splits/`
contenga `train.csv`, `val.csv` y `test.csv`. Se generan con
`notebooks/01_eda_plantvillage_crispmlq.ipynb`. Estos manifiestos protegen las
hojas agrupadas cuando los metadatos están disponibles; la limitación de
cobertura documentada por el EDA se conserva explícitamente.

Si una fila del manifiesto de `segmented` no tiene imagen equivalente en
`color`, no se reemplaza: queda registrada en
`unresolved_color_manifest_rows.csv` dentro del experimento.
