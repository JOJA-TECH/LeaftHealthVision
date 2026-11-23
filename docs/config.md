# Configuración base

La configuración vive en dos capas:

- `src/leafthealthvision/config/settings.py`: valores compartidos por todo el
  equipo (semillas, rutas, tamaños, MLflow). Solo constantes; sin lógica de
  entrenamiento, inferencia ni preprocesamiento.
- `configs/*.yaml`: valores vigentes por experimento (los que de verdad usan
  los scripts). Hoy: `efficientnetb0_baseline.yaml` y
  `efficientnetb0_finetune.yaml`.

## Semillas (reproducibilidad)
- `SEED = 42`: valor base usado por `numpy`, `tensorflow` y `random` de Python.
  Cada script fija además determinismo de TensorFlow
  (`tf.config.experimental.enable_op_determinism`).

## Datasets
- **PlantVillage**: fuente oficial de GitHub, licencia `cc-by-sa-3.0`, se guarda
  en `data/plantvillage/`. Se usan las imágenes a color (`raw/color`) y la
  partición por manifiestos versionados en `data/splits/*.csv`
  (43 486 / 5 398 / 5 401), agrupada por hoja física (`leaf_grouping`) para
  evitar fuga de datos. Las filas del manifiesto sin equivalente en `color` no
  se sustituyen: quedan en `unresolved_color_manifest_rows.csv` del experimento.
- **PlantDoc**: fuente verificada, licencia `CC-BY-4.0`, pendiente de descarga.

## Imágenes
- Tamaño objetivo: `224x224`, RGB. El reescalado depende de la arquitectura:
  - **EfficientNetB0** (Keras 3) normaliza internamente (rescale + normalización
    dentro del modelo) y espera `float32` en `[0, 255]`; **no** se encadena
    `1/255` externo ni existe el parámetro `include_preprocessing`.
  - **MobileNetV2** requiere su preprocesamiento/rescale externo propio.

## Entrenamiento (implementado; valores de referencia)
- Batch 32, hasta 50 épocas, lr 0.001, optimizador Adam, pérdida
  `SparseCategoricalCrossentropy`, métrica accuracy, early stopping sobre
  `val_loss` con paciencia 5 y `restore_best_weights`, pesos por clase
  calculados solo con train (`N / (K·n_clase)`), aumento de datos
  (flip horizontal, rotación 0.05, zoom 0.10).
- Limitadores opcionales `steps_per_epoch` y `validation_steps` en el config
  (`null` = usar todo el manifiesto); útiles en hardware limitado y pruebas.

## Fine-tuning (`efficientnetb0_finetune.yaml`)
- Parte del checkpoint del baseline
  (`experiments/efficientnetb0/baseline/seed_42/best_model.keras`); requiere
  haber corrido el baseline primero.
- Descongela las últimas 20 capas del extractor con BatchNorm siempre
  congelada; lr `1e-5`; hasta 5 épocas; early stopping paciencia 3.
- `image_size` y `dropout_rate` deben coincidir con el baseline o
  `load_weights` fallará al cargar el checkpoint.

## MLflow
- Tracking URI local: `./mlruns` (ruta plana; `file://./mlruns` se parsea como
  `\\.\mlruns` en Windows y falla). Experimento: `leafthealthvision`; runs
  `efficientnetb0_baseline` y `efficientnetb0_finetune`.
- `configure_mlflow` activa automáticamente `MLFLOW_ALLOW_FILE_STORE=true`
  (MLflow 3 lo exige incluso para tracking directo con backend de archivos).
- UI: `MLFLOW_ALLOW_FILE_STORE=true uv run mlflow ui --backend-store-uri ./mlruns`.

## Artefactos por corrida
`experiments/efficientnetb0/{baseline,finetune}/seed_<seed>/`:
`config.yaml` (copia), `model.keras`, `best_model.keras`, `metrics.json`,
`classification_report.csv`, `confusion_matrix.csv/.png`,
`misclassifications.csv/.png`, `keras_training_log.csv`, `emissions.csv`
(codecarbon), `metadata.json` (git commit, specs del equipo, tamaños de
partición, filas no resueltas) y todos ellos registrados como artefactos del
run de MLflow.
