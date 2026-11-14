# Configuración base

Todos los valores compartidos por el equipo viven en `config/settings.py`.
No contiene lógica de entrenamiento, inferencia ni preprocesamiento; solo
constantes para que todo el mundo parta de la misma base reproducible.

## Semillas (reproducibilidad)
- `SEED = 42` — valor base usado por `numpy`, `tensorflow` y `random` de Python.

## Datasets
- **PlantVillage**: fuente oficial de GitHub, licencia `cc-by-sa-3.0`, se guarda en `data/plantvillage/`. Se usan las imágenes a color (`raw/color`) y se respeta el agrupamiento por hoja (`leaf_grouping`) para evitar data leakage.
- **PlantDoc**: fuente verificada, licencia `CC-BY-4.0`, pendiente de descarga por el equipo.

## Imágenes
- Tamaño objetivo: `224x224`, RGB, rescale `1/255`. Coincide con el input esperado por MobileNetV2 y EfficientNetB0.

## Entrenamiento (referencia, aún sin implementar)
- Batch 32, 50 épocas, lr 0.001, optimizador Adam, pérdida `categorical_crossentropy`, métrica accuracy, early stopping paciencia 5.

## MLflow
- Tracking URI local: `file://./mlruns`, experimento `leafthealthvision`.
