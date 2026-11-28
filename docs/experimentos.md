# Estado de experimentos EfficientNetB0

## Trabajo realizado

He implementado el baseline congelado y el fine-tuning sobre la misma partición, resolución, mapping, semilla, métricas y test.

He auditado los manifiestos: 43 490 / 5 398 / 5 396, 38 clases, 0 duplicados, 0 hojas compartidas y 0 filas sin resolver a `color`.

He ejecutado las pruebas unitarias y una corrida sintética en CPU.

He dejado la evidencia de la partición en `data/splits/SHA256SUMS` y `data/splits/partition_audit.json`.

## Verificación real reducida

He ejecutado el 27 de noviembre de 2026 una corrida real sobre `raw/color` con los configs `configs/efficientnetb0_verificacion_baseline.yaml` y `configs/efficientnetb0_verificacion_finetune.yaml`: 40 pasos de train y 10 de validación por época, 2 épocas, semilla 42 y test completo de 5 396 imágenes. No modifiqué los manifiestos; sus hashes siguen siendo los de `docs/particion.md`.

| Etapa | accuracy | f1_macro | Artefactos |
|---|---:|---:|---|
| Baseline | 0,7393 | 0,6615 | `experiments/efficientnetb0/verificacion/seed_42/baseline/` |
| Fine-tuning | 0,7281 | 0,6481 | `experiments/efficientnetb0/verificacion/seed_42/finetune/` |

Estos valores comprueban el recorrido completo del pipeline; no son una comparación de modelos entrenados con la partición completa. Ambos runs terminaron con estado `FINISHED` en el experimento `leafthealthvision_verificacion` de MLflow. En MLflow la serie `accuracy` por época mide train; el accuracy de test quedó en `metrics.json` de cada corrida.

## Corridas reales pendientes

| Etapa | Config | Comando | Estado |
|---|---|---|---|
| Baseline | `configs/efficientnetb0_baseline.yaml` | `uv run python -m leafthealthvision.training.efficientnetb0 --config configs/efficientnetb0_baseline.yaml` | Pendiente |
| Fine-tuning | `configs/efficientnetb0_finetune.yaml` | `uv run python -m leafthealthvision.training.finetune --config configs/efficientnetb0_finetune.yaml` | Pendiente |

El fine-tuning lo ejecuto después del baseline, cuando exista `experiments/efficientnetb0/baseline/seed_42/best_model.keras`.

## Requisitos

He descargado el dataset en `data/plantvillage/` (`raw/color` y `leaf_grouping/filtered_leafmaps/`).

Uso GPU recomendable; en CPU, 43 490 imágenes × hasta 50 épocas no son viables en un tiempo razonable. No reduzco el test ni lo uso para tomar decisiones.

Reservo espacio para checkpoints y artefactos; `experiments/` no está versionado.

## Artefactos esperados

Guardo en `experiments/efficientnetb0/{baseline,finetune}/seed_42/` los checkpoints, `metrics.json`, la matriz de confusión, el reporte por clase, la cola de errores, el historial y las curvas, `split_audit.json`, `class_weights.csv`, `metadata.json` y `emissions.csv`, junto con el run correspondiente en MLflow.

## Reglas

- No modifico `data/splits/*.csv` sin autorización y sin documentar el motivo.
- No copio las métricas del notebook 02 como resultados locales: provienen de una corrida Colab con batch 620, 10 épocas y `mixed_float16`.
- Reviso personalmente al menos 10 errores reales antes de dar por cerrada la revisión.
- Evalúo el test una sola vez al final; selecciono el checkpoint por `val_loss`.
