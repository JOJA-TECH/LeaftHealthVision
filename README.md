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

Hay una verificación local real de EfficientNetB0 en
`experiments/efficientnetb0/verificacion/seed_42/` (baseline y fine-tuning): 2 épocas con
pasos limitados, evaluadas sobre las 5 396 imágenes del test canónico. La corrida de
MobileNetV2 en `experiments/local/mobilenetv2/frozen/seed_42/` se importó de
`origin/feature/MobileNetV2` para la demo: usó otros manifiestos, hardware distinto y sus
métricas se calculan sobre 160 imágenes. Ese origen está documentado en `provenance.json`
y la app lo advierte: **no es una comparación controlada**. Los valores del notebook 02
son de Colab y tampoco se copian como resultados locales. Los pendientes están en
`docs/experimentos.md`.

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

## MobileNetV2 (Ruta F)

El pipeline de MobileNetV2 sigue la misma estructura que EfficientNetB0 para que
ambas arquitecturas sean comparables (mismos manifiestos, resolución, preprocesamiento
y métricas).

## Ejecutar MobileNetV2 baseline congelado

```bash
uv run python -m leafthealthvision.training.mobilenetv2 --config configs/mobilenetv2_baseline.yaml
```

Entrena la cabeza sobre MobileNetV2 ImageNet congelada con `raw/color`, pesos por
clase calculados solo en train, semilla 42 y preprocesamiento oficial de MobileNetV2
(`[0,255]` -> `[-1,1]` como capa interna, aplicado una sola vez). Artefactos en
`experiments/mobilenetv2/baseline/seed_42/`.

## Ejecutar MobileNetV2 fine-tuning

```bash
uv run python -m leafthealthvision.training.finetune_mobilenetv2 --config configs/mobilenetv2_finetune.yaml
```

Requiere haber corrido el baseline: carga
`experiments/mobilenetv2/baseline/seed_42/best_model.keras`, descongela las últimas
20 capas (BatchNorm congelada recursivamente) y ajusta con lr 1e-5 sobre la misma
partición y semilla.

## Interfaz de evaluación comparativa

```bash
uv run streamlit run src/leafthealthvision/app.py
```

La paleta de la interfaz se define en `.streamlit/config.toml` (tonos azules). La app
lee dos fuentes reales: `experiments/efficientnetb0/colab_results/` (métricas finales de
MLflow, baseline y finetune) y `experiments/mobilenetv2/colab_results/`
(`metrics_baseline.json`, configuración efectiva, historial y artefactos). Halla tres
corridas: dos de EfficientNetB0 y una de MobileNetV2. Por defecto selecciona
EfficientNetB0 baseline como A y MobileNetV2 baseline como B. Cuatro secciones:

- **1. Resultados**: exactitud, exactitud equilibrada, precisión/recuperación/F1 macro y
  ponderado, número de muestras, CO₂eq, energía total y desglose CPU/RAM/GPU, duración,
  épocas, tamaño del artefacto y coste energético en COP (energía × tarifa editable en
  COP/kWh). Incluye definiciones operativas y señala diferencias de protocolo.
- **2. Inferencia**: una misma imagen para ambos modelos, clases con mayor probabilidad,
  acuerdo en la clase principal y latencia medida. MobileNetV2 devuelve logits y
  EfficientNetB0 probabilidades; se aplica softmax solo cuando hace falta.
- **3. Evidencias**: matriz de confusión, series históricas por época, curvas, reportes,
  cola de errores, hashes de manifiestos y procedencia de cada corrida.
- **4. Marco metodológico**: fuentes, criterios de comparabilidad, procedimiento de
  medición ambiental y referencias bibliográficas.

La tarifa eléctrica es editable en COP/kWh, con rango de referencia de 870 a 1200 y valor
inicial de 1035; no constituye una factura real. La latencia en vivo no incluye energía
por inferencia. Los resultados son descriptivos y no sustituyen un diagnóstico agrícola.

### Posibles ampliaciones

- Ejecutar MobileNetV2 con los manifiestos canónicos para que la comparación sea controlada.
- Medir energía y emisiones por inferencia (CodeCarbon) y mostrar coste por imagen en COP.
- Añadir eficiencia (F1 por kWh / por COP) y curvas de calibración desde validación.
- Comparar las clases más confundidas y exportar el informe a CSV/PDF.

## Despliegue en Vercel

Vercel no ejecuta Streamlit como función serverless: necesita un proceso HTTP con
WebSocket. Se usa el **Container Runtime** (beta) con `framework: container` en
`vercel.json` y el `Dockerfile` de la raíz. La imagen incluye el dashboard comparativo
completo (`deploy/app.py`, copia de `src/leafthealthvision/app.py` sin TensorFlow), la
capa de normalización `comparison.py`, los artefactos de las corridas de Colab y los
tres modelos en TFLite. `.dockerignore` excluye entrenamiento, notebooks, datasets y
corridas locales.

Se despliegan **solo los modelos entrenados en Colab**, con sus gemelos TFLite en
`deploy/models/`:

| Corrida | Artefacto de origen | TFLite | Accuracy (test) |
| --- | --- | --- | --- |
| EfficientNetB0 baseline | `experiments/efficientnetb0/colab_results/` | `efficientnet_baseline.tflite` | 0,9429 |
| EfficientNetB0 finetune | `experiments/efficientnetb0/colab_results/` | `efficientnet_finetune.tflite` | 0,95 |
| MobileNetV2 baseline | `experiments/mobilenetv2/colab_results/` | `mobilenet_baseline.tflite` | 0,9433 |

Los `.keras` originales se mantienen en la imagen como evidencia (tamaño y
procedencia), pero la inferencia usa TFLite: se eliminaron las capas de aumento de
datos, se sustituyó el `Lambda` de MobileNetV2 por `Rescaling(1/127.5, -1)` y se
forzó `float32` en EfficientNetB0 (el original usaba `mixed_float16`). La diferencia
máxima de probabilidades frente al modelo Keras es `8e-7`.

```bash
vercel deploy          # o conectar el repo en la UI de Vercel, sin comandos de build
```

Prueba local de la misma imagen:

```bash
docker build -t leafthealthvision .
docker run --rm -p 8080:80 leafthealthvision   # http://localhost:8080
```

`deploy/check.py` valida forma, rango y suma de probabilidades de los tres modelos
durante el build. Límites a tener en cuenta: escalado a cero tras 5 min de inactividad
(con pérdida de la sesión de Streamlit), hasta 4,5 MB por imagen subida y 2 GB de RAM
en Hobby. Para cambiar un modelo, reemplaza su TFLite en `deploy/models/` manteniendo
entrada `(1, 224, 224, 3)` float32 y salida de 38 probabilidades en orden alfabético.

## Pruebas MobileNetV2

```bash
uv run python tests/test_mobilenetv2_pipeline.py
```

Cubren integridad de manifiestos, preprocesamiento [-1,1], consistencia entrenamiento
vs inferencia, forma/rango de probabilidades y carga de modelo guardado.
