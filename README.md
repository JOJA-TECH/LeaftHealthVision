# LeaftHealthVision

Clasificación de enfermedades de hojas de cultivos con **MobileNetV2** y **EfficientNetB0** (transferencia de aprendizaje, PlantVillage), prototipo en **Streamlit**. Python 3.10 gestionado con **uv**.

**Ruta del proyecto:** F. Visión por computador.  
**Técnica central:** redes convolucionales y transferencia de aprendizaje.

## Estado actual (Rama `feature/MobileNetV2`)
- `src/leafthealthvision/training/mobilenet.py`: baseline MobileNetV2 ImageNet congelado + variante fine-tuned (config separada, opt-in). Carga train/val/test desde manifiestos (`data/splits/{train,val,test}.csv`), pesos de clase **sobre train**, preprocesamiento `mobilenet_v2.preprocess_input` ([0,255] → [-1,1]) como capa interna del modelo (una sola vez: entrenamiento == inferencia, sin doble aplicación), umbral de baja confianza definido **solo con validación** (nunca con test), evaluación final bloqueada en test y guardado de artefactos en `experiments/mobilenetv2/<variante>/seed_<n>/`.
- `src/leafthealthvision/data.py`: carga por manifiestos con integridad (sin hojas/hashes compartidos entre splits) + carga histórica por lotes conservada para interop con `feature/efficentnetb0`.
- `src/leafthealthvision/app.py`: prototipo Streamlit (clase, cultivo, probabilidades top-K, aviso de baja confianza, advertencia orientativa).
- `src/leafthealthvision/error_analysis.py`: guarda casos erróneos (real/predicha/confianza) para revisión humana.
- `src/leafthealthvision/evaluate_plantdoc.py`: evaluación externa en PlantDoc solo sobre clases mapeadas y verificadas por el equipo.
- `tests/test_mobilenetv2_pipeline.py`: pruebas ligeras sin descargar PlantVillage ni ImageNet.
- **BLOQUEO:** no hay manifiestos acordados ni imágenes en este entorno (`data/` no existe; red bloqueada en sandbox). `main()` no entrena ni divide por lotes. Entregue el bloqueo hasta que el equipo aporte los manifiestos y acuerde la fuente de verdad de los conteos.

## Comandos (menos de cinco)

```bash
# 1. Entorno (una sola vez)
source .venv/bin/activate        # o: uv sync

# 2. Descargar PlantVillage (requiere red; script listo para ejecutar localmente)
uv run python scripts/download_plantvillage.py

# 3. Generar manifiestos de partición (canónica, sin fuga por hoja)
#    Ejecutar notebooks/01_eda_plantvillage_crispmlq.ipynb -> data/splits/{train,val,test}.csv
#    [POR DEFINIR] fuente de verdad: 43.486/5.398/5.401 (README/EDA) vs 43.490/5.398/5.396 (CSV EfficientNet)

# 4. Entrenar MobileNetV2 (bloquea si faltan manifiestos del paso 3)
uv run python -m leafthealthvision.training.mobilenet
#    Variante fine-tuned: editar FINE_TUNE_ENABLED=True en config/settings.py
#    Testeo rápido del sistema (5-10 min en CPU, NO para resultados finales):
#    uv run python -m leafthealthvision.training.mobilenet --quick

# 5. Resultados finales en Colab gratuito (ejecutar notebooks/mobilenet_colab.ipynb)
#    y/o prototipo Streamlit:
uv run streamlit run src/leafthealthvision/app.py

# Pruebas automáticas (ligero, sin PlantVillage completo ni ImageNet)
uv run python tests/test_mobilenetv2_pipeline.py
```

## Ubicación de artefactos
Cada corrida guarda en:
`experiments/mobilenetv2/<variante>/seed_<n>/`
- `model.keras`, `config_effective.json`, `metrics.json` (incluye `umbral_confianza_validacion`)
- `classification_report.txt` / `.json`, `confusion_matrix.csv` / `.png`
- `history.json`, `training_curves.png`
- `casos_error_test.csv` / `.json` (análisis de errores, revisión humana)
- `plantdoc/` (métricas externas solo si el equipo aporta PlantDoc + mapeo verificado)
- Commit, versiones (Python/TensorFlow/MLflow/Keras), hardware, semilla, hashes SHA-256 de manifiestos y tamaños reales de split.

`variante`: `frozen` (baseline, por defecto) o `finetuned` (opt-in). `seed`: determinista (por defecto 42).

`data/`, `mlruns/`, `mlflow.db`, `.venv` y `experiments/` no se versionan (voluminosos/locales).

## Interoperabilidad con `feature/efficentnetb0`
Mismo tamaño de entrada (224×224 RGB), mismo orden de clases (`PLANTVILLAGE_CLASSES`), misma partición canónica por manifiestos y mismo procedimiento de evaluación (accuracy, macro-F1, matriz de confusión, reporte por clase, balanced accuracy, precisión/recall macro y weighted). Cada arquitectura usa su propio preprocesamiento oficial, aplicado una sola vez.

## Métricas y criterio de éxito (propuesta, no resultados garantizados)
- Métrica principal: **Macro-F1** (test de PlantVillage).
- Meta previa a experimentos: al menos un modelo ajustado alcance **Macro-F1 ≥ 0,75** y supere en **≥ 0,03** la línea base. Estos son objetivos, no resultados confirmados.
- Umbral de baja confianza: se define con validación, no con test.

## Análisis de errores
Al menos 10 casos erróneos del test se guardan en `casos_error_test.csv` con clase real, clase predicha, confianza y archivo. Tipo e hipótesis de causa se completan tras revisión humana; se marcan explícitamente como pendientes hasta inspeccionarse.

## Limitaciones
- No se ha ejecutado una corrida completa de entrenamiento (no hay manifiestos ni imágenes).
- No se declaran resultados de accuracy/F1 de una corrida real ni se presenta ninguna imagen de demostración como evaluación PlantDoc.
- No se declara cumplimiento completo del proyecto ni generalización a campo.
- PlantDoc no garantiza representatividad de cultivos/enfermedades/condiciones de Cartagena o el Caribe.
