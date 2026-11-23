"""Configuración base del proyecto LeaftHealthVision.

Valores comunes para todo el equipo: semillas aleatorias, rutas de dataset,
parámetros de imagen, referencias de entrenamiento y MLflow.
Este archivo no contiene lógica de entrenamiento, inferencia ni preprocesamiento.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # raíz del proyecto

# ---------------------------------------------------------------------------
# Semillas para reproducibilidad
# ---------------------------------------------------------------------------
SEED = 42
NUMPY_SEED = 42
TENSORFLOW_SEED = 42
PYTHON_RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
DATA_DIR = ROOT / "data"

PLANTVILLAGE = {
    "source": "https://github.com/spMohanty/PlantVillage-Dataset",
    "license": "cc-by-sa-3.0",
    "local_path": DATA_DIR / "plantvillage",
    "image_subfolder": "raw/color",
    "image_size" : {
        "x" : 256,
        "y" : 256
    },
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "stratify_by_label": True,
    "respect_leaf_grouping": True,  # evitar data leakage por hoja física
}

PLANTDOC = {
    "source": "https://github.com/pratikkayal/PlantDoc-Dataset",
    "license": "CC-BY-4.0",
    "local_path": DATA_DIR / "plantdoc",
    "status": "pendiente_de_descarga",
}

# ---------------------------------------------------------------------------
# Imágenes
# ---------------------------------------------------------------------------
# Nota: EfficientNetB0(include_preprocessing=True) espera float32 en [0, 255] y
# normaliza internamente; no se aplica un Rescaling(1/255) externo.
IMAGE_TARGET_SIZE = (224, 224)  # estándar para MobileNetV2 / EfficientNetB0
IMAGE_COLOR_MODE = "rgb"

# ---------------------------------------------------------------------------
# Entrenamiento (valores de referencia del protocolo; los scripts usan configs/)
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
OPTIMIZER = "adam"
METRICS = ["accuracy"]
EARLY_STOPPING_PATIENCE = 5

# ---------------------------------------------------------------------------
# MLflow (seguimiento local de experimentos)
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = "./mlruns"  # "file://./mlruns" se parsea como UNC \\.\mlruns en Windows
MLFLOW_EXPERIMENT_NAME = "leafthealthvision"
MLFLOW_ARTIFACT_LOCATION = None

# ---------------------------------------------------------------------------
# Modelos candidatos (etapa futura)
# ---------------------------------------------------------------------------
MODEL_CANDIDATES = ["mobilenetv2", "efficientnetb0"]
INPUT_SHAPE = (224, 224, 3)
