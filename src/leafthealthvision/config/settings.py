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

PLANTVILLAGE_CLASSES = (
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
)

# ---------------------------------------------------------------------------
# Imágenes
# ---------------------------------------------------------------------------
IMAGE_TARGET_SIZE = (224, 224)  # estándar para MobileNetV2 / EfficientNetB0
IMAGE_COLOR_MODE = "rgb"
IMAGE_RESCALE = 1.0 / 255.0

# ---------------------------------------------------------------------------
# Entrenamiento (valores de referencia, lógica aún no implementada)
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
OPTIMIZER = "adam"
LOSS = "categorical_crossentropy"
METRICS = ["accuracy"]
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5

# ---------------------------------------------------------------------------
# MLflow (seguimiento local de experimentos)
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "leafthealthvision"
MLFLOW_ARTIFACT_LOCATION = str(ROOT / "mlruns")

# ---------------------------------------------------------------------------
# Modelos candidatos (etapa futura)
# ---------------------------------------------------------------------------
MODEL_CANDIDATES = ["mobilenetv2", "efficientnetb0"]
INPUT_SHAPE = (224, 224, 3)
