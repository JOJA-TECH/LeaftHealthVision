"""Streamlit de despliegue: clasificación preliminar con MobileNetV2 en TFLite.

Resultado orientativo: no sustituye el diagnóstico de un especialista.
"""

from pathlib import Path

import numpy as np
import streamlit as st
from ai_edge_litert.interpreter import Interpreter
from PIL import Image

MODEL_PATH = Path(__file__).with_name("model.tflite")
IMAGE_SIZE = (224, 224)
UMBRAL_DEFECTO = 0.5

# Orden alfabético de las 38 clases del manifiesto canónico (data/splits/train.csv).
PLANTVILLAGE_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
    "Apple___healthy", "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot",
    "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def _cultivo_de_clase(clase: str) -> str:
    """Devuelve el cultivo asociado a una etiqueta PlantVillage (antes de '___')."""
    return clase.split("___", 1)[0] if "___" in clase else clase


@st.cache_resource
def cargar_modelo() -> Interpreter:
    interpreter = Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    return interpreter


def probabilidades(interpreter: Interpreter, image: Image.Image) -> np.ndarray:
    entrada = interpreter.get_input_details()[0]
    # El TFLite ya incluye el preprocesamiento MobileNetV2 ([0,255] -> [-1,1]).
    lote = np.expand_dims(np.asarray(image, dtype=entrada["dtype"]), 0)
    interpreter.set_tensor(entrada["index"], lote)
    interpreter.invoke()
    salida = interpreter.get_output_details()[0]
    return interpreter.get_tensor(salida["index"])[0]


def main() -> None:
    st.set_page_config(
        page_title="LeaftHealthVision — Clasificación de hojas", page_icon="🌿"
    )
    st.title("LeaftHealthVision — Clasificación de enfermedades de hojas")
    st.caption(
        "Prototipo educativo y experimental. Resultado orientativo, "
        "no es un diagnóstico agrícola."
    )

    st.sidebar.header("Modelo")
    st.sidebar.caption("MobileNetV2 (PlantVillage, TFLite).")
    umbral = st.sidebar.number_input(
        "Umbral de baja confianza",
        min_value=0.0,
        max_value=1.0,
        value=UMBRAL_DEFECTO,
        step=0.05,
        help="Por debajo de este valor se avisa de baja confianza. "
        "Defínelo con validación, nunca con el test.",
    )
    top_k = st.sidebar.slider("Clases más probables a mostrar", 1, 5, 3)

    st.sidebar.markdown("---")
    st.sidebar.warning(
        "Advertencia: el resultado es orientativo y no constituye un diagnóstico "
        "agrícola. No se recomiendan tratamientos ni acciones agronómicas a partir "
        "de esta clasificación. PlantVillage tiene imágenes en condiciones "
        "controladas; el desempeño puede disminuir en fotos reales."
    )

    archivo = st.file_uploader("Cargar imagen de una hoja", type=["jpg", "jpeg", "png"])
    if not archivo:
        st.info("Carga una imagen de hoja para obtener una clasificación preliminar.")
        return

    with st.spinner("Clasificando..."):
        image = Image.open(archivo).convert("RGB").resize(
            IMAGE_SIZE, Image.Resampling.BILINEAR
        )
        probs = probabilidades(cargar_modelo(), image)

    top_idx = [int(i) for i in np.argsort(probs)[::-1][:top_k]]
    clase_pred = PLANTVILLAGE_CLASSES[top_idx[0]]
    confianza = float(probs[top_idx[0]])

    st.subheader("Resultado")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Cultivo asociado", _cultivo_de_clase(clase_pred))
        st.metric("Clase estimada", clase_pred)
    with col2:
        st.metric("Confianza", f"{confianza:.2%}")

    if confianza < umbral:
        st.warning(
            f"Baja confianza ({confianza:.2%} < umbral {umbral:.2f}). "
            "Resultado poco fiable; revisar con un especialista."
        )
    else:
        st.success(f"Confianza {confianza:.2%} (umbral {umbral:.2f}).")

    st.markdown("### Clases más probables")
    for idx in top_idx:
        nombre = PLANTVILLAGE_CLASSES[idx]
        st.write(
            f"- **{nombre}**: {probs[idx]:.2%}  (cultivo: {_cultivo_de_clase(nombre)})"
        )

    st.image(image, caption="Imagen analizada", width="stretch")
    st.markdown("---")
    st.caption(
        "Limitaciones: PlantVillage (condiciones controladas) y PlantDoc (evaluación externa) "
        "no garantizan representatividad de Cartagena ni la región Caribe. Una puntuación alta "
        "tampoco garantiza que la predicción sea correcta."
    )


if __name__ == "__main__":
    main()
