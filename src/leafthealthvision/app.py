"""Prototipo Streamlit de LeaftHealthVision.

Clasificación preliminar de enfermedades de hojas con MobileNetV2 (Ruta F).
Resultado orientativo: no sustituye el diagnóstico de un especialista.
"""

from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

from leafthealthvision.config.settings import IMAGE_TARGET_SIZE
from leafthealthvision.plantvillage import load_splits
from leafthealthvision.training.common import project_root

UMBRAL_DEFECTO = 0.5


def _cultivo_de_clase(clase: str) -> str:
    """Devuelve el cultivo asociado a una etiqueta PlantVillage (antes de '___')."""
    if "___" in clase:
        return clase.split("___", 1)[0]
    return clase


def _modelo_por_defecto():
    """Devuelve el directorio de experimento más reciente con model.keras."""
    experiments_dir = project_root() / "experiments"
    if not experiments_dir.exists():
        return None
    modelos = list(experiments_dir.rglob("model.keras"))
    if not modelos:
        return None
    modelos.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(modelos[0])


def _clases_plantvillage():
    """Devuelve los 38 nombres de clase de PlantVillage en orden alfabético."""
    root = project_root()
    manifest_dir = root / "data" / "splits"
    if manifest_dir.exists():
        splits = load_splits(manifest_dir, root / "data" / "plantvillage" / "raw/color")
        return splits["train"].class_names
    # Fallback: lista canónica si no hay manifiestos
    return [
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


PLANTVILLAGE_CLASSES = _clases_plantvillage()


def main():
    st.set_page_config(
        page_title="LeaftHealthVision — Clasificación de hojas", page_icon="🌿"
    )
    st.title("LeaftHealthVision — Clasificación de enfermedades de hojas")
    st.caption(
        "Prototipo educativo y experimental. Resultado orientativo, "
        "no es un diagnóstico agrícola."
    )

    st.sidebar.header("Modelo")
    modelo_defecto = _modelo_por_defecto()
    modelo_path = st.sidebar.text_input(
        "Ruta al modelo (.keras)",
        value=str(modelo_defecto) if modelo_defecto else "",
        help="Ej: experiments/mobilenetv2/baseline/seed_42/model.keras",
    )
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

    if not modelo_path or not Path(modelo_path).exists():
        st.error("No se encontró el modelo. Indica una ruta válida a model.keras.")
        st.stop()

    # Cargar modelo y predecir
    with st.spinner("Cargando modelo y clasificando..."):
        try:
            model = tf.keras.models.load_model(modelo_path, compile=False)
        except Exception as exc:  # noqa: BLE001
            st.error(f"No se pudo cargar el modelo: {exc}")
            st.stop()

        image = tf.keras.utils.load_img(archivo, target_size=IMAGE_TARGET_SIZE)
        batch = np.expand_dims(np.asarray(image), 0).astype(np.float32)
        probabilities = model.predict(batch, verbose=0)
        probs = tf.nn.softmax(probabilities).numpy()[0]

    top_idx = np.argsort(probs)[::-1][:top_k]
    top_idx_int = [int(i) for i in top_idx]
    clase_pred = (
        PLANTVILLAGE_CLASSES[top_idx_int[0]]
        if top_idx_int[0] < len(PLANTVILLAGE_CLASSES)
        else str(top_idx_int[0])
    )
    confianza = float(probs[top_idx_int[0]])
    cultivo = _cultivo_de_clase(clase_pred)

    st.subheader("Resultado")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Cultivo asociado", cultivo)
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
    for idx in top_idx_int:
        nombre = (
            PLANTVILLAGE_CLASSES[idx]
            if idx < len(PLANTVILLAGE_CLASSES)
            else str(idx)
        )
        st.write(
            f"- **{nombre}**: {probs[idx]:.2%}  (cultivo: {_cultivo_de_clase(nombre)})"
        )

    st.image(image, caption="Imagen analizada", use_container_width=True)

    st.markdown("---")
    st.caption(
        "Limitaciones: PlantVillage (condiciones controladas) y PlantDoc (evaluación externa) "
        "no garantizan representatividad de Cartagena ni la región Caribe. Una puntuación alta "
        "tampoco garantiza que la predicción sea correcta."
    )


if __name__ == "__main__":
    main()
