"""Prototipo Streamlit de LeaftHealthVision.

Clasificación preliminar de enfermedades de hojas con MobileNetV2 (Ruta F).
Resultado orientativo: no sustituye el diagnóstico de un especialista.
"""

from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

from leafthealthvision.config.settings import (
    EXPERIMENTS_DIR,
    IMAGE_TARGET_SIZE,
    PLANTVILLAGE_CLASSES,
)
from leafthealthvision.inference import load_model

UMBRAL_DEFECTO = 0.5


def _cultivo_de_clase(clase: str) -> str:
    """Devuelve el cultivo asociado a una etiqueta PlantVillage (antes de '___')."""
    if "___" in clase:
        return clase.split("___", 1)[0]
    return clase


def _modelo_por_defecto():
    """Devuelve el directorio de experimento más reciente con model.keras."""
    if not EXPERIMENTS_DIR.exists():
        return None
    # Busca recursivamente: experiments/mobilenetv2/<variant>/<seed>/model.keras
    modelos = list(EXPERIMENTS_DIR.rglob("model.keras"))
    if not modelos:
        return None
    modelos.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(modelos[0])


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
        help="Ej: experiments/mobilenetv2/frozen/seed_42/model.keras",
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
            model = load_model(modelo_path)
        except Exception as exc:  # noqa: BLE001
            st.error(f"No se pudo cargar el modelo: {exc}")
            st.stop()

        image = tf.keras.utils.load_img(archivo, target_size=IMAGE_TARGET_SIZE)
        batch = np.expand_dims(np.asarray(image), 0).astype(np.float32)
        logits = model.predict(batch, verbose=0)
        probs = tf.nn.softmax(logits).numpy()[0]

    top_idx = np.argsort(probs)[::-1][:top_k]
    clase_pred = (
        PLANTVILLAGE_CLASSES[int(top_idx[0])]
        if int(top_idx[0]) < len(PLANTVILLAGE_CLASSES)
        else str(int(top_idx[0]))
    )
    confianza = float(probs[top_idx[0]])
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
    for idx in top_idx:
        nombre = (
            PLANTVILLAGE_CLASSES[int(idx)]
            if int(idx) < len(PLANTVILLAGE_CLASSES)
            else str(int(idx))
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
