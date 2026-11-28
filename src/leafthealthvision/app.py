"""Interfaz de evaluación comparativa de los modelos locales de LeaftHealthVision.

Presenta métricas de prueba, huella de carbono, energía, coste eléctrico y evidencias
de cada corrida, además de la inferencia de una misma imagen con dos modelos.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from leafthealthvision.config.settings import IMAGE_TARGET_SIZE
from leafthealthvision.evaluation.comparison import (
    METRIC_LABELS,
    RunReport,
    comparability_notes,
    discover_runs,
    estimate_cost,
)
from leafthealthvision.models.mobilenet_v2 import MobileNetV2Preprocess  # noqa: F401 (registra la capa)

ROOT = Path(__file__).resolve().parents[2]
UMBRAL_DEFECTO = 0.5
TARIFA_COP_DEFECTO = 1035.0  # COP/kWh, punto medio del rango de referencia 870–1200

# Los modelos de Colab se guardaron con una clave `quantization_config` que la
# versión local de Keras no acepta; se descarta solo esa clave.
_original_dense_from_config = tf.keras.layers.Dense.from_config.__func__


def _dense_from_config(cls, config):
    config = dict(config)
    config.pop("quantization_config", None)
    return _original_dense_from_config(cls, config)


tf.keras.layers.Dense.from_config = classmethod(_dense_from_config)

# El MobileNetV2 de Colab serializó una Lambda con bytecode de Python 3.13;
# se sustituye por la función equivalente para poder cargarla en Python local.
def _mobilenet_preprocess(inputs):
    return inputs / 127.5 - 1.0


_original_lambda_from_config = tf.keras.layers.Lambda.from_config.__func__


def _lambda_from_config(cls, config, custom_objects=None, safe_mode=None):
    if config.get("name") == "mobilenetv2_preprocess":
        return cls(
            function=_mobilenet_preprocess,
            name=config["name"],
            trainable=config.get("trainable", True),
            dtype=config.get("dtype"),
        )
    return _original_lambda_from_config(
        cls, config, custom_objects=custom_objects, safe_mode=safe_mode
    )


tf.keras.layers.Lambda.from_config = classmethod(_lambda_from_config)

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

DEFINICIONES = [
    ("Exactitud", "Proporción de predicciones correctas sobre el conjunto de prueba."),
    ("Exactitud equilibrada", "Media de la recuperación por clase; evita el sesgo de clases mayoritarias."),
    ("Precisión y recuperación macro", "Media no ponderada de la precisión y la recuperación de las 38 clases."),
    ("F1 macro", "Media no ponderada de la media armónica entre precisión y recuperación por clase."),
    ("F1 ponderado", "Media de F1 por clase ponderada por el número de muestras de cada clase."),
    ("kg CO₂eq", "Emisiones de gases de efecto invernadero estimadas por CodeCarbon, expresadas en dióxido de carbono equivalente."),
    ("kWh", "Energía eléctrica estimada por CodeCarbon para el proceso de entrenamiento."),
]


def _cultivo_de_clase(clase: str) -> str:
    """Devuelve el cultivo asociado a una etiqueta PlantVillage."""
    return clase.split("___", 1)[0] if "___" in clase else clase


def _fmt(value, key: str | None = None, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "N/D"
    if key == "n_test":
        return f"{int(value):,}"
    if isinstance(value, int):
        return str(value)
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def _fmt_cop(value) -> str:
    return "N/D" if value is None else f"${value:,.4f} COP"


def _default_index(runs: list[RunReport], *needles: str) -> int:
    for index, run in enumerate(runs):
        if all(needle.lower() in run.name.lower() for needle in needles):
            return index
    return 0


def _clases_de(run: RunReport, num_classes: int) -> list[str]:
    names = run.class_names if len(run.class_names) == num_classes else PLANTVILLAGE_CLASSES
    return names if len(names) == num_classes else [str(i) for i in range(num_classes)]


@st.cache_resource(show_spinner="Cargando artefacto del modelo…")
def _load_model(model_path: str):
    # MobileNetV2 usa una capa personalizada registrada; el mapa también sirve
    # para EfficientNetB0, que no la necesita.
    # Los artefactos son locales; el MobileNetV2 de Colab serializó una capa Lambda.
    return tf.keras.models.load_model(
        model_path,
        compile=False,
        safe_mode=False,
        custom_objects={"MobileNetV2Preprocess": MobileNetV2Preprocess},
    )


def _probabilities(model, batch: np.ndarray) -> np.ndarray:
    """Respeta softmax del modelo; aplica softmax solo si la cabeza da logits."""
    output = np.asarray(model(batch, training=False))[0]
    activation = getattr(getattr(model.layers[-1], "activation", None), "__name__", "")
    if "softmax" not in activation:
        output = tf.nn.softmax(output).numpy()
    return output


def _tab_resultados(first: RunReport, second: RunReport, tarifa: float) -> None:
    notas = comparability_notes(first, second)
    if notas:
        st.info(
            "**Condición de comparabilidad.** " + " ".join(notas)
            + " Los valores se transcripten de los artefactos originales y no constituyen un experimento controlado."
        )
    else:
        st.info("**Condición de comparabilidad.** No se detectaron diferencias de protocolo entre las corridas seleccionadas.")

    st.header("1.1 Desempeño en el conjunto de prueba")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(f"Exactitud · {first.model}", _fmt(first.metrics.get("accuracy")))
        st.metric("F1 macro", _fmt(first.metrics.get("f1_macro")))
    with col_b:
        st.metric(f"Exactitud · {second.model}", _fmt(second.metrics.get("accuracy")))
        st.metric("F1 macro", _fmt(second.metrics.get("f1_macro")))

    filas = []
    for clave, etiqueta in METRIC_LABELS.items():
        valor_a = first.metrics.get(clave)
        valor_b = second.metrics.get(clave)
        delta = valor_b - valor_a if valor_a is not None and valor_b is not None and clave != "n_test" else None
        filas.append(
            {
                "Indicador": etiqueta,
                "Corrida A": _fmt(valor_a, clave),
                "Corrida B": _fmt(valor_b, clave),
                "Diferencia (B−A)": _fmt(delta, clave, signed=True) if clave != "n_test" else "—",
            }
        )
    st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)
    st.markdown("**Definiciones operativas.** La tabla siguiente precisa el significado de cada indicador.")
    st.dataframe(
        pd.DataFrame(DEFINICIONES, columns=["Indicador", "Definición operativa"]),
        hide_index=True,
        use_container_width=True,
    )

    st.header("1.2 Huella de carbono y coste computacional")

    def _tamano_mb(run: RunReport):
        return round(run.model_path.stat().st_size / 1_000_000, 1) if run.model_path else None

    filas_coste = [
        ("Emisiones de entrenamiento (kg CO₂eq)", first.emissions_kg, second.emissions_kg, _fmt),
        ("Energía de entrenamiento (kWh)", first.energy_kwh, second.energy_kwh, _fmt),
        ("Duración del entrenamiento (s)", first.training_seconds, second.training_seconds, _fmt),
        ("Coste energético estimado (COP)", estimate_cost(first.energy_kwh, tarifa), estimate_cost(second.energy_kwh, tarifa), _fmt_cop),
        ("Épocas ejecutadas", first.epochs, second.epochs, _fmt),
        ("Tamaño del artefacto (MB)", _tamano_mb(first), _tamano_mb(second), _fmt),
    ]
    st.dataframe(
        pd.DataFrame(
            [
                {"Concepto": etiqueta, "Corrida A": formato(valor_a), "Corrida B": formato(valor_b)}
                for etiqueta, valor_a, valor_b, formato in filas_coste
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.markdown(
        f"**Procedimiento de estimación.** El coste se calcula como $C = E \\times p$, donde $E$ es la energía "
        f"registrada por CodeCarbon (kWh) y $p = {tarifa:,.0f}$ COP/kWh es una tarifa supuesta, modificable en la "
        "barra lateral. La estimación cubre el entrenamiento registrado y no incluye infraestructura externa."
    )
    desglose = pd.DataFrame({"A": first.energy_breakdown, "B": second.energy_breakdown}).T.rename(
        columns={"cpu_energy": "CPU (kWh)", "ram_energy": "RAM (kWh)", "gpu_energy": "GPU (kWh)"}
    )
    if not desglose.empty:
        st.markdown("**Distribución de la energía por componente.**")
        st.dataframe(desglose.fillna("N/D"), use_container_width=True)

    datos = []
    for metrica in ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"):
        datos.append({"Indicador": METRIC_LABELS[metrica], "A": first.metrics.get(metrica), "B": second.metrics.get(metrica)})
    grafico = pd.DataFrame(datos).set_index("Indicador")
    if grafico.notna().any().any():
        st.markdown("**Comparación gráfica de indicadores seleccionados.**")
        st.bar_chart(grafico)


def _tab_inferencia(first: RunReport, second: RunReport, umbral: float, top_k: int) -> None:
    st.markdown(
        "Se presenta una única imagen RGB, redimensionada a 224 × 224 píxeles y entregada a ambos modelos en el "
        "intervalo [0, 255]. Cada arquitectura aplica su preprocesamiento interno: EfficientNetB0 normaliza en su "
        "propia grafo y MobileNetV2 transforma a [−1, 1] mediante una capa registrada. La conversión softmax se "
        "aplica únicamente a las cabezas que entregan logits."
    )
    archivo = st.file_uploader("Seleccione una imagen de hoja", type=["jpg", "jpeg", "png"])
    if not archivo:
        st.info("Se requiere una imagen para ejecutar la inferencia comparativa.")
        return

    image = tf.keras.utils.load_img(archivo, target_size=IMAGE_TARGET_SIZE)
    batch = np.expand_dims(np.asarray(image), 0).astype(np.float32)
    resultados: dict[str, tuple[np.ndarray, list[str], float]] = {}

    with st.spinner("Ejecutando inferencia con ambos modelos…"):
        for clave, run in (("A", first), ("B", second)):
            if not run.model_path or not Path(run.model_path).is_file():
                st.info(f"Corrida {clave}: no hay artefacto `model.keras` en `{run.path}`.")
                continue
            try:
                model = _load_model(str(run.model_path))
                inicio = time.perf_counter()
                probabilidades = _probabilities(model, batch)
                latencia = time.perf_counter() - inicio
            except Exception as exc:  # noqa: BLE001
                st.info(f"Corrida {clave}: no fue posible cargar el artefacto `{run.model_path}`. Detalle: {exc}")
                continue
            resultados[clave] = (probabilidades, _clases_de(run, len(probabilidades)), latencia)

    st.image(image, caption="Imagen de entrada", use_container_width=True)
    columnas = st.columns(2)
    for columna, clave in zip(columnas, ("A", "B")):
        with columna:
            st.subheader(f"Corrida {clave} · {first.model if clave == 'A' else second.model}")
            if clave not in resultados:
                continue
            probabilidades, nombres, latencia = resultados[clave]
            indices = np.argsort(probabilidades)[::-1][:top_k]
            mejor = int(indices[0])
            confianza = float(probabilidades[mejor])
            st.metric("Clase estimada", nombres[mejor])
            st.metric("Cultivo", _cultivo_de_clase(nombres[mejor]))
            st.metric("Probabilidad máxima", f"{confianza:.2%}")
            st.markdown(f"**Latencia medida:** {latencia * 1000:.0f} ms en CPU, sin incluir la carga del modelo.")
            if confianza < umbral:
                st.info(
                    f"La probabilidad máxima ({confianza:.2%}) es inferior al umbral definido ({umbral:.2f}). "
                    "Se recomienda contraste con un especialista."
                )
            st.markdown("**Clases con mayor probabilidad**")
            for indice in indices:
                st.markdown(f"- {nombres[int(indice)]}: **{probabilidades[int(indice)]:.2%}**")

    if len(resultados) == 2:
        mejor_a = int(np.argmax(resultados["A"][0]))
        mejor_b = int(np.argmax(resultados["B"][0]))
        nombre_a = resultados["A"][1][mejor_a]
        nombre_b = resultados["B"][1][mejor_b]
        if nombre_a == nombre_b:
            st.info(f"**Acuerdo entre modelos.** Ambas corridas coinciden en la clase de mayor probabilidad: {nombre_a}.")
        else:
            st.info(
                "**Acuerdo entre modelos.** Las corridas difieren en la clase de mayor probabilidad: "
                f"A → {nombre_a}; B → {nombre_b}."
            )


def _tab_evidencias(first: RunReport, second: RunReport) -> None:
    st.markdown(
        "Se relacionan los artefactos generados por cada corrida. Las imágenes y tablas se leen directamente del "
        "directorio del experimento; no se recalculan en esta interfaz."
    )
    for clave, run in (("A", first), ("B", second)):
        with st.expander(f"Corrida {clave} · {run.label} · `{run.name}`", expanded=True):
            st.markdown(f"**Ruta del experimento.** `{run.path}`")
            if run.provenance:
                st.markdown("**Procedencia.**")
                st.json(run.provenance)
            if run.history:
                st.markdown("**Serie histórica por época.**")
                st.line_chart(pd.DataFrame(run.history))
            if run.split_sizes:
                st.markdown("**Partición declarada.** " + ", ".join(f"{k}={v:,}" for k, v in run.split_sizes.items()))
            if run.split_test_sha256:
                st.markdown(f"**Hash SHA-256 del manifiesto de prueba.** `{run.split_test_sha256}`")
            if run.hardware:
                st.markdown(f"**Plataforma de ejecución.** {run.hardware}")
            for etiqueta, ruta in run.artifacts.items():
                st.markdown(f"**{etiqueta}**")
                if ruta.suffix == ".png":
                    st.image(str(ruta), caption=etiqueta, use_container_width=True)
                elif ruta.suffix == ".txt":
                    st.text(ruta.read_text(encoding="utf-8", errors="replace")[:5000])
                else:
                    st.dataframe(pd.read_csv(ruta).head(20), hide_index=True, use_container_width=True)


def _tab_marco(first: RunReport, second: RunReport, tarifa: float) -> None:
    st.header("2.1 Fuentes y trazabilidad")
    st.markdown(
        "Las corridas se leen de `experiments/efficientnetb0/colab_results/` —métricas finales del SQLite de "
        "MLflow— y de `experiments/mobilenetv2/colab_results/` —`metrics_baseline.json`, "
        "`config_effective_baseline.json`, `history_baseline.json` y artefactos—. En ambos casos la energía y "
        "las emisiones se toman de `emissions.csv` y el reporte de clasificación aporta las clases y el soporte. "
        "Las series por época se presentan únicamente como histórico de entrenamiento, nunca como cifras de "
        f"test. La tarifa empleada en la estimación económica es de {tarifa:,.0f} COP/kWh."
    )
    for clave, run in (("A", first), ("B", second)):
        st.markdown(
            f"- **Corrida {clave}** ({run.label}): {run.n_test or 'N/D'} imágenes de prueba; "
            f"{run.epochs or 'N/D'} épocas; procedencia: {run.provenance.get('origen', 'artefactos del repositorio')}."
        )

    st.header("2.2 Criterios de comparabilidad")
    notas = comparability_notes(first, second)
    if notas:
        st.markdown("Se identifican las siguientes diferencias observables entre protocolos:")
        for nota in notas:
            st.markdown(f"- {nota}")
    else:
        st.markdown("No se identifican diferencias observables en muestra, manifiesto, épocas, plataforma u origen.")
    st.markdown(
        "La comparación solo es válida cuando coinciden la partición, el número de muestras evaluadas, el protocolo "
        "de entrenamiento y la plataforma de medición. En caso contrario, las cifras se presentan como evidencia "
        "descriptiva y no como conclusión de superioridad de una arquitectura."
    )

    st.header("2.3 Medición ambiental y económica")
    st.markdown(
        "CodeCarbon estima la energía a partir de la potencia de CPU, RAM y GPU y la convierte a emisiones mediante "
        "un factor de intensidad carbonosa regional. El coste se deriva de la energía estimada y de una tarifa "
        "eléctrica supuesta, con rango de referencia de 870 a 1200 COP/kWh; no constituye una factura ni incluye "
        "costes de infraestructura, almacenamiento o mantenimiento."
    )

    st.header("2.4 Referencias")
    st.markdown(
        """
        1. Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML.
        2. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR.
        3. Lacoste, A., Luccioni, A., Schmidt, V., & Dandres, T. (2019). *Quantifying Carbon Footprint for Training and Evaluation of Deep Learning Models*. arXiv:1907.10597.
        4. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825–2830.
        5. Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). *PlantVillage Dataset*. https://github.com/spMohanty/PlantVillage-Dataset
        """
    )


def main() -> None:
    st.set_page_config(page_title="LeaftHealthVision | Evaluación comparativa", layout="wide")
    st.title("Evaluación comparativa de modelos locales para clasificación de enfermedades foliares")
    st.markdown(
        "**Objetivo.** Contrastar el desempeño, la huella de carbono y el coste computacional de dos arquitecturas "
        "entrenadas sobre PlantVillage, conservando la procedencia de cada corrida y las condiciones de medición."
    )

    runs = discover_runs(ROOT / "experiments")
    if not runs:
        st.info("No se encontraron corridas con `metrics.json` en `experiments/`. Ejecute previamente un entrenamiento.")
        st.stop()

    etiquetas = [run.label for run in runs]
    st.sidebar.header("Configuración del análisis")
    etiqueta_a = st.sidebar.selectbox("Corrida A", etiquetas, index=_default_index(runs, "efficientnetb0", "baseline"))
    etiqueta_b = st.sidebar.selectbox("Corrida B", etiquetas, index=_default_index(runs, "mobilenetv2"))
    first = runs[etiquetas.index(etiqueta_a)]
    second = runs[etiquetas.index(etiqueta_b)]
    tarifa = st.sidebar.number_input(
        "Tarifa eléctrica (COP/kWh)", min_value=0.0, max_value=5000.0,
        value=TARIFA_COP_DEFECTO, step=10.0,
        help="Rango de referencia: 870–1200 COP/kWh. Valor empleado en la estimación; no representa una factura real.",
    )
    umbral = st.sidebar.slider("Umbral de baja confianza", 0.0, 1.0, UMBRAL_DEFECTO, 0.05)
    top_k = st.sidebar.slider("Número de clases a presentar", 1, 5, 3)
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Las corridas disponibles pueden diferir en partición, plataforma y protocolo. Consulte la sección "
        "**Condición de comparabilidad** antes de interpretar las diferencias."
    )

    pestaña_resultados, pestaña_inferencia, pestaña_evidencias, pestaña_marco = st.tabs(
        ["1. Resultados", "2. Inferencia", "3. Evidencias", "4. Marco metodológico"]
    )
    with pestaña_resultados:
        _tab_resultados(first, second, tarifa)
    with pestaña_inferencia:
        _tab_inferencia(first, second, umbral, top_k)
    with pestaña_evidencias:
        _tab_evidencias(first, second)
    with pestaña_marco:
        _tab_marco(first, second, tarifa)


if __name__ == "__main__":
    main()
