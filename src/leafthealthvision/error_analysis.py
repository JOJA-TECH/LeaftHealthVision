"""Análisis de errores del modelo sobre el conjunto de prueba.

Guarda casos erróneos (clase real, clase predicha, confianza) para revisión
humana. Cada caso se clasifica por tipo y se deja espacio para la hipótesis de
causa, que solo puede formularse tras inspección humana: nunca se inventan ni
se presentan como confirmadas sin evidencia.
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from leafthealthvision.config.settings import PLANTVILLAGE_CLASSES


def clasificar_error(tipo_propuesto: str | None = None) -> dict:
    """Devuelve una plantilla de caso de error con campos para revisión humana.

    ``tipo_propuesto`` es solo una sugerencia preliminar; el tipo definitivo y
    la hipótesis de causa deben ser revisados y validados por una persona.
    """
    return {
        "clase_real": "",
        "clase_predicha": "",
        "confianza": 0.0,
        "archivo": "",
        "tipo_error": tipo_propuesto or "pendiente_revision_humana",
        "hipotesis_causa": "pendiente_revision_humana",
        "origen": "datos_etiquetado_calidad_modelo_pendiente",
        "confirmado_por": "",
        "fecha_revision": "",
    }


def analizar_errores_test(
    model,
    manifest_test_path: Path,
    image_dir: Path,
    run_dir: Path,
    class_names=None,
    max_casos: int = 10,
    umbral_confianza: float | None = None,
) -> list[dict]:
    """Itera el test desde el manifiesto, predice imagen a imagen y guarda los
    casos erróneos en ``run_dir/casos_error_test.csv``.

    No usa el tf.data.Dataset (que no expone rutas), sino las rutas del
    manifiesto, para poder asociar cada error a su archivo.
    """
    import csv as _csv

    class_names = class_names or list(PLANTVILLAGE_CLASSES)
    class_to_index = {n: i for i, n in enumerate(class_names)}
    index_to_class = {i: n for n, i in class_to_index.items()}

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    casos = []
    with manifest_test_path.open(newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            if len(casos) >= max_casos:
                break
            archivo = row["archivo"]
            categoria = row["categoria"]
            if categoria not in class_to_index:
                continue
            ruta = image_dir / categoria / archivo
            if not ruta.exists():
                ruta = image_dir / archivo
            if not ruta.exists():
                continue

            image = tf.keras.utils.load_img(str(ruta), target_size=(224, 224))
            batch = np.expand_dims(np.asarray(image), 0).astype(np.float32)
            logits = model.predict(batch, verbose=0)
            probs = tf.nn.softmax(logits).numpy()[0]
            pred_index = int(np.argmax(probs))
            confianza = float(probs[pred_index])
            real_index = class_to_index[categoria]

            if pred_index != real_index:
                caso = clasificar_error()
                caso["clase_real"] = categoria
                caso["clase_predicha"] = index_to_class[pred_index]
                caso["confianza"] = confianza
                caso["archivo"] = str(ruta)
                caso["umbral_confianza"] = (
                    float(umbral_confianza) if umbral_confianza is not None else None
                )
                casos.append(caso)

    # Guardar CSV de casos erróneos
    out_csv = run_dir / "casos_error_test.csv"
    if casos:
        claves = list(casos[0].keys())
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=claves)
            writer.writeheader()
            writer.writerows(casos)
    else:
        out_csv.write_text("clase_real,clase_predicha,confianza,archivo,tipo_error,hipotesis_causa\n", encoding="utf-8")

    # Guardar JSON también para facilitar lectura
    with (run_dir / "casos_error_test.json").open("w", encoding="utf-8") as f:
        json.dump(casos, f, indent=2, ensure_ascii=False)

    print(f"Casos erróneos guardados: {len(casos)} en {out_csv}")
    return casos
