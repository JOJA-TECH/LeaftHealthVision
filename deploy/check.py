"""Comprobación de los modelos empaquetados; se ejecuta al construir la imagen."""

from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter

for nombre in (
    "efficientnet_baseline.tflite",
    "efficientnet_finetune.tflite",
    "mobilenet_baseline.tflite",
):
    interpreter = Interpreter(model_path=str(Path(__file__).parent / "models" / nombre))
    interpreter.allocate_tensors()
    entrada, salida = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    assert tuple(entrada["shape"]) == (1, 224, 224, 3)
    assert tuple(salida["shape"]) == (1, 38)
    interpreter.set_tensor(entrada["index"], np.zeros(entrada["shape"], dtype=entrada["dtype"]))
    interpreter.invoke()
    probabilidades = interpreter.get_tensor(salida["index"])[0]
    assert np.isfinite(probabilidades).all() and abs(float(probabilidades.sum()) - 1.0) < 1e-3
    print(f"{nombre} OK: entrada (1, 224, 224, 3), salida 38 probabilidades")
