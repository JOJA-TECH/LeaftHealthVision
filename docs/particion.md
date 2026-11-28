# Partición: decisión y procedencia

## Decisión

He conservado la partición de los CSV versionados, generada desde `raw/color`:

| Split | Imágenes |
|---|---:|
| train | 43 490 |
| val | 5 398 |
| test | 5 396 |
| **total** | **54 284** |

No modifiqué los CSV. Las cifras 43 486 / 5 398 / 5 401 corresponden al EDA anterior sobre `raw/segmented`; las mantengo solo como referencia histórica.

## Verificación

El 27 de noviembre de 2026 comprobé lo siguiente:

- `color`: 54 305 imágenes, 38 clases. `segmented`: 54 306 imágenes, 38 clases.
- Los 54 284 nombres de los manifiestos existen en `color`; ninguno existe en `segmented` ni lleva el sufijo `_final_masked`.
- Reproduje `scripts/generate_splits.py` sobre `color`: los tres CSV coinciden exactamente (`same_set = true` en train, val y test; 0 diferencias en ambos sentidos).
- Reproduje el EDA sobre `segmented`: obtuve 54 285 únicas y 43 486 / 5 398 / 5 401; no coincide con los CSV vigentes.
- `color` tiene 21 grupos de duplicados exactos (42 imágenes, 21 copias extra); tras deduplicar quedan 54 284 únicas.
- Verifiqué 0 duplicados `(categoria, archivo)`, 0 hojas compartidas entre splits y 0 filas sin resolver a `color`; 41 104 filas (75,72 %) tienen grupo de hoja, con 7 595 hojas.

Calculé los hashes de las imágenes de `color` y `segmented` y reproduje ambos algoritmos de partición sin modificar los CSV. La generación desde `color` es reproducible con `scripts/generate_splits.py` en un workspace aislado. Para la comprobación habitual de los manifiestos ejecuto:

```bash
python scripts/audit_splits.py
```

Este comando escribe `data/splits/SHA256SUMS` y `data/splits/partition_audit.json` sin tocar los CSV.

## Hashes

```text
55d2f58f421d1e41c4603fc27cfb6284365d32c8496a1bda25262b6f8d93bfc8  train.csv
b68230c19898d42f36d94b1496bb905149e5769fccfe5724264b023fbb0238d9  val.csv
259261fd581849f089f30bca6cc7249625bcea0625a11eedb0e33e5d84b75462  test.csv
```

## Regla

No sobrescribo `data/splits/*.csv` sin autorización y sin documentar el motivo. Si los regenero, lo hago en un workspace aislado y los comparo con estos hashes antes de reemplazarlos.
