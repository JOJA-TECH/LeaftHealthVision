"""Genera manifiestos canónicos train/val/test para PlantVillage (raw/color).

Motivo: no existían `data/splits/{train,val,test}.csv` en el entorno; se
crean a partir de la descarga disponible con semilla determinista (42).

Estrategia de integridad:
- Se agrupan las imágenes por hash SHA-256 (contenido). Cada grupo (imagen
  única) se asigna a un solo split, lo que garantiza 0 duplicados y 0
  hashes compartidos entre splits.
- Si un mismo hash aparece bajo varias categorías (conflicto de etiquetado),
  se conserva la categoría más frecuente y se documenta el conflicto.
- Se barajan los items únicos con semilla 42 y se reparten 80/10/10.

No usa `leaf_grouping` porque sus nombres de archivo
(`FREC_<enfermedad> <NNN>.JPG`) no matchean directamente con los de
`raw/color` (`<Clase>___<enfermedad>_<UUID>.jpg`); un cruce fiable requeriría
mapeo no disponible. El barajado por imagen única con semilla fija es
reproducible y evita fuga de imágenes entre splits.

Salida: data/splits/{train,val,test}.csv con columnas
archivo,categoria,split,hash,hoja (hoja = nombre sin UUID ni extensión).
"""

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLOR_DIR = ROOT / "data" / "plantvillage" / "raw" / "color"
SPLIT_DIR = ROOT / "data" / "splits"
SEED = 42
TRAIN, VAL, TEST = 0.8, 0.1, 0.1

random.seed(SEED)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def leaf_name(nombre: str) -> str:
    """Nombre de 'hoja' aproximado: clase+enfermedad sin UUID ni extensión."""
    stem = Path(nombre).stem
    stem = stem.removesuffix("_final_masked")
    partes = stem.split("___")
    if len(partes) >= 2:
        return f"{partes[0]}___{partes[1]}"
    return stem


def main():
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    # hash -> lista de (categoria, nombre)
    hash_a_items = defaultdict(list)
    total = 0
    for clase_dir in sorted(COLOR_DIR.iterdir()):
        if not clase_dir.is_dir():
            continue
        categoria = clase_dir.name
        for img in sorted(clase_dir.iterdir()):
            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                h = sha256_file(img)
                hash_a_items[h].append((categoria, img.name))
                total += 1

    # Resolver conflictos de etiquetado: misma imagen bajo varias categorías.
    items_unicos = []  # (hash, categoria_elegida, nombre_representante)
    conflictos = []
    for h, items in hash_a_items.items():
        cats = Counter(c for c, _ in items)
        categoria_elegida = cats.most_common(1)[0][0]
        if len(cats) > 1:
            conflictos.append((h, dict(cats), categoria_elegida))
        nombre_rep = items[0][1]
        items_unicos.append((h, categoria_elegida, nombre_rep))

    total_unicos = len(items_unicos)
    print(
        f"Archivos totales: {total} | Imágenes únicas (por hash): {total_unicos} | "
        f"Conflictos de etiquetado (hash en varias categorías): {len(conflictos)}"
    )

    # Barajar y repartir 80/10/10 sobre items únicos.
    random.shuffle(items_unicos)
    n = total_unicos
    n_train = max(1, round(n * TRAIN))
    n_val = max(1, round(n * VAL))
    n_test = n - n_train - n_val
    if n_test < 1:
        n_train = max(1, n - 2)
        n_val = 1
        n_test = 1
    grupos = {
        "train": items_unicos[:n_train],
        "val": items_unicos[n_train : n_train + n_val],
        "test": items_unicos[n_train + n_val :],
    }

    splits_rows = {"train": [], "val": [], "test": []}
    conteos = {"train": Counter(), "val": Counter(), "test": Counter()}

    for split, items in grupos.items():
        for h, categoria, nombre_rep in items:
            # Escribir TODOS los archivos que comparten este hash en el mismo split.
            for categoria_arch, nombre_arch in hash_a_items[h]:
                # Usar la categoría resuelta para consistencia de clase.
                cat_final = categoria_elegida_para(h, hash_a_items)
                splits_rows[split].append(
                    {
                        "archivo": nombre_arch,
                        "categoria": cat_final,
                        "split": split,
                        "hash": h,
                        "hoja": leaf_name(nombre_arch),
                    }
                )
                conteos[split][cat_final] += 1

    # Verificación de integridad: ningún hash aparezca en más de un split.
    hash_a_split = {}
    for split in ("train", "val", "test"):
        for r in splits_rows[split]:
            h = r["hash"]
            if h in hash_a_split and hash_a_split[h] != split:
                raise AssertionError(
                    f"¡hash compartido entre splits ({hash_a_split[h]} / {split}): {h}!"
                )
            hash_a_split[h] = split
    # Dentro de un split puede haber duplicados exactos (mismo hash, varios
    # archivos); eso está permitido y no es fuga entre splits.

    clases = sorted(
        {c for items in hash_a_items.values() for c, _ in items}
    )
    for split in ("train", "val", "test"):
        cats = {r["categoria"] for r in splits_rows[split]}
        assert cats == set(clases), f"{split} no tiene las {len(clases)} clases"
        path = SPLIT_DIR / f"{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=["archivo", "categoria", "split", "hash", "hoja"]
            )
            w.writeheader()
            w.writerows(splits_rows[split])
        print(
            f"{split}: {len(splits_rows[split])} archivos, "
            f"{len({r['hash'] for r in splits_rows[split]})} imágenes únicas, "
            f"{len(cats)} clases, sha256={sha256_file(path)}"
        )

    if conflictos:
        with (SPLIT_DIR / "conflictos_etiquetado.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(conflictos, f, indent=2, ensure_ascii=False)
        print("Conflictos documentados en data/splits/conflictos_etiquetado.json")

    resumen = {
        "total_archivos": total,
        "imagenes_unicas": total_unicos,
        "clases": len(clases),
        "seed": SEED,
        "split_porcentaje": [TRAIN, VAL, TEST],
        "conteos_archivos": {s: sum(c.values()) for s, c in conteos.items()},
        "conteos_unicos": {
            s: len({r["hash"] for r in splits_rows[s]}) for s in ("train", "val", "test")
        },
        "sha256": {s: sha256_file(SPLIT_DIR / f"{s}.csv") for s in ("train", "val", "test")},
        "conflictos_etiquetado": len(conflictos),
    }
    with (SPLIT_DIR / "resumen.json").open("w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print("Resumen:", json.dumps(resumen, indent=2))


def categoria_elegida_para(h, hash_a_items):
    items = hash_a_items[h]
    cats = Counter(c for c, _ in items)
    return cats.most_common(1)[0][0]


if __name__ == "__main__":
    main()
