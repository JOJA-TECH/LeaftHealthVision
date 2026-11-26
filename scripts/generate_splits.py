"""Genera data/splits/{train,val,test}.csv para Colab (PlantVillage).

Replica la metodología del EDA local (01_eda_plantvillage_crispmlq.ipynb):
  * inventario de imágenes en raw/color
  * deduplicación exacta por SHA-256
  * grupos por hoja desde leaf_grouping/filtered_leafmaps/*.csv (si existen)
  * split 80/10/10 por grupo dentro de cada clase, sin fuga entre splits
  * exporta train.csv, val.csv, test.csv con columnas: archivo,categoria,split,hoja,origen

Uso en Colab (pegar el contenido en una celda y ejecutar):
    %run generate_splits.py
o bien copiar/pegar todo el cuerpo de main() directamente.

Requiere: pandas, numpy (ya presentes en Colab; si no: !pip install pandas numpy)
"""

from __future__ import annotations

import csv
import hashlib
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Rutas (ajusta si tu notebook usa otra raíz; en Colab el notebook usa
# /content/leafthealthvision. Sobrescribe con LEAF_WIP si pruebas en local).
# ---------------------------------------------------------------------------
WORK = Path(os.environ.get("LEAF_WORK", "/content/leafthealthvision"))
DATA_DIR = WORK / "data" / "plantvillage"
COLOR_DIR = DATA_DIR / "raw" / "color"
MANIFEST_DIR = WORK / "data" / "splits"
GROUPING_DIR = DATA_DIR / "leaf_grouping"
CSVS_DIR = GROUPING_DIR / "filtered_leafmaps"

SEED = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.80, 0.10, 0.10
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SUFFIX = "_final_masked"


def normalized_stem(name: str) -> str:
    """Nombre base sin UUID, extensión y sufijo _final_masked."""
    stem = Path(name).stem.lower().strip().removesuffix(SUFFIX)
    return stem.split("___", 1)[-1]


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def build_index(color_dir: Path):
    rows = []
    index = {}
    core_index = defaultdict(list)
    for class_dir in sorted(p for p in color_dir.iterdir() if p.is_dir()):
        category = class_dir.name
        for img in sorted(class_dir.iterdir()):
            if not img.is_file() or img.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            ns = normalized_stem(img.name)
            rows.append(
                {
                    "archivo": img.name,
                    "categoria": category,
                    "path": str(img),
                    "stem": ns,
                }
            )
            index[(category, ns)] = img
            core_index[(category, ns)].append(img)
    return rows, index, core_index


def load_leaf_groups(rows: list[dict]) -> dict[str, str]:
    """Devuelve dict ruta_imagen -> 'archivocsv|Leaf #' (vacío si no hay CSVs)."""
    grupo_de = {}
    if not CSVS_DIR.is_dir():
        return grupo_de

    key_to_paths = defaultdict(list)
    for r in rows:
        key_to_paths[(r["categoria"], r["stem"])].append(r["path"])

    classes_set = {r["categoria"] for r in rows}
    csvs = sorted(CSVS_DIR.glob("*.csv"))
    for csv_path in csvs:
        try:
            csv_df = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace")
        except Exception:
            csv_df = pd.read_csv(csv_path, encoding="latin-1")
        if "File Name" not in csv_df.columns:
            continue
        stems = [normalized_stem(str(v).strip()) for v in csv_df["File Name"]]
        leaves = [str(v).strip() for v in csv_df.get("Leaf #", [""] * len(stems))]

        if csv_path.stem in classes_set:
            category = csv_path.stem
        else:
            votos = Counter(c for s in stems for c in classes_set if (c, s) in key_to_paths)
            category = votos.most_common(1)[0][0] if votos else None
        if category is None:
            continue
        for stem, leaf_no in zip(stems, leaves):
            for img_path in key_to_paths.get((category, stem), []):
                grupo_de[img_path] = f"{csv_path.name}|{leaf_no}"
    return grupo_de


def main() -> None:
    if not COLOR_DIR.is_dir():
        raise FileNotFoundError(
            f"No existe {COLOR_DIR}. Descarga primero PlantVillage "
            "(el notebook lo clona en la celda '2. Descargar PlantVillage')."
        )

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    rows, _, _ = build_index(COLOR_DIR)
    print(f"Imágenes en color: {len(rows)}")

    # Grupos por hoja (vacío si no existe leaf_grouping/filtered_leafmaps)
    grupo_de = load_leaf_groups(rows)
    for r in rows:
        r["hoja"] = grupo_de.get(r["path"])

    # --- SHA-256 para deduplicar contenido exacto ---
    for r in rows:
        r["sha256"] = sha256_file(Path(r["path"]))

    dup_rows = [r for r in rows]
    dup_df = pd.DataFrame(dup_rows)
    dup_groups = dup_df.groupby("sha256").filter(lambda g: len(g) > 1)
    if len(dup_groups):
        print(
            f"Grupos de duplicados exactos: {dup_groups.sha256.nunique()} "
            f"| imágenes implicadas: {len(dup_groups)}"
        )

    # Agg por sha256: conservar primera copia; hoja = primera no nula entre copias.
    agg = {}
    for r in rows:
        h = r["sha256"]
        if h not in agg:
            agg[h] = {
                "path": r["path"],
                "categoria": r["categoria"],
                "archivo": r["archivo"],
                "hoja": r["hoja"],
            }
        elif agg[h]["hoja"] is None and r["hoja"] is not None:
            agg[h]["hoja"] = r["hoja"]

    hash_to_group = {}
    for h, a in agg.items():
        grupo = a["hoja"] if a["hoja"] else f"IMAGE|{a['path']}"
        hash_to_group[h] = grupo

    # Únicas por sha256, ordenadas por path para determinismo.
    unique_rows = []
    seen = set()
    for r in sorted(rows, key=lambda x: x["path"]):
        if r["sha256"] in seen:
            continue
        seen.add(r["sha256"])
        r["group_id"] = hash_to_group[r["sha256"]]
        unique_rows.append(r)

    df_unique = pd.DataFrame(unique_rows)
    print(f"Únicas tras deduplicar: {len(df_unique)}")

    # --- split por grupo dentro de cada clase (sin fuga) ---
    rng = np.random.default_rng(SEED)
    assignments = {}
    for category, subset in df_unique.groupby("categoria", sort=True):
        groups = list(subset.group_id.unique())
        rng.shuffle(groups)
        n = len(groups)
        n_train = max(1, int(round(TRAIN_FRAC * n)))
        n_val = max(1, int(round(VAL_FRAC * n))) if n >= 3 else 0
        if n >= 3:
            n_train = min(n_train, n - 2)
            n_val = min(n_val, n - n_train - 1)
        for g in groups[:n_train]:
            assignments[g] = "train"
        for g in groups[n_train : n_train + n_val]:
            assignments[g] = "val"
        for g in groups[n_train + n_val :]:
            assignments[g] = "test"

    df_unique["split"] = df_unique.group_id.map(assignments)

    print(
        df_unique.groupby("split").size().reindex(["train", "val", "test"]).fillna(0).astype(int).to_string()
    )
    print("Grupos compartidos entre splits:", int((df_unique.groupby("group_id").split.nunique() > 1).sum()))
    print(
        "Hashes compartidos entre splits:",
        int((df_unique.groupby("sha256").split.nunique() > 1).sum()),
    )
    tabla = pd.crosstab(df_unique.categoria, df_unique.split).reindex(
        columns=["train", "val", "test"], fill_value=0
    )
    print("Clases sin test:", int((tabla.test == 0).sum()))
    hoja_col = df_unique["hoja"] if "hoja" in df_unique.columns else pd.Series([None] * len(df_unique))
    print("Cobertura por hoja: %.1f%%" % (hoja_col.notna().mean() * 100))

    # --- exportar manifiestos ---
    export = pd.DataFrame(
        {
            "archivo": df_unique["archivo"],
            "categoria": df_unique["categoria"],
            "split": df_unique["split"],
            "hoja": df_unique.get("hoja", pd.Series([None] * len(df_unique))),
        }
    )
    export["origen"] = np.where(export.hoja.notna(), "hoja", "imagen")
    for nombre in ("train", "val", "test"):
        g = (
            export[export.split == nombre][["archivo", "categoria", "split", "hoja", "origen"]]
            .sort_values("archivo")
        )
        g.to_csv(MANIFEST_DIR / f"{nombre}.csv", index=False)

    print("Manifiestos guardados en:", MANIFEST_DIR)
    print(
        "Clases por split:",
        {
            n: int(export.loc[export.split.eq(n), "categoria"].nunique())
            for n in ("train", "val", "test")
        },
    )


if __name__ == "__main__":
    main()
