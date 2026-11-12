"""Descarga el repositorio oficial de PlantVillage y deja solo lo necesario.

Clona https://github.com/spMohanty/PlantVillage-Dataset en data/plantvillage/
y elimina todo lo que no se usa
(scripts de generación, logs, datos para SVM, loader de Hugging Face, .git, etc.).

Se conserva:
  - raw/            -> imágenes (color, grayscale, segmented)
  - leaf_grouping/  -> metadatos de agrupación por hoja (evita data leakage)
  - leaf-map.json   -> mapeo de clases
  - README.md, CITATION.cff -> documentación y cita
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPO_URL = "https://github.com/spMohanty/PlantVillage-Dataset.git"
DEST = DATA_DIR / "plantvillage"

# Carpetas y archivos que se conservan del repositorio original.
KEEP = {
    "raw",
    "leaf_grouping",
    "leaf-map.json",
    "README.md",
    "CITATION.cff",
}


def main() -> None:
    if DEST.exists():
        print(f"{DEST} ya existe. Elimínalo si quieres volver a descargar.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Clonando {REPO_URL} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(DEST)],
        check=True,
    )

    # Quitar el .git anidado para no crear repositorio dentro de repositorio.
    git_dir = DEST / ".git"
    if git_dir.exists():
        shutil.rmtree(git_dir)
        print(".git del dataset eliminado.")

    # Eliminar todo lo que no se conserva.
    for item in DEST.iterdir():
        if item.name not in KEEP:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            print(f"Eliminado: {item.name}")

    print(f"\nDataset listo en: {DEST}")
    print("Contenido conservado:")
    for item in sorted(DEST.iterdir()):
        mark = "(dir)" if item.is_dir() else "(file)"
        print(f"  {item.name} {mark}")


if __name__ == "__main__":
    main()
