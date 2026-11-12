# Datasets — PlantVillage y PlantDoc

## PlantVillage — fuente oficial
- Fuente verificada: https://github.com/spMohanty/PlantVillage-Dataset
- Licencia verificada: **cc-by-sa-3.0** (verificada en la página de Hugging Face del dataset)
- Paper: Mohanty et al. (2016), "Using Deep Learning for Image-Based Plant Disease Detection", Frontiers in Plant Science, DOI 10.3389/fpls.2016.01419.

### Cómo descargar (ejecutar en tu máquina)

```bash
source .venv/bin/activate
python scripts/download_plantvillage.py
# o: uv run python scripts/download_plantvillage.py
```

El script clona `https://github.com/spMohanty/PlantVillage-Dataset` (profundidad 1) en `data/plantvillage/`, elimina el `.git` anidado y **solo conserva lo necesario para clasificación por deep learning**:

Se conserva:
- `raw/` — imágenes (subcarpetas `color/`, `grayscale/`, `segmented/`)
- `leaf_grouping/` — metadatos de agrupación por hoja (crítico para train/test sin data leakage)
- `leaf-map.json` — mapeo de clases
- `README.md`, `CITATION.cff` — documentación y cita

Se elimina:
- `.git/` (no anidar repositorios)
- `scripts/`, `utils/` — scripts de generación/utilidades internas
- `logs/` — logs de ejecución
- `data_distribution_for_SVM/` — específico para SVM, no para deep learning
- `generated_for_paper/` — figuras del paper
- `plant_village.py`, `generate_data_for_SVM.py`, `generate_lmdb.sh`, `generate_mapstring.py` — scripts de generación y loader de Hugging Face
- `.gitignore`, `README_HF.md` — específicos del repositorio original

`data/` está en `.gitignore` (no versionar datos voluminosos).

## PlantDoc — solo documentado (no descargado)

- Fuente verificada: https://github.com/pratikkayal/PlantDoc-Dataset
- Licencia verificada: **CC-BY-4.0** (`LICENSE.txt`)
- Contenido: 2 598 datos; 13 especies vegetales; hasta 17 clases de enfermedades.
- Paper: "PlantDoc: A Dataset for Visual Plant Disease Detection", CoDS-COMAD 2020, arXiv 1911.10317, DOI 10.1145/3371158.3371196.
