# CellCounterAIModel
Cell Counter Application to count cells in a given tissue sample from an image.

Project scaffold created. Key files and folders:

- `environment.yml` - conda environment to create for development
- `requirements.txt` - minimal pip requirements
- `configs/default.yaml` - default experiment config
- `data/` - data folders (raw, marked, crops, manifests, synthetic)
- `src/` - source code (cli, data, models, train, eval, infer, utils)
- `scripts/` - helper scripts like `make_manifest.py`
- `experiments/` - place for checkpoints and logs
- `notebooks/` - EDA and visualization notebooks
- `docs/labeling_guidelines.md` - labeling instructions

Quick start:

1. Create the conda env: `conda env create -f environment.yml`
2. Activate it: `conda activate cell-counter`
3. Install pip requirements if needed: `pip install -r requirements.txt`

Development notes:
- Many modules are placeholders to be implemented. Start with `src/data/manifest_utils.py` and `src/data/dataset.py` when preparing your dataset.
