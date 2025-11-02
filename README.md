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

1. Download and install python 3.11: https://www.python.org/downloads/release/python-3119/
2. Open main project root and run this command to tell the python launcher to use 3.11 and a create  virtual enviroment for it: py -3.11 -m venv .venv
3. If using VScode, set change to python intrepeter: on windows (ctrl + shift + p) then type Python: Select Interpreter, then from the list choose the one with '.venv' in its name
4. Install libraries, run the following commnad to activate the enviroment: .\.venv\Scripts\Activate.ps1
5. Install the packages: python -m pip install torch torchvision scikit-learn scikit-image albumentations opencv-python tqdm matplotlib numpy
6. Testing the data-loader: python test_classifier_loader.py
7. Training the model, saves the model to models/cell_classifier_best.pth: python src/train/train_classifier.py
8. After training, run inference (a sliding window that will scan an image of muscle tissue scanning cells. It will use a heatmap to assign confidence scores when it thinks it has found a cell): python src/train/train_classifier.py

Notes for Running/Testing Model. This will run on 1C RGB raw image as it is currently in the testing phase. It will output in quickcheck folder.
Run the following commands in sequential order:

.venv\Scripts\activate

python .\infer_single_overlay_improved.py --image "data\raw\1c RGB.tif" --weights models\cell_classifier_best.pth --out_dir experiments\quickcheck --device cpu --color_order rgb --window 200 --use_edge_gate --keep_border --border_px 2 --green_k 0.24 --tR 0.55 --tG 0.55 --tB 0.42 --delta_rg 0.16 --rg_ratio 0.82 --min_cc_area_gate 8 --max_cc_area_gate 240 --candidate_union_radius 5 --nms_radius 20 --post_tight_pair_floor 18 --fp_gate_proximity --fp_rededge --snap --snap_search_r 10 --snap_jump_limit 12 --snap_s_lo 150 --snap_v_lo 170 --h_lo 24 --h_hi 34 --s_lo 210 --v_lo 210 --min_area 8 --max_area 240 --threshold 0.92