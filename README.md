# Xilinx Package

End-to-end graph ML pipeline for Xilinx netlists:

- HAL netlist cleanup/export (`.v` -> `.gexf`)
- data preparation and labeling (`.gexf` -> labeled split `.gexf` graphs)
- tensorization to PyG (`.gexf` -> `.pt`)
- model training (`gcn-4-skip`, `gat`, `gatv2`)
- OOD-style inference using no-BP/BP energy thresholding

Run commands from the Package root `Xilinx_Package`

## Repository layout

- `Data/`
  - raw and generated data roots used by preprocessing/training
- `Scripts/Data_Preparation_Scripts/`
  - `raw_to_processed_gexf.py`, `label_train_ips.py`, `label_soc.py`, `data_prep.py`
- `Scripts/Model_Training_and_Inference_Scripts/`
  - `models.py`, `train_eval.py`, `ood_inference.py`
- `Scripts/HAL_Verilog_Scripts/`
  - HAL-side preprocessing notes/scripts
- `work_dirs_xilinx_<TARGET_SUFFIX>/`
  - training outputs per run tag
- `Inference_Output_<TARGET_SUFFIX>/`
  - inference/OOD outputs per run tag and BP mode
- `experiment_results.txt`
  - global append-only run ledger

For detailed script docs:

- HAL workflow: `Scripts/HAL_Verilog_Scripts/HAL_README.md`
- data prep: `Scripts/Data_Preparation_Scripts/DATA_PREP_README.md`
- model + OOD: `Scripts/Model_Training_and_Inference_Scripts/MODEL_README.md`

## Environment setup

This project assumes Conda is installed.

If `conda` is not available on your system, install Conda first (Miniconda or Anaconda), then continue.

Quick check:

```bash
conda --version
```

### Recommended: use the setup script

Use the provided `setup_env.sh` to create and configure a new Conda environment with pinned versions.

```bash
# from Xilinx_Package root
bash setup_env.sh --cuda 121
# or
bash setup_env.sh --cuda 128
```

Notes:

- `--cuda 121`: use when your system/driver supports CUDA 12.1 wheels.
- `--cuda 128`: use when your system/driver supports CUDA 12.8 wheels.
- the script creates environment `project_env`.

Activate after setup:

```bash
conda activate project_env
```

### Alternative: manual environment setup using requirements.txt

If you prefer manual setup:

```bash
conda create -y -n project_env python=3.10
conda activate project_env
pip install -r requirements.txt
```

If CUDA-specific PyTorch wheel compatibility issues occur with manual install, prefer `setup_env.sh`.

## Standard run (test case: microblaze target)

What this does overall:

- starts from HAL-exported raw `.gexf` netlist graphs,
- builds normalized preprocessed graphs,
- applies target-specific labels for train/eval/test/inference splits,
- converts labeled graphs into PyG `.pt` tensors,
- trains one model run,
- runs OOD-style inference for no-BP and BP modes.

HAL prerequisite:

- run the HAL workflow first (GUI + HAL Python shell) to generate/refresh:
  - `Data/Raw_Files/*.gexf` (IP graphs)
  - `Data/Test_Raw_Files/*.gexf` (SoC graphs)
- see `Scripts/HAL_Verilog_Scripts/HAL_README.md` before running the standard CLI flow below.

```bash
TARGET_SUFFIX=MB
DATA_ROOT="Data/Data_Xilinx_${TARGET_SUFFIX}"

# 1) Preprocess raw graphs
#    Purpose: normalize attributes and produce pre_processed_*.gexf/.json
#    Inputs: Data/Raw_Files, Data/Test_Raw_Files
#    Outputs: Data/Preprocessed_Files, Data/Test_Preprocessed_Files
python Scripts/Data_Preparation_Scripts/raw_to_processed_gexf.py --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --dataset ip --verbose
python Scripts/Data_Preparation_Scripts/raw_to_processed_gexf.py --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --dataset soc --verbose

# 2) Label split graphs for the selected target IP
#    Purpose: create binary labels and split raw graphs for train/eval/test/inference
#    Inputs: preprocessed graphs from step 1 -> Data/Preprocessed_Files/*.gexf, Data/Test_Preprocessed_Files/*.gexf
#    Outputs: Data/Processed_Files_${TARGET_SUFFIX}, Data/Test_Processed_Files_${TARGET_SUFFIX}, Data/Data_Xilinx_${TARGET_SUFFIX}/{train,eval,test,inference}/raw
python Scripts/Data_Preparation_Scripts/label_train_ips.py --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --target-ip microblaze --ip-list axi_pcie,ila,jtag_axi,microblaze,pcie_7x,mig_7series
python Scripts/Data_Preparation_Scripts/label_soc.py --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --target-ip microblaze --soc-file mb_axi_pcie_jtag_ila

# 3) Build PyG tensors
#    Purpose: format node attributes and convert split raw .gexf graphs into model-ready .pt tensors 
#    Inputs: Data/Data_Xilinx_${TARGET_SUFFIX}/*/raw
#    Outputs: Data/Data_Xilinx_${TARGET_SUFFIX}/*/processed
python Scripts/Data_Preparation_Scripts/data_prep.py --raw-root "$DATA_ROOT" --output-root "$DATA_ROOT"

# 4) Train model (example: GAT)
#    Purpose: train + validate, save best checkpoint and run manifest
#    Outputs: work_dirs_xilinx_${TARGET_SUFFIX}/${RUN_TAG}/
RUN_TAG=gcn4_run01
CKPT_DIR="work_dirs_xilinx_${TARGET_SUFFIX}/${RUN_TAG}"
mkdir -p "$CKPT_DIR"
python Scripts/Model_Training_and_Inference_Scripts/train_eval.py \
  --data-root "$DATA_ROOT" \
  --checkpoint-dir "$CKPT_DIR" \
  --model gcn4 \
  --device auto \
  --hidden-channels 128 --num-layers 4 --dropout 0.1 --heads 1 \
  --epochs 700 --lr 1e-3 --weight-decay 5e-5 --optimizer adam \
  --train-ratio 0.7 --full-batch --seed 42 --patience 1000 \
  --metrics-csv "$CKPT_DIR/metrics.csv" 2>&1 | tee "$CKPT_DIR/train.log"

# 5) Run inference/OOD for bp=0,2,4
#    Purpose: evaluate energy-thresholded inference in no-BP and BP modes to evaluate OOD detection performance
#    Outputs: Inference_Output_${TARGET_SUFFIX}/${RUN_TAG}__ood__*
for BP in 0 2 4; do
  if [ "$BP" -eq 0 ]; then
    OUT_DIR="Inference_Output_${TARGET_SUFFIX}/${RUN_TAG}__ood__no-bp"
  else
    OUT_DIR="Inference_Output_${TARGET_SUFFIX}/${RUN_TAG}__ood__no-bp+bp${BP}"
  fi
  mkdir -p "$OUT_DIR"
  python Scripts/Model_Training_and_Inference_Scripts/ood_inference.py \
    --data-root "$DATA_ROOT" \
    --checkpoint "$CKPT_DIR/best_val.pth" \
    --model gcn4 \
    --device auto \
    --hidden-channels 128 --num-layers 4 --dropout 0.1 --heads 1 \
    --keep-ratios 0.9 0.95 --bp-steps "$BP" \
    --output-dir "$OUT_DIR" 2>&1 | tee "$OUT_DIR/infer.log"
done
```

Device note:
- `--device auto` (default) resolves to `cuda:0` if CUDA is available, else `cpu`.
- you can override manually with values like `cuda:1`, `cuda:2`, or `cpu`.

Patience note:
- if `patience` is set higher than `epochs`, early stopping will not be invoked in that run.

Suffix convention:
- `<TARGET_SUFFIX>` is set from selected target IPs in data labeling.
- single target: use that target shortform (example: `microblaze` -> `MB`).
- multi-target: use unique shortforms sorted alphabetically and joined with `_` (example: `microblaze ila` -> `ILA_MB`).
- keep the same `<TARGET_SUFFIX>` across `DATA_ROOT`, `work_dirs_xilinx_<TARGET_SUFFIX>`, and `Inference_Output_<TARGET_SUFFIX>`.

## Versioning policy for run tags

Use explicit human-managed run tags; do not rely on timestamps.

- format: `<model>_runNN`
- examples: `gcn_run01`, `gatv2_run01`, `gat_run01`, `gat_run02`

Policy:

1. `run01` = first baseline attempt for that model.
2. Increment to `run02`, `run03`, ... for any tuned rerun.
3. Reuse the same `RUN_TAG` only if you intentionally want overwrite behavior.
4. Keep the same `RUN_TAG` between training and inference for traceability.

Output naming standard:

- train outputs: `work_dirs_xilinx_<TARGET_SUFFIX>/<RUN_TAG>/...`
- OOD outputs:
  - `Inference_Output_<TARGET_SUFFIX>/<RUN_TAG>__ood__no-bp` for `--bp-steps 0`
  - `Inference_Output_<TARGET_SUFFIX>/<RUN_TAG>__ood__no-bp+bp2` for `--bp-steps 2`
  - `Inference_Output_<TARGET_SUFFIX>/<RUN_TAG>__ood__no-bp+bp4` for `--bp-steps 4`

## Data assumptions

The pipeline expects these paths and naming conventions.

Required raw graph inputs:

- `Data/Raw_Files/*.gexf` (pure IP graphs)
- `Data/Test_Raw_Files/*.gexf` (SoC graphs)

Expected config location:

- `Scripts/Data_Preparation_Scripts/graph_labeling_config.json`

Assumptions about file naming:

- IP raw files match config `ips.*.train_gexf` values (examples: `microblaze.gexf`, `ila.gexf`).
- SoC raw files match config `soc_files` values (example: `mb_axi_pcie_jtag_ila.gexf`).
- Labeling scripts write `processed_*.gexf` into split `raw/` folders.
- `data_prep.py` reads split roots at:
  - `Data/Data_Xilinx_<TARGET_SUFFIX>/train/raw`
  - `Data/Data_Xilinx_<TARGET_SUFFIX>/eval/raw`
  - `Data/Data_Xilinx_<TARGET_SUFFIX>/test/raw`
  - `Data/Data_Xilinx_<TARGET_SUFFIX>/inference/raw`

Assumptions about execution:

- run from `Xilinx_Package` root
- scripts rely on relative paths from this root

## Reproducibility outputs

Per run tag:

- `work_dirs_xilinx_<TARGET_SUFFIX>/<RUN_TAG>/run_config_results.txt` is the canonical record
  - train parameters/outcomes
  - appended inference invocations
  - artifact paths

Global index:

- `experiment_results.txt` contains concise entries and pointers to each run's `run_config_results.txt`
