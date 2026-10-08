# Model Training and Inference (Xilinx Package)

This guide documents how to train and evaluate graph models using:

- `Scripts/Model_Training_and_Inference_Scripts/models.py`
- `Scripts/Model_Training_and_Inference_Scripts/train_eval.py`
- `Scripts/Model_Training_and_Inference_Scripts/ood_inference.py`

Run commands from the Package root `Xilinx_Package`

## 1) Script overview

### `models.py`

Defines supported model architectures and the `build_model(...)` factory.

Supported model names:

- `gcn-4-skip` (aliases include `gcn`, `gcn4`, `gcn4skip`, `circuitgnn`)
- `gatv2` (aliases include `gat`, `gat4v1`, `gat4-v1`, `gat4v1skip`, `gat4-v1-skip`, `gat-v1`)
- `gat` (aliases include `gatv2`, `gat-v2`, `gat4v2`, `gat4-v2`, `gat4v2skip`, `gat4-v2-skip`)

### Model selection quick guide

- `gcn-4-skip`: strong baseline, generally stable and fast to tune.
- `gat`: attention-based neighbor weighting, often strongest but more hyperparameter-sensitive.
- `gatv2`: GATv2 attention variant; often improves attention flexibility over baseline GAT.

### `train_eval.py`

Trains node-level classifiers to perform node classification on the graph netlists and saves the best checkpoint by validation macro-F1.

Reads from:

- `<data-root>/train/processed/*.pt`
- `<data-root>/test/processed/*.pt`

Writes:

- `<checkpoint-dir>/best_val.pth`
- `<checkpoint-dir>/metrics.csv` (if `--metrics-csv` is provided)
- `<checkpoint-dir>/train.log` (when command output is redirected/piped by user)
- `<checkpoint-dir>/run_config_results.txt` (results to compare for reproducibility)
- `experiment_results.txt` block at package root (includes pointer to run_config_results)

### `ood_inference.py`

Runs inference and OOD rejection using energy thresholds with optional belief-propagation smoothing.

Reads from:

- `<data-root>/eval/processed/*.pt`
- `<data-root>/inference/processed/*.pt`

Writes under `<output-dir>`:

- `ood_summary.txt`
- energy score CSV files (no-BP and BP)
- energy distribution plots
- optional `inference_output.gexf` and `inference_output.xlsx`
- `<output-dir>/infer.log` (when command output is redirected/piped by user)
- inference entry appended into `<checkpoint-dir>/run_config_results.txt`
- `experiment_results.txt` block at package root (includes pointer to run_config_results)

## 2) Output roots and run naming

Create output roots once:

```bash
TARGET_SUFFIX=MB
DATA_ROOT="Data/Data_Xilinx_${TARGET_SUFFIX}"
```

Use manual run tags for reproducible terminal workflows:

- `gat_run01`
- `gat_run02`
- `gatv2_run01`
- `gcn_run01`

This keeps runs easy to launch, compare, and rerun from terminal history.

Suffix convention:

- `<TARGET_SUFFIX>` is derived from selected target IPs in data labeling.
- single target: target shortform (example: `microblaze` -> `MB`).
- multi-target: unique shortforms sorted alphabetically and joined with `_` (example: `microblaze ila` -> `ILA_MB`).
- keep `<TARGET_SUFFIX>` consistent between `DATA_ROOT`, `work_dirs_xilinx_<TARGET_SUFFIX>`, and `Inference_Output_<TARGET_SUFFIX>`.

## 3) Training command template

```bash
RUN_TAG=gat_run01
CKPT_DIR="work_dirs_xilinx_${TARGET_SUFFIX}/${RUN_TAG}"
mkdir -p "$CKPT_DIR"
python Scripts/Model_Training_and_Inference_Scripts/train_eval.py \
  --data-root "$DATA_ROOT" \
  --checkpoint-dir "$CKPT_DIR" \
  --model gat \
  --device auto \
  --hidden-channels 128 \
  --num-layers 4 \
  --dropout 0.1 \
  --heads 1 \
  --epochs 700 \
  --lr 1e-3 \
  --weight-decay 5e-5 \
  --optimizer adam \
  --train-ratio 0.7 \
  --full-batch \
  --seed 42 \
  --patience 1000 \
  --metrics-csv "$CKPT_DIR/metrics.csv" 2>&1 | tee "$CKPT_DIR/train.log"
```

Use the same template with different model names:

- `RUN_TAG=gcn_run01` with `--model gcn-4-skip`
- `RUN_TAG=gatv2_run01` with `--model gatv2`
- `RUN_TAG=gat_run01` with `--model gat`

For `gat`, keep `--heads` explicit so comparisons remain unambiguous.

### Training arguments explained (`train_eval.py`)

- `--data-root`
  - required path.
  - points to dataset root containing `train/processed` and `test/processed`.
  - changing this switches the full dataset source for training/validation/testing.

- `--checkpoint-dir`
  - required path.
  - where training artifacts are written: `best_val.pth`, `metrics.csv` (if enabled), `train.log`, `run_config_results.txt`.
  - use a unique run folder (`gcn_run01`, `gin_run01`, etc.) to avoid overwriting.

- `--model`
  - values: `gcn-4-skip`, `gat`, `gatv2`.
  - selects architecture; this changes parameterization, convergence behavior, and final metrics.

- `--hidden-channels`
  - integer (default `128`).
  - larger values increase model capacity and GPU memory/time usage.
  - too small may underfit; too large may overfit or train slower.

- `--num-layers`
  - integer (default `4`).
  - deeper models aggregate information from larger neighborhoods.
  - too deep can over-smooth node features on some graphs.

- `--dropout`
  - float in `[0,1]` (default `0.1`).
  - higher values increase regularization and may improve generalization.
  - very high values can hurt training signal and lower peak accuracy/F1.

- `--heads`
  - integer (default `1`), only meaningful for `--model gat`.
  - increases attention heads in GAT.
  - higher heads can improve expressiveness but raise compute/memory cost.

- `--epochs`
  - integer (default `300`).
  - maximum training epochs.
  - actual stop can be earlier because of `--patience` early stopping.

- `--lr`
  - float (default `1e-3`).
  - optimizer step size.
  - larger values train faster but may destabilize; smaller values are stable but slower.

- `--weight-decay`
  - float (default `5e-5`).
  - L2 regularization strength.
  - larger values reduce overfitting but can underfit if too strong.

- `--optimizer`
  - values: `adam`, `rmsprop` (default `adam`).
  - changes update dynamics and sometimes convergence quality/speed.

- `--train-ratio`
  - float in `(0,1)` (default `0.7`).
  - split ratio for train/val masks generated from train graphs.
  - larger ratio gives more train nodes, fewer val nodes.

- `--full-batch`
  - flag (default off).
  - when set, trains on full merged graph each epoch.
  - useful for deterministic behavior if graph fits memory.

- `--train-batch-size`
  - integer, optional (default unset).
  - in non-`--full-batch` mode, enables NeighborLoader mini-batch training.
  - if omitted or `<=0`, loader path is disabled and training falls back to full-batch behavior.

- `--val-batch-size`
  - integer, optional (default unset).
  - in non-`--full-batch` mode, controls mini-batch validation.
  - if omitted or `<=0`, validation uses full-batch evaluation.

- `--test-batch-size`
  - integer, optional (default unset).
  - in non-`--full-batch` mode, controls mini-batch testing.
  - if omitted or `<=0`, testing uses full-batch evaluation.

- `--full-batch-eval`
  - flag (default off).
  - when training in non-full-batch mode, forces validation/testing to run full-batch.
  - useful for more stable evaluation comparisons across runs.

- `--seed`
  - integer (default `42`).
  - controls randomness in mask splitting and training initialization.
  - keep fixed for fair model/hyperparameter comparisons.

- `--patience`
  - integer (default `1000`).
  - early-stopping patience based on validation macro-F1.
  - larger values allow longer training before stopping.
  - if `patience > epochs`, early stopping is effectively disabled for that run.

- `--device`
  - string (default `auto`).
  - `auto` resolves to `cuda:0` when CUDA is available, else `cpu`.
  - examples: `auto`, `cpu`, `cuda:0`, `cuda:1`.
  - selects execution device; wrong GPU index will fail at runtime.

- `--metrics-csv`
  - optional path.
  - if set, writes per-epoch metrics timeline CSV.
  - useful for plotting and run-to-run diagnostics.

## 4) Inference/OOD command template

```bash
RUN_TAG=gat_run01
CKPT_DIR="work_dirs_xilinx_${TARGET_SUFFIX}/${RUN_TAG}"
OUT_DIR="Inference_Output_${TARGET_SUFFIX}/${RUN_TAG}__ood__no-bp+bp4"
mkdir -p "$OUT_DIR"

python Scripts/Model_Training_and_Inference_Scripts/ood_inference.py \
  --data-root "$DATA_ROOT" \
  --checkpoint "$CKPT_DIR/best_val.pth" \
  --model gat \
  --device auto \
  --hidden-channels 128 \
  --num-layers 4 \
  --dropout 0.1 \
  --heads 1 \
  --keep-ratios 0.9 0.95 \
  --bp-steps 4 \
  --output-dir "$OUT_DIR" 2>&1 | tee "$OUT_DIR/infer.log"
```

BP naming reference:

- `--bp-steps 0` -> `Inference_Output_<TARGET_SUFFIX>/<RUN_TAG>__ood__no-bp`
- `--bp-steps 2` -> `Inference_Output_<TARGET_SUFFIX>/<RUN_TAG>__ood__no-bp+bp2`
- `--bp-steps 4` -> `Inference_Output_<TARGET_SUFFIX>/<RUN_TAG>__ood__no-bp+bp4`

Important behavior:

- For `--bp-steps > 0`, `ood_inference.py` writes both no-BP and BP outputs in the same run.
- Use `--bp-steps 0` only when you explicitly want a no-BP-only run.

Example progression for one model:

- training: `RUN_TAG=gat_run01`
- inference (same checkpoint):
  - `Inference_Output_<TARGET_SUFFIX>/gat_run01__ood__no-bp` (optional no-BP only)
  - `Inference_Output_<TARGET_SUFFIX>/gat_run01__ood__no-bp+bp2`
  - `Inference_Output_<TARGET_SUFFIX>/gat_run01__ood__no-bp+bp4`
- next tuned run: `RUN_TAG=gat_run02`

### Inference arguments explained (`ood_inference.py`)

- `--data-root`
  - required path.
  - points to processed data root used for thresholding/evaluation flows.
  - changing this changes validation distributions and resulting OOD thresholds.

- `--checkpoint`
  - required path to `best_val.pth`.
  - loads trained weights and saved training config context.
  - must match architecture arguments (`--model`, channels/layers/dropout/heads).

- `--model`
  - values: `gcn-4-skip`, `gat`, `gatv2`.
  - must be consistent with training; mismatch causes load errors or invalid results.

- `--hidden-channels`
  - integer (default `128`).
  - must match trained model dimensions for checkpoint load compatibility.

- `--num-layers`
  - integer (default `4`).
  - must match checkpoint architecture.

- `--dropout`
  - float in `[0,1]` (default `0.1`).
  - should match training config; this parameter is part of model construction.

- `--heads`
  - integer (default `1`), relevant for `--model gat`.
  - must match training for GAT checkpoint compatibility.

- `--keep-ratios`
  - one or more floats in `(0,1)` (default `0.9 0.95`).
  - each value defines percentile retention on positive validation energies.
  - lower keep-ratio => stricter thresholding (fewer accepted positives, typically higher precision).
  - higher keep-ratio => looser thresholding (more accepted positives, typically higher recall).

- `--bp-steps`
  - integer (default `4`).
  - `0`: no-BP only outputs.
  - `>0`: outputs include both no-BP and BP-smoothed metrics/artifacts in the same run.
  - larger BP steps increase smoothing strength; can improve or degrade class separation depending on data.

- `--device`
  - string (default `auto`).
  - `auto` resolves to `cuda:0` when CUDA is available, else `cpu`.
  - examples: `auto`, `cpu`, `cuda:0`, `cuda:1`.
  - should generally match training/inference hardware intent for reproducible timing and behavior.

- `--output-dir`
  - required path.
  - where OOD artifacts are written (`ood_summary.txt`, plots, CSVs, optional exports, `infer.log`).
  - use naming convention:
    - `...__ood__no-bp` for `--bp-steps 0`
    - `...__ood__no-bp+bp2` for `--bp-steps 2`
    - `...__ood__no-bp+bp4` for `--bp-steps 4`

- `--export-gexf`
  - flag (default off).
  - when set, writes `inference_output.gexf` with predictions and energy attributes.

- `--inference-gexf`
  - optional path, but required when `--export-gexf` is set.
  - source graph used for export overlay.

- `--excel`
  - flag (default off).
  - writes `inference_output.xlsx` alongside exported GEXF.
  - valid only with `--export-gexf`; otherwise the script exits with argument error.

## 5) Run Config JSON File (reproducibility)

Each run now uses a JSON manifest as the run record:

- `work_dirs_xilinx_<TARGET_SUFFIX>/<RUN_TAG>/run_config_results.txt`

This file stores:

- training parameters and outcome (`best_epoch`, `best_val_f1`, final test metrics)
- key directories (`data_root`, `checkpoint_dir`, `output_dir`)
- artifact locations (checkpoint, metrics CSV, summary files, plots, CSV outputs, logs)
- executed command strings and timestamps
- inference invocations appended under `inference_runs`

`experiment_results.txt` remains the global ledger, and points to this results file rather than duplicating full parameter blocks.

## 6) What each output file means

### Training artifacts (`<checkpoint-dir>`)

- `best_val.pth`
  - model checkpoint chosen by best validation macro-F1
  - includes model weights, model config snapshot, best-epoch information

- `metrics.csv`
  - per-epoch training timeline (train loss, val loss, val accuracy, val macro-F1)
  - useful for convergence checks and overfitting analysis

- `train.log`
  - full training console trace (epoch progress, final metrics, artifact paths)

- `run_config_results.txt`
  - canonical metadata + artifact manifest for this run
  - includes training block and appended inference blocks

### Inference/OOD artifacts (`<output-dir>`)

- `ood_summary.txt`
  - compact OOD metrics report across keep-ratios and BP/no-BP modes

- `train_energy_scores_no_bp.csv`, `val_energy_scores_no_bp.csv`
  - node-level no-BP energies and predictions

- `train_bp_energyprop_<BP>.csv`, `val_bp_energyprop_<BP>.csv`
  - node-level BP-smoothed energies and predictions

- `infer.log`
  - full inference console trace (thresholds, pre/post OOD metrics)

- optional `inference_output.gexf`, `inference_output.xlsx`
  - graph exports with prediction and energy annotations (enabled by `--export-gexf` and optionally `--excel`)

## 7) Plot guide

Plots are saved under `<output-dir>`.

- `val_energy_no_bp.png`
  - validation energy distribution (no-BP)
  - used to inspect separation and threshold calibration behavior

- `infer_energy_no_bp.png`
  - inference energy distribution (no-BP)
  - shows confidence spread on inference nodes before BP smoothing

- `infer_energy_bp.png`
  - inference energy distribution after BP smoothing (created when `bp_steps > 0`)

- `energy_train_val_test_no_bp_thr*.png`
  - train/val/test histograms for class-1 correct vs incorrect predictions (no-BP)
  - generated per keep-ratio threshold

- `energy_train_val_test_bp*_thr*.png`
  - same train/val/test histograms after BP smoothing

Interpretation:

- lower energy indicates higher confidence
- thresholding prunes class-1 predictions with energy above threshold
- keep-ratio controls how strict thresholding is on positive validation samples

## 8) Artifact paths at a glance

- checkpoint: `work_dirs_xilinx_<TARGET_SUFFIX>/<RUN_TAG>/best_val.pth`
- epoch metrics: `work_dirs_xilinx_<TARGET_SUFFIX>/<RUN_TAG>/metrics.csv`
- training console log: `work_dirs_xilinx_<TARGET_SUFFIX>/<RUN_TAG>/train.log`
- inference outputs: `Inference_Output_<TARGET_SUFFIX>/<RUN_TAG>__ood__no-bp(+bpN)/`
- inference console log: `Inference_Output_<TARGET_SUFFIX>/<RUN_TAG>__ood__no-bp(+bpN)/infer.log`
- run manifest: `work_dirs_xilinx_<TARGET_SUFFIX>/<RUN_TAG>/run_config_results.txt`
- global run ledger: `experiment_results.txt`

## 9) Global experiment log

`experiment_results.txt` at package root stores append-only run blocks for both training and inference:

- command used
- key split metrics
- OOD threshold outcomes
- pointer to `work_dirs_xilinx_<TARGET_SUFFIX>/<RUN_TAG>/run_config_results.txt`

`run_config_results.txt` is the canonical detailed record; `experiment_results.txt` is the quick index.

## 10) Quick run progression

Example manual progression:

1. run `gat_run01`
2. inspect `work_dirs_xilinx_<TARGET_SUFFIX>/gat_run01/metrics.csv`
3. inspect `Inference_Output_<TARGET_SUFFIX>/gat_run01__ood__no-bp+bp4/ood_summary.txt`
4. tune parameters
5. rerun as `gat_run02`
6. compare both runs via `experiment_results.txt`, metrics CSV, and OOD summaries

## 11) Common notes

- Keep architecture arguments consistent between training and inference.
- Use the same `data-root` for train and inference when comparing runs.
- `--excel` requires `--export-gexf`.
