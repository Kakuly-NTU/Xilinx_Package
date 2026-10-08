# Data Preparation README

This README documents the data-preparation pipeline implemented by:

- `raw_to_processed_gexf.py`
- `label_train_ips.py`
- `label_soc.py`
- `data_prep.py`

Together these scripts form a data preprocessing package.

- `raw_to_processed_gexf.py`, `label_train_ips.py`, `label_soc.py` are config-driven.
- `data_prep.py` is CLI-driven (no config required).

Use the first three scripts to build labeled split folders, then run `data_prep.py` to convert labeled graphs into PyTorch Geometric tensors.

Suffix convention:
- `<TARGET_SUFFIX>` is derived from selected target IPs (`--target-ip`).
- single target: target shortform (example: `microblaze` -> `MB`).
- multi-target: unique shortforms sorted alphabetically and joined with `_` (example: `microblaze ila` -> `ILA_MB`).
- this same suffix should be used consistently in root README/model README variables:
  - `TARGET_SUFFIX=...`
  - `DATA_ROOT="Data/Data_Xilinx_${TARGET_SUFFIX}"`

## 1) What each script does

### 1.1 `raw_to_processed_gexf.py`

Purpose: convert raw `.gexf` files into preprocessed graph + metadata files.

Input:
- raw `.gexf` files from either IP raw directory or SoC raw directory

Output:
- `pre_processed_<name>.gexf`
- `pre_processed_<name>.json`

Notebook-equivalent transformations to format input features:
- normalize node `INIT` to 64-bit hex (`lut_mask`)
- copy original node `label` to `cell_name`
- set node `label` from node id
- emit JSON summary containing:
  - graph name
  - number of nodes / edges
  - `type` histogram
  - `module` histogram with placeholder labels (`"NA"`)

---

### 1.2 `label_train_ips.py`

Purpose: label pure-IP training/eval `.gexf` graph netlists for one or more selected target IPs.

Behavior:
- selected target IP graphs: all node labels set to `"1"`
- all other included IP graphs: all node labels set to `"0"`

Writes outputs to:
- `Processed_Files_<TARGET_SUFFIX>/processed_*.gexf|json`
- `Data_Xilinx_<TARGET_SUFFIX>/train/raw/processed_*.gexf`
- `Data_Xilinx_<TARGET_SUFFIX>/eval/raw/processed_*.gexf`

`<TARGET_SUFFIX>` rules:
- single target: its acronym (example: `--target-ip microblaze` -> `MB`)
- multi-target: deduplicated acronym set sorted alphabetically and joined with `_`
  - example: `--target-ip microblaze ila` -> `ILA_MB`

Important:
- this script performs **cleanup before writing** (details in Section 5)

---

### 1.3 `label_soc.py`

Purpose: label SoC graph(s) using target-module mapping for one or more selected IPs.

Behavior:
- node `label="1"` if `node.module` is in the union of configured target module lists
- otherwise `label="0"`

For multi-target IP runs, the module union is deduplicated automatically.

Writes outputs to:
- `Test_Processed_Files_<TARGET_SUFFIX>/processed_<soc>.gexf|json`
- `Data_Xilinx_<TARGET_SUFFIX>/test/raw/processed_<soc>.gexf`
- `Data_Xilinx_<TARGET_SUFFIX>/inference/raw/processed_<soc>.gexf`

Important:
- this script performs **cleanup before writing** (details in Section 5)

---

### 1.4 `data_prep.py`

Purpose: convert labeled split `.gexf` graphs into PyG `.pt` tensors for training/inference pipelines.

Why use it:
- creates model-ready graph tensors (`torch_geometric.data.Data`)
- applies consistent node feature generation for every split
- ensures deterministic split outputs under one root (`train/eval/test/inference`)

Feature and label construction:
- node feature `x` = LUT mask bits (64) + type one-hot vector (63) = 127 dims
- node label `y` is derived from node `label` (cast to int)
- `train_mask` is set for all nodes (`y >= 0`)

Default behavior:
- extracts largest connected component first (`extract_cc=True` by default)
- converts directed graphs to undirected (`keep_directed=False` by default)

Input layout expected under `--raw-root`:
- `<raw-root>/train/raw/*.gexf`
- `<raw-root>/eval/raw/*.gexf`
- `<raw-root>/test/raw/*.gexf`
- `<raw-root>/inference/raw/*.gexf`

Output layout under `--output-root`:
- `<output-root>/<split>/CC_extracted/*.gexf` (if CC extraction enabled)
- `<output-root>/<split>/processed/*.pt`

---

## 2) Config file (`graph_labeling_config.json`)

The first three scripts are driven by:

- `Scripts/Data_Preparation_Scripts/graph_labeling_config.json`

You should treat this file as the source of truth for paths, IP definitions, and SoC file keys.

`data_prep.py` does not use this config file.

### 2.1 Current schema

```json
{
  "paths": {
    "ip_raw_dir": "Data/Raw_Files",
    "soc_raw_dir": "Data/Test_Raw_Files",
    "ip_preprocessed_dir": "Data/Preprocessed_Files",
    "soc_preprocessed_dir": "Data/Test_Preprocessed_Files",
    "ip_processed_base": "Data/Processed_Files",
    "soc_processed_base": "Data/Test_Processed_Files",
    "data_base": "Data/Data_Xilinx"
  },
  "ips": {
    "microblaze": {
      "shortform": "MB",
      "train_gexf": "microblaze.gexf",
      "target_modules": ["microblaze_0", "microblaze_0/U0"]
    }
  },
  "ip_processing_order": ["microblaze"],
  "soc_files": {
    "mb_axi_pcie_jtag_ila": "mb_axi_pcie_jtag_ila.gexf"
  }
}
```

### 2.2 Meaning of each section

`paths`
- `ip_raw_dir`: raw pure-IP `.gexf` input folder
- `soc_raw_dir`: raw SoC `.gexf` input folder
- `ip_preprocessed_dir`: output for IP `pre_processed_*`
- `soc_preprocessed_dir`: output for SoC `pre_processed_*`
- `ip_processed_base`: base name for labeled IP processed folder
  - final folder is `Data/Processed_Files_<TARGET_SUFFIX>`
- `soc_processed_base`: base name for labeled SoC processed folder
  - final folder is `Data/Test_Processed_Files_<TARGET_SUFFIX>`
- `data_base`: base name for labeled dataset root
  - final folder is `Data/Data_Xilinx_<TARGET_SUFFIX>`

`ips`
- key: IP id used on CLI (`--target-ip`)
- `shortform`: acronym suffix used for output folders
- `train_gexf`: pure-IP graph netlist filename
- `target_modules`: module names used by `label_soc.py` for positive labeling

`ip_processing_order`
- controls default iteration order for `label_train_ips.py`
- if omitted, scripts use sorted IP keys

`soc_files`
- mapping from friendly key to SoC netlist filename
- used by `label_soc.py --soc-file <key1> [key2 ...]`

### 2.3 How output folder naming is derived

For `target-ip=microblaze` with `shortform=MB`:

- `ip_processed_base=Data/Processed_Files` -> `Data/Processed_Files_MB`
- `soc_processed_base=Data/Test_Processed_Files` -> `Data/Test_Processed_Files_MB`
- `data_base=Data/Data_Xilinx` -> `Data/Data_Xilinx_MB`

For `target-ip microblaze ila`:

- acronyms are deduplicated and alphabetically sorted: `ILA`, `MB`
- suffix is `ILA_MB`
- outputs become:
  - `Data/Processed_Files_ILA_MB`
  - `Data/Test_Processed_Files_ILA_MB`
  - `Data/Data_Xilinx_ILA_MB`

So if you change `data_base` to `Data/MyBase`, outputs become `Data/MyBase_<TARGET_SUFFIX>`.

### 2.4 How to update config safely

Common updates:

1. Add a new IP or target IP
- add a new key under `ips`
- set `shortform`, `train_gexf`, and `target_modules`
- include the key in `ip_processing_order`

2. Add a new SoC file option
- add entry under `soc_files`:
  - key: CLI-friendly name
  - value: actual filename in SoC raw/preprocessed folders

3. Change directories
- update values in `paths`
- keep relative paths relative to this config file location

Validation tips:
- ensure every `ips.*.train_gexf` exists in `ip_raw_dir` or preprocessed dir depending on stage
- ensure every `soc_files.*` value exists in `soc_raw_dir` for preprocess, and `soc_preprocessed_dir` for labeling
- keep `shortform` stable if downstream tooling expects fixed folder names
- ensure new IP keys are added to `ip_processing_order` if order matters for your use case (default is config order, fallback is sorted keys)
- ensure every target IP is included in the `ips` section with correct `target_modules` for SoC labeling

---

## 3) CLI usage

Run commands from `Xilinx_Package` root.

Path resolution rule:
- All relative paths in CLI arguments and config are resolved from the current working directory.
- Expected working directory is `Xilinx_Package` root.
- Scripts fail fast if `Data/` and `Scripts/` are not present in the current working directory.

### 3.1 `raw_to_processed_gexf.py`

Command:

```bash
python Scripts/Data_Preparation_Scripts/raw_to_processed_gexf.py --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --dataset ip
```

Flags:
- `--config PATH` (required): config file
- `--dataset {ip,soc}` (required): choose path set from config
- `--file NAME.gexf` (optional): process one file only
  - if omitted, processes **all** `.gexf` files in selected raw dir
- `--raw-dir PATH` (optional): override raw input directory
- `--output-dir PATH` (optional): override preprocessed output directory
- `--verbose` (optional): print per-file progress

Examples:

```bash
# preprocess all IP raw graphs
python Scripts/Data_Preparation_Scripts/raw_to_processed_gexf.py --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --dataset ip --verbose

# preprocess only microblaze
python Scripts/Data_Preparation_Scripts/raw_to_processed_gexf.py --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --dataset ip --file microblaze.gexf --verbose

# preprocess all SoC raw graphs
python Scripts/Data_Preparation_Scripts/raw_to_processed_gexf.py --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --dataset soc --verbose
```

---

### 3.2 `label_train_ips.py`

Command:

```bash
python Scripts/Data_Preparation_Scripts/label_train_ips.py --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --target-ip microblaze
```

Flags:
- `--config PATH` (required)
- `--target-ip KEY [KEY ...]` (required): one or many target keys
- values may be space-separated or comma-separated
  - examples:
    - `--target-ip microblaze`
    - `--target-ip microblaze ila`
    - `--target-ip microblaze,ila`
- all target keys must exist in `ips`
- `--input-dir PATH` (optional): override preprocessed IP input dir
- `--processed-out-dir PATH` (optional): override processed base dir
- `--data-root PATH` (optional): override data base dir
- `--ip-list key1,key2,...` (optional): label only selected IP keys
  - useful for excluding specific IPs

Examples:

```bash
# full train/eval labeling for microblaze
python Scripts/Data_Preparation_Scripts/label_train_ips.py --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --target-ip microblaze

# multi-target train/eval labeling (suffix becomes ILA_MB)
python Scripts/Data_Preparation_Scripts/label_train_ips.py --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --target-ip microblaze ila

# exclude microblaze_mcs by explicit include list
python Scripts/Data_Preparation_Scripts/label_train_ips.py \
  --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json \
  --target-ip microblaze \
  --ip-list axi_pcie,ila,jtag_axi,microblaze,pcie_7x,mig_7series
```

---

### 3.3 `label_soc.py`

Command:

```bash
python Scripts/Data_Preparation_Scripts/label_soc.py --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --target-ip microblaze --soc-file mb_axi_pcie_jtag_ila
```

Flags:
- `--config PATH` (required)
- `--target-ip KEY [KEY ...]` (required): one or many target keys
- values may be space-separated or comma-separated
- for multi-target, modules are unioned and deduplicated
- exactly one of:
  - `--soc-file KEY_OR_FILENAME [KEY_OR_FILENAME ...]`
    - accepts one or many keys/filenames
    - values may be space-separated or comma-separated
  - `--all-soc`
- `--input-dir PATH` (optional): override preprocessed SoC input dir
- `--processed-out-dir PATH` (optional): override SoC processed base dir
- `--data-root PATH` (optional): override data base dir
- `--module-list m1,m2,...` (optional): runtime override for target modules

Examples:

```bash
# one SoC graph using key from config.soc_files
python Scripts/Data_Preparation_Scripts/label_soc.py \
  --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json \
  --target-ip microblaze \
  --soc-file mb_axi_pcie_jtag_ila

# multi-target SoC labeling (union of target modules)
python Scripts/Data_Preparation_Scripts/label_soc.py \
  --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json \
  --target-ip microblaze ila \
  --soc-file mb_axi_pcie_jtag_ila

# multiple specific SoC files (keys and/or direct filenames)
python Scripts/Data_Preparation_Scripts/label_soc.py \
  --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json \
  --target-ip microblaze ila \
  --soc-file mb_axi_pcie_jtag_ila mb_axi_pcie.gexf

# all configured SoC files
python Scripts/Data_Preparation_Scripts/label_soc.py \
  --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json \
  --target-ip microblaze \
  --all-soc
```

---

## Dependencies (for `data_prep.py`)

`data_prep.py` requires:

- `torch`
- `torch_geometric`
- `networkx`
- `numpy`

Quick check:

```bash
python -c "import torch, torch_geometric, networkx, numpy; print('ok')"
```

If this fails with `No module named 'torch_geometric'`, install PyG for your PyTorch/CUDA setup from:

- `https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html`

---

### 3.4 `data_prep.py`

Command:

```bash
TARGET_SUFFIX=MB
DATA_ROOT="Data/Data_Xilinx_${TARGET_SUFFIX}"
python Scripts/Data_Preparation_Scripts/data_prep.py --raw-root "$DATA_ROOT" --output-root "$DATA_ROOT"
```

Flags:
- `--raw-root PATH` (required): root containing split raw folders
- `--output-root PATH` (required): root where `CC_extracted` and `processed` are written
- `--type-dict PATH` (optional): node type one-hot dictionary JSON
  - default: `Scripts/Data_Preparation_Scripts/type_IP_onehot_dict_incl_sparc.json`
- `--max-degree INT` (optional): compatibility argument (currently not used for filtering)
  - default: `600`
- `--no-extract-cc` (optional): disable CC extraction
  - default behavior is CC extraction enabled
- `--keep-directed` (optional): keep directed edges
  - default behavior is undirected output

Examples:

```bash
# default run (CC extraction on, undirected conversion on)
TARGET_SUFFIX=MB
DATA_ROOT="Data/Data_Xilinx_${TARGET_SUFFIX}"
python Scripts/Data_Preparation_Scripts/data_prep.py --raw-root "$DATA_ROOT" --output-root "$DATA_ROOT"

# disable CC extraction
python Scripts/Data_Preparation_Scripts/data_prep.py --raw-root "$DATA_ROOT" --output-root "$DATA_ROOT" --no-extract-cc

# keep directed graphs
python Scripts/Data_Preparation_Scripts/data_prep.py --raw-root "$DATA_ROOT" --output-root "$DATA_ROOT" --keep-directed

# custom type dictionary
python Scripts/Data_Preparation_Scripts/data_prep.py \
  --raw-root "$DATA_ROOT" \
  --output-root "$DATA_ROOT" \
  --type-dict Scripts/Data_Preparation_Scripts/type_IP_onehot_dict_incl_sparc.json
```

---

## 4) Recommended full pipeline

Single-target end-to-end (Microblaze + SoC `mb_axi_pcie_jtag_ila`):

```bash
TARGET_SUFFIX=MB
DATA_ROOT="Data/Data_Xilinx_${TARGET_SUFFIX}"

python Scripts/Data_Preparation_Scripts/raw_to_processed_gexf.py --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --dataset ip --verbose
python Scripts/Data_Preparation_Scripts/raw_to_processed_gexf.py --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --dataset soc --verbose
python Scripts/Data_Preparation_Scripts/label_train_ips.py --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --target-ip microblaze
python Scripts/Data_Preparation_Scripts/label_soc.py --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --target-ip microblaze --soc-file mb_axi_pcie_jtag_ila
python Scripts/Data_Preparation_Scripts/data_prep.py --raw-root "$DATA_ROOT" --output-root "$DATA_ROOT"
```

Multi-target end-to-end (Microblaze + ILA):

```bash
TARGET_SUFFIX=ILA_MB
DATA_ROOT="Data/Data_Xilinx_${TARGET_SUFFIX}"

python Scripts/Data_Preparation_Scripts/raw_to_processed_gexf.py --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --dataset ip --verbose
python Scripts/Data_Preparation_Scripts/raw_to_processed_gexf.py --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --dataset soc --verbose
python Scripts/Data_Preparation_Scripts/label_train_ips.py --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --target-ip microblaze ila
python Scripts/Data_Preparation_Scripts/label_soc.py --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --target-ip microblaze ila --soc-file mb_axi_pcie_jtag_ila
python Scripts/Data_Preparation_Scripts/data_prep.py --raw-root "$DATA_ROOT" --output-root "$DATA_ROOT"
```

Multi-target end-to-end with multiple SoCs (single labeling run):

```bash
TARGET_SUFFIX=ILA_MB
DATA_ROOT="Data/Data_Xilinx_${TARGET_SUFFIX}"

python Scripts/Data_Preparation_Scripts/raw_to_processed_gexf.py --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --dataset ip --verbose
python Scripts/Data_Preparation_Scripts/raw_to_processed_gexf.py --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --dataset soc --verbose
python Scripts/Data_Preparation_Scripts/label_train_ips.py --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --target-ip microblaze ila
python Scripts/Data_Preparation_Scripts/label_soc.py --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --target-ip microblaze ila --soc-file mb_axi_pcie_jtag_ila mb_axi_pcie.gexf
python Scripts/Data_Preparation_Scripts/data_prep.py --raw-root "$DATA_ROOT" --output-root "$DATA_ROOT"
```

Skip preprocess when raw inputs are unchanged:

- Re-run `raw_to_processed_gexf.py` only if files were added/changed in `Data/Raw_Files` or `Data/Test_Raw_Files`.
- If raw files are unchanged and you only want different target IP labeling, skip preprocess and run:
  1. `label_train_ips.py`
  2. `label_soc.py`
  3. `data_prep.py`

Example (relabel only for a new target set):

```bash
python Scripts/Data_Preparation_Scripts/label_train_ips.py \
  --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json \
  --target-ip microblaze ila

python Scripts/Data_Preparation_Scripts/label_soc.py \
  --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json \
  --target-ip microblaze ila \
  --soc-file mb_axi_pcie_jtag_ila

python Scripts/Data_Preparation_Scripts/data_prep.py \
  --raw-root "$DATA_ROOT" \
  --output-root "$DATA_ROOT"
```

---

## 5) Cleanup behavior (destructive by design)

Both labeling scripts clear previous outputs before writing new ones.

### `label_train_ips.py` cleanup targets
- `Data/Processed_Files_<TARGET_SUFFIX>/`
- `Data/Data_Xilinx_<TARGET_SUFFIX>/train/`
- `Data/Data_Xilinx_<TARGET_SUFFIX>/eval/`

### `label_soc.py` cleanup targets
- `Data/Test_Processed_Files_<TARGET_SUFFIX>/`
- `Data/Data_Xilinx_<TARGET_SUFFIX>/test/`
- `Data/Data_Xilinx_<TARGET_SUFFIX>/inference/`

Important for multiple SoCs:
- `label_soc.py` cleans SoC output folders before writing.
- If you run one SoC per command, each new run replaces previous SoC outputs for the same target suffix.
- To keep multiple SoCs together, pass all desired SoCs in one command (`--soc-file ...`) or use `--all-soc`.

Logging format:
- live run: `[CLEAN] Removing: <path>`
- missing path: `[CLEAN] Not found, skipping: <path>`

This is intentional so each run produces a clean, deterministic dataset state for the selected acronym.

---

## 6) Troubleshooting

1. `FileNotFoundError` for SoC preprocessed files
- Run SoC preprocess first:
  - `raw_to_processed_gexf.py --dataset soc`

1a. `label_train_ips.py` fails because preprocessed IP folder is missing/empty or selected IP files are missing
- Run IP preprocess first:
  - `python Scripts/Data_Preparation_Scripts/raw_to_processed_gexf.py --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --dataset ip`

1b. `label_soc.py` fails because preprocessed SoC folder is missing/empty or selected SoC files are missing
- Run SoC preprocess first:
  - `python Scripts/Data_Preparation_Scripts/raw_to_processed_gexf.py --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --dataset soc`

2. Unknown target IP
- Check `ips` keys in config
- Use exact key string in `--target-ip`

3. Nothing written after run
- Check script output for upstream input errors (scripts are fail-fast)

4. Missing raw SoC folder
- Ensure `paths.soc_raw_dir` exists and contains `.gexf` files
- Example expected input: `Data/Test_Raw_Files/mb_axi_pcie_jtag_ila.gexf`

5. Wrong output root
- Verify `paths.data_base` and `ips.<target>.shortform`
- Final data dir is always `<data_base>_<TARGET_SUFFIX>`
- For multiple targets, suffix uses unique acronyms sorted alphabetically

6. `data_prep.py` cannot find split folders
- Check input layout under `--raw-root`:
  - `<split>/raw/*.gexf` must exist for `train`, `eval`, `test`, `inference`
- If folders are missing/empty, run labeling first:
  - `python Scripts/Data_Preparation_Scripts/label_train_ips.py --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --target-ip <target_ip>`
  - `python Scripts/Data_Preparation_Scripts/label_soc.py --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --target-ip <target_ip> --soc-file <soc_key_or_file>`

7. `data_prep.py` type dictionary load error
- Verify `--type-dict` path is correct and readable
- Ensure JSON has keys for expected node types

8. Unexpected directed behavior in `.pt`
- Default is undirected conversion
- Use `--keep-directed` only when directed message passing is intended

---

## 7) Sanity precheck before `data_prep.py`

```bash
TARGET_SUFFIX=ILA_MB
DATA_ROOT="Data/Data_Xilinx_${TARGET_SUFFIX}"

# ensure split raw directories exist and contain .gexf files
ls "$DATA_ROOT"/train/raw
ls "$DATA_ROOT"/eval/raw
ls "$DATA_ROOT"/test/raw
ls "$DATA_ROOT"/inference/raw

# ensure type dictionary is present
ls Scripts/Data_Preparation_Scripts/type_IP_onehot_dict_incl_sparc.json
```

Then run:

```bash
python Scripts/Data_Preparation_Scripts/data_prep.py --raw-root "$DATA_ROOT" --output-root "$DATA_ROOT"
```
