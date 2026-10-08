# HAL Netlist Preprocessing README

This document covers the HAL-side preprocessing step used before the VSCode/Python data preparation scripts.

## Important boundary: HAL vs VSCode

- `netlist_removal.py` runs inside HAL's Python environment (`hal_py`) from the HAL GUI.
- Note that this step is not run as a normal VSCode Python script.
- HAL is used to clean netlists and export `.gexf` graphs.
- After `.gexf` export, continue with the scripts in `Scripts/Data_Preparation_Scripts`.

## Inputs and outputs in this repository

- Verilog inputs: `Data/Raw_Verilog_Files/*.v`
- HAL-cleaned graph outputs: `Data/Raw_Files/*.gexf`
- Test Verilog inputs: `Data/Test_Raw_Verilog_Files/*.v`
- HAL-cleaned test graph outputs: `Data/Test_Raw_Files/*.gexf`

In other words:
- raw `.v` files come from `Data/Raw_Verilog_Files` and `Data/Test_Raw_Verilog_Files`
- HAL processing produces the `.gexf` files that should be placed in `Data/Raw_Files` and `Data/Test_Raw_Files` respectively.

## What `netlist_removal.py` removes

`Scripts/HAL_Verilog_Scripts/netlist_removal.py` identifies nets connected to special pin types and removes those nets.

Target pin types:
- power
- ground
- clock
- reset
- enable
- set

## HAL installation

If HAL is not installed, follow instructions from the official HAL wiki:
- https://github.com/emsec/hal/wiki/Building-HAL

## HAL GUI launch

Once installed, launch HAL GUI from HAL build directory:

```bash
./bin/hal -g
```

## Recommended HAL workflow

1. Launch HAL GUI.
2. Load one of the Verilog netlists from `Data/Raw_Verilog_Files`.
3. Open HAL's Python interface (with `hal_py` available).
4. Paste the contents of `Scripts/HAL_Verilog_Scripts/netlist_removal.py` into the HAL Python context and run it.
5. Export/save the processed graph as `.gexf` into `Data/Raw_Files` or `Data/Test_Raw_Files/*.gexf`.
6. Repeat for other Verilog files.

## Next step after HAL export

After `Data/Raw_Files/*.gexf` and `Data/Test_Raw_Files/*.gexf` are ready, run the data-preparation pipeline from `Xilinx_Package` root:

```bash
python Scripts/Data_Preparation_Scripts/raw_to_processed_gexf.py --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --dataset ip --verbose
python Scripts/Data_Preparation_Scripts/raw_to_processed_gexf.py --config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --dataset soc --verbose
```

