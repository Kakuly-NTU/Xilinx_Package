import argparse
import json
import os
import shutil
from pathlib import Path
import networkx as nx


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(value: str) -> Path:
    if os.path.isabs(value):
        return Path(value)
    return Path(os.path.abspath(os.path.join(os.getcwd(), value)))


def ensure_package_root_cwd() -> None:
    cwd = Path(os.getcwd())
    required = [cwd / "Data", cwd / "Scripts"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        joined = "\n".join(f"- {p}" for p in missing)
        raise RuntimeError(
            "Run this script from Xilinx_Package root. Missing required paths in current working directory:\n"
            f"{joined}"
        )


def save_processed_gexf(graph: nx.Graph, original_file_name: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"processed_{Path(original_file_name).name}"
    nx.write_gexf(graph, out_path)
    return out_path


def save_processed_json(data: dict, original_file_name: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"processed_{Path(original_file_name).name.replace('.gexf', '.json')}"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    return out_path


def enumerate_remove_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("*"):
        paths.append(path)
    paths.sort(key=lambda p: (p.is_dir(), str(p)))
    paths.append(root)
    return paths


def clean_paths(paths_to_clean: list[Path]):
    for root in paths_to_clean:
        if not root.exists():
            print(f"[CLEAN] Not found, skipping: {root}")
            continue

        for item in enumerate_remove_paths(root):
            print(f"[CLEAN] Removing: {item}")

        shutil.rmtree(root)


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def parse_target_ips(args: argparse.Namespace, ips: dict) -> list[str]:
    raw_targets: list[str] = []
    for token in args.target_ip:
        raw_targets.extend([x.strip() for x in token.split(",") if x.strip()])

    target_keys = dedupe_preserve_order(raw_targets)
    if not target_keys:
        raise ValueError("At least one target IP must be provided.")

    unknown = [key for key in target_keys if key not in ips]
    if unknown:
        raise KeyError(f"Unknown target IP key(s): {unknown}. Available: {sorted(ips.keys())}")

    return target_keys


def build_target_suffix(target_keys: list[str], ips: dict) -> str:
    acronyms = sorted({ips[key]["shortform"] for key in target_keys})
    return "_".join(acronyms)


def union_target_modules(target_keys: list[str], ips: dict) -> list[str]:
    modules: set[str] = set()
    for key in target_keys:
        for module in ips[key].get("target_modules", []):
            modules.add(module)
    return sorted(modules)


def validate_preprocessed_inputs_for_soc(input_dir: Path, soc_filenames: list[str]) -> None:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(
            "Preprocessed SoC input folder is missing. "
            f"Expected: {input_dir}\n"
            "Run preprocess first:\n"
            "python Scripts/Data_Preparation_Scripts/raw_to_processed_gexf.py "
            "--config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --dataset soc"
        )

    has_any_preprocessed = any(input_dir.glob("pre_processed_*.gexf"))
    if not has_any_preprocessed:
        raise FileNotFoundError(
            "Preprocessed SoC input folder is empty (no pre_processed_*.gexf files found). "
            f"Folder: {input_dir}\n"
            "Run preprocess first:\n"
            "python Scripts/Data_Preparation_Scripts/raw_to_processed_gexf.py "
            "--config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --dataset soc"
        )

    missing: list[str] = []
    for soc_filename in soc_filenames:
        pre_gexf = input_dir / f"pre_processed_{soc_filename}"
        pre_json = input_dir / f"pre_processed_{soc_filename.replace('.gexf', '.json')}"
        if not pre_gexf.exists():
            missing.append(str(pre_gexf))
        if not pre_json.exists():
            missing.append(str(pre_json))

    if missing:
        joined = "\n".join(f"- {p}" for p in missing)
        raise FileNotFoundError(
            "Missing required preprocessed SoC files for selected --soc-file/--all-soc:\n"
            f"{joined}\n"
            "Run preprocess first:\n"
            "python Scripts/Data_Preparation_Scripts/raw_to_processed_gexf.py "
            "--config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --dataset soc"
        )


def choose_soc_files(args, soc_files_cfg: dict) -> list[str]:
    if args.all_soc:
        return list(soc_files_cfg.values())

    if args.soc_file:
        raw_tokens: list[str] = []
        for token in args.soc_file:
            raw_tokens.extend([x.strip() for x in token.split(",") if x.strip()])

        selected: list[str] = []
        for key in raw_tokens:
            if key in soc_files_cfg:
                selected.append(soc_files_cfg[key])
            else:
                selected.append(key)

        return dedupe_preserve_order(selected)

    raise ValueError("Provide one of --soc-file or --all-soc.")


def label_soc_file(
    soc_filename: str,
    target_modules: list[str],
    input_dir: Path,
    processed_out_dir: Path,
    data_root: Path,
):
    pre_gexf = input_dir / f"pre_processed_{soc_filename}"
    pre_json = input_dir / f"pre_processed_{soc_filename.replace('.gexf', '.json')}"

    if not pre_gexf.exists():
        raise FileNotFoundError(f"Missing preprocessed GEXF: {pre_gexf}")
    if not pre_json.exists():
        raise FileNotFoundError(f"Missing preprocessed JSON: {pre_json}")

    graph = nx.read_gexf(pre_gexf)
    target_set = set(target_modules)

    for node, _ in graph.nodes(data=True):
        module_name = graph.nodes[node].get("module")
        graph.nodes[node]["label"] = "1" if module_name in target_set else "0"

    save_processed_gexf(graph, soc_filename, processed_out_dir)
    save_processed_gexf(graph, soc_filename, data_root / "test" / "raw")
    save_processed_gexf(graph, soc_filename, data_root / "inference" / "raw")

    with pre_json.open("r", encoding="utf-8") as f:
        json_data = json.load(f)

    for module_name, module_data in json_data.get("module", {}).items():
        module_data["label"] = "1" if module_name in target_set else "0"

    save_processed_json(json_data, soc_filename, processed_out_dir)
    print(f"Processed: {soc_filename}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Label SoC preprocessed graph(s) by target IP module mapping.")
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    parser.add_argument(
        "--target-ip",
        required=True,
        nargs="+",
        help="One or more target IP keys from config.ips (e.g., --target-ip microblaze ila).",
    )

    sel = parser.add_mutually_exclusive_group(required=True)
    sel.add_argument(
        "--soc-file",
        nargs="+",
        help="One or more SoC file keys from config.soc_files or direct .gexf filenames.",
    )
    sel.add_argument("--all-soc", action="store_true", help="Process all SoC files in config.soc_files.")
    parser.add_argument("--input-dir", help="Override SoC preprocessed input dir.")
    parser.add_argument("--processed-out-dir", help="Override SoC processed output base dir.")
    parser.add_argument("--data-root", help="Override data root base dir.")
    parser.add_argument("--module-list", help="Comma-separated module list override for target labeling.")
    return parser


def main():
    args = build_parser().parse_args()
    ensure_package_root_cwd()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    ips = config.get("ips", {})
    target_keys = parse_target_ips(args, ips)

    soc_files_cfg = config.get("soc_files", {})
    if not soc_files_cfg:
        raise ValueError("config.soc_files is required.")

    paths = config.get("paths", {})
    input_dir_value = args.input_dir or paths.get("soc_preprocessed_dir")
    processed_base_value = args.processed_out_dir or paths.get("soc_processed_base")
    data_base_value = args.data_root or paths.get("data_base")
    if not input_dir_value or not processed_base_value or not data_base_value:
        raise ValueError("Missing soc_preprocessed_dir/soc_processed_base/data_base in config.paths or CLI overrides.")

    input_dir = resolve_path(input_dir_value)
    processed_base = resolve_path(processed_base_value)
    data_base = resolve_path(data_base_value)

    target_suffix = build_target_suffix(target_keys, ips)
    processed_out_dir = processed_base.parent / f"{processed_base.name}_{target_suffix}"
    data_root = data_base.parent / f"{data_base.name}_{target_suffix}"

    clean_paths(
        [
            processed_out_dir,
            data_root / "test",
            data_root / "inference",
        ]
    )

    if args.module_list:
        target_modules = sorted(set([x.strip() for x in args.module_list.split(",") if x.strip()]))
    else:
        target_modules = union_target_modules(target_keys, ips)

    soc_filenames = choose_soc_files(args, soc_files_cfg)
    validate_preprocessed_inputs_for_soc(input_dir=input_dir, soc_filenames=soc_filenames)

    for soc_filename in soc_filenames:
        label_soc_file(
            soc_filename=soc_filename,
            target_modules=target_modules,
            input_dir=input_dir,
            processed_out_dir=processed_out_dir,
            data_root=data_root,
        )


if __name__ == "__main__":
    main()
