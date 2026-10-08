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


def validate_preprocessed_inputs_for_ips(input_dir: Path, selected_keys: list[str], ips: dict) -> None:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(
            "Preprocessed IP input folder is missing. "
            f"Expected: {input_dir}\n"
            "Run preprocess first:\n"
            "python Scripts/Data_Preparation_Scripts/raw_to_processed_gexf.py "
            "--config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --dataset ip"
        )

    has_any_preprocessed = any(input_dir.glob("pre_processed_*.gexf"))
    if not has_any_preprocessed:
        raise FileNotFoundError(
            "Preprocessed IP input folder is empty (no pre_processed_*.gexf files found). "
            f"Folder: {input_dir}\n"
            "Run preprocess first:\n"
            "python Scripts/Data_Preparation_Scripts/raw_to_processed_gexf.py "
            "--config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --dataset ip"
        )

    missing: list[str] = []
    for ip_key in selected_keys:
        filename = ips[ip_key]["train_gexf"]
        pre_gexf = input_dir / f"pre_processed_{filename}"
        pre_json = input_dir / f"pre_processed_{filename.replace('.gexf', '.json')}"
        if not pre_gexf.exists():
            missing.append(str(pre_gexf))
        if not pre_json.exists():
            missing.append(str(pre_json))

    if missing:
        joined = "\n".join(f"- {p}" for p in missing)
        raise FileNotFoundError(
            "Missing required preprocessed IP files for selected targets/IP list:\n"
            f"{joined}\n"
            "Run preprocess first:\n"
            "python Scripts/Data_Preparation_Scripts/raw_to_processed_gexf.py "
            "--config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --dataset ip"
        )


def mark_file(
    ip_key: str,
    ip_info: dict,
    target_ips: set[str],
    input_dir: Path,
    processed_out_dir: Path,
    data_root: Path,
):
    filename = ip_info["train_gexf"]
    pre_gexf = input_dir / f"pre_processed_{filename}"
    pre_json = input_dir / f"pre_processed_{filename.replace('.gexf', '.json')}"

    label_val = "1" if ip_key in target_ips else "0"

    if not pre_gexf.exists():
        raise FileNotFoundError(f"Missing preprocessed GEXF: {pre_gexf}")
    if not pre_json.exists():
        raise FileNotFoundError(f"Missing preprocessed JSON: {pre_json}")

    graph = nx.read_gexf(pre_gexf)
    for node, _ in graph.nodes(data=True):
        graph.nodes[node]["label"] = label_val

    save_processed_gexf(graph, filename, processed_out_dir)
    save_processed_gexf(graph, filename, data_root / "train" / "raw")
    save_processed_gexf(graph, filename, data_root / "eval" / "raw")

    with pre_json.open("r", encoding="utf-8") as f:
        json_data = json.load(f)

    for _, module_data in json_data.get("module", {}).items():
        module_data["label"] = label_val

    save_processed_json(json_data, filename, processed_out_dir)
    print(f"Processed: {filename} label={label_val}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Label preprocessed pure-IP graphs for train/eval by target IP.")
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    parser.add_argument(
        "--target-ip",
        required=True,
        nargs="+",
        help="One or more target IP keys from config.ips (e.g., --target-ip microblaze ila).",
    )

    parser.add_argument("--input-dir", help="Override preprocessed input dir (expects pre_processed_* files).")
    parser.add_argument("--processed-out-dir", help="Override processed output dir (processed_* files).")
    parser.add_argument("--data-root", help="Override data root (writes train/raw and eval/raw).")
    parser.add_argument("--ip-list", help="Comma-separated subset of IP keys from config.ips.")
    return parser


def main():
    args = build_parser().parse_args()
    ensure_package_root_cwd()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    ips = config.get("ips", {})
    target_keys = parse_target_ips(args, ips)
    target_key_set = set(target_keys)

    paths = config.get("paths", {})
    input_dir_value = args.input_dir or paths.get("ip_preprocessed_dir")
    processed_base_value = args.processed_out_dir or paths.get("ip_processed_base")
    data_base_value = args.data_root or paths.get("data_base")
    if not input_dir_value or not processed_base_value or not data_base_value:
        raise ValueError("Missing ip_preprocessed_dir/ip_processed_base/data_base in config.paths or CLI overrides.")

    input_dir = resolve_path(input_dir_value)
    processed_base = resolve_path(processed_base_value)
    data_base = resolve_path(data_base_value)

    target_suffix = build_target_suffix(target_keys, ips)
    processed_out_dir = processed_base.parent / f"{processed_base.name}_{target_suffix}"
    data_root = data_base.parent / f"{data_base.name}_{target_suffix}"

    clean_paths(
        [
            processed_out_dir,
            data_root / "train",
            data_root / "eval",
        ]
    )

    if args.ip_list:
        selected_keys = [x.strip() for x in args.ip_list.split(",") if x.strip()]
    else:
        order = config.get("ip_processing_order")
        selected_keys = [k for k in order if k in ips] if order else sorted(ips.keys())

    unknown = [k for k in selected_keys if k not in ips]
    if unknown:
        raise KeyError(f"Unknown IP keys in --ip-list/ip_processing_order: {unknown}")

    validate_preprocessed_inputs_for_ips(input_dir=input_dir, selected_keys=selected_keys, ips=ips)

    for ip_key in selected_keys:
        mark_file(
            ip_key=ip_key,
            ip_info=ips[ip_key],
            target_ips=target_key_set,
            input_dir=input_dir,
            processed_out_dir=processed_out_dir,
            data_root=data_root,
        )


if __name__ == "__main__":
    main()
