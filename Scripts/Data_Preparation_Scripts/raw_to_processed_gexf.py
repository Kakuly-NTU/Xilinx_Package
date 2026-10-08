import argparse
import json
import os
from collections import Counter
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


def to_64bit_hex(value):
    try:
        val = int(value, 16)
        return f"64'h{val & 0xFFFFFFFFFFFFFFFF:016X}"
    except (ValueError, TypeError):
        return "64'h0000000000000000"


def normalize_node_attribute(graph: nx.Graph, attribute_name: str):
    for node, data in graph.nodes(data=True):
        val = data.get(attribute_name, "0")
        graph.nodes[node][attribute_name] = to_64bit_hex(val)


def cleanup_node_attribute(graph: nx.Graph, attribute_name: str):
    for node, data in graph.nodes(data=True):
        if attribute_name in data:
            del graph.nodes[node][attribute_name]


def edit_node_attributes(graph: nx.Graph, old_attr: str, new_attr: str):
    cleanup_node_attribute(graph, new_attr)
    capitalize_old = old_attr.capitalize()
    for node, data in graph.nodes(data=True):
        if old_attr == "id":
            graph.nodes[node][new_attr] = str(node)
        elif old_attr in data:
            graph.nodes[node][new_attr] = data[old_attr]
        elif capitalize_old in data:
            graph.nodes[node][new_attr] = data[capitalize_old]
        else:
            graph.nodes[node][new_attr] = "N/A"


def count_attr(graph: nx.Graph, attribute_name: str) -> dict:
    values = [data.get(attribute_name, None) for _, data in graph.nodes(data=True)]
    return dict(Counter(values))


def save_preprocessed_graph(graph: nx.Graph, original_file_name: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"pre_processed_{Path(original_file_name).name}"
    nx.write_gexf(graph, out_path)
    return out_path


def save_preprocessed_json(summary: dict, original_file_name: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_name = f"pre_processed_{Path(original_file_name).name.replace('.gexf', '.json')}"
    out_path = output_dir / json_name
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    return out_path


def process_one_file(file_path: Path, output_dir: Path, verbose: bool = False):
    if verbose:
        print(f"Processing: {file_path}")

    graph = nx.read_gexf(file_path)

    normalize_node_attribute(graph, "INIT")
    edit_node_attributes(graph, "INIT", "lut_mask")
    cleanup_node_attribute(graph, "INIT")
    edit_node_attributes(graph, "label", "cell_name")
    edit_node_attributes(graph, "id", "label")

    gexf_path = save_preprocessed_graph(graph, file_path.name, output_dir)
    type_counts = count_attr(graph, "type")
    module_counts = count_attr(graph, "module")
    module_counts_with_labels = {
        key: {"label": "NA", "nodes": value} for key, value in module_counts.items()
    }

    summary = {
        "name": file_path.name,
        "number of nodes": graph.number_of_nodes(),
        "number of edges": graph.number_of_edges(),
        "type": type_counts,
        "module": module_counts_with_labels,
    }
    json_path = save_preprocessed_json(summary, file_path.name, output_dir)

    print(f"Saved: {gexf_path}")
    print(f"Saved: {json_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert raw GEXF files into pre_processed_* GEXF/JSON files.")
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    parser.add_argument("--dataset", required=True, choices=["ip", "soc"], help="Dataset kind to process.")

    parser.add_argument("--file", help="Single .gexf file name to process from the selected dataset raw dir. If omitted, all .gexf files are processed.")

    parser.add_argument("--raw-dir", help="Override raw input directory.")
    parser.add_argument("--output-dir", help="Override preprocessed output directory.")
    parser.add_argument("--verbose", action="store_true", help="Print extra details.")
    return parser


def main():
    args = build_parser().parse_args()
    ensure_package_root_cwd()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    paths = config.get("paths", {})
    if args.dataset == "ip":
        raw_dir_value = args.raw_dir or paths.get("ip_raw_dir")
        out_dir_value = args.output_dir or paths.get("ip_preprocessed_dir")
    else:
        raw_dir_value = args.raw_dir or paths.get("soc_raw_dir")
        out_dir_value = args.output_dir or paths.get("soc_preprocessed_dir")

    if not raw_dir_value or not out_dir_value:
        raise ValueError("Missing raw/output directory. Set in config.paths or pass --raw-dir/--output-dir.")

    raw_dir = resolve_path(raw_dir_value)
    output_dir = resolve_path(out_dir_value)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    if args.file:
        file_path = raw_dir / args.file
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        process_one_file(file_path, output_dir, verbose=args.verbose)
        return

    files = sorted([p for p in raw_dir.iterdir() if p.suffix.lower() == ".gexf"])
    if not files:
        raise FileNotFoundError(f"No .gexf files found under: {raw_dir}")
    for file_path in files:
        process_one_file(file_path, output_dir, verbose=args.verbose)


if __name__ == "__main__":
    main()
