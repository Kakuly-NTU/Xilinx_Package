import argparse
import os
import json
import numpy as np
import networkx as nx
import torch
from torch_geometric.utils import from_networkx


def resolve_runtime_path(path_value):
    candidate = os.path.expanduser(path_value)
    if os.path.isabs(candidate):
        return candidate
    return os.path.abspath(os.path.join(os.getcwd(), candidate))


def ensure_package_root_cwd():
    cwd = os.getcwd()
    required = [os.path.join(cwd, "Data"), os.path.join(cwd, "Scripts")]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        joined = "\n".join(f"- {p}" for p in missing)
        raise RuntimeError(
            "Run this script from Xilinx_Package root. Missing required paths in current working directory:\n"
            f"{joined}"
        )


def validate_type_dict_path(type_dict_file):
    if not os.path.isfile(type_dict_file):
        raise FileNotFoundError(f"Type dictionary file not found: {type_dict_file}")
    try:
        with open(type_dict_file, "r", encoding="utf-8") as f:
            json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Type dictionary is not valid JSON: {type_dict_file}") from e
    except OSError as e:
        raise OSError(f"Could not read type dictionary file: {type_dict_file}") from e


def validate_split_raw_inputs(raw_root, splits):
    if not os.path.isdir(raw_root):
        raise FileNotFoundError(f"Raw root not found: {raw_root}")

    missing_dirs = []
    empty_splits = []
    for split in splits:
        split_raw_dir = os.path.join(raw_root, split, "raw")
        if not os.path.isdir(split_raw_dir):
            missing_dirs.append(split_raw_dir)
            continue

        gexf_files = [f for f in os.listdir(split_raw_dir) if f.endswith(".gexf")]
        if not gexf_files:
            empty_splits.append(split_raw_dir)

    if missing_dirs:
        joined = "\n".join(f"- {p}" for p in missing_dirs)
        raise FileNotFoundError(
            "Missing required split raw directories:\n"
            f"{joined}\n"
            "Run labeling scripts first to create split raw inputs:\n"
            "Example labeling script commands:\n"
            "python Scripts/Data_Preparation_Scripts/label_train_ips.py "
            "--config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --target-ip <target_ip>\n"
            "python Scripts/Data_Preparation_Scripts/label_soc.py "
            "--config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --target-ip <target_ip> --soc-file <soc_key_or_file>"
        )

    if empty_splits:
        joined = "\n".join(f"- {p}" for p in empty_splits)
        raise FileNotFoundError(
            "No .gexf files found in split raw directories:\n"
            f"{joined}\n"
            "Run labeling scripts first to populate split raw inputs:\n"
            "Example labeling script commands:\n"
            "python Scripts/Data_Preparation_Scripts/label_train_ips.py "
            "--config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --target-ip <target_ip>\n"
            "python Scripts/Data_Preparation_Scripts/label_soc.py "
            "--config Scripts/Data_Preparation_Scripts/graph_labeling_config.json --target-ip <target_ip> --soc-file <soc_key_or_file>"
        )


def _remove_attribute(G, tnode, attr):
    G.nodes[tnode].pop(attr, None)


def relabel(G):
    mapping = {}
    it = 0
    for n in G.nodes:
        mapping[n] = str(it)
        it += 1
    G = nx.relabel_nodes(G, mapping)
    return G


def RemoveAttr(G, attrs_to_keep):
    for n in G.nodes:
        att_to_remove = []
        for att in G.nodes[n]:
            if att not in attrs_to_keep:
                att_to_remove.append(att)
        for att in att_to_remove:
            _remove_attribute(G, n, att)
    return G


def ExtractLUTMaskFeature(G, named_attr="lut_mask"):
    for n in G.nodes:
        if named_attr not in G.nodes[n]:
            G.nodes[n][named_attr] = "64'h0000000000000000"
        G.nodes[n][named_attr] = G.nodes[n][named_attr].replace("64'h", "0x")
    for n in G.nodes:
        if G.nodes[n][named_attr] == "0x0000000000000000":
            bincode = bin(0)[2:].zfill(64)
        else:
            hex_int = int(G.nodes[n][named_attr], 16)
            bincode = bin(hex_int)[2:].zfill(64)
        G.nodes[n]["x"] = [int(d) for d in bincode]
    return G


def ExtractTypeFeature(G, named_attr="type", type_dict_file="Scripts/Data_Preparation_Scripts/type_IP_onehot_dict_incl_sparc.json"):
    with open(type_dict_file, "r", encoding="utf-8") as f:
        type_dict_json = json.load(f)
    type_dict = {k: np.array(v, dtype=np.float32) for k, v in type_dict_json.items()}
    for n in G.nodes:
        node_type = G.nodes[n].get(named_attr, G.nodes[n].get("cell_type", ""))
        type_feature = type_dict.get(node_type, np.zeros(63, dtype=np.float32))
        G.nodes[n]["x"] = np.append(G.nodes[n]["x"], type_feature)
        assert G.nodes[n]["x"].shape == (127,), f"Feature shape mismatch at node {n}: {G.nodes[n]['x'].shape}"
    return G


def process_gexf_file(gexf_path, type_dict_file="Scripts/Data_Preparation_Scripts/type_IP_onehot_dict_incl_sparc.json", max_degree=600, force_undirected=True):
    try:
        G = nx.read_gexf(gexf_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read GEXF file: {gexf_path}") from e

    G = relabel(G)
    G = ExtractLUTMaskFeature(G, named_attr="lut_mask")
    G = ExtractTypeFeature(G, named_attr="type", type_dict_file=type_dict_file)
    G = RemoveAttr(G, attrs_to_keep=["x", "label"])

    for n in G.nodes:
        label = G.nodes[n].get("label", 0)
        try:
            label_int = int(label)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid node label '{label}' in {gexf_path} at node '{n}'. Expected integer-like label.") from e

        G.nodes[n]["y"] = label_int
        G.nodes[n]["label"] = label_int
        G.nodes[n]["mask"] = 0

    if G.is_directed() and force_undirected:
        G = G.to_undirected()

    G.graph = {}
    data = from_networkx(G)
    data.x = torch.tensor(np.stack([G.nodes[n]["x"] for n in G.nodes]), dtype=torch.float)
    data.y = torch.tensor([G.nodes[n]["y"] for n in G.nodes], dtype=torch.long)
    data.train_mask = data.y >= 0
    return data


def process_all_gexf(raw_dir, processed_dir, type_dict_file="Scripts/Data_Preparation_Scripts/type_IP_onehot_dict_incl_sparc.json", max_degree=600, force_undirected=True):
    os.makedirs(processed_dir, exist_ok=True)
    gexf_files = [f for f in os.listdir(raw_dir) if f.endswith(".gexf")]
    data_list = []
    for fname in gexf_files:
        gexf_path = os.path.join(raw_dir, fname)
        try:
            data = process_gexf_file(gexf_path, type_dict_file=type_dict_file, max_degree=max_degree, force_undirected=force_undirected)
            torch.save(data, os.path.join(processed_dir, fname.replace(".gexf", ".pt")))
        except Exception as e:
            raise RuntimeError(f"Failed processing file in split directory '{raw_dir}': {gexf_path}") from e
        data_list.append(data)
    return data_list


def extract_CC_gexf_file(gexf_path):
    try:
        G = nx.read_gexf(gexf_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read GEXF file: {gexf_path}") from e

    if G.number_of_nodes() == 0:
        raise ValueError(f"Graph has no nodes, cannot extract connected component: {gexf_path}")

    if G.is_directed():
        components = nx.weakly_connected_components(G)
    else:
        components = nx.connected_components(G)

    Largest_CC = max(components, key=len)
    H = G.subgraph(Largest_CC).copy()
    return H


def extract_CC_all_gexf(raw_dir, processed_dir):
    os.makedirs(processed_dir, exist_ok=True)
    gexf_files = [f for f in os.listdir(raw_dir) if f.endswith(".gexf")]
    for fname in gexf_files:
        gexf_path = os.path.join(raw_dir, fname)
        G = extract_CC_gexf_file(gexf_path)
        nx.write_gexf(G, os.path.join(processed_dir, fname))


def main():
    parser = argparse.ArgumentParser(description="Preprocess Xilinx IP netlist graphs into PyG Data objects.")
    parser.add_argument("--raw-root", required=True, help="Root directory containing train/eval/test/inference raw folders.")
    parser.add_argument("--output-root", required=True, help="Root directory to write processed data.")
    parser.add_argument(
        "--type-dict",
        default="Scripts/Data_Preparation_Scripts/type_IP_onehot_dict_incl_sparc.json",
        help="Path to node type one-hot dictionary JSON.",
    )
    parser.add_argument("--max-degree", type=int, default=600, help="Maximum node degree filter (kept for compatibility).")
    parser.add_argument(
        "--no-extract-cc",
        dest="extract_cc",
        action="store_false",
        help="Disable largest connected component extraction (default: extraction enabled).",
    )
    parser.set_defaults(extract_cc=True)
    parser.add_argument(
        "--keep-directed",
        action="store_true",
        help="Skip conversion to undirected graphs (default: disabled; undirected output).",
    )
    args = parser.parse_args()

    ensure_package_root_cwd()

    args.type_dict = resolve_runtime_path(args.type_dict)
    args.raw_root = resolve_runtime_path(args.raw_root)
    args.output_root = resolve_runtime_path(args.output_root)

    splits = ["train", "eval", "test", "inference"]
    validate_type_dict_path(args.type_dict)
    validate_split_raw_inputs(args.raw_root, splits)

    for split in splits:
        raw_dir = os.path.join(args.raw_root, split, "raw")
        cc_dir = os.path.join(args.output_root, split, "CC_extracted")
        processed_dir = os.path.join(args.output_root, split, "processed")

        if args.extract_cc:
            extract_CC_all_gexf(raw_dir, cc_dir)
            src_dir = cc_dir
        else:
            src_dir = raw_dir

        process_all_gexf(
            src_dir,
            processed_dir,
            type_dict_file=args.type_dict,
            max_degree=args.max_degree,
            force_undirected=not args.keep_directed,
        )


if __name__ == "__main__":
    main()
