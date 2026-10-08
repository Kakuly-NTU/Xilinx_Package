import argparse
import json
import os
import shlex
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import networkx as nx
from torch_sparse import SparseTensor
from torch_geometric.data import Batch
from sklearn.metrics import precision_recall_fscore_support
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from models import build_model
from train_eval import split_train_val_masks, merge_graphs_and_masks


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_RESULTS_FILE = PACKAGE_ROOT / "experiment_results.txt"


def ensure_standard_output_roots():
    for dirname in ("work_dirs_xilinx_MB", "Inference_Output_MB"):
        (PACKAGE_ROOT / dirname).mkdir(parents=True, exist_ok=True)


def load_split_data(processed_dir: str):
    pt_files = [f for f in os.listdir(processed_dir) if f.endswith(".pt")]
    data_list = []
    for file in pt_files:
        file_path = os.path.join(processed_dir, file)
        data = torch.load(file_path, weights_only=False)
        data_list.append(data)
    return data_list


def merge_graphs(dataset):
    merged_graph = Batch.from_data_list(dataset)
    merged_graph.train_mask = torch.cat([data.train_mask for data in dataset], dim=0)
    if hasattr(dataset[0], "val_mask"):
        merged_graph.val_mask = torch.cat([data.val_mask for data in dataset], dim=0)
    return merged_graph


def compute_energy(logits: torch.Tensor, temperature: float = 1.0):
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


def belief_propagation_energy(energy: torch.Tensor, edge_index, steps: int, damping: float = 0.5):
    """
    Propagate scalar energies (as in the notebook): iteratively replace each node's energy
    with a convex combination of its own energy and the mean of neighbors' energies.
    """
    num_nodes = energy.size(0)
    if isinstance(edge_index, SparseTensor):
        adj = edge_index
    else:
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes))
    deg = adj.sum(dim=1).clamp(min=1)

    e = energy.unsqueeze(1)
    for _ in range(steps):
        neigh = adj.matmul(e) / deg.unsqueeze(1)
        e = damping * e + (1 - damping) * neigh
    return e.squeeze(1)


def compute_threshold(energies: torch.Tensor, labels: torch.Tensor, keep_ratio: float = 0.95, positive_class: int = 1):
    pos_mask = labels == positive_class
    pos_energies = energies[pos_mask].cpu().numpy()
    if len(pos_energies) == 0:
        raise ValueError("No positive samples found for threshold computation.")
    return float(np.quantile(pos_energies, keep_ratio))


def apply_energy_threshold(preds: torch.Tensor, energies: torch.Tensor, threshold: float):
    adjusted = preds.clone()
    pos_mask = preds == 1
    adjusted[pos_mask & (energies > threshold)] = 0
    return adjusted


def classification_stats(labels: torch.Tensor, preds: torch.Tensor):
    labels_np = labels.cpu().numpy()
    preds_np = preds.cpu().numpy()
    tp = int(((labels_np == 1) & (preds_np == 1)).sum())
    fp = int(((labels_np == 0) & (preds_np == 1)).sum())
    fn = int(((labels_np == 1) & (preds_np == 0)).sum())
    tn = int(((labels_np == 0) & (preds_np == 0)).sum())

    per_class_prec, per_class_rec, per_class_f1, _ = precision_recall_fscore_support(
        labels_np, preds_np, labels=[0, 1], zero_division=0
    )
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        labels_np, preds_np, average="macro", zero_division=0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accepted_pos": tp + fp,
        "class0_precision": float(per_class_prec[0]),
        "class0_recall": float(per_class_rec[0]),
        "class0_f1": float(per_class_f1[0]),
        "class1_precision": float(per_class_prec[1]),
        "class1_recall": float(per_class_rec[1]),
        "class1_f1": float(per_class_f1[1]),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
    }


def format_metrics_block(stats: dict) -> str:
    return (
        f"class1={stats['class1_precision']:.4f}/{stats['class1_recall']:.4f}/{stats['class1_f1']:.4f}; "
        f"class0={stats['class0_precision']:.4f}/{stats['class0_recall']:.4f}/{stats['class0_f1']:.4f}; "
        f"macro={stats['macro_precision']:.4f}/{stats['macro_recall']:.4f}/{stats['macro_f1']:.4f}"
    )


def write_or_update_run_config_results(
    args,
    run_params: dict,
    pre_infer_stats: dict,
    infer_metrics_no_bp: dict,
    infer_metrics_bp: dict,
) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cmd = " ".join(shlex.quote(x) for x in sys.argv)
    run_config_path = Path(run_params["checkpoint_dir"]) / "run_config_results.txt"

    legacy_json_path = Path(run_params["checkpoint_dir"]) / "run_config.json"

    existing = {}
    if run_config_path.exists():
        try:
            existing = json.loads(run_config_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    elif legacy_json_path.exists():
        try:
            existing = json.loads(legacy_json_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    inference_entry = {
        "timestamp": timestamp,
        "command": cmd,
        "parameters": {
            **run_params,
            "keep_ratios": args.keep_ratios,
            "bp_steps": args.bp_steps,
            "output_dir": args.output_dir,
            "checkpoint": args.checkpoint,
            "export_gexf": bool(args.export_gexf),
            "excel": bool(args.excel),
        },
        "directories": {
            "data_root": args.data_root,
            "output_dir": args.output_dir,
            "checkpoint_dir": run_params["checkpoint_dir"],
        },
        "artifacts": {
            "summary": os.path.join(args.output_dir, "ood_summary.txt"),
            "infer_log": os.path.join(args.output_dir, "infer.log"),
            "plots": [
                os.path.join(args.output_dir, "val_energy_no_bp.png"),
                os.path.join(args.output_dir, "infer_energy_no_bp.png"),
                os.path.join(args.output_dir, "infer_energy_bp.png"),
            ],
            "energy_csv": [
                os.path.join(args.output_dir, "train_energy_scores_no_bp.csv"),
                os.path.join(args.output_dir, "val_energy_scores_no_bp.csv"),
                os.path.join(args.output_dir, f"train_bp_energyprop_{args.bp_steps}.csv"),
                os.path.join(args.output_dir, f"val_bp_energyprop_{args.bp_steps}.csv"),
            ],
            "gexf": os.path.join(args.output_dir, "inference_output.gexf") if args.export_gexf else None,
            "excel": os.path.join(args.output_dir, "inference_output.xlsx") if args.excel else None,
            "experiment_results": str(EXPERIMENT_RESULTS_FILE),
        },
        "metrics": {
            "pre_ood": {
                "inference": pre_infer_stats,
            },
            "ood_no_bp": {
                "inference": infer_metrics_no_bp,
            },
            "ood_bp": {
                "bp_steps": args.bp_steps,
                "inference": infer_metrics_bp,
            },
        },
    }

    payload = {
        "schema_version": 1,
        "last_updated": timestamp,
        "train_run": existing.get("train_run"),
        "inference_runs": existing.get("inference_runs", []) + [inference_entry],
    }

    run_config_path.parent.mkdir(parents=True, exist_ok=True)
    run_config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return run_config_path


def append_experiment_results(args, run_config_path: Path, pre_infer_stats: dict, thresholds_no_bp: dict, thresholds_bp: dict, infer_metrics_no_bp: dict, infer_metrics_bp: dict):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cmd = " ".join(shlex.quote(x) for x in sys.argv)

    lines = [
        f"{timestamp} ood_inference run",
        f"- Cmd: {cmd}",
        f"- Checkpoint: {args.checkpoint}",
        f"- Output dir: {args.output_dir}",
        f"- Run config results: {run_config_path}",
        f"- Inference metrics before OOD: {format_metrics_block(pre_infer_stats)}.",
    ]

    if thresholds_no_bp:
        for keep_ratio in args.keep_ratios:
            stats_i = infer_metrics_no_bp.get(keep_ratio)
            if stats_i is None:
                continue
            lines.append(
                f"- OOD no-BP @keep={keep_ratio}: inference thr={stats_i['threshold']:.4f}, accepted_pos={stats_i['accepted_pos']}, {format_metrics_block(stats_i)}."
            )

    if thresholds_bp:
        for keep_ratio in args.keep_ratios:
            stats_i = infer_metrics_bp.get(keep_ratio)
            if stats_i is None:
                continue
            lines.append(
                f"- OOD BP={args.bp_steps} @keep={keep_ratio}: inference thr={stats_i['threshold']:.4f}, accepted_pos={stats_i['accepted_pos']}, {format_metrics_block(stats_i)}."
            )

    artifact_parts = [
        f"summary={os.path.join(args.output_dir, 'ood_summary.txt')}",
        f"plots={os.path.join(args.output_dir, 'val_energy_no_bp.png')}|{os.path.join(args.output_dir, 'infer_energy_no_bp.png')}",
        (
            f"energy_csv={os.path.join(args.output_dir, 'train_energy_scores_no_bp.csv')}|"
            f"{os.path.join(args.output_dir, 'val_energy_scores_no_bp.csv')}|"
            f"{os.path.join(args.output_dir, f'train_bp_energyprop_{args.bp_steps}.csv')}|"
            f"{os.path.join(args.output_dir, f'val_bp_energyprop_{args.bp_steps}.csv')}|"
            f"inference_only"
        ),
        f"experiment_log={EXPERIMENT_RESULTS_FILE}",
    ]
    if args.export_gexf:
        artifact_parts.append(f"gexf={os.path.join(args.output_dir, 'inference_output.gexf')}")
    if args.excel:
        artifact_parts.append(f"excel={os.path.join(args.output_dir, 'inference_output.xlsx')}")

    lines.append("- Artifacts: " + ", ".join(artifact_parts) + ".")
    lines.append("")

    EXPERIMENT_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENT_RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def evaluate_with_thresholds(labels, preds, energies, thresholds):
    results = {}
    for keep_ratio, threshold in thresholds.items():
        adjusted = apply_energy_threshold(preds, energies, threshold)
        results[keep_ratio] = classification_stats(labels, adjusted)
        results[keep_ratio]["threshold"] = float(threshold)
    return results


def run_model(model, graph, device):
    model.eval()
    with torch.no_grad():
        logits = model(graph.x.to(device), graph.edge_index.to(device)).cpu()
    preds = logits.argmax(dim=1)
    energies = compute_energy(logits)
    return logits, preds, energies


def save_gexf_with_predictions(src_gexf: str, dst_gexf: str, preds, energies, energies_bp=None):
    G = nx.read_gexf(src_gexf)
    nodes_sorted = sorted(G.nodes, key=lambda x: int(x))
    for idx, n in enumerate(nodes_sorted):
        G.nodes[n]["pred"] = int(preds[idx])
        G.nodes[n]["energy_score_no_prop"] = float(energies[idx])
        if energies_bp is not None:
            G.nodes[n]["energy_score_bp"] = float(energies_bp[idx])
    nx.write_gexf(G, dst_gexf)


def maybe_export_excel(gexf_file: str, excel_file: str):
    G = nx.read_gexf(gexf_file)
    node_df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient="index")
    node_df.index.name = "node_id"
    node_df.reset_index(inplace=True)
    node_df.to_excel(excel_file, index=False)


def plot_energy(energies, labels, thresholds, title, out_path, xlim=None):
    plt.figure(figsize=(8, 5))
    if labels is not None:
        sns.kdeplot(energies[labels == 0], label="Class 0", fill=True, alpha=0.5, color="#1f77b4")
        sns.kdeplot(energies[labels == 1], label="Class 1", fill=True, alpha=0.5, color="#d62728")
    else:
        sns.kdeplot(energies, fill=True, alpha=0.5, color="#1f77b4")
    for t in thresholds:
        plt.axvline(t, color="k", linestyle="--", linewidth=1, label=f"thr {t:.3f}")
    plt.title(title)
    plt.xlabel("Energy (lower = higher confidence)")
    plt.ylabel("Density")
    if xlim:
        plt.xlim(xlim)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def percentile_from_array(arr, percentile):
    arr = np.asarray(arr)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, percentile))


def compute_logits_and_energies(model, graph, device, bp_steps):
    graph = graph.to("cpu")
    with torch.no_grad():
        logits = model(graph.x.to(device), graph.edge_index.to(device)).cpu()
    energy = compute_energy(logits)
    bp_energy = belief_propagation_energy(energy.clone(), graph.edge_index, steps=bp_steps)
    return logits, energy, bp_energy


def export_energy_split(graph, mask_attr, logits, energy, bp_energy, output_dir, split_tag, bp_steps):
    mask = getattr(graph, mask_attr, None)
    if mask is None:
        return None
    mask = mask.bool()
    if mask.sum().item() == 0:
        return None
    indices = torch.where(mask)[0]
    node_idx = indices.cpu().numpy()
    labels = graph.y[indices].cpu().numpy()
    preds = logits.argmax(dim=1)[indices].cpu().numpy()
    energy_vals = energy[indices].cpu().numpy()
    df_no_bp = pd.DataFrame(
        {
            "node_idx": node_idx,
            "energy_score_no_bp": energy_vals,
            "pred_label_no_bp": preds,
            "true_label_no_bp": labels,
        }
    )
    df_no_bp.to_csv(os.path.join(output_dir, f"{split_tag}_energy_scores_no_bp.csv"), index=False)

    bp_vals = bp_energy[indices].cpu().numpy()
    df_bp = pd.DataFrame(
        {
            "node_idx": node_idx,
            f"energy_score_bp_{bp_steps}": bp_vals,
            f"pred_label_bp_{bp_steps}": preds,
            f"true_label_bp_{bp_steps}": labels,
        }
    )
    df_bp.to_csv(os.path.join(output_dir, f"{split_tag}_bp_energyprop_{bp_steps}.csv"), index=False)

    correct_mask = (labels == 1) & (preds == 1)
    incorrect_mask = (labels == 0) & (preds == 1)

    return {
        "no_bp": {
            "correct": energy_vals[correct_mask],
            "incorrect": energy_vals[incorrect_mask],
        },
        "bp": {
            "correct": bp_vals[correct_mask],
            "incorrect": bp_vals[incorrect_mask],
        },
    }


def plot_energy_histograms(energy_stats, keep_ratios, output_dir, bp_steps):
    split_order = ["train", "val", "test"]

    def gather_arrays(variant):
        arrays = []
        for split in split_order:
            split_stats = energy_stats.get(split)
            if not split_stats or variant not in split_stats:
                continue
            for arr in split_stats[variant].values():
                if arr.size > 0:
                    arrays.append(arr)
        return arrays

    def plot_variant(variant, threshold, ratio_label):
        arrays = gather_arrays(variant)
        if not arrays:
            return
        energy_min = min(arr.min() for arr in arrays)
        energy_max = max(arr.max() for arr in arrays)
        if variant == "no_bp":
            energy_max = max(energy_max, 0.0)
        bin_edges = np.linspace(energy_min, energy_max, 31)
        fig, axes = plt.subplots(1, len(split_order), figsize=(15, 4))
        for ax, split in zip(axes, split_order):
            split_stats = energy_stats.get(split)
            corr = split_stats[variant]["correct"] if split_stats else np.array([])
            inc = split_stats[variant]["incorrect"] if split_stats else np.array([])
            if corr.size > 0:
                ax.hist(corr, bins=bin_edges, alpha=0.7, label="Correct Class 1", color="green")
            if inc.size > 0:
                ax.hist(inc, bins=bin_edges, alpha=0.7, label="Incorrect Class 1", color="red")
            ax.set_xlabel("Energy Score")
            ax.set_ylabel("Count")
            title_suffix = "No Propagation" if variant == "no_bp" else f"BP steps = {bp_steps}"
            ax.set_title(f"{split.capitalize()}: Energy Scores for Class 1\n({title_suffix})")
            ax.grid(True)
            ax.set_xlim(energy_min, energy_max)
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            show_threshold = variant == "bp" or split in ("val", "test")
            if show_threshold and not np.isnan(threshold):
                ax.axvline(threshold, color="orange", linestyle="--", linewidth=2)
            if split == "train":
                ax.legend()
        plt.tight_layout()
        variant_name = "no_bp" if variant == "no_bp" else f"bp{bp_steps}"
        out_path = os.path.join(output_dir, f"energy_train_val_test_{variant_name}_thr{ratio_label}.png")
        plt.savefig(out_path, dpi=200)
        plt.close(fig)

    for keep_ratio in keep_ratios:
        ratio_label = int(round(keep_ratio * 100))
        val_stats = energy_stats.get("val")
        if not val_stats:
            continue
        val_correct_no_bp = val_stats["no_bp"]["correct"]
        val_correct_bp = val_stats["bp"]["correct"]
        threshold_no_bp = percentile_from_array(val_correct_no_bp, keep_ratio * 100)
        threshold_bp = percentile_from_array(val_correct_bp, keep_ratio * 100)
        if not np.isnan(threshold_no_bp):
            plot_variant("no_bp", threshold_no_bp, ratio_label)
        if not np.isnan(threshold_bp):
            plot_variant("bp", threshold_bp, ratio_label)


def generate_energy_reports(model, data_root, output_dir, keep_ratios, bp_steps, cfg, device):
    train_dir = os.path.join(data_root, "train", "processed")
    test_dir = os.path.join(data_root, "test", "processed")
    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        return
    train_ratio = cfg.get("train_ratio", 0.7)
    seed = cfg.get("seed", 42)

    train_dataset = load_split_data(train_dir)
    split_train_val_masks(train_dataset, train_ratio=train_ratio, seed=seed)
    train_graph = merge_graphs_and_masks(train_dataset)

    test_dataset = load_split_data(test_dir)
    test_graph = merge_graphs_and_masks(test_dataset)

    train_logits, train_energy, train_bp_energy = compute_logits_and_energies(model, train_graph, device, bp_steps)
    test_logits, test_energy, test_bp_energy = compute_logits_and_energies(model, test_graph, device, bp_steps)

    energy_stats = {}
    energy_stats["train"] = export_energy_split(train_graph, "train_mask", train_logits, train_energy, train_bp_energy, output_dir, "train", bp_steps)
    energy_stats["val"] = export_energy_split(train_graph, "val_mask", train_logits, train_energy, train_bp_energy, output_dir, "val", bp_steps)
    energy_stats["test"] = export_energy_split(test_graph, "train_mask", test_logits, test_energy, test_bp_energy, output_dir, "test", bp_steps)

    # Remove None entries (e.g., missing masks)
    energy_stats = {k: v for k, v in energy_stats.items() if v is not None}
    if "val" not in energy_stats:
        return
    plot_energy_histograms(energy_stats, keep_ratios, output_dir, bp_steps)


def main():
    parser = argparse.ArgumentParser(description="Inference + OOD rejection using energy scores (optional BP smoothing).")
    parser.add_argument("--data-root", required=True, help="Root directory containing eval/test/inference processed data.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint (.pth).")
    parser.add_argument("--model", default="gcn-4-skip", help="Model name: gcn-4-skip, gat, or gatv2.")
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--heads", type=int, default=1, help="GAT attention heads (only used for model=gat).")
    parser.add_argument("--keep-ratios", type=float, nargs="+", default=[0.9, 0.95], help="Fractions of positive validation nodes to retain.")
    parser.add_argument("--bp-steps", type=int, default=4, help="Belief propagation steps (0 disables BP).")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device string, e.g., auto, cuda:0, cuda:1, or cpu. auto resolves to cuda:0 if available, else cpu.",
    )
    parser.add_argument("--export-gexf", action="store_true", help="Export inference_output.gexf (default: disabled).")
    parser.add_argument("--inference-gexf", default=None, help="Path to the CC-extracted inference gexf file (required with --export-gexf).")
    parser.add_argument("--output-dir", required=True, help="Directory to write inference outputs.")
    parser.add_argument("--excel", action="store_true", help="Also export Excel version of the GEXF output (requires --export-gexf).")
    args = parser.parse_args()

    ensure_standard_output_roots()

    if args.excel and not args.export_gexf:
        parser.error("--excel requires --export-gexf.")
    if args.export_gexf and not args.inference_gexf:
        parser.error("--inference-gexf is required when --export-gexf is set.")

    device = "cuda:0" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)

    val_dataset = load_split_data(os.path.join(args.data_root, "eval", "processed"))
    infer_dataset = load_split_data(os.path.join(args.data_root, "inference", "processed"))

    val_graph = merge_graphs(val_dataset)
    infer_graph = merge_graphs(infer_dataset)

    in_channels = val_graph.num_features
    out_channels = int(val_graph.y.max().item()) + 1

    raw_model = build_model(
        args.model,
        in_channels,
        out_channels,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        drop_out=args.dropout,
        heads=args.heads,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_config = ckpt["config"] if isinstance(ckpt, dict) and "config" in ckpt else {}
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    raw_model.load_state_dict(state)
    model = raw_model

    run_params = {
        "data_root": args.data_root,
        "model_name": args.model,
        "hidden_channels": args.hidden_channels,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "heads": args.heads,
        "epochs": ckpt_config.get("epochs", "unknown"),
        "learning_rate": ckpt_config.get("lr", "unknown"),
        "weight_decay": ckpt_config.get("weight_decay", "unknown"),
        "optimizer": ckpt_config.get("optimizer", "unknown"),
        "train_ratio": ckpt_config.get("train_ratio", "unknown"),
        "mode": "full_batch" if ckpt_config.get("full_batch", False) else "batched",
        "train_batch_size": ckpt_config.get("train_batch_size", "unknown"),
        "val_batch_size": ckpt_config.get("val_batch_size", "unknown"),
        "test_batch_size": ckpt_config.get("test_batch_size", "unknown"),
        "seed": ckpt_config.get("seed", "unknown"),
        "patience": ckpt_config.get("patience", "unknown"),
        "best_epoch": ckpt.get("best_epoch", "unknown") if isinstance(ckpt, dict) else "unknown",
        "checkpoint_dir": str(Path(args.checkpoint).resolve().parent),
        "device": device,
    }

    val_logits, val_preds, val_energy = run_model(model, val_graph, device)
    infer_logits, infer_preds, infer_energy = run_model(model, infer_graph, device)

    val_bp_energy = None
    infer_bp_energy = None
    if args.bp_steps > 0:
        val_bp_energy = belief_propagation_energy(val_energy.clone(), val_graph.edge_index, steps=args.bp_steps)
        infer_bp_energy = belief_propagation_energy(infer_energy.clone(), infer_graph.edge_index, steps=args.bp_steps)

    thresholds_no_bp = {
        r: compute_threshold(val_energy, val_graph.y.cpu(), keep_ratio=r, positive_class=1) for r in args.keep_ratios
    }
    for r, t in thresholds_no_bp.items():
        print(f"[no-bp] keep_ratio={r}: threshold={t:.4f}")

    infer_metrics_no_bp = evaluate_with_thresholds(infer_graph.y.cpu(), infer_preds, infer_energy, thresholds_no_bp)

    thresholds_bp = {}
    if val_bp_energy is not None:
        thresholds_bp = {
            r: compute_threshold(val_bp_energy, val_graph.y.cpu(), keep_ratio=r, positive_class=1) for r in args.keep_ratios
        }
        for r, t in thresholds_bp.items():
            print(f"[bp] keep_ratio={r}: threshold={t:.4f}")

    infer_metrics_bp = {}
    if thresholds_bp and infer_bp_energy is not None:
        infer_metrics_bp = evaluate_with_thresholds(infer_graph.y.cpu(), infer_preds, infer_bp_energy, thresholds_bp)

    def append_metrics_lines(summary_lines, split_name, variant_name, metrics_dict):
        for keep_ratio in args.keep_ratios:
            stats = metrics_dict.get(keep_ratio)
            if not stats:
                continue
            summary_lines.append(
                f"[{variant_name}][keep_ratio={keep_ratio}] {split_name}: "
                f"thr={stats['threshold']:.4f}, TP={stats['tp']}, FP={stats['fp']}, FN={stats['fn']}, accepted_pos={stats['accepted_pos']}, "
                f"class1={stats['class1_precision']:.4f}/{stats['class1_recall']:.4f}/{stats['class1_f1']:.4f}, "
                f"class0={stats['class0_precision']:.4f}/{stats['class0_recall']:.4f}/{stats['class0_f1']:.4f}, "
                f"macro={stats['macro_precision']:.4f}/{stats['macro_recall']:.4f}/{stats['macro_f1']:.4f}"
            )

    pre_infer_stats = classification_stats(infer_graph.y.cpu(), infer_preds)
    print(f"Pre-OOD inference metrics: {format_metrics_block(pre_infer_stats)}")

    summary_lines = []
    append_metrics_lines(summary_lines, "inference", "no-bp", infer_metrics_no_bp)
    if infer_metrics_bp:
        append_metrics_lines(summary_lines, "inference", "bp", infer_metrics_bp)

    for line in summary_lines:
        print(line)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.export_gexf:
        out_gexf = os.path.join(args.output_dir, "inference_output.gexf")
        save_gexf_with_predictions(
            args.inference_gexf,
            out_gexf,
            infer_preds.numpy(),
            infer_energy.numpy(),
            energies_bp=infer_bp_energy.numpy() if infer_bp_energy is not None else None,
        )
        print(f"Saved inference GEXF with predictions to {out_gexf}")

        if args.excel:
            out_excel = os.path.join(args.output_dir, "inference_output.xlsx")
            maybe_export_excel(out_gexf, out_excel)
            print(f"Saved Excel export to {out_excel}")
    else:
        print("Skipping GEXF/Excel export (use --export-gexf to enable).")

    thresholds_plot = [thresholds_no_bp[r] for r in args.keep_ratios]

    plot_energy(
        val_energy.numpy(),
        val_graph.y.cpu().numpy(),
        thresholds_plot,
        title="Validation energy (no BP)",
        out_path=os.path.join(args.output_dir, "val_energy_no_bp.png"),
        xlim=(-10, 2),
    )
    plot_energy(
        infer_energy.numpy(),
        None,
        thresholds_plot,
        title="Inference energy (no BP)",
        out_path=os.path.join(args.output_dir, "infer_energy_no_bp.png"),
        xlim=(-10, 2),
    )
    if infer_bp_energy is not None and thresholds_bp:
        thresholds_plot_bp = [thresholds_bp[r] for r in args.keep_ratios]
        plot_energy(
            infer_bp_energy.numpy(),
            None,
            thresholds_plot_bp,
            title="Inference energy (BP)",
            out_path=os.path.join(args.output_dir, "infer_energy_bp.png"),
            xlim=(-10, 2),
        )

    generate_energy_reports(
        model,
        args.data_root,
        args.output_dir,
        args.keep_ratios,
        args.bp_steps,
        ckpt_config,
        device,
    )

    summary_path = os.path.join(args.output_dir, "ood_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"Wrote OOD summary to {summary_path}")

    run_config_path = write_or_update_run_config_results(
        args=args,
        run_params=run_params,
        pre_infer_stats=pre_infer_stats,
        infer_metrics_no_bp=infer_metrics_no_bp,
        infer_metrics_bp=infer_metrics_bp,
    )

    append_experiment_results(
        args=args,
        run_config_path=run_config_path,
        pre_infer_stats=pre_infer_stats,
        thresholds_no_bp=thresholds_no_bp,
        thresholds_bp=thresholds_bp,
        infer_metrics_no_bp=infer_metrics_no_bp,
        infer_metrics_bp=infer_metrics_bp,
    )
    print(f"Appended experiment summary to {EXPERIMENT_RESULTS_FILE}")


if __name__ == "__main__":
    main()
