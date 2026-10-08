import argparse
import json
import os
import random
import shlex
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import precision_recall_fscore_support

from models import build_model


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_RESULTS_FILE = PACKAGE_ROOT / "experiment_results.txt"


def ensure_standard_output_roots():
    for dirname in ("work_dirs_xilinx_MB", "Inference_Output_MB"):
        (PACKAGE_ROOT / dirname).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split_data(processed_dir: str):
    pt_files = [f for f in os.listdir(processed_dir) if f.endswith(".pt")]
    data_list = []
    for file in pt_files:
        file_path = os.path.join(processed_dir, file)
        data = torch.load(file_path, weights_only=False)
        data_list.append(data)
    return data_list


def split_train_val_masks(dataset, train_ratio=0.8, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    for data in dataset:
        num_nodes = data.num_nodes
        indices = np.arange(num_nodes)
        np.random.shuffle(indices)
        split = int(train_ratio * num_nodes)
        train_idx = indices[:split]
        val_idx = indices[split:]
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.train_mask[train_idx] = True
        data.val_mask[val_idx] = True


def merge_graphs_and_masks(dataset):
    merged_graph = Batch.from_data_list(dataset)
    merged_graph.train_mask = torch.cat([data.train_mask for data in dataset], dim=0)
    if hasattr(dataset[0], "val_mask"):
        merged_graph.val_mask = torch.cat([data.val_mask for data in dataset], dim=0)
    return merged_graph


@dataclass
class TrainConfig:
    model: str
    hidden_channels: int = 128
    num_layers: int = 4
    drop_out: float = 0.1
    heads: int = 1
    lr: float = 1e-3
    weight_decay: float = 5e-5
    optimizer: str = "adam"
    epochs: int = 300
    train_ratio: float = 0.7
    train_batch_size: Optional[int] = None
    val_batch_size: Optional[int] = None
    test_batch_size: Optional[int] = None
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    patience: int = 1000
    full_batch: bool = False
    full_batch_eval: bool = False
    seed: int = 42


def make_loaders(train_graph, val_graph, test_graph, num_layers, cfg: TrainConfig):
    neighbor_sizes = [-1] * num_layers

    def build_loader(graph, batch_size, shuffle, mask):
        if batch_size is None or batch_size <= 0:
            return None
        input_nodes = mask if mask is not None else None
        return NeighborLoader(
            graph,
            num_neighbors=neighbor_sizes,
            batch_size=batch_size,
            shuffle=shuffle,
            input_nodes=input_nodes,
        )

    train_loader = build_loader(train_graph, cfg.train_batch_size, True, train_graph.train_mask)
    val_mask = val_graph.val_mask if hasattr(val_graph, "val_mask") else None
    val_loader = build_loader(val_graph, cfg.val_batch_size, False, val_mask)
    test_loader = build_loader(test_graph, cfg.test_batch_size, False, test_graph.train_mask)
    return train_loader, val_loader, test_loader


def evaluate(model, loader, device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index)
            seed_attr = "input_id" if hasattr(batch, "input_id") else ("seed" if hasattr(batch, "seed") else None)
            if seed_attr is not None:
                seed_size = getattr(batch, seed_attr).size(0)
                logits = logits[:seed_size]
                labels = batch.y[:seed_size]
            else:
                labels = batch.y
            cls, counts = torch.unique(labels, return_counts=True)
            weights = torch.ones(logits.size(1), device=device)
            if cls.numel() > 0:
                weights[cls] = 1.0 / counts.float()
            weights = weights / weights.sum() * logits.size(1)
            total_loss += F.cross_entropy(logits, labels, weight=weights, reduction="sum").item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    acc = correct / labels.numel()
    f1 = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)[2]
    loss = total_loss / labels.numel()
    return loss, acc, f1


def evaluate_full_batch_graph(model, graph, mask_attr, device, out_channels):
    model.eval()
    with torch.no_grad():
        logits = model(graph.x, graph.edge_index)
        mask = getattr(graph, mask_attr, None)
        if mask is None:
            raise ValueError(f"{mask_attr} not found on graph for full-batch evaluation.")
        labels = graph.y[mask]
        cls, counts = torch.unique(labels, return_counts=True)
        weights = torch.ones(out_channels, device=device)
        if cls.numel() > 0:
            weights[cls] = 1.0 / counts.float()
        weights = weights / weights.sum() * out_channels
        loss = F.cross_entropy(logits[mask], labels, weight=weights).item()
        preds = logits.argmax(dim=1)
        preds_mask = preds[mask]
        acc = (preds_mask == labels).sum().item() / mask.sum().item()
        f1 = precision_recall_fscore_support(labels.cpu(), preds_mask.cpu(), average="macro", zero_division=0)[2]
    return loss, acc, f1


def compute_classification_metrics(labels: torch.Tensor, preds: torch.Tensor):
    labels_cpu = labels.detach().cpu()
    preds_cpu = preds.detach().cpu()
    per_class_prec, per_class_rec, per_class_f1, _ = precision_recall_fscore_support(
        labels_cpu.numpy(), preds_cpu.numpy(), labels=[0, 1], zero_division=0
    )
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        labels_cpu.numpy(), preds_cpu.numpy(), average="macro", zero_division=0
    )
    acc = (preds_cpu == labels_cpu).float().mean().item()
    return {
        "accuracy": float(acc),
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


def evaluate_split_metrics(model, graph, mask_attr: str, device: str):
    model.eval()
    with torch.no_grad():
        logits = model(graph.x.to(device), graph.edge_index.to(device))
    mask = getattr(graph, mask_attr, None)
    if mask is None:
        raise ValueError(f"{mask_attr} not found on graph.")
    mask = mask.bool()
    logits_m = logits[mask]
    labels_m = graph.y.to(device)[mask]
    if labels_m.numel() == 0:
        raise ValueError(f"{mask_attr} has no selected nodes.")
    cls, counts = torch.unique(labels_m, return_counts=True)
    weights = torch.ones(logits_m.size(1), device=device)
    if cls.numel() > 0:
        weights[cls] = 1.0 / counts.float()
    weights = weights / weights.sum() * logits_m.size(1)
    loss = F.cross_entropy(logits_m, labels_m, weight=weights).item()
    preds_m = logits_m.argmax(dim=1)
    metrics = compute_classification_metrics(labels_m, preds_m)
    metrics["loss"] = float(loss)
    return metrics


def format_metrics_line(metrics: dict) -> str:
    return (
        f"class1 P/R/F1={metrics['class1_precision']:.4f}/{metrics['class1_recall']:.4f}/{metrics['class1_f1']:.4f}; "
        f"class0={metrics['class0_precision']:.4f}/{metrics['class0_recall']:.4f}/{metrics['class0_f1']:.4f}; "
        f"macro={metrics['macro_precision']:.4f}/{metrics['macro_recall']:.4f}/{metrics['macro_f1']:.4f}; "
        f"acc={metrics['accuracy']:.4f}; loss={metrics['loss']:.4f}"
    )


def write_run_config_results(
    args,
    run_params: dict,
    best_epoch: int,
    best_val_f1: float,
    train_metrics: dict,
    val_metrics: dict,
    test_metrics: dict,
) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cmd = " ".join(shlex.quote(x) for x in sys.argv)
    run_config_path = Path(args.checkpoint_dir) / "run_config_results.txt"

    legacy_json_path = Path(args.checkpoint_dir) / "run_config.json"

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

    payload = {
        "schema_version": 1,
        "last_updated": timestamp,
        "train_run": {
            "timestamp": timestamp,
            "command": cmd,
            "parameters": {
                **run_params,
                "best_epoch": int(best_epoch),
            },
            "directories": {
                "data_root": args.data_root,
                "checkpoint_dir": args.checkpoint_dir,
            },
            "artifacts": {
                "checkpoint": os.path.join(args.checkpoint_dir, "best_val.pth"),
                "metrics_csv": args.metrics_csv,
                "train_log": os.path.join(args.checkpoint_dir, "train.log"),
                "experiment_results": str(EXPERIMENT_RESULTS_FILE),
            },
            "outcome": {
                "best_epoch": int(best_epoch),
                "best_val_f1": float(best_val_f1),
                "test_loss": float(test_metrics["loss"]),
                "test_acc": float(test_metrics["accuracy"]),
            },
            "metrics": {
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
            },
        },
        "inference_runs": existing.get("inference_runs", []),
    }

    run_config_path.parent.mkdir(parents=True, exist_ok=True)
    run_config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return run_config_path


def append_experiment_results(args, run_config_path: Path, best_epoch: int, best_val_f1: float, test_metrics: dict):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cmd = " ".join(shlex.quote(x) for x in sys.argv)
    lines = [
        f"{timestamp} train_eval run",
        f"- Cmd: {cmd}",
        (
            f"- Outcome: best_epoch={best_epoch}; best_val_f1={best_val_f1:.4f}; "
            f"test_loss={test_metrics['loss']:.4f}; test_acc={test_metrics['accuracy']:.4f}."
        ),
        f"- Run config results: {run_config_path}",
        f"- Artifacts: checkpoint={args.checkpoint_dir}/best_val.pth"
        + (f", metrics_csv={args.metrics_csv}" if args.metrics_csv else "")
        + f", experiment_log={EXPERIMENT_RESULTS_FILE}.",
        "",
    ]
    EXPERIMENT_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENT_RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def train_one_epoch(model, loader, device, optimizer):
    model.train()
    total_loss = 0.0
    total = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index)
        seed_attr = "input_id" if hasattr(batch, "input_id") else ("seed" if hasattr(batch, "seed") else None)
        if seed_attr is not None:
            seed_size = getattr(batch, seed_attr).size(0)
            logits_seed = logits[:seed_size]
            labels_seed = batch.y[:seed_size]
        else:
            logits_seed = logits
            labels_seed = batch.y
        # class-weighted CE per batch (as notebook)
        cls, counts = torch.unique(labels_seed, return_counts=True)
        weights = torch.ones(logits.size(1), device=device)
        weights[cls] = 1.0 / counts.float()
        weights = weights / weights.sum() * logits.size(1)
        loss = F.cross_entropy(logits_seed, labels_seed, weight=weights)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels_seed.numel()
        total += labels_seed.numel()
    return total_loss / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models for IP detection.")
    parser.add_argument("--data-root", required=True, help="Root directory containing train/eval/test/inference/processed folders.")
    parser.add_argument("--model", default="gcn-4-skip", help="Model name: gcn-4-skip, gat, or gatv2.")
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--heads", type=int, default=1, help="GAT attention heads (only used for model=gat).")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--optimizer", choices=["adam", "rmsprop"], default="adam")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="NeighborLoader batch size for training; omit or <=0 to run full-batch training.",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=None,
        help="NeighborLoader batch size for validation; omit or <=0 to evaluate full-batch.",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=None,
        help="NeighborLoader batch size for testing; omit or <=0 to evaluate full-batch.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=1000, help="Early stopping patience based on val accuracy.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device string, e.g., auto, cuda:0, cuda:1, or cpu. auto resolves to cuda:0 if available, else cpu.",
    )
    parser.add_argument("--full-batch", action="store_true", help="Use full-batch training instead of neighbor sampling.")
    parser.add_argument("--full-batch-eval", action="store_true", help="Force full-batch validation/test passes when not training in full-batch mode.")
    parser.add_argument("--metrics-csv", default=None, help="Optional path to save per-epoch metrics CSV.")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory to save checkpoints.")
    args = parser.parse_args()

    ensure_standard_output_roots()

    resolved_device = "cuda:0" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)

    set_seed(args.seed)

    cfg = TrainConfig(
        model=args.model,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        drop_out=args.dropout,
        heads=args.heads,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        epochs=args.epochs,
        train_ratio=args.train_ratio,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size,
        patience=args.patience,
        full_batch=args.full_batch,
        device=resolved_device,
        full_batch_eval=args.full_batch_eval,
        seed=args.seed,
    )

    train_dataset = load_split_data(os.path.join(args.data_root, "train", "processed"))
    # Notebook uses train split for both train/val via masks
    val_dataset = train_dataset
    test_dataset = load_split_data(os.path.join(args.data_root, "test", "processed"))

    split_train_val_masks(train_dataset, train_ratio=cfg.train_ratio, seed=args.seed)

    train_graph = merge_graphs_and_masks(train_dataset)
    val_graph = merge_graphs_and_masks(val_dataset)
    test_graph = merge_graphs_and_masks(test_dataset)

    in_channels = train_graph.num_features
    out_channels = int(train_graph.y.max().item()) + 1

    model = build_model(
        cfg.model,
        in_channels,
        out_channels,
        hidden_channels=cfg.hidden_channels,
        num_layers=cfg.num_layers,
        drop_out=cfg.drop_out,
        heads=cfg.heads,
    ).to(cfg.device)

    opt_name = cfg.optimizer.lower()
    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    best_path = os.path.join(args.checkpoint_dir, "best_val.pth")
    metrics = []

    train_loader: Optional[NeighborLoader] = None
    val_loader: Optional[NeighborLoader] = None
    test_loader: Optional[NeighborLoader] = None
    if not cfg.full_batch:
        train_loader, val_loader, test_loader = make_loaders(train_graph, val_graph, test_graph, cfg.num_layers, cfg)

    use_full_batch_training = cfg.full_batch or train_loader is None

    run_params = {
        "data_root": args.data_root,
        "model_name": cfg.model,
        "hidden_channels": cfg.hidden_channels,
        "num_layers": cfg.num_layers,
        "dropout": cfg.drop_out,
        "heads": cfg.heads,
        "epochs": cfg.epochs,
        "learning_rate": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "optimizer": cfg.optimizer,
        "train_ratio": cfg.train_ratio,
        "mode": "full_batch" if use_full_batch_training else "batched",
        "train_batch_size": cfg.train_batch_size,
        "val_batch_size": cfg.val_batch_size,
        "test_batch_size": cfg.test_batch_size,
        "seed": cfg.seed,
        "patience": cfg.patience,
        "checkpoint_dir": args.checkpoint_dir,
        "device": cfg.device,
        "best_epoch": None,
    }

    if use_full_batch_training:
        train_graph = train_graph.to(cfg.device)
        val_graph = val_graph.to(cfg.device)
        test_graph = test_graph.to(cfg.device)

        train_indices = torch.where(train_graph.train_mask)[0]
        chunk_size = cfg.train_batch_size if cfg.train_batch_size else 0
        chunked_full_batch = 0 < chunk_size < train_indices.numel()

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            cls, counts = torch.unique(train_graph.y[train_graph.train_mask], return_counts=True)
            train_weights = torch.ones(out_channels, device=cfg.device)
            train_weights[cls] = 1.0 / counts.float()
            train_weights = train_weights / train_weights.sum() * out_channels

            if chunked_full_batch:
                perm = torch.randperm(train_indices.numel(), device=cfg.device)
                total_loss = 0.0
                total_nodes = 0
                for start in range(0, train_indices.numel(), chunk_size):
                    batch_idx = train_indices[perm[start : start + chunk_size]]
                    optimizer.zero_grad()
                    logits = model(train_graph.x, train_graph.edge_index)
                    loss = F.cross_entropy(logits[batch_idx], train_graph.y[batch_idx], weight=train_weights)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * batch_idx.numel()
                    total_nodes += batch_idx.numel()
                epoch_train_loss = total_loss / max(total_nodes, 1)
            else:
                optimizer.zero_grad()
                logits = model(train_graph.x, train_graph.edge_index)
                loss = F.cross_entropy(logits[train_graph.train_mask], train_graph.y[train_graph.train_mask], weight=train_weights)
                loss.backward()
                optimizer.step()
                epoch_train_loss = float(loss.item())

            with torch.no_grad():
                model.eval()
                val_logits = model(val_graph.x, val_graph.edge_index)
                val_mask = val_graph.val_mask
                cls_v, counts_v = torch.unique(val_graph.y[val_mask], return_counts=True)
                val_weights = torch.ones(out_channels, device=cfg.device)
                val_weights[cls_v] = 1.0 / counts_v.float()
                val_weights = val_weights / val_weights.sum() * out_channels
                val_loss = F.cross_entropy(val_logits[val_mask], val_graph.y[val_mask], weight=val_weights).item()
                val_preds = val_logits.argmax(dim=1)
                val_preds_mask = val_preds[val_mask]
                val_labels_mask = val_graph.y[val_mask]
                val_correct = (val_preds_mask == val_labels_mask).sum().item()
                val_acc = val_correct / val_mask.sum().item()
                val_f1 = precision_recall_fscore_support(
                    val_labels_mask.cpu(), val_preds_mask.cpu(), average="macro", zero_division=0
                )[2]

            metrics.append(
                {
                    "epoch": epoch,
                    "train_loss": float(epoch_train_loss),
                    "val_loss": float(val_loss),
                    "val_acc": float(val_acc),
                    "val_f1": float(val_f1),
                }
            )
            print(f"Epoch {epoch:03d} | train_loss={epoch_train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                patience_counter = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": asdict(cfg),
                        "in_channels": in_channels,
                        "out_channels": out_channels,
                        "best_epoch": best_epoch,
                        "best_val_f1": best_val_f1,
                    },
                    best_path,
                )
                print(f"Saved new best checkpoint to {best_path}")
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    print(f"Early stopping at epoch {epoch} (best val f1 {best_val_f1:.4f} at epoch {best_epoch})")
                    break

        with torch.no_grad():
            model.load_state_dict(torch.load(best_path, map_location=cfg.device, weights_only=False)["model_state"])
            model.eval()
            test_logits = model(test_graph.x, test_graph.edge_index)
            test_preds = test_logits.argmax(dim=1)
            test_labels = test_graph.y[test_graph.train_mask]
            cls_t, counts_t = torch.unique(test_labels, return_counts=True)
            test_weights = torch.ones(out_channels, device=cfg.device)
            if cls_t.numel() > 0:
                test_weights[cls_t] = 1.0 / counts_t.float()
            test_weights = test_weights / test_weights.sum() * out_channels
            test_loss = F.cross_entropy(test_logits[test_graph.train_mask], test_labels, weight=test_weights).item()
            test_acc = (test_preds[test_graph.train_mask] == test_graph.y[test_graph.train_mask]).sum().item() / test_graph.train_mask.sum().item()
    else:
        val_graph_full = None
        test_graph_full = None
        if cfg.full_batch_eval or val_loader is None:
            val_graph_full = val_graph.clone().to(cfg.device)
        if cfg.full_batch_eval or test_loader is None:
            test_graph_full = test_graph.clone().to(cfg.device)
        for epoch in range(1, cfg.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, cfg.device, optimizer)
            if cfg.full_batch_eval:
                val_loss, val_acc, val_f1 = evaluate_full_batch_graph(
                    model, val_graph_full, "val_mask", cfg.device, out_channels
                )
            elif val_loader is not None:
                val_loss, val_acc, val_f1 = evaluate(model, val_loader, cfg.device)
            else:
                val_loss, val_acc, val_f1 = evaluate_full_batch_graph(
                    model, val_graph_full, "val_mask", cfg.device, out_channels
                )
            metrics.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "val_acc": float(val_acc),
                    "val_f1": float(val_f1),
                }
            )
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                patience_counter = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": asdict(cfg),
                        "in_channels": in_channels,
                        "out_channels": out_channels,
                        "best_epoch": best_epoch,
                        "best_val_f1": best_val_f1,
                    },
                    best_path,
                )
                print(f"Saved new best checkpoint to {best_path}")
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    print(f"Early stopping at epoch {epoch} (best val f1 {best_val_f1:.4f} at epoch {best_epoch})")
                    break
        if cfg.full_batch_eval:
            test_loss, test_acc, _ = evaluate_full_batch_graph(model, test_graph_full, "train_mask", cfg.device, out_channels)
        elif test_loader is not None:
            test_loss, test_acc, _ = evaluate(model, test_loader, cfg.device)
        else:
            test_loss, test_acc, _ = evaluate_full_batch_graph(model, test_graph_full, "train_mask", cfg.device, out_channels)

    best_ckpt = torch.load(best_path, map_location=cfg.device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    best_epoch = int(best_ckpt.get("best_epoch", best_epoch))
    best_val_f1 = float(best_ckpt.get("best_val_f1", best_val_f1))
    train_eval_graph = train_graph if use_full_batch_training else train_graph.to(cfg.device)
    val_eval_graph = val_graph if use_full_batch_training else val_graph.to(cfg.device)
    test_eval_graph = test_graph if use_full_batch_training else test_graph.to(cfg.device)

    train_summary_metrics = evaluate_split_metrics(model, train_eval_graph, "train_mask", cfg.device)
    val_summary_metrics = evaluate_split_metrics(model, val_eval_graph, "val_mask", cfg.device)
    test_summary_metrics = evaluate_split_metrics(model, test_eval_graph, "train_mask", cfg.device)

    print(f"Final test_loss={test_summary_metrics['loss']:.4f} | test_acc={test_summary_metrics['accuracy']:.4f} | best_epoch={best_epoch} | best_val_f1={best_val_f1:.4f}")
    print(f"Train metrics (train_mask): {format_metrics_line(train_summary_metrics)}")
    print(f"Val metrics (val_mask): {format_metrics_line(val_summary_metrics)}")
    print(f"Test metrics (test_mask): {format_metrics_line(test_summary_metrics)}")

    run_config_path = write_run_config_results(
        args=args,
        run_params=run_params,
        best_epoch=best_epoch,
        best_val_f1=best_val_f1,
        train_metrics=train_summary_metrics,
        val_metrics=val_summary_metrics,
        test_metrics=test_summary_metrics,
    )

    append_experiment_results(
        args,
        run_config_path=run_config_path,
        best_epoch=best_epoch,
        best_val_f1=best_val_f1,
        test_metrics=test_summary_metrics,
    )
    print(f"Appended experiment summary to {EXPERIMENT_RESULTS_FILE}")

    if args.metrics_csv:
        import pandas as pd

        pd.DataFrame(metrics).to_csv(args.metrics_csv, index=False)


if __name__ == "__main__":
    main()
