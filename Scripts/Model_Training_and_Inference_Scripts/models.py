import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from torch_sparse import SparseTensor, matmul
from typing import Optional, Tuple
from torch import Tensor


class CircuitGNN(nn.Module):
    """
    GCN stack with skip via concatenation of intermediate embeddings,
    followed by a linear classifier (matches notebook CircuitGNN).
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 4, drop_out: float = 0.1):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >=1"
        self.drop_out = drop_out
        convs = [GCNConv(in_channels, hidden_channels, cached=False)]
        for _ in range(num_layers - 1):
            convs.append(GCNConv(hidden_channels, hidden_channels, cached=False))
        self.convs = nn.ModuleList(convs)
        self.lin = nn.Linear(hidden_channels * num_layers, out_channels)

    def forward(self, x: torch.Tensor, edge_index):
        node_embs = []
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.drop_out, training=self.training)
            node_embs.append(x)
        final_embed = torch.cat(node_embs, dim=1)
        return self.lin(final_embed)


class GATNet(nn.Module):
    """
    GAT stack with concatenated intermediate embeddings.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 4,
        drop_out: float = 0.1,
        heads: int = 1,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >=1"
        self.drop_out = drop_out
        convs = []
        in_dim = in_channels
        for _ in range(num_layers):
            convs.append(GATConv(in_dim, hidden_channels, heads=heads, concat=False, dropout=drop_out))
            in_dim = hidden_channels
        self.convs = nn.ModuleList(convs)
        self.lin = nn.Linear(hidden_channels * num_layers, out_channels)

    def forward(self, x: torch.Tensor, edge_index):
        node_embs = []
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.drop_out, training=self.training)
            node_embs.append(x)
        final_embed = torch.cat(node_embs, dim=1)
        return self.lin(final_embed)

class GATNetV2(nn.Module):
    """
    GATv2 stack with concatenated intermediate embeddings.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 4,
        drop_out: float = 0.1,
        heads: int = 1,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >=1"
        self.drop_out = drop_out
        convs = []
        in_dim = in_channels
        for _ in range(num_layers):
            convs.append(GATv2Conv(in_dim, hidden_channels, heads=heads, concat=False, dropout=drop_out))
            in_dim = hidden_channels
        self.convs = nn.ModuleList(convs)
        self.lin = nn.Linear(hidden_channels * num_layers, out_channels)

    def forward(self, x: torch.Tensor, edge_index):
        node_embs = []
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.drop_out, training=self.training)
            node_embs.append(x)
        final_embed = torch.cat(node_embs, dim=1)
        return self.lin(final_embed)

def build_model(
    name: str,
    in_channels: int,
    out_channels: int,
    hidden_channels: int = 128,
    num_layers: int = 4,
    drop_out: float = 0.1,
    heads: int = 1,
):
    name = name.lower()
    if name in ["gcn4", "gcn4skip", "circuitgnn", "gcn-4-skip", "gcn"]:
        return CircuitGNN(in_channels, hidden_channels, out_channels, num_layers=num_layers, drop_out=drop_out)
    if name in ["gat", "gat4v1", "gat4-v1", "gat4v1skip", "gat4-v1-skip", "gat-v1"]:
        return GATNet(
            in_channels,
            hidden_channels,
            out_channels,
            num_layers=num_layers,
            drop_out=drop_out,
            heads=heads,
        )
    if name in ["gatv2", "gat-v2", "gat4v2", "gat4-v2", "gat4v2skip", "gat4-v2-skip"]:
        return GATNetV2(
            in_channels,
            hidden_channels,
            out_channels,
            num_layers=num_layers,
            drop_out=drop_out,
            heads=heads,
        )

    raise ValueError(f"Unknown model name: {name}")
