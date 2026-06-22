from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GNNBatch:
    x: torch.Tensor
    edge_index: torch.Tensor
    node_names: list[str]
    date: object


class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        from torch_geometric.nn import SAGEConv

        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.conv1(x, edge_index))
        x = self.dropout(x)
        return self.conv2(x, edge_index)


class GATEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        from torch_geometric.nn import GATv2Conv

        heads = 4
        self.conv1 = GATv2Conv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=dropout)
        self.activation = nn.ELU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


def make_encoder(model_name: str, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Module:
    if model_name.lower() == "gat":
        return GATEncoder(in_dim, hidden_dim, out_dim, dropout)
    if model_name.lower() in {"graphsage", "sage"}:
        return GraphSAGEEncoder(in_dim, hidden_dim, out_dim, dropout)
    raise ValueError(f"Unsupported model '{model_name}'. Use 'graphsage' or 'gat'.")


def link_reconstruction_loss(z: torch.Tensor, edge_index: torch.Tensor, negative_samples: int = 1) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=z.device, requires_grad=True)
    src, dst = edge_index
    pos_score = (z[src] * z[dst]).sum(dim=1)
    pos_loss = nn.functional.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
    num_nodes = z.shape[0]
    neg_src = torch.randint(0, num_nodes, (src.numel() * negative_samples,), device=z.device)
    neg_dst = torch.randint(0, num_nodes, (dst.numel() * negative_samples,), device=z.device)
    neg_score = (z[neg_src] * z[neg_dst]).sum(dim=1)
    neg_loss = nn.functional.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
    return pos_loss + neg_loss


def temporal_consistency_loss(current: torch.Tensor, previous: torch.Tensor | None) -> torch.Tensor:
    if previous is None or previous.shape != current.shape:
        return torch.tensor(0.0, device=current.device)
    return nn.functional.mse_loss(current, previous.detach())
