from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from mantis_financial_graph.graph.builder import GraphSnapshot
from mantis_financial_graph.models.gnn import GNNBatch, link_reconstruction_loss, make_encoder, temporal_consistency_loss
from mantis_financial_graph.utils import ensure_dir

LOGGER = logging.getLogger(__name__)


def _numeric_feature_matrix(snapshot: GraphSnapshot) -> tuple[np.ndarray, list[str]]:
    node_features = snapshot.node_features.set_index("entity")
    numeric = node_features.select_dtypes(include=[np.number])
    if numeric.empty:
        numeric = pd.DataFrame(index=node_features.index, data={"bias": 1.0})
    return numeric.to_numpy(dtype=np.float32), list(numeric.columns)


def _snapshot_to_batch(snapshot: GraphSnapshot, feature_columns: list[str] | None = None) -> tuple[GNNBatch, list[str]]:
    import torch

    nodes = list(snapshot.graph.nodes)
    node_features = snapshot.node_features.set_index("entity")
    numeric = node_features.select_dtypes(include=[np.number])
    if feature_columns is None:
        feature_columns = list(numeric.columns)
    numeric = numeric.reindex(nodes).reindex(columns=feature_columns, fill_value=0.0).fillna(0.0)
    x = torch.tensor(numeric.to_numpy(dtype=np.float32))
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    edges = []
    for source, target in snapshot.graph.edges:
        edges.append((node_to_idx[source], node_to_idx[target]))
        edges.append((node_to_idx[target], node_to_idx[source]))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    return GNNBatch(x=x, edge_index=edge_index, node_names=nodes, date=snapshot.date), feature_columns


def train_embeddings(snapshots: list[GraphSnapshot], raw_config: dict[str, Any]) -> pd.DataFrame:
    try:
        import torch
        import torch_geometric  # noqa: F401
    except ImportError:
        LOGGER.warning("PyTorch Geometric is unavailable; using deterministic SVD graph-feature fallback embeddings.")
        return fallback_embeddings(snapshots, raw_config)

    train_cfg = raw_config["training"]
    feat_cfg = raw_config["features"]
    embedding_dim = int(feat_cfg["embedding_dim"])
    hidden_dim = int(train_cfg["hidden_dim"])
    dropout = float(train_cfg["dropout"])
    lr = float(train_cfg["learning_rate"])
    weight_decay = float(train_cfg["weight_decay"])
    epochs = int(train_cfg["epochs"])
    temporal_weight = float(train_cfg["temporal_consistency_weight"])
    negative_samples = int(train_cfg["negative_samples"])

    first_batch, feature_columns = _snapshot_to_batch(snapshots[0])
    scaler = StandardScaler().fit(first_batch.x.numpy())
    batches: list[GNNBatch] = []
    for snapshot in snapshots:
        batch, _ = _snapshot_to_batch(snapshot, feature_columns)
        scaled_x = scaler.transform(np.nan_to_num(batch.x.numpy(), nan=0.0, posinf=0.0, neginf=0.0))
        batch.x = torch.tensor(np.nan_to_num(scaled_x, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32)
        batches.append(batch)

    model = make_encoder(train_cfg["model"], first_batch.x.shape[1], hidden_dim, embedding_dim, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        previous_z = None
        for batch in batches:
            optimizer.zero_grad()
            z = model(batch.x, batch.edge_index)
            loss = link_reconstruction_loss(z, batch.edge_index, negative_samples)
            loss = loss + temporal_weight * temporal_consistency_loss(z, previous_z)
            loss.backward()
            optimizer.step()
            previous_z = z.detach()
            total_loss += float(loss.item())
        if epoch == 0 or (epoch + 1) % 10 == 0:
            LOGGER.info("epoch=%s loss=%.4f", epoch + 1, total_loss / max(len(batches), 1))

    checkpoint = Path(train_cfg["checkpoint_path"])
    ensure_dir(checkpoint.parent)
    torch.save({"model_state_dict": model.state_dict(), "feature_columns": feature_columns, "config": raw_config}, checkpoint)

    model.eval()
    rows = []
    with torch.no_grad():
        for batch in batches:
            z = model(batch.x, batch.edge_index).cpu().numpy()
            for idx, entity in enumerate(batch.node_names):
                row = {"entity": entity, "date": pd.Timestamp(batch.date)}
                row.update({f"gnn_{i + 1}": float(z[idx, i]) for i in range(z.shape[1])})
                rows.append(row)
    return pd.DataFrame(rows)


def fallback_embeddings(snapshots: list[GraphSnapshot], raw_config: dict[str, Any]) -> pd.DataFrame:
    embedding_dim = int(raw_config["features"]["embedding_dim"])
    rows = []
    matrices = []
    keys = []
    feature_cols = None
    for snapshot in snapshots:
        graph_nodes = list(snapshot.graph.nodes)
        features = snapshot.node_features.set_index("entity").select_dtypes(include=[np.number])
        if feature_cols is None:
            feature_cols = list(features.columns)
        features = features.reindex(graph_nodes).reindex(columns=feature_cols, fill_value=0.0).fillna(0.0)
        adjacency = nx.to_numpy_array(snapshot.graph, nodelist=graph_nodes, weight="weight")
        feature_matrix = np.nan_to_num(features.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        adjacency = np.nan_to_num(adjacency, nan=0.0, posinf=0.0, neginf=0.0)
        matrix = np.concatenate([feature_matrix, adjacency], axis=1)
        matrices.append(matrix)
        keys.extend((entity, snapshot.date) for entity in graph_nodes)
    combined = np.vstack(matrices)
    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
    scaled = StandardScaler().fit_transform(combined)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    n_components = min(embedding_dim, max(1, min(scaled.shape) - 1))
    z = TruncatedSVD(n_components=n_components, random_state=int(raw_config["project"]["seed"])).fit_transform(scaled)
    if n_components < embedding_dim:
        z = np.pad(z, ((0, 0), (0, embedding_dim - n_components)))
    for idx, (entity, date) in enumerate(keys):
        row = {"entity": entity, "date": pd.Timestamp(date)}
        row.update({f"gnn_{i + 1}": float(z[idx, i]) for i in range(embedding_dim)})
        rows.append(row)
    return pd.DataFrame(rows)
