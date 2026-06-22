from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class GraphSnapshot:
    date: pd.Timestamp
    graph: nx.Graph
    node_features: pd.DataFrame
    edge_table: pd.DataFrame


def _add_edge(graph: nx.Graph, source: str, target: str, edge_type: str, weight: float, date: pd.Timestamp) -> None:
    if source == target:
        return
    graph.add_edge(source, target, edge_type=edge_type, weight=float(weight), timestamp=date)


def build_temporal_snapshots(
    financial_features: pd.DataFrame,
    world_features: pd.DataFrame,
    metadata: pd.DataFrame,
    etf_holdings: pd.DataFrame,
    raw_config: dict[str, Any],
) -> list[GraphSnapshot]:
    feature_cfg = raw_config["features"]
    data_cfg = raw_config["data"]
    threshold = float(feature_cfg["correlation_threshold"])
    window_days = int(feature_cfg["graph_window_days"])
    max_corr_edges_per_node = int(feature_cfg.get("max_corr_edges_per_node", 25))
    dates = sorted(financial_features["date"].unique())
    snapshots: list[GraphSnapshot] = []
    sectors = data_cfg["sectors"]

    for date in dates:
        graph = nx.Graph()
        for row in metadata.to_dict("records"):
            graph.add_node(row["entity"], node_type=row["node_type"], sector=row["sector"], country=row["country"])

        history_start = pd.Timestamp(date) - pd.Timedelta(days=window_days)
        hist = financial_features[(financial_features["date"] <= date) & (financial_features["date"] >= history_start)]
        pivot = hist.pivot_table(index="date", columns="entity", values="return")
        corr = pivot.corr().fillna(0.0)
        added_pairs: set[tuple[str, str]] = set()
        for source in corr.columns:
            candidates = corr[source].drop(labels=[source], errors="ignore")
            candidates = candidates[candidates.abs() >= threshold].reindex(candidates.abs().sort_values(ascending=False).index)
            for target, weight in candidates.head(max_corr_edges_per_node).items():
                pair = tuple(sorted((str(source), str(target))))
                if pair in added_pairs:
                    continue
                added_pairs.add(pair)
                _add_edge(graph, source, target, "stock_to_stock_rolling_correlation", float(weight), pd.Timestamp(date))

        for stock, sector in sectors.items():
            _add_edge(graph, stock, sector, "stock_to_sector_membership", 1.0, pd.Timestamp(date))
        for row in etf_holdings.to_dict("records"):
            _add_edge(graph, row["etf"], row["stock"], "etf_to_stock_ownership", row["weight"], pd.Timestamp(date))

        for stock in data_cfg["tickers"]:
            sector = sectors.get(stock, "Equity Universe")
            sector_lower = sector.lower()
            if any(token in sector_lower for token in ["semiconductor", "information technology", "technology"]):
                _add_edge(graph, stock, "AI Boom", "technology_narrative_exposure", 0.8, pd.Timestamp(date))
                _add_edge(graph, stock, "Taiwan", "geopolitical_supply_chain_exposure", 0.7, pd.Timestamp(date))
                _add_edge(graph, stock, "China Tariffs", "tariff_exposure", 0.55, pd.Timestamp(date))
            if any(token in sector_lower for token in ["energy", "oil", "gas"]):
                _add_edge(graph, stock, "Oil Market", "commodity_dependence", 0.85, pd.Timestamp(date))
                _add_edge(graph, stock, "Middle East Conflict", "geopolitical_influence", 0.7, pd.Timestamp(date))
            if any(token in sector_lower for token in ["bank", "financial"]):
                _add_edge(graph, stock, "Federal Reserve", "macroeconomic_pressure", 0.9, pd.Timestamp(date))
                _add_edge(graph, stock, "US CPI Release", "interest_rate_sensitivity", 0.7, pd.Timestamp(date))
            if any(token in sector_lower for token in ["ev", "automobile", "consumer discretionary"]):
                _add_edge(graph, stock, "Copper", "commodity_dependence", 0.5, pd.Timestamp(date))
                _add_edge(graph, stock, "China Tariffs", "regulatory_impact", 0.65, pd.Timestamp(date))

        base_nodes = metadata[["entity", "node_type", "sector", "country"]].copy()
        base_nodes["date"] = pd.Timestamp(date)
        day_fin = financial_features[financial_features["date"] == date].drop(columns=["sector"], errors="ignore")
        day_world = world_features[world_features["date"] == date]
        node_features = base_nodes.merge(day_fin, on=["entity", "date"], how="left").merge(day_world, on=["entity", "date"], how="left", suffixes=("", "_world"))
        numeric_cols = node_features.select_dtypes(include=[np.number]).columns
        node_features[numeric_cols] = node_features[numeric_cols].fillna(0.0)
        node_features = node_features.fillna("")

        edge_table = pd.DataFrame(
            [
                {"source": s, "target": t, **attrs}
                for s, t, attrs in graph.edges(data=True)
            ]
        )
        snapshots.append(GraphSnapshot(date=pd.Timestamp(date), graph=graph, node_features=node_features, edge_table=edge_table))
    return snapshots
