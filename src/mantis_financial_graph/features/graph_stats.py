from __future__ import annotations

import networkx as nx
import pandas as pd


def graph_statistics(graph: nx.Graph, date: pd.Timestamp) -> pd.DataFrame:
    if graph.number_of_nodes() == 0:
        return pd.DataFrame()
    influence_graph = graph.copy()
    for source, target, attrs in influence_graph.edges(data=True):
        magnitude = abs(float(attrs.get("weight", 0.0)))
        attrs["influence_weight"] = magnitude
        attrs["distance"] = 1.0 / max(magnitude, 1e-6)
    degree = nx.degree_centrality(influence_graph)
    if influence_graph.number_of_nodes() > 300:
        betweenness = nx.betweenness_centrality(
            influence_graph,
            k=min(120, influence_graph.number_of_nodes()),
            weight="distance",
            normalized=True,
            seed=42,
        )
    else:
        betweenness = nx.betweenness_centrality(influence_graph, weight="distance", normalized=True)
    clustering = nx.clustering(influence_graph.to_undirected(), weight="influence_weight")
    if influence_graph.number_of_edges():
        try:
            pagerank = nx.pagerank(influence_graph, weight="influence_weight", max_iter=500, tol=1e-8)
        except nx.PowerIterationFailedConvergence:
            weighted_degree = dict(influence_graph.degree(weight="influence_weight"))
            total_weighted_degree = sum(weighted_degree.values()) or 1.0
            pagerank = {node: value / total_weighted_degree for node, value in weighted_degree.items()}
    else:
        pagerank = {n: 0.0 for n in influence_graph.nodes}
    density = nx.density(influence_graph)
    rows = []
    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        edge_weights = [abs(graph[node][nbr].get("weight", 0.0)) for nbr in neighbors]
        rows.append(
            {
                "entity": node,
                "date": date,
                "degree_centrality": degree.get(node, 0.0),
                "betweenness_centrality": betweenness.get(node, 0.0),
                "clustering_coefficient": clustering.get(node, 0.0),
                "pagerank": pagerank.get(node, 0.0),
                "graph_density": density,
                "neighborhood_similarity": sum(edge_weights) / max(len(edge_weights), 1),
                "local_market_influence": sum(edge_weights),
            }
        )
    return pd.DataFrame(rows)
