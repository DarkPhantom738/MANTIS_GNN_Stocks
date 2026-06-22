from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from mantis_financial_graph.config import load_config
from mantis_financial_graph.data.synthetic import make_synthetic_ecosystem
from mantis_financial_graph.data.real_feeds import make_real_feed_ecosystem
from mantis_financial_graph.data.yfinance_loader import make_yfinance_ecosystem
from mantis_financial_graph.export.csv_exporter import export_mantis_csv
from mantis_financial_graph.features.financial import compute_financial_features
from mantis_financial_graph.features.graph_stats import graph_statistics
from mantis_financial_graph.features.world_events import build_world_event_features
from mantis_financial_graph.graph.builder import build_temporal_snapshots
from mantis_financial_graph.training.train import train_embeddings
from mantis_financial_graph.utils import configure_logging, ensure_dir, set_seed

LOGGER = logging.getLogger(__name__)


def run(config_path: str | Path) -> pd.DataFrame:
    config = load_config(config_path)
    configure_logging()
    set_seed(config.seed)
    ensure_dir(config.output_dir)

    mode = config.raw["data"].get("mode", "synthetic")
    LOGGER.info("Loading multimodal ecosystem data with mode=%s", mode)
    if mode == "synthetic":
        data = make_synthetic_ecosystem(config.raw, config.seed)
    elif mode == "yfinance":
        data = make_yfinance_ecosystem(config.raw, config.seed)
    elif mode == "real_feeds":
        data = make_real_feed_ecosystem(config.raw, config.seed)
    else:
        raise ValueError(f"Unsupported data.mode '{mode}'. Use 'synthetic', 'yfinance', or 'real_feeds'.")
    LOGGER.info("Computing engineered financial descriptors")
    financial = compute_financial_features(data.prices, config.raw["features"]["rolling_windows"])
    LOGGER.info("Computing world-event and NLP-style descriptors")
    world = build_world_event_features(data.prices, data.events, data.news)
    LOGGER.info("Building temporal heterogeneous graph snapshots")
    snapshots = build_temporal_snapshots(financial, world, data.entity_metadata, data.etf_holdings, config.raw)

    graph_stats_frames = [graph_statistics(snapshot.graph, snapshot.date) for snapshot in snapshots]
    graph_features = pd.concat(graph_stats_frames, ignore_index=True)
    graph_features["sector_correlation_average"] = graph_features["neighborhood_similarity"]
    graph_features["etf_exposure_score"] = graph_features["entity"].map(
        data.etf_holdings.groupby("stock")["weight"].sum()
    )
    graph_features["contagion_metric"] = graph_features["degree_centrality"] * graph_features["local_market_influence"]

    LOGGER.info("Training graph representation model and extracting node embeddings")
    embeddings = train_embeddings(snapshots, config.raw)
    output_path = config.raw["export"]["final_csv"]
    LOGGER.info("Exporting MANTIS-ready CSV to %s", output_path)
    dataset = export_mantis_csv(financial, world, graph_features, embeddings, output_path)

    sample_path = Path(config.raw["export"]["sample_csv"])
    ensure_dir(sample_path.parent)
    dataset.head(25).to_csv(sample_path, index=False)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MANTIS financial ecosystem graph-learning CSV.")
    parser.add_argument("--config", default="configs/sample_config.yaml", help="Path to YAML pipeline config.")
    args = parser.parse_args()
    dataset = run(args.config)
    print(f"Exported {len(dataset):,} rows and {len(dataset.columns):,} columns.")


if __name__ == "__main__":
    main()
