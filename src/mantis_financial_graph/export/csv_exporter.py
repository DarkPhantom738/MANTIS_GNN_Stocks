from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from mantis_financial_graph.utils import ensure_dir


def export_mantis_csv(
    financial_features: pd.DataFrame,
    world_features: pd.DataFrame,
    graph_features: pd.DataFrame,
    embeddings: pd.DataFrame,
    output_path: str | Path,
) -> pd.DataFrame:
    fin = financial_features.drop(columns=["sector"], errors="ignore")
    dataset = (
        fin.merge(world_features, on=["entity", "date"], how="outer", suffixes=("", "_world"))
        .merge(graph_features, on=["entity", "date"], how="left")
        .merge(embeddings, on=["entity", "date"], how="left")
    )
    dataset = dataset.sort_values(["date", "entity"]).reset_index(drop=True)
    numeric_cols = dataset.select_dtypes(include="number").columns
    dataset[numeric_cols] = dataset[numeric_cols].replace([np.inf, -np.inf], np.nan)
    dataset = dataset.fillna("")
    path = Path(output_path)
    ensure_dir(path.parent)
    dataset.to_csv(path, index=False)
    return dataset
