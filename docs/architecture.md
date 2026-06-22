# Architecture

## Pipeline Stages

1. Load multimodal data.
2. Compute engineered financial descriptors.
3. Compute world-event and NLP-style descriptors.
4. Build temporal heterogeneous graph snapshots.
5. Train an unsupervised graph representation model.
6. Extract node embeddings for each `(entity, date)`.
7. Join descriptors and embeddings into a MANTIS-ready CSV.

## Graph Model

The graph supports heterogeneous node categories:

- stocks
- sectors
- ETFs
- countries
- commodities
- currencies
- central banks
- macroeconomic reports
- geopolitical events
- regulatory events
- technology narratives

Supported edge classes include:

- `stock_to_stock_rolling_correlation`
- `stock_to_sector_membership`
- `etf_to_stock_ownership`
- `technology_narrative_exposure`
- `geopolitical_supply_chain_exposure`
- `tariff_exposure`
- `commodity_dependence`
- `geopolitical_influence`
- `macroeconomic_pressure`
- `interest_rate_sensitivity`
- `regulatory_impact`

Edges carry weights and timestamps. The current implementation stores snapshots as NetworkX graphs and converts them to PyTorch Geometric tensors for training.

## Representation Learning

The model path supports:

- GraphSAGE
- GATv2

Training combines:

- link reconstruction
- negative edge sampling
- temporal consistency loss

This keeps the objective focused on ecosystem structure rather than price prediction.

## Extension Points

Add new node or edge types in `src/mantis_financial_graph/graph/builder.py`.

Add new descriptors in:

- `src/mantis_financial_graph/features/financial.py`
- `src/mantis_financial_graph/features/world_events.py`
- `src/mantis_financial_graph/features/graph_stats.py`

Add real ingestion sources in `src/mantis_financial_graph/data/` and keep the normalized table schemas documented in `README.md`.
