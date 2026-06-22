# MANTIS GNN Stocks

MANTIS GNN Stocks builds a graph-based financial ecosystem dataset. It takes market prices, public web data, macroeconomic series, and news/event signals, turns them into a temporal heterogeneous graph, and exports one CSV where each row represents an `entity` on a `date`.

The project is not meant to be a trading bot. It is a representation-learning pipeline for studying how stocks, sectors, ETFs, macro conditions, geopolitical events, commodities, and narratives move together inside one connected system.

## Core Idea

Most stock datasets describe each ticker as a separate time series: price, volume, indicators, maybe sentiment. This project adds the missing relational layer.

A stock is represented not only by its own price behavior, but also by:

- which sector it belongs to
- which ETFs hold it
- which other stocks it correlates with
- which commodities or currencies it is exposed to
- which macro regimes affect it
- which geopolitical or regulatory narratives touch it
- which news topics are active around its sector
- where it sits in the graph of the broader market ecosystem

The final output is a table that combines traditional financial features, event/news features, graph statistics, and learned GNN embeddings.

## What The Dataset Contains

The exported CSV has this shape:

```text
entity,date,<interpretable features>,<graph features>,gnn_1,...,gnn_N
```

An `entity` can be a stock, sector, ETF, country, commodity, currency, central bank topic, macro report, geopolitical event, regulatory event, or technology narrative.

The interpretable columns include:

- OHLCV-derived market values: open, high, low, close, volume
- return features: simple return, log return, momentum, drawdown
- technical descriptors: rolling volatility, RSI, MACD, Bollinger z-score, ATR
- risk/statistical descriptors: beta, Sharpe-style ratios, autocorrelation, rolling entropy, volatility clustering
- macro/event descriptors: inflation pressure, interest-rate sensitivity, tariff exposure, commodity dependence, macro surprise
- geopolitical descriptors: geopolitical instability, conflict intensity, regulatory uncertainty
- news/narrative descriptors: sentiment, article volume, sentiment variance, narrative momentum, narrative novelty, narrative persistence, topic cluster
- graph descriptors: degree centrality, betweenness centrality, PageRank, clustering coefficient, graph density, neighborhood similarity, ETF exposure, contagion, local market influence

The learned columns are `gnn_1` through `gnn_N`. In the provided configs, `N` is typically `32`.

## How The Web/Data Collection Works

The real-feed pipeline uses public data sources and normalizes them into one internal ecosystem format.

Market prices come from Yahoo Finance through `yfinance`. For every ticker, the loader collects historical OHLCV rows and standardizes them into:

```text
entity,date,open,high,low,close,volume,sector
```

The stock universe can be built from scraped public listings:

- Wikipedia S&P 500 constituents are read from the public S&P 500 table. The project extracts ticker symbols and GICS sectors.
- Nasdaq Trader listed symbols are read from the public `nasdaqlisted.txt` file. Test issues, ETFs, malformed symbols, and unsupported symbols are filtered out.

Macro data comes from FRED CSV endpoints. The pipeline currently pulls series such as:

- CPIAUCSL: inflation
- FEDFUNDS: federal funds rate
- DGS10: 10-year Treasury yield
- UNRATE: unemployment
- DCOILWTICO: WTI oil price

Those series are aligned to the project date range, forward/back filled where necessary, and converted into change-based macro pressure features.

News and topic signals come from the GDELT 2.1 Doc API. The project queries topic baskets such as:

- Federal Reserve interest rates inflation
- CPI inflation price index
- Middle East conflict oil shipping
- China tariffs trade restrictions
- artificial intelligence semiconductors chips

For each returned article, the loader keeps the seen date, title, source domain, tone-derived sentiment, article volume, and a rough geopolitical intensity score. These article-level rows are aggregated into daily topic signals and then mapped back onto stocks through sector/topic relationships.

The yfinance-only mode still uses real market prices, but its news and event fields are proxy features derived from realized market stress, returns, and volume. The broader real-feed mode is the one that uses Yahoo Finance, Wikipedia, Nasdaq Trader, FRED, and GDELT together.

## How Raw Data Becomes A Graph

For each date, the project builds a graph snapshot. The graph is heterogeneous because it contains many kinds of nodes, not just stocks.

Node types include:

- stocks
- sectors
- ETFs
- countries
- commodities
- currencies
- central-bank topics
- macroeconomic reports
- geopolitical events
- regulatory events
- technology narratives

Edges describe relationships between those nodes. Examples include:

- `stock_to_stock_rolling_correlation`: connects stocks whose recent returns are strongly correlated
- `stock_to_sector_membership`: connects a stock to its sector
- `etf_to_stock_ownership`: connects ETFs to their holdings
- `technology_narrative_exposure`: connects technology/semiconductor names to the AI narrative
- `geopolitical_supply_chain_exposure`: connects exposed companies to countries or geopolitical nodes
- `tariff_exposure`: connects affected companies to tariff/regulatory narratives
- `commodity_dependence`: connects energy or manufacturing-sensitive names to commodities
- `macroeconomic_pressure`: connects rate-sensitive companies to Federal Reserve or macro nodes
- `interest_rate_sensitivity`: connects financial names to inflation/rate-report nodes
- `regulatory_impact`: connects exposed sectors to regulatory events

Every graph snapshot has node features and weighted edges. A stock like `NVDA`, for example, can simultaneously be connected to semiconductors, AI, Taiwan supply-chain risk, tariff exposure, ETFs, and other stocks with similar rolling return behavior.

## What The GNN Adds

The hand-engineered columns are useful, but they are mostly local: they describe one entity's direct values on one date. The GNN adds a learned relational representation.

In the full training path, the project converts each NetworkX graph snapshot into PyTorch Geometric tensors and trains either a GraphSAGE or GAT encoder. The model receives:

- node feature matrix `x`: numeric descriptors for every node
- edge index: graph connections for that date
- weighted ecosystem structure through the graph construction step

The training objective is unsupervised. It does not train on future returns or price labels. Instead, it learns embeddings that help reconstruct the graph itself:

- observed edges are treated as positive relationships
- randomly sampled non-edges are treated as negative relationships
- a temporal consistency loss discourages embeddings from changing too violently between adjacent graph snapshots

This means the GNN is learning a compact coordinate system for market structure. Nodes that play similar roles in the ecosystem, share exposures, or sit in similar graph neighborhoods should end up closer together in embedding space.

## What `gnn_1...gnn_N` Mean

The `gnn_*` columns are the learned embedding vector for an entity on a specific date.

For example:

```text
entity,date,gnn_1,gnn_2,...,gnn_32
NVDA,2026-01-05,0.184,-0.077,...,0.412
```

These values should be interpreted as latent coordinates, not individually named financial indicators. `gnn_7` does not directly mean "inflation risk" or "AI exposure" by itself. The useful information is in the vector as a whole.

The embeddings can be used for:

- similarity search: find entities with similar graph roles
- clustering: discover market regimes or groups that are not obvious from sector labels
- manifold visualization: project graph embeddings into 2D/3D for MANTIS-style cartographic analysis
- drift analysis: measure how an entity's ecosystem position changes over time
- contagion analysis: identify entities that move closer together during stress
- feature augmentation: combine `gnn_*` with interpretable columns for downstream models

The project also exports graph-statistic columns such as PageRank, centrality, clustering, and local influence. Those are interpretable graph metrics. The `gnn_*` values are learned relational features that capture higher-dimensional structure beyond those single metrics.


## Output Files

The repository includes generated examples in `outputs/` and `data/sample/`.

The very large raw real-feed CSV is intentionally ignored because it exceeds GitHub's normal file-size limit. Compact, compressed, and under-50MB versions are included instead.

For detailed column definitions, see [docs/final_csv_schema.md](docs/final_csv_schema.md). For the lower-level architecture, see [docs/architecture.md](docs/architecture.md).
