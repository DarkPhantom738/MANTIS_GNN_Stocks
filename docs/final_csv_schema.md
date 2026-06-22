# Final CSV Schema

The final CSV is an entity-time table:

```text
entity,date,<interpretable descriptors>,<graph descriptors>,gnn_1,...,gnn_N
```

## Key Columns

- `entity`: stock, sector, ETF, commodity, country, currency, event, or narrative node
- `date`: graph snapshot date
- `open`, `high`, `low`, `close`, `volume`: OHLCV values where available
- `return`, `log_return`: simple and log returns
- `rolling_volatility_*`: rolling realized volatility
- `rsi`: relative strength index
- `macd`: moving average convergence divergence
- `bollinger_z_*`: Bollinger band z-score
- `atr`: average true range
- `beta`: market beta versus the synthetic market factor or configured universe
- `sharpe_*`: rolling Sharpe-style return over volatility
- `momentum`: rolling price momentum
- `drawdown`: current drawdown from running maximum
- `autocorrelation`: rolling return autocorrelation
- `rolling_entropy`: entropy of recent returns
- `volatility_clustering`: autocorrelation of absolute returns
- `degree_centrality`: normalized graph degree
- `betweenness_centrality`: shortest-path bridge importance
- `clustering_coefficient`: local graph clustering
- `pagerank`: recursive influence measure
- `graph_density`: snapshot-level edge density
- `neighborhood_similarity`: average absolute edge weight around the node
- `sector_correlation_average`: interpretable proxy for local sector co-movement
- `etf_exposure_score`: aggregate ETF holding exposure
- `contagion_metric`: degree multiplied by local edge influence
- `local_market_influence`: sum of local edge weights
- `inflation_pressure`
- `interest_rate_sensitivity`
- `tariff_exposure`
- `commodity_dependence`
- `geopolitical_instability`
- `conflict_intensity`
- `regulatory_uncertainty`
- `macroeconomic_surprise`
- `narrative_momentum`
- `technology_hype_cycle`
- `public_sentiment_shift`
- `news_sentiment`
- `news_sentiment_variance`
- `article_volume`
- `geopolitical_intensity`
- `macro_score`
- `narrative_novelty`
- `narrative_persistence`
- `embedding_centroid_shift`
- `topic_cluster`
- `gnn_1 ... gnn_N`: learned graph embedding dimensions

## MANTIS Usage

Use the descriptor columns for explainability overlays and the `gnn_*` columns as latent graph-representation axes. MANTIS can ingest both sets together to discover ecosystem similarity, latent regimes, and cross-domain relationships.
