from __future__ import annotations

import numpy as np
import pandas as pd


WORLD_EVENT_COLUMNS = [
    "inflation_pressure",
    "interest_rate_sensitivity",
    "tariff_exposure",
    "commodity_dependence",
    "geopolitical_instability",
    "conflict_intensity",
    "regulatory_uncertainty",
    "macroeconomic_surprise",
    "narrative_momentum",
    "technology_hype_cycle",
    "public_sentiment_shift",
]


def build_world_event_features(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    news: pd.DataFrame,
) -> pd.DataFrame:
    market_dates = pd.DataFrame({"date": sorted(prices["date"].unique())})
    event_daily = events.groupby("date")[WORLD_EVENT_COLUMNS].mean().reset_index()
    news_daily = (
        news.groupby(["entity", "date"])
        .agg(
            news_sentiment=("sentiment", "mean"),
            news_sentiment_variance=("sentiment", "var"),
            article_volume=("article_volume", lambda values: values.sum(min_count=1)),
            geopolitical_intensity=("geopolitical_intensity", "mean"),
        )
        .reset_index()
    )

    stock_rows = prices[["entity", "date", "sector"]].drop_duplicates()
    out = stock_rows.merge(event_daily, on="date", how="left").merge(news_daily, on=["entity", "date"], how="left")
    out["macro_score"] = (
        0.35 * out["inflation_pressure"]
        + 0.25 * out["interest_rate_sensitivity"]
        + 0.2 * out["macroeconomic_surprise"]
        + 0.2 * out["regulatory_uncertainty"]
    )
    out["narrative_novelty"] = out.groupby("entity")["news_sentiment"].diff().abs()
    out["narrative_persistence"] = out.groupby("entity")["news_sentiment"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    out["embedding_centroid_shift"] = np.sqrt(out["narrative_novelty"] ** 2 + out["geopolitical_intensity"] ** 2)
    out["topic_cluster"] = out["sector"].astype("category").cat.codes

    entity_event_rows = []
    for event_entity in events["entity"].unique():
        event_slice = events[events["entity"] == event_entity].copy()
        event_slice["sector"] = "World Event"
        event_slice["news_sentiment"] = np.nan
        event_slice["news_sentiment_variance"] = np.nan
        event_slice["article_volume"] = np.nan
        event_slice["macro_score"] = event_slice[WORLD_EVENT_COLUMNS].mean(axis=1)
        event_slice["narrative_novelty"] = event_slice["narrative_momentum"].diff().abs()
        event_slice["narrative_persistence"] = event_slice["narrative_momentum"].rolling(3, min_periods=1).mean()
        event_slice["embedding_centroid_shift"] = event_slice["narrative_novelty"]
        event_slice["topic_cluster"] = 999
        entity_event_rows.append(event_slice[["entity", "date", "sector", *WORLD_EVENT_COLUMNS, "news_sentiment", "news_sentiment_variance", "article_volume", "macro_score", "narrative_novelty", "narrative_persistence", "embedding_centroid_shift", "topic_cluster"]])

    base_cols = ["entity", "date", "sector", *WORLD_EVENT_COLUMNS, "news_sentiment", "news_sentiment_variance", "article_volume", "macro_score", "narrative_novelty", "narrative_persistence", "embedding_centroid_shift", "topic_cluster"]
    return pd.concat([out[base_cols], *entity_event_rows], ignore_index=True).merge(market_dates, on="date", how="right")
