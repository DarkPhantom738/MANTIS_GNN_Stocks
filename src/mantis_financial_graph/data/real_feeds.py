from __future__ import annotations

import json
import logging
import ssl
import time
import urllib.parse
import urllib.request
from typing import Any

import numpy as np
import pandas as pd

from mantis_financial_graph.data.synthetic import EcosystemData
from mantis_financial_graph.data.universe import resolve_universe
from mantis_financial_graph.data.yfinance_loader import load_yfinance_prices

LOGGER = logging.getLogger(__name__)

FRED_SERIES = {
    "CPIAUCSL": "inflation",
    "FEDFUNDS": "fed_funds",
    "DGS10": "ten_year_yield",
    "UNRATE": "unemployment",
    "DCOILWTICO": "oil_price",
}

DEFAULT_TOPICS = {
    "Federal Reserve": "Federal Reserve interest rates inflation",
    "US CPI Release": "CPI inflation price index",
    "Middle East Conflict": "Middle East conflict oil shipping",
    "China Tariffs": "China tariffs trade restrictions",
    "AI Boom": "artificial intelligence semiconductors chips",
}


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": "mantis-financial-graph-research/0.1"})


def _read_csv_url(url: str) -> pd.DataFrame:
    request = _request(url)
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return pd.read_csv(response)
    except (ssl.SSLError, urllib.error.URLError):
        with urllib.request.urlopen(request, timeout=45, context=ssl._create_unverified_context()) as response:
            return pd.read_csv(response)


def load_fred_macro(start: str, end: str) -> pd.DataFrame:
    frames = []
    for series_id, name in FRED_SERIES.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        try:
            frame = _read_csv_url(url)
        except Exception as exc:
            LOGGER.warning("FRED load failed for %s: %s", series_id, exc)
            continue
        frame.columns = ["date", name]
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame[name] = pd.to_numeric(frame[name].replace(".", np.nan), errors="coerce")
        frames.append(frame)

    dates = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    macro = dates
    for frame in frames:
        macro = macro.merge(frame, on="date", how="left")
    macro = macro.sort_values("date").ffill().bfill()
    for col in macro.columns:
        if col != "date":
            macro[f"{col}_change"] = macro[col].pct_change().replace([np.inf, -np.inf], np.nan)
    return macro


def _gdelt_articles(query: str, start: str, end: str, max_records: int) -> list[dict[str, Any]]:
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": str(max_records),
        "sort": "HybridRel",
        "startdatetime": pd.Timestamp(start).strftime("%Y%m%d%H%M%S"),
        "enddatetime": pd.Timestamp(end).strftime("%Y%m%d%H%M%S"),
    }
    url = "https://api.gdeltproject.org/api/v2/doc/doc?" + urllib.parse.urlencode(params)
    request = _request(url)
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except (ssl.SSLError, urllib.error.URLError):
        with urllib.request.urlopen(request, timeout=45, context=ssl._create_unverified_context()) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    return payload.get("articles", [])


def load_gdelt_topic_news(start: str, end: str, topic_queries: dict[str, str] | None = None, max_records: int = 250) -> pd.DataFrame:
    topic_queries = topic_queries or DEFAULT_TOPICS
    rows = []
    for topic, query in topic_queries.items():
        try:
            articles = _gdelt_articles(query, start, end, max_records)
        except Exception as exc:
            LOGGER.warning("GDELT load failed for %s: %s", topic, exc)
            articles = []
        for article in articles:
            seen = pd.to_datetime(article.get("seendate"), errors="coerce")
            if pd.isna(seen):
                continue
            if seen.tzinfo is not None:
                seen = seen.tz_convert(None)
            tone = float(article.get("tone", 0.0) or 0.0)
            rows.append(
                {
                    "topic": topic,
                    "date": seen.normalize(),
                    "text": article.get("title", ""),
                    "source": article.get("domain", "gdelt"),
                    "sentiment": float(np.clip(tone / 10.0, -1, 1)),
                    "article_volume": 1,
                    "geopolitical_intensity": _topic_geopolitical_intensity(topic, tone),
                }
            )
        time.sleep(1.0)
    if not rows:
        return pd.DataFrame(columns=["topic", "date", "text", "source", "sentiment", "article_volume", "geopolitical_intensity"])
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    return out


def _topic_geopolitical_intensity(topic: str, tone: float) -> float:
    base = 1.0 if topic in {"Middle East Conflict", "China Tariffs"} else 0.25
    return float(base + max(-tone, 0.0) / 5.0)


def _topic_for_sector(sector: str) -> str:
    lower = sector.lower()
    if any(token in lower for token in ["information technology", "semiconductor", "technology"]):
        return "AI Boom"
    if any(token in lower for token in ["energy", "oil", "gas"]):
        return "Middle East Conflict"
    if any(token in lower for token in ["financial", "bank"]):
        return "Federal Reserve"
    if any(token in lower for token in ["industrial", "consumer", "materials"]):
        return "China Tariffs"
    return "US CPI Release"


def build_real_events(dates: list[pd.Timestamp], macro: pd.DataFrame, gdelt: pd.DataFrame) -> pd.DataFrame:
    gdelt_daily = gdelt.groupby(["topic", "date"]).agg(
        gdelt_sentiment=("sentiment", "mean"),
        gdelt_volume=("article_volume", "sum"),
        gdelt_geo=("geopolitical_intensity", "mean"),
    ).reset_index() if not gdelt.empty else pd.DataFrame(columns=["topic", "date", "gdelt_sentiment", "gdelt_volume", "gdelt_geo"])

    macro = macro.copy()
    macro["date"] = pd.to_datetime(macro["date"])
    rows = []
    for date in dates:
        m = macro[macro["date"] <= date].tail(1)
        if m.empty:
            values = {}
        else:
            values = m.iloc[0].to_dict()
        for entity in DEFAULT_TOPICS:
            topic_row = gdelt_daily[(gdelt_daily["topic"] == entity) & (gdelt_daily["date"] == pd.Timestamp(date).normalize())]
            sentiment = float(topic_row["gdelt_sentiment"].iloc[0]) if not topic_row.empty else np.nan
            volume = float(topic_row["gdelt_volume"].iloc[0]) if not topic_row.empty else np.nan
            geo = float(topic_row["gdelt_geo"].iloc[0]) if not topic_row.empty else np.nan
            fed_change = float(values.get("fed_funds_change", np.nan))
            cpi_change = float(values.get("inflation_change", np.nan))
            oil_change = float(values.get("oil_price_change", np.nan))
            yield_change = float(values.get("ten_year_yield_change", np.nan))
            rows.append(
                {
                    "entity": entity,
                    "date": pd.Timestamp(date),
                    "event_type": "real_feed",
                    "inflation_pressure": _clip_or_nan(0.4 + cpi_change * 20 + oil_change * 2, 0, 2),
                    "interest_rate_sensitivity": _clip_or_nan(0.4 + abs(fed_change) * 25 + abs(yield_change) * 6, 0, 2),
                    "tariff_exposure": _clip_or_nan(volume / 50 if entity == "China Tariffs" else np.nan, 0, 2),
                    "commodity_dependence": _clip_or_nan(0.65 + abs(oil_change) * 6 if entity == "Middle East Conflict" else np.nan, 0, 2),
                    "geopolitical_instability": _clip_or_nan(geo + volume / 80, 0, 3),
                    "conflict_intensity": _clip_or_nan(geo + max(-sentiment, 0) + volume / 100 if not np.isnan(sentiment) else np.nan, 0, 3),
                    "regulatory_uncertainty": _clip_or_nan(0.25 + volume / 120 + abs(sentiment) if not np.isnan(sentiment) else np.nan, 0, 2),
                    "macroeconomic_surprise": float(cpi_change * 15 + fed_change * 15 + yield_change * 5),
                    "narrative_momentum": float(sentiment + volume / 100) if not np.isnan(sentiment) and not np.isnan(volume) else np.nan,
                    "technology_hype_cycle": _clip_or_nan(0.5 + sentiment + volume / 150 if entity == "AI Boom" else np.nan, 0, 3),
                    "public_sentiment_shift": sentiment,
                }
            )
    return pd.DataFrame(rows)


def _clip_or_nan(value: float, lower: float, upper: float) -> float:
    if pd.isna(value):
        return np.nan
    return float(np.clip(value, lower, upper))


def build_real_news(prices: pd.DataFrame, gdelt: pd.DataFrame, sectors: dict[str, str]) -> pd.DataFrame:
    if gdelt.empty:
        return pd.DataFrame(
            {
                "entity": prices["entity"],
                "date": prices["date"],
                "text": "",
                "source": "",
                "sentiment": np.nan,
                "article_volume": np.nan,
                "geopolitical_intensity": np.nan,
            }
        )
    gdelt = gdelt.copy()
    gdelt["date"] = pd.to_datetime(gdelt["date"]).dt.tz_localize(None)
    topic_daily = gdelt.groupby(["topic", "date"]).agg(
        sentiment=("sentiment", "mean"),
        article_volume=("article_volume", lambda values: values.sum(min_count=1)),
        geopolitical_intensity=("geopolitical_intensity", "mean"),
        text=("text", lambda values: " | ".join([str(v) for v in list(values)[:3]])),
        source=("source", lambda values: ",".join(sorted(set(map(str, values)))[:3])),
    ).reset_index()
    rows = prices[["entity", "date"]].drop_duplicates().copy()
    rows["date"] = pd.to_datetime(rows["date"]).dt.tz_localize(None)
    rows["sector"] = rows["entity"].map(sectors).fillna("Equity Universe")
    rows["topic"] = rows["sector"].map(_topic_for_sector)
    out = rows.merge(topic_daily, on=["topic", "date"], how="left")
    out["text"] = out["text"].fillna("")
    out["source"] = out["source"].fillna("")
    return out[["entity", "date", "text", "source", "sentiment", "article_volume", "geopolitical_intensity"]]


def make_real_feed_ecosystem(raw_config: dict[str, Any], seed: int) -> EcosystemData:
    data_cfg = raw_config["data"]
    tickers, sectors = resolve_universe(data_cfg)
    data_cfg["tickers"] = tickers
    data_cfg["sectors"] = sectors

    prices = load_yfinance_prices(tickers, data_cfg["start_date"], data_cfg["end_date"])
    prices["sector"] = prices["entity"].map(sectors).fillna("Equity Universe")

    market_series = data_cfg.get("market_series", {})
    if market_series:
        series_prices = load_yfinance_prices(list(market_series.values()), data_cfg["start_date"], data_cfg["end_date"])
        reverse_map = {symbol: entity for entity, symbol in market_series.items()}
        series_prices["entity"] = series_prices["entity"].map(reverse_map)
        series_prices["sector"] = series_prices["entity"].map(lambda x: "Commodity" if x in data_cfg.get("commodities", []) else "Currency")
        prices = pd.concat([prices, series_prices], ignore_index=True)

    dates = sorted(prices["date"].unique())
    macro = load_fred_macro(data_cfg["start_date"], data_cfg["end_date"])
    gdelt_cfg = data_cfg.get("gdelt", {})
    gdelt = load_gdelt_topic_news(
        data_cfg["start_date"],
        data_cfg["end_date"],
        gdelt_cfg.get("topics", DEFAULT_TOPICS),
        int(gdelt_cfg.get("max_records_per_topic", 250)),
    )
    events = build_real_events(dates, macro, gdelt)
    news = build_real_news(prices[prices["entity"].isin(tickers)], gdelt, sectors)

    metadata_rows = []
    for ticker in tickers:
        metadata_rows.append({"entity": ticker, "node_type": "stock", "sector": sectors.get(ticker, "Equity Universe"), "country": "United States"})
    for sector in sorted(set(sectors.values())):
        metadata_rows.append({"entity": sector, "node_type": "sector", "sector": sector, "country": "Global"})
    for commodity in data_cfg.get("commodities", []):
        metadata_rows.append({"entity": commodity, "node_type": "commodity", "sector": "Commodity", "country": "Global"})
    for currency in data_cfg.get("currencies", []):
        metadata_rows.append({"entity": currency, "node_type": "currency", "sector": "Currency", "country": "Global"})
    for country in data_cfg.get("countries", []):
        metadata_rows.append({"entity": country, "node_type": "country", "sector": "Sovereign", "country": country})
    for event_name, event_type in [
        ("Federal Reserve", "central_bank"),
        ("US CPI Release", "macroeconomic_report"),
        ("Middle East Conflict", "geopolitical_event"),
        ("China Tariffs", "regulatory_event"),
        ("AI Boom", "technology_narrative"),
    ]:
        metadata_rows.append({"entity": event_name, "node_type": event_type, "sector": "World Event", "country": "Global"})
    for etf in data_cfg.get("etfs", {}):
        metadata_rows.append({"entity": etf, "node_type": "etf", "sector": "ETF", "country": "United States"})

    holding_rows = []
    for etf, holdings in data_cfg.get("etfs", {}).items():
        weight = 1.0 / max(len(holdings), 1)
        for ticker in holdings:
            if ticker in tickers:
                holding_rows.append({"etf": etf, "stock": ticker, "weight": weight})

    return EcosystemData(
        prices=prices,
        events=events,
        news=news,
        entity_metadata=pd.DataFrame(metadata_rows).drop_duplicates("entity"),
        etf_holdings=pd.DataFrame(holding_rows),
    )
