from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EcosystemData:
    prices: pd.DataFrame
    events: pd.DataFrame
    news: pd.DataFrame
    entity_metadata: pd.DataFrame
    etf_holdings: pd.DataFrame


def make_synthetic_ecosystem(raw_config: dict[str, Any], seed: int) -> EcosystemData:
    rng = np.random.default_rng(seed)
    data_cfg = raw_config["data"]
    dates = pd.date_range(data_cfg["start_date"], data_cfg["end_date"], freq="D")
    tickers = data_cfg["tickers"]
    sectors = data_cfg["sectors"]

    market_factor = rng.normal(0.0008, 0.015, len(dates))
    geopolitical_factor = rng.normal(0.0, 0.012, len(dates))
    ai_factor = rng.normal(0.001, 0.02, len(dates))
    oil_factor = rng.normal(0.0002, 0.018, len(dates))

    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        price = 100 + rng.normal(0, 8)
        sector = sectors[ticker]
        beta = 0.8 + rng.uniform(0.1, 0.8)
        for i, date in enumerate(dates):
            thematic = 0.0
            if "Semiconductor" in sector or ticker in {"NVDA", "AMD"}:
                thematic += 0.65 * ai_factor[i] - 0.25 * geopolitical_factor[i]
            if "Energy" in sector:
                thematic += 0.75 * oil_factor[i] + 0.2 * geopolitical_factor[i]
            if "Banking" in sector:
                thematic += -0.35 * geopolitical_factor[i] + rng.normal(0, 0.004)
            if "EV" in sector:
                thematic += 0.4 * ai_factor[i] - 0.35 * oil_factor[i]
            ret = beta * market_factor[i] + thematic + rng.normal(0, 0.012)
            open_price = price * (1 + rng.normal(0, 0.004))
            close = price * (1 + ret)
            high = max(open_price, close) * (1 + abs(rng.normal(0, 0.01)))
            low = min(open_price, close) * (1 - abs(rng.normal(0, 0.01)))
            volume = int(rng.lognormal(15.5, 0.4) * (1 + abs(ret) * 8))
            rows.append(
                {
                    "entity": ticker,
                    "date": date,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "sector": sector,
                }
            )
            price = close

    event_rows: list[dict[str, Any]] = []
    for i, date in enumerate(dates):
        event_rows.extend(
            [
                {
                    "entity": "Federal Reserve",
                    "date": date,
                    "event_type": "central_bank",
                    "inflation_pressure": 0.55 + 0.2 * np.sin(i / 2),
                    "interest_rate_sensitivity": 0.7 + 0.08 * rng.normal(),
                    "tariff_exposure": 0.1,
                    "commodity_dependence": 0.2,
                    "geopolitical_instability": abs(geopolitical_factor[i]) * 35,
                    "conflict_intensity": abs(geopolitical_factor[i]) * 20,
                    "regulatory_uncertainty": 0.45 + 0.08 * rng.normal(),
                    "macroeconomic_surprise": market_factor[i] * 25,
                    "narrative_momentum": ai_factor[i] * 20,
                    "technology_hype_cycle": 0.5 + ai_factor[i] * 8,
                    "public_sentiment_shift": rng.normal(0, 0.2),
                },
                {
                    "entity": "Middle East Conflict",
                    "date": date,
                    "event_type": "geopolitical_event",
                    "inflation_pressure": 0.25 + abs(oil_factor[i]) * 10,
                    "interest_rate_sensitivity": 0.2,
                    "tariff_exposure": 0.15,
                    "commodity_dependence": 0.85,
                    "geopolitical_instability": abs(geopolitical_factor[i]) * 55,
                    "conflict_intensity": abs(geopolitical_factor[i]) * 60,
                    "regulatory_uncertainty": 0.35,
                    "macroeconomic_surprise": oil_factor[i] * 25,
                    "narrative_momentum": geopolitical_factor[i] * 15,
                    "technology_hype_cycle": 0.05,
                    "public_sentiment_shift": -abs(geopolitical_factor[i]) * 8,
                },
                {
                    "entity": "AI Boom",
                    "date": date,
                    "event_type": "technology_narrative",
                    "inflation_pressure": 0.1,
                    "interest_rate_sensitivity": 0.35,
                    "tariff_exposure": 0.4,
                    "commodity_dependence": 0.3,
                    "geopolitical_instability": abs(geopolitical_factor[i]) * 15,
                    "conflict_intensity": 0.05,
                    "regulatory_uncertainty": 0.5 + 0.1 * rng.normal(),
                    "macroeconomic_surprise": ai_factor[i] * 12,
                    "narrative_momentum": ai_factor[i] * 45,
                    "technology_hype_cycle": 0.8 + ai_factor[i] * 12,
                    "public_sentiment_shift": ai_factor[i] * 10,
                },
            ]
        )

    news_rows: list[dict[str, Any]] = []
    topics = {
        "NVDA": "AI accelerators and semiconductor supply chain",
        "AMD": "AI chips and server CPU competition",
        "MSFT": "cloud infrastructure and AI platform demand",
        "XOM": "oil supply shock and energy security",
        "JPM": "interest rates bank credit and regulation",
        "TSLA": "EV demand batteries autonomous driving",
    }
    for ticker in tickers:
        for i, date in enumerate(dates):
            volume = int(rng.poisson(7 if ticker in {"NVDA", "TSLA"} else 4))
            sentiment = float(np.clip(rng.normal(0.08, 0.35) + ai_factor[i], -1, 1))
            news_rows.append(
                {
                    "entity": ticker,
                    "date": date,
                    "text": f"{ticker} coverage: {topics[ticker]} macro regime day {i}",
                    "source": "synthetic_news",
                    "sentiment": sentiment,
                    "article_volume": volume,
                    "geopolitical_intensity": abs(geopolitical_factor[i]) * (8 if ticker in {"NVDA", "AMD", "XOM"} else 3),
                }
            )

    metadata_rows = []
    for ticker in tickers:
        metadata_rows.append({"entity": ticker, "node_type": "stock", "sector": sectors[ticker], "country": "United States"})
    for sector in sorted(set(sectors.values())):
        metadata_rows.append({"entity": sector, "node_type": "sector", "sector": sector, "country": "Global"})
    for commodity in data_cfg["commodities"]:
        metadata_rows.append({"entity": commodity, "node_type": "commodity", "sector": "Commodity", "country": "Global"})
    for currency in data_cfg["currencies"]:
        metadata_rows.append({"entity": currency, "node_type": "currency", "sector": "Currency", "country": "Global"})
    for country in data_cfg["countries"]:
        metadata_rows.append({"entity": country, "node_type": "country", "sector": "Sovereign", "country": country})
    for event_name, event_type in [
        ("Federal Reserve", "central_bank"),
        ("US CPI Release", "macroeconomic_report"),
        ("Middle East Conflict", "geopolitical_event"),
        ("China Tariffs", "regulatory_event"),
        ("AI Boom", "technology_narrative"),
    ]:
        metadata_rows.append({"entity": event_name, "node_type": event_type, "sector": "World Event", "country": "Global"})
    for etf in data_cfg["etfs"]:
        metadata_rows.append({"entity": etf, "node_type": "etf", "sector": "ETF", "country": "United States"})

    holding_rows = []
    for etf, holdings in data_cfg["etfs"].items():
        weight = 1.0 / len(holdings)
        for ticker in holdings:
            holding_rows.append({"etf": etf, "stock": ticker, "weight": weight})

    return EcosystemData(
        prices=pd.DataFrame(rows),
        events=pd.DataFrame(event_rows),
        news=pd.DataFrame(news_rows),
        entity_metadata=pd.DataFrame(metadata_rows).drop_duplicates("entity"),
        etf_holdings=pd.DataFrame(holding_rows),
    )
