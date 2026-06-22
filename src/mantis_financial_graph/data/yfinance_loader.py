from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from mantis_financial_graph.data.synthetic import EcosystemData
from mantis_financial_graph.utils import ensure_dir


def load_yfinance_prices(tickers: Sequence[str], start: str, end: str, batch_size: int = 100) -> pd.DataFrame:
    import yfinance as yf

    cache_dir = ensure_dir("/private/tmp/mantis_yfinance_cache")
    try:
        yf.set_tz_cache_location(str(cache_dir))
    except AttributeError:
        pass
    frames: list[pd.DataFrame] = []
    ticker_list = list(dict.fromkeys(tickers))
    for offset in range(0, len(ticker_list), batch_size):
        batch = ticker_list[offset : offset + batch_size]
        raw = yf.download(batch, start=start, end=end, auto_adjust=False, progress=False, group_by="ticker", threads=True)
        if raw.empty:
            continue
        for ticker in batch:
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker in raw.columns.get_level_values(0):
                    ticker_frame = raw[ticker].copy()
                else:
                    continue
            else:
                ticker_frame = raw.copy()
            ticker_frame.columns = [str(col).lower().replace(" ", "_") for col in ticker_frame.columns]
            ticker_frame = ticker_frame.reset_index().rename(columns={"Date": "date"})
            if "date" not in ticker_frame.columns and "datetime" in ticker_frame.columns:
                ticker_frame = ticker_frame.rename(columns={"datetime": "date"})
            if "adj_close" in ticker_frame.columns and "close" not in ticker_frame.columns:
                ticker_frame["close"] = ticker_frame["adj_close"]
            ticker_frame["entity"] = ticker
            required = ["entity", "date", "open", "high", "low", "close", "volume"]
            if set(required).issubset(ticker_frame.columns):
                frames.append(ticker_frame[required])
    if not frames:
        raise RuntimeError("yfinance returned no price rows. Check tickers, dates, and network access.")
    if not frames:
        raise RuntimeError("No usable yfinance OHLCV frames were returned.")
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    return out.dropna(subset=["open", "high", "low", "close"])


def make_yfinance_ecosystem(raw_config: dict[str, Any], seed: int) -> EcosystemData:
    rng = np.random.default_rng(seed)
    data_cfg = raw_config["data"]
    tickers = data_cfg["tickers"]
    sectors = data_cfg["sectors"]
    prices = load_yfinance_prices(tickers, data_cfg["start_date"], data_cfg["end_date"])
    prices["sector"] = prices["entity"].map(sectors).fillna("Unknown Sector")

    dates = sorted(prices["date"].unique())
    market_return = prices.pivot_table(index="date", columns="entity", values="close").pct_change().mean(axis=1).fillna(0.0)
    realized_stress = market_return.abs().rolling(5, min_periods=1).mean()
    volume_pressure = (
        prices.groupby("date")["volume"].sum().pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0).rolling(5, min_periods=1).mean()
    )

    event_rows: list[dict[str, Any]] = []
    for date in dates:
        stress = float(realized_stress.loc[date])
        volume = float(abs(volume_pressure.loc[date]))
        ret = float(market_return.loc[date])
        event_rows.extend(
            [
                {
                    "entity": "Federal Reserve",
                    "date": date,
                    "event_type": "central_bank_proxy",
                    "inflation_pressure": 0.45 + min(volume * 4, 0.35),
                    "interest_rate_sensitivity": 0.55 + min(stress * 8, 0.35),
                    "tariff_exposure": 0.1,
                    "commodity_dependence": 0.2,
                    "geopolitical_instability": min(stress * 20, 1.0),
                    "conflict_intensity": min(stress * 12, 1.0),
                    "regulatory_uncertainty": 0.35 + min(volume * 3, 0.4),
                    "macroeconomic_surprise": ret * 10,
                    "narrative_momentum": ret * 5,
                    "technology_hype_cycle": 0.4 + max(ret, 0) * 8,
                    "public_sentiment_shift": ret * 6,
                },
                {
                    "entity": "Middle East Conflict",
                    "date": date,
                    "event_type": "geopolitical_proxy",
                    "inflation_pressure": 0.25 + min(stress * 6, 0.4),
                    "interest_rate_sensitivity": 0.2,
                    "tariff_exposure": 0.15,
                    "commodity_dependence": 0.75,
                    "geopolitical_instability": min(stress * 35, 1.0),
                    "conflict_intensity": min(stress * 28, 1.0),
                    "regulatory_uncertainty": 0.3 + min(volume * 2, 0.3),
                    "macroeconomic_surprise": ret * 8,
                    "narrative_momentum": -abs(ret) * 4,
                    "technology_hype_cycle": 0.05,
                    "public_sentiment_shift": -abs(ret) * 5,
                },
                {
                    "entity": "AI Boom",
                    "date": date,
                    "event_type": "technology_narrative_proxy",
                    "inflation_pressure": 0.1,
                    "interest_rate_sensitivity": 0.3,
                    "tariff_exposure": 0.35,
                    "commodity_dependence": 0.25,
                    "geopolitical_instability": min(stress * 8, 1.0),
                    "conflict_intensity": 0.05,
                    "regulatory_uncertainty": 0.4 + min(volume * 2, 0.35),
                    "macroeconomic_surprise": ret * 9,
                    "narrative_momentum": ret * 12,
                    "technology_hype_cycle": 0.65 + max(ret, 0) * 10,
                    "public_sentiment_shift": ret * 8,
                },
            ]
        )

    news_rows: list[dict[str, Any]] = []
    returns = prices.sort_values("date").groupby("entity")["close"].pct_change().fillna(0.0)
    prices_with_ret = prices.copy()
    prices_with_ret["return_proxy"] = returns
    for row in prices_with_ret.to_dict("records"):
        sector = row["sector"]
        ret = float(row["return_proxy"])
        vol = float(row["volume"])
        sentiment = float(np.clip(ret * 15 + rng.normal(0, 0.08), -1, 1))
        article_volume = int(max(1, np.log1p(vol) - 10))
        news_rows.append(
            {
                "entity": row["entity"],
                "date": row["date"],
                "text": f"{row['entity']} real-market proxy narrative for {sector}",
                "source": "yfinance_market_proxy",
                "sentiment": sentiment,
                "article_volume": article_volume,
                "geopolitical_intensity": abs(ret) * (8 if "Semiconductor" in sector or "Energy" in sector else 4),
            }
        )

    metadata_rows = []
    for ticker in tickers:
        metadata_rows.append({"entity": ticker, "node_type": "stock", "sector": sectors.get(ticker, "Unknown Sector"), "country": "United States"})
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
            holding_rows.append({"etf": etf, "stock": ticker, "weight": weight})

    return EcosystemData(
        prices=prices,
        events=pd.DataFrame(event_rows),
        news=pd.DataFrame(news_rows),
        entity_metadata=pd.DataFrame(metadata_rows).drop_duplicates("entity"),
        etf_holdings=pd.DataFrame(holding_rows),
    )
