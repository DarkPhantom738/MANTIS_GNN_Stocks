from __future__ import annotations

import io
import ssl
import urllib.request
from typing import Any

import pandas as pd


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": "mantis-financial-graph-research/0.1"})


def load_sp500_universe() -> tuple[list[str], dict[str, str]]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    request = _request(url)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            html = response.read()
    except (ssl.SSLError, urllib.error.URLError):
        with urllib.request.urlopen(request, timeout=30, context=ssl._create_unverified_context()) as response:
            html = response.read()
    tables = pd.read_html(io.BytesIO(html))
    table = tables[0]
    tickers = [str(symbol).replace(".", "-") for symbol in table["Symbol"].tolist()]
    sectors = {
        str(row["Symbol"]).replace(".", "-"): str(row["GICS Sector"])
        for _, row in table.iterrows()
    }
    return tickers, sectors


def load_nasdaqtrader_universe() -> list[str]:
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    request = _request(url)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            text = response.read().decode("utf-8", errors="replace")
    except (ssl.SSLError, urllib.error.URLError):
        with urllib.request.urlopen(request, timeout=30, context=ssl._create_unverified_context()) as response:
            text = response.read().decode("utf-8", errors="replace")
    table = pd.read_csv(io.StringIO(text), sep="|")
    table = table[
        (table["Test Issue"] == "N")
        & (table["ETF"] == "N")
        & table["Symbol"].notna()
    ]
    symbols = table["Symbol"].astype(str).tolist()
    return [symbol for symbol in symbols if symbol.isascii() and "^" not in symbol and "." not in symbol]


def resolve_universe(data_cfg: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    universe_cfg = data_cfg.get("universe", {})
    source = universe_cfg.get("source")
    max_tickers = int(universe_cfg.get("max_tickers", len(data_cfg.get("tickers", [])) or 100))

    configured_tickers = list(data_cfg.get("tickers", []))
    configured_sectors = dict(data_cfg.get("sectors", {}))
    if source == "sp500":
        tickers, sectors = load_sp500_universe()
    elif source == "nasdaqtrader":
        tickers = load_nasdaqtrader_universe()
        sectors = {}
    elif source == "sp500_plus_nasdaq":
        sp_tickers, sp_sectors = load_sp500_universe()
        nasdaq_tickers = load_nasdaqtrader_universe()
        tickers = list(dict.fromkeys([*sp_tickers, *nasdaq_tickers]))
        sectors = sp_sectors
    else:
        tickers = configured_tickers
        sectors = configured_sectors

    tickers = list(dict.fromkeys([*configured_tickers, *tickers]))[:max_tickers]
    sectors = {**{ticker: "Equity Universe" for ticker in tickers}, **sectors, **configured_sectors}
    return tickers, {ticker: sectors.get(ticker, "Equity Universe") for ticker in tickers}
