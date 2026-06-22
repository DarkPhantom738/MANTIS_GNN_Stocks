from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=2).mean()
    loss = -delta.clip(upper=0).rolling(window, min_periods=2).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _entropy(values: pd.Series, bins: int = 8) -> float:
    clean = values.dropna()
    if clean.empty:
        return 0.0
    counts, _ = np.histogram(clean, bins=bins)
    probs = counts[counts > 0] / max(counts.sum(), 1)
    return float(-(probs * np.log(probs)).sum())


def compute_financial_features(prices: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for entity, group in prices.sort_values("date").groupby("entity", sort=False):
        g = group.copy()
        g["return"] = g["close"].pct_change()
        g["log_return"] = np.log(g["close"]).diff()
        g["rsi"] = _rsi(g["close"], window=max(windows))
        ema12 = g["close"].ewm(span=12, adjust=False).mean()
        ema26 = g["close"].ewm(span=26, adjust=False).mean()
        g["macd"] = ema12 - ema26
        typical_range = g["high"] - g["low"]
        g["atr"] = typical_range.rolling(max(windows), min_periods=1).mean()
        g["momentum"] = g["close"].pct_change(max(windows))
        running_max = g["close"].cummax()
        g["drawdown"] = g["close"] / running_max - 1.0
        g["autocorrelation"] = g["return"].rolling(max(windows), min_periods=3).apply(lambda x: pd.Series(x).autocorr(), raw=False)
        g["rolling_entropy"] = g["return"].rolling(max(windows), min_periods=2).apply(_entropy, raw=False)
        g["volatility_clustering"] = g["return"].abs().rolling(max(windows), min_periods=3).apply(lambda x: pd.Series(x).autocorr(), raw=False)
        for window in windows:
            g[f"rolling_volatility_{window}"] = g["log_return"].rolling(window, min_periods=2).std()
            g[f"sharpe_{window}"] = (
                g["return"].rolling(window, min_periods=2).mean()
                / g["return"].rolling(window, min_periods=2).std().replace(0, np.nan)
            )
            mean = g["close"].rolling(window, min_periods=1).mean()
            std = g["close"].rolling(window, min_periods=1).std()
            g[f"bollinger_z_{window}"] = (g["close"] - mean) / std.replace(0, np.nan)
        frames.append(g)

    out = pd.concat(frames, ignore_index=True)
    market_return = out.groupby("date")["return"].mean().rename("market_return")
    out = out.merge(market_return, on="date", how="left")
    beta = (
        out.groupby("entity")
        .apply(lambda x: x["return"].cov(x["market_return"]) / max(x["market_return"].var(), 1e-9), include_groups=False)
        .rename("beta")
        .reset_index()
    )
    return out.merge(beta, on="entity", how="left")
