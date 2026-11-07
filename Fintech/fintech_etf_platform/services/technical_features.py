"""Technical feature engineering for ETF market data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_technical_features(market: pd.DataFrame) -> pd.DataFrame:
    frame = market.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)

    close = frame["close"].astype(float)
    volume = frame["volume"].astype(float)
    frame["daily_return"] = close.pct_change().fillna(0.0)
    frame["cumulative_return"] = close / close.iloc[0] - 1 if len(close) else 0.0
    for window in [5, 20, 50, 200]:
        frame[f"MA{window}"] = close.rolling(window=window, min_periods=1).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    frame["RSI"] = (100 - (100 / (1 + rs))).fillna(50).clip(0, 100)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    frame["MACD"] = ema12 - ema26
    frame["MACD_signal"] = frame["MACD"].ewm(span=9, adjust=False).mean()
    frame["MACD_histogram"] = frame["MACD"] - frame["MACD_signal"]
    frame["volatility_5d"] = frame["daily_return"].rolling(window=5, min_periods=2).std().fillna(0.0)
    frame["volatility_20d"] = frame["daily_return"].rolling(window=20, min_periods=2).std().fillna(0.0)
    frame["volume_ratio_5d"] = volume / volume.rolling(window=5, min_periods=1).mean().replace(0, np.nan)
    frame["volume_ratio_20d"] = volume / volume.rolling(window=20, min_periods=1).mean().replace(0, np.nan)
    frame[["volume_ratio_5d", "volume_ratio_20d"]] = frame[["volume_ratio_5d", "volume_ratio_20d"]].fillna(1.0)

    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    return frame.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
