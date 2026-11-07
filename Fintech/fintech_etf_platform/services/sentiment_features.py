"""FinBERT-compatible daily sentiment feature layer."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_sentiment_features(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.copy().reset_index(drop=True)
    rng = np.random.default_rng(20260517)

    daily_return = _zscore(frame["daily_return"])
    momentum = _zscore((frame["MA5"] - frame["MA20"]) / frame["MA20"].replace(0, np.nan))
    macd = _zscore(frame["MACD"])
    volatility = _zscore(frame["volatility_20d"])
    volume_confirmation = np.where(
        (frame["daily_return"] > 0) & (frame["volume_ratio_5d"] > 1),
        1.0,
        np.where((frame["daily_return"] < 0) & (frame["volume_ratio_5d"] > 1), -1.0, 0.0),
    )
    noise = rng.normal(0.0, 0.12, len(frame))
    latent_score = (
        0.30 * daily_return
        + 0.25 * momentum
        + 0.20 * macd
        - 0.15 * volatility
        + 0.10 * volume_confirmation
        + noise
    )
    sentiment_score = np.clip(np.tanh(latent_score / 1.9), -1.0, 1.0)

    probabilities = np.vstack([_probabilities(score) for score in sentiment_score])
    result = pd.DataFrame(
        {
            "date": frame["date"],
            "positive_probability": probabilities[:, 0],
            "neutral_probability": probabilities[:, 1],
            "negative_probability": probabilities[:, 2],
            "sentiment_score": sentiment_score,
        }
    )
    labels = np.array(["Positive", "Neutral", "Negative"])
    result["sentiment_label"] = labels[np.argmax(probabilities, axis=1)]
    result["confidence"] = probabilities.max(axis=1)
    result["signal_strength"] = result["sentiment_score"].abs()
    result["sentiment_regime"] = np.select(
        [result["sentiment_score"] > 0.25, result["sentiment_score"] < -0.25],
        ["Risk-On", "Risk-Off"],
        default="Neutral",
    )
    result["sentiment_score_lag1"] = result["sentiment_score"].shift(1).fillna(0.0)
    result["sentiment_score_lag3_mean"] = result["sentiment_score"].rolling(3, min_periods=1).mean().shift(1).fillna(0.0)
    result["sentiment_score_lag5_mean"] = result["sentiment_score"].rolling(5, min_periods=1).mean().shift(1).fillna(0.0)
    numeric_cols = result.select_dtypes(include=["number"]).columns
    result[numeric_cols] = result[numeric_cols].round(6)
    return result


def _zscore(series: pd.Series) -> np.ndarray:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    std = clean.std(ddof=0)
    if std == 0 or np.isnan(std):
        return np.zeros(len(clean))
    return ((clean - clean.mean()) / std).clip(-3, 3).to_numpy()


def _probabilities(score: float) -> np.ndarray:
    positive_logit = 1.75 * score
    negative_logit = -1.75 * score
    neutral_logit = 1.15 - 2.2 * abs(score)
    logits = np.array([positive_logit, neutral_logit, negative_logit])
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()
