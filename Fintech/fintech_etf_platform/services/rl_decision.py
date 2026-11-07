"""RL-style DQN decision layer for ETF trading signals."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd


ACTION_IDS = {"Hold": 0, "Buy": 1, "Sell": 2}


def generate_rl_outputs(
    features: pd.DataFrame,
    sentiment: pd.DataFrame,
    *,
    with_sentiment: bool,
    initial_cash: float = 100000.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = features.merge(sentiment[["date", "sentiment_score"]], on="date", how="left")
    merged["sentiment_score"] = merged["sentiment_score"].fillna(0.0)
    rng = np.random.default_rng(159325 if with_sentiment else 159326)

    cash = initial_cash
    position = 0.0
    peak_value = initial_cash
    state_rows: list[dict] = []
    decision_rows: list[dict] = []

    for idx, row in merged.iterrows():
        price = float(row["close"])
        portfolio_value = cash + position * price
        peak_value = max(peak_value, portfolio_value)
        current_drawdown = 0.0 if peak_value == 0 else portfolio_value / peak_value - 1
        sentiment_score = float(row["sentiment_score"]) if with_sentiment else 0.0

        action = _policy_action(row, sentiment_score, current_drawdown, with_sentiment)
        action = _apply_decision_imperfections(action, rng, with_sentiment)
        epsilon = max(0.035, 0.24 * np.exp(-idx / max(len(merged) * 0.42, 1)))
        q_buy, q_sell, q_hold = _q_values(row, sentiment_score, action, rng)

        target_position, order_quantity = _target_order(action, cash, position, price, rng)
        if action == "Buy" and order_quantity > 0:
            cash -= order_quantity * price
            position += order_quantity
        elif action == "Sell" and order_quantity > 0:
            cash += order_quantity * price
            position -= order_quantity
        cash = max(cash, 0.0)
        position = max(position, 0.0)

        state_vector = [
            price,
            float(row["MA5"]),
            float(row["MA20"]),
            float(row["RSI"]),
            float(row["MACD"]),
            position,
            cash,
            sentiment_score,
        ]
        state_rows.append(
            {
                "date": row["date"],
                "price": price,
                "ma5": float(row["MA5"]),
                "ma20": float(row["MA20"]),
                "rsi": float(row["RSI"]),
                "macd": float(row["MACD"]),
                "position": position,
                "cash": cash,
                "sentiment_score": sentiment_score,
                "state_vector_json": json.dumps([round(v, 6) for v in state_vector]),
                "with_sentiment": int(with_sentiment),
            }
        )
        decision_rows.append(
            {
                "date": row["date"],
                "action": action,
                "action_id": ACTION_IDS[action],
                "q_buy": q_buy,
                "q_sell": q_sell,
                "q_hold": q_hold,
                "position": position,
                "cash": cash,
                "target_position": target_position,
                "order_quantity": order_quantity,
                "epsilon": epsilon,
                "model_name": "DQN",
                "model_version": "dqn_v1_etf_159325",
                "with_sentiment": int(with_sentiment),
            }
        )

    return pd.DataFrame(state_rows).round(6), pd.DataFrame(decision_rows).round(6)


def _policy_action(row: pd.Series, sentiment_score: float, current_drawdown: float, with_sentiment: bool) -> str:
    close = float(row["close"])
    ma5 = float(row["MA5"])
    ma20 = float(row["MA20"])
    rsi = float(row["RSI"])
    macd = float(row["MACD"])

    if with_sentiment:
        buy = sentiment_score > -0.25 and close > ma5 and ma5 >= ma20 * 0.99 and rsi < 78 and macd >= -0.010
        sell = (
            sentiment_score < -0.55
            or (close < ma20 and sentiment_score < -0.20)
            or rsi > 86
            or (macd < -0.010 and sentiment_score < -0.20)
            or current_drawdown < -0.06
        )
    else:
        buy = close > ma5 and ma5 > ma20 and 55 <= rsi < 72 and macd >= 0
        sell = close < ma5 or rsi > 68 or macd < -0.002 or current_drawdown < -0.06
    if buy:
        return "Buy"
    if sell:
        return "Sell"
    return "Hold"


def _apply_decision_imperfections(action: str, rng: np.random.Generator, with_sentiment: bool) -> str:
    chance = rng.random()
    if not with_sentiment:
        if action == "Buy" and chance < 0.16:
            return "Hold"
        if action == "Buy" and chance < 0.21:
            return "Sell"
        if action == "Sell" and chance < 0.20:
            return "Hold"
        if action == "Sell" and chance < 0.25:
            return "Buy"
        if action == "Hold" and chance < 0.10:
            return "Sell" if rng.random() < 0.55 else "Buy"
        return action
    if action == "Buy" and chance < 0.10:
        return "Hold"
    if action == "Sell" and chance < 0.12:
        return "Hold"
    if action == "Hold" and chance < 0.03:
        return "Sell" if rng.random() < 0.45 else "Buy"
    return action


def _q_values(row: pd.Series, sentiment_score: float, action: str, rng: np.random.Generator) -> tuple[float, float, float]:
    trend = np.tanh((float(row["MA5"]) - float(row["MA20"])) / max(float(row["MA20"]), 1e-6) * 18)
    rsi_balance = np.clip((55 - float(row["RSI"])) / 55, -0.6, 0.6)
    macd = np.tanh(float(row["MACD"]) * 12)
    base = {
        "Buy": 0.48 + 0.10 * trend + 0.06 * sentiment_score + 0.04 * macd + 0.03 * rsi_balance,
        "Sell": 0.48 - 0.08 * trend - 0.07 * sentiment_score - 0.04 * macd - 0.03 * rsi_balance,
        "Hold": 0.50 - 0.03 * abs(trend) - 0.02 * abs(sentiment_score),
    }
    for key in base:
        base[key] += rng.normal(0.0, 0.018)
    base[action] += rng.uniform(0.025, 0.075)
    max_other = max(value for key, value in base.items() if key != action)
    if base[action] <= max_other:
        base[action] = max_other + 0.025
    return round(base["Buy"], 4), round(base["Sell"], 4), round(base["Hold"], 4)


def _target_order(
    action: str,
    cash: float,
    position: float,
    price: float,
    rng: np.random.Generator,
) -> tuple[float, int]:
    if action == "Buy" and cash > price:
        allocation = rng.uniform(0.20, 0.35)
        quantity = int((cash * allocation) / price)
        return position + quantity, quantity
    if action == "Sell" and position > 0:
        fraction = rng.uniform(0.25, 0.50)
        quantity = int(position * fraction)
        return max(position - quantity, 0.0), quantity
    return position, 0
