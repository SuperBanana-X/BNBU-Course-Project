"""Portfolio backtesting and performance evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


INITIAL_CASH = 100000.0
TRANSACTION_COST_BPS = 8
SLIPPAGE_BPS = 3


def run_backtest(
    features: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    with_sentiment: bool,
    initial_cash: float = INITIAL_CASH,
) -> tuple[pd.DataFrame, dict]:
    merged = features[["date", "close"]].merge(decisions[["date", "action"]], on="date", how="left")
    merged["action"] = merged["action"].fillna("Hold")
    rng = np.random.default_rng(9001 if with_sentiment else 9002)

    cash = initial_cash
    position = 0.0
    previous_value = initial_cash
    peak = initial_cash
    rows: list[dict] = []
    trade_returns: list[float] = []

    for _, row in merged.iterrows():
        price = float(row["close"])
        action = row["action"]
        transaction_cost = 0.0
        slippage_cost = 0.0
        turnover = 0.0
        pre_trade_value = cash + position * price

        if action == "Buy" and cash > price:
            allocation = rng.uniform(0.20, 0.35)
            gross_amount = cash * allocation
            quantity = int(gross_amount / price)
            gross_trade = quantity * price
            transaction_cost = gross_trade * TRANSACTION_COST_BPS / 10000
            slippage_cost = gross_trade * SLIPPAGE_BPS / 10000
            total_cost = gross_trade + transaction_cost + slippage_cost
            if quantity > 0 and total_cost <= cash:
                cash -= total_cost
                position += quantity
                turnover = gross_trade / max(pre_trade_value, 1e-9)
        elif action == "Sell" and position > 0:
            quantity = int(position * rng.uniform(0.25, 0.50))
            gross_trade = quantity * price
            transaction_cost = gross_trade * TRANSACTION_COST_BPS / 10000
            slippage_cost = gross_trade * SLIPPAGE_BPS / 10000
            if quantity > 0:
                cash += gross_trade - transaction_cost - slippage_cost
                position -= quantity
                turnover = gross_trade / max(pre_trade_value, 1e-9)
                trade_returns.append((gross_trade - transaction_cost - slippage_cost) / max(pre_trade_value, 1e-9))

        market_value = position * price
        portfolio_value = cash + market_value
        daily_return = portfolio_value / previous_value - 1 if previous_value else 0.0
        cumulative_return = portfolio_value / initial_cash - 1
        peak = max(peak, portfolio_value)
        current_drawdown = portfolio_value / peak - 1 if peak else 0.0
        position_weight = market_value / portfolio_value if portfolio_value else 0.0
        rows.append(
            {
                "date": row["date"],
                "cash": cash,
                "position": position,
                "market_value": market_value,
                "portfolio_value": portfolio_value,
                "daily_return": daily_return,
                "cumulative_return": cumulative_return,
                "current_drawdown": current_drawdown,
                "transaction_cost": transaction_cost,
                "slippage_cost": slippage_cost,
                "turnover": turnover,
                "position_weight": position_weight,
                "with_sentiment": int(with_sentiment),
            }
        )
        previous_value = portfolio_value

    snapshots = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    metrics = calculate_metrics(
        snapshots,
        benchmark_cumulative_return=_buy_and_hold_cumulative_return(features),
        run_name="DQN with Sentiment Feature" if with_sentiment else "DQN without Sentiment Feature",
        with_sentiment=with_sentiment,
    )
    return snapshots.round(6), metrics


def buy_and_hold_series(features: pd.DataFrame, initial_cash: float = INITIAL_CASH) -> pd.DataFrame:
    close = pd.to_numeric(features["close"], errors="coerce").ffill().bfill()
    first_close = close.iloc[0] if len(close) else 1.0
    value = initial_cash * close / max(first_close, 1e-9)
    daily_return = value.pct_change().fillna(0.0)
    peak = value.cummax()
    return pd.DataFrame(
        {
            "date": features["date"],
            "portfolio_value": value,
            "daily_return": daily_return,
            "cumulative_return": value / initial_cash - 1,
            "current_drawdown": value / peak - 1,
        }
    ).round(6)


def buy_and_hold_metrics(features: pd.DataFrame) -> dict:
    series = buy_and_hold_series(features)
    metrics = calculate_metrics(
        series,
        benchmark_cumulative_return=float(series["cumulative_return"].iloc[-1]) if len(series) else 0.0,
        run_name="Buy-and-Hold Benchmark",
        with_sentiment=False,
    )
    metrics["n_trades"] = 1
    metrics["excess_return"] = 0.0
    return metrics


def build_ablation_results(*metrics: dict) -> pd.DataFrame:
    rows = []
    for item in metrics:
        rows.append(
            {
                "model_name": item["run_name"],
                "cumulative_return": item["cumulative_return"],
                "annualized_return": item["annualized_return"],
                "annualized_volatility": item["annualized_volatility"],
                "sharpe_ratio": item["sharpe_ratio"],
                "max_drawdown": item["max_drawdown"],
                "win_rate": item["win_rate"],
                "n_trades": item["n_trades"],
                "profit_factor": item["profit_factor"],
            }
        )
    return pd.DataFrame(rows).round(6)


def calculate_metrics(
    snapshots: pd.DataFrame,
    *,
    benchmark_cumulative_return: float,
    run_name: str,
    with_sentiment: bool,
) -> dict:
    if snapshots.empty:
        return _empty_metrics(run_name, with_sentiment, benchmark_cumulative_return)
    returns = pd.to_numeric(snapshots["daily_return"], errors="coerce").fillna(0.0)
    cumulative_return = float(snapshots["cumulative_return"].iloc[-1])
    periods = max(len(snapshots), 1)
    annualized_return = (1 + cumulative_return) ** (252 / periods) - 1 if cumulative_return > -1 else -1.0
    annualized_volatility = float(returns.std(ddof=0) * np.sqrt(252))
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0.0
    max_drawdown = float(snapshots["current_drawdown"].min())
    win_rate = float((returns > 0).mean())
    n_trades = int((pd.to_numeric(snapshots.get("turnover", pd.Series([0] * len(snapshots))), errors="coerce") > 0).sum())
    positive = returns[returns > 0].sum()
    negative = abs(returns[returns < 0].sum())
    profit_factor = float(positive / negative) if negative > 1e-12 else float(positive > 0)
    return {
        "run_name": run_name,
        "with_sentiment": int(with_sentiment),
        "cumulative_return": round(cumulative_return, 6),
        "annualized_return": round(float(annualized_return), 6),
        "annualized_volatility": round(annualized_volatility, 6),
        "sharpe_ratio": round(float(sharpe_ratio), 6),
        "max_drawdown": round(max_drawdown, 6),
        "win_rate": round(win_rate, 6),
        "n_trades": n_trades,
        "profit_factor": round(profit_factor, 6),
        "benchmark_cumulative_return": round(float(benchmark_cumulative_return), 6),
        "excess_return": round(cumulative_return - float(benchmark_cumulative_return), 6),
    }


def _buy_and_hold_cumulative_return(features: pd.DataFrame) -> float:
    if features.empty:
        return 0.0
    close = pd.to_numeric(features["close"], errors="coerce").ffill().bfill()
    return float(close.iloc[-1] / close.iloc[0] - 1) if close.iloc[0] else 0.0


def _empty_metrics(run_name: str, with_sentiment: bool, benchmark_cumulative_return: float) -> dict:
    return {
        "run_name": run_name,
        "with_sentiment": int(with_sentiment),
        "cumulative_return": 0.0,
        "annualized_return": 0.0,
        "annualized_volatility": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "n_trades": 0,
        "profit_factor": 0.0,
        "benchmark_cumulative_return": benchmark_cumulative_return,
        "excess_return": 0.0,
    }
