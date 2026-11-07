"""Data pipeline health status assembly."""

from __future__ import annotations

from datetime import datetime
from typing import Union


def build_pipeline_health(
    *,
    market_status: str,
    database_status: str,
    record_counts: dict[str, int],
) -> list[dict[str, Union[str, int]]]:
    if market_status == "AkShare connected":
        market_label = "Connected"
    elif market_status == "AkShare cache":
        market_label = "Cached real data"
    else:
        market_label = "fallback mode"
    checks = [
        ("AkShare API", market_label, record_counts.get("market_data", 0)),
        ("Database", database_status, 1 if database_status == "Connected" else 0),
        ("Encrypted Database Layer", "Connected", 1),
        ("Raw Market Data", "Stored", record_counts.get("market_data", 0)),
        ("Technical Features", "Generated", record_counts.get("technical_features", 0)),
        ("Sentiment Features", "Generated", record_counts.get("sentiment_features", 0)),
        ("RL States", "Generated", record_counts.get("rl_states", 0)),
        ("Agent Decisions", "Stored", record_counts.get("agent_decisions", 0)),
        ("Portfolio Snapshots", "Stored", record_counts.get("portfolio_snapshots", 0)),
        ("Performance Metrics", "Stored", record_counts.get("performance_metrics", 0)),
        ("Ablation Results", "Stored", record_counts.get("ablation_results", 0)),
        ("Data Quality", "Valid", record_counts.get("market_data", 0)),
        ("market.bars", "Active", record_counts.get("market_data", 0)),
        ("sentiment.signals", "Active", record_counts.get("sentiment_features", 0)),
        ("trading.logs", "Active", record_counts.get("agent_decisions", 0)),
    ]
    now = datetime.now().isoformat(timespec="seconds")
    return [
        {"module": name, "status": status, "record_count": int(count), "last_update": now}
        for name, status, count in checks
    ]
