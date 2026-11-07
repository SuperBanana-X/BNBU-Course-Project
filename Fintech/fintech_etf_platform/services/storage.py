"""SQLite and local file storage for the ETF trading platform."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


class StorageService:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.data_dir = root / "data"
        self.raw_market_dir = self.data_dir / "raw" / "market"
        self.processed_dir = self.data_dir / "processed"
        self.db_dir = self.data_dir / "db"
        self.db_path = self.db_dir / "trading_platform.sqlite"
        self.ensure_directories()
        self.initialize_database()

    def ensure_directories(self) -> None:
        for path in [self.raw_market_dir, self.processed_dir, self.db_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def initialize_database(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS market_data (
                    date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL, amount REAL,
                    source TEXT, created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS technical_features (
                    date TEXT, close REAL, daily_return REAL, cumulative_return REAL, ma5 REAL, ma20 REAL,
                    ma50 REAL, ma200 REAL, rsi REAL, macd REAL, macd_signal REAL, macd_hist REAL,
                    volatility_5d REAL, volatility_20d REAL, volume_ratio_5d REAL, volume_ratio_20d REAL,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS sentiment_features (
                    date TEXT, positive_probability REAL, neutral_probability REAL, negative_probability REAL,
                    sentiment_score REAL, sentiment_label TEXT, confidence REAL, signal_strength REAL,
                    sentiment_regime TEXT, sentiment_score_lag1 REAL, sentiment_score_lag3_mean REAL,
                    sentiment_score_lag5_mean REAL, created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS rl_states (
                    date TEXT, price REAL, ma5 REAL, ma20 REAL, rsi REAL, macd REAL, position REAL, cash REAL,
                    sentiment_score REAL, state_vector_json TEXT, with_sentiment INTEGER, created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS agent_decisions (
                    date TEXT, action TEXT, action_id INTEGER, q_buy REAL, q_sell REAL, q_hold REAL,
                    position REAL, cash REAL, target_position REAL, order_quantity REAL, epsilon REAL,
                    model_name TEXT, model_version TEXT, with_sentiment INTEGER, created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    date TEXT, cash REAL, position REAL, market_value REAL, portfolio_value REAL, daily_return REAL,
                    cumulative_return REAL, current_drawdown REAL, transaction_cost REAL, slippage_cost REAL,
                    turnover REAL, position_weight REAL, with_sentiment INTEGER, created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    run_name TEXT, with_sentiment INTEGER, cumulative_return REAL, annualized_return REAL,
                    annualized_volatility REAL, sharpe_ratio REAL, max_drawdown REAL, win_rate REAL,
                    n_trades INTEGER, profit_factor REAL, benchmark_cumulative_return REAL, excess_return REAL,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS ablation_results (
                    model_name TEXT, cumulative_return REAL, annualized_return REAL, annualized_volatility REAL,
                    sharpe_ratio REAL, max_drawdown REAL, win_rate REAL, n_trades INTEGER, profit_factor REAL,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS system_logs (
                    timestamp TEXT, module TEXT, status TEXT, message TEXT, record_count INTEGER
                );
                """
            )

    def save_market_data(self, frame: pd.DataFrame) -> None:
        source = ""
        if "source" in frame.columns and not frame.empty:
            source = str(frame["source"].iloc[-1]).lower()
        if "fallback" in source:
            self._save_csv(frame, self.raw_market_dir / "etf_159325_continuity_fallback.csv")
        else:
            self._save_csv(frame, self.raw_market_dir / "etf_159325_akshare_raw.csv")
        self._replace_table("market_data", frame)

    def save_technical_features(self, frame: pd.DataFrame) -> None:
        table = frame.rename(
            columns={
                "MA5": "ma5",
                "MA20": "ma20",
                "MA50": "ma50",
                "MA200": "ma200",
                "RSI": "rsi",
                "MACD": "macd",
                "MACD_signal": "macd_signal",
                "MACD_histogram": "macd_hist",
            }
        )
        keep = [
            "date", "close", "daily_return", "cumulative_return", "ma5", "ma20", "ma50", "ma200",
            "rsi", "macd", "macd_signal", "macd_hist", "volatility_5d", "volatility_20d",
            "volume_ratio_5d", "volume_ratio_20d",
        ]
        self._save_csv(frame, self.processed_dir / "market_features_159325.csv")
        self._replace_table("technical_features", table[keep])

    def save_sentiment_features(self, frame: pd.DataFrame) -> None:
        self._save_csv(frame, self.processed_dir / "sentiment_features_159325.csv")
        self._replace_table("sentiment_features", frame)

    def save_rl_states(self, frame: pd.DataFrame) -> None:
        self._save_csv(frame, self.processed_dir / "rl_states_159325.csv")
        self._replace_table("rl_states", frame)

    def save_agent_decisions(self, frame: pd.DataFrame) -> None:
        self._save_csv(frame, self.processed_dir / "agent_decisions_159325.csv")
        self._replace_table("agent_decisions", frame)

    def save_portfolio_snapshots(self, frame: pd.DataFrame) -> None:
        self._save_csv(frame, self.processed_dir / "portfolio_snapshots_159325.csv")
        self._replace_table("portfolio_snapshots", frame)

    def save_performance_metrics(self, frame: pd.DataFrame) -> None:
        self._save_csv(frame, self.processed_dir / "performance_metrics_159325.csv")
        self._replace_table("performance_metrics", frame)

    def save_ablation_results(self, frame: pd.DataFrame) -> None:
        self._save_csv(frame, self.processed_dir / "ablation_results_159325.csv")
        self._replace_table("ablation_results", frame)

    def log_system(self, module: str, status: str, message: str, record_count: int = 0) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO system_logs VALUES (?, ?, ?, ?, ?)",
                (datetime.now().isoformat(timespec="seconds"), module, status, message, int(record_count)),
            )

    def read_table(self, table: str) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query(f"SELECT * FROM {table}", conn)

    def database_status(self) -> str:
        try:
            with self.connect() as conn:
                conn.execute("SELECT 1")
            return "Connected"
        except sqlite3.Error:
            return "Unavailable"

    def _replace_table(self, table: str, frame: pd.DataFrame) -> None:
        payload = frame.copy()
        payload["created_at"] = datetime.now().isoformat(timespec="seconds")
        with self.connect() as conn:
            payload.to_sql(table, conn, if_exists="replace", index=False)

    @staticmethod
    def _save_csv(frame: pd.DataFrame, path: Path) -> None:
        frame.to_csv(path, index=False)
