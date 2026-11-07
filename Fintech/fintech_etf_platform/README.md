# NLP-RL Quantitative Trading Platform for Southern Semiconductor ETF

## 1. Project Overview

This Flask project implements a course-demo quantitative trading workstation for the Southern Semiconductor ETF. It integrates ETF market data ingestion, technical feature engineering, a FinBERT-compatible sentiment feature layer, an RL trading decision layer, portfolio backtesting, local storage, and dashboard monitoring.

The platform is designed to be runnable from a single entry point and suitable for a live demonstration.

## 2. Target ETF

- Asset: Southern Semiconductor ETF
- Symbol: `159325`
- Display window: `2026-01-05` to the current date

## 3. Data Source

The required external market data source is AkShare:

- API: `akshare.fund_etf_hist_em`
- Symbol: `159325`
- Frequency: daily
- Normalized fields: `date`, `open`, `high`, `low`, `close`, `volume`, `amount`

If AkShare Eastmoney is unavailable or returns empty data, the application tries AkShare's Sina ETF historical endpoint (`fund_etf_hist_sina("sz159325")`) and then a direct Eastmoney k-line endpoint with browser-style request headers. If live sources fail, it reuses the last cached real AkShare CSV in `data/raw/market/etf_159325_akshare_raw.csv`.

Only when live and cached real ETF records are unavailable does the application use a deterministic continuity dataset so the dashboard remains available during a live demo. The dashboard shows:

- `Market data source status: AkShare connected`
- `Market data source status: AkShare cache`
- or `Market data source status: fallback mode`

When fallback mode is active, the price path is a continuity dataset for system demonstration and should not be interpreted as live ETF market prices.

The fallback path uses A-share trading dates. It tries to load the AkShare/Sina China trading calendar first and then falls back to a weekday calendar with major Mainland exchange holiday exclusions. It does not use ordinary calendar days, and it does not include weekends.

## 4. System Architecture

```text
AkShare ETF OHLCV
        |
        v
Bronze Layer: raw market data CSV + SQLite market_data
        |
        v
Silver Layer: technical indicators
        |
        v
Gold Layer: sentiment features, RL states, decisions, portfolio snapshots,
            performance metrics, ablation results
        |
        v
Flask Dashboard
```

## 5. Flask Routes

- `/` redirects to `/login`
- `/login` displays the login page and handles authentication
- `/loading` displays the financial system loading sequence
- `/dashboard` displays the main workstation and requires login
- `/logout` clears the session and returns to `/login`

Login credentials:

- Username: `SuperBanana`
- Password: `5201314`

## 6. Data Storage Layer

The project uses both structured local files and SQLite:

- SQLite database: `data/db/trading_platform.sqlite`
- Raw files: `data/raw/`
- Processed files: `data/processed/`

Real AkShare raw market files are stored as `data/raw/market/etf_159325_akshare_raw.csv`. If fallback mode is active, the continuity file is stored separately as `data/raw/market/etf_159325_continuity_fallback.csv` so it does not overwrite the real raw-data cache.

SQLite tables:

- `market_data`
- `technical_features`
- `sentiment_features`
- `rl_states`
- `agent_decisions`
- `portfolio_snapshots`
- `performance_metrics`
- `ablation_results`
- `system_logs`

## 7. Bronze / Silver / Gold Mapping

- Bronze Layer: raw AkShare ETF OHLCV data
- Silver Layer: market technical features
- Gold Layer: sentiment features, RL state vectors, agent decisions, portfolio snapshots, performance metrics, ablation results
- SQLite: local relational storage for dashboard readback and auditability

## 8. Technical Indicators

The technical feature service calculates:

- `daily_return`
- `cumulative_return`
- `MA5`
- `MA20`
- `MA50`
- `MA200`
- `RSI`
- `MACD`
- `MACD_signal`
- `MACD_histogram`
- `volatility_5d`
- `volatility_20d`
- `volume_ratio_5d`
- `volume_ratio_20d`

The dashboard emphasizes MA5 and MA20 while still storing MA50 and MA200 when the data window supports them.

## 9. Sentiment Feature Layer

The sentiment service generates a FinBERT-compatible daily feature layer. It does not display article-level data and does not claim that raw news articles were processed.

The daily sentiment score is derived from market-implied inputs:

- standardized daily return
- MA5 vs MA20 momentum
- MACD direction
- volatility pressure
- volume confirmation

Generated fields:

- positive, neutral, and negative probabilities
- sentiment score
- sentiment label
- confidence
- signal strength
- sentiment regime
- lagged sentiment features

## 10. RL Decision Layer

The RL layer creates DQN-style Buy / Sell / Hold records with:

- action ID: Hold = 0, Buy = 1, Sell = 2
- Q-values for Buy, Sell, and Hold
- epsilon decay
- model metadata: `DQN`, `dqn_v1_etf_159325`

The RL state vector is:

```text
[price, MA5, MA20, RSI, MACD, position, cash, sentiment_score]
```

Two strategy modes are generated:

- DQN with Sentiment Feature
- DQN without Sentiment Feature

The decision rules use only current and past information.

## 11. Portfolio Backtest

The backtest engine uses:

- Initial cash: `100000`
- Transaction cost: `8 bps`
- Slippage: `3 bps`
- Buy allocation: 20% to 35% of available cash
- Sell allocation: 25% to 50% of current position

Daily portfolio fields:

- cash
- position
- market value
- portfolio value
- daily return
- cumulative return
- current drawdown
- transaction cost
- slippage cost
- turnover
- position weight

## 12. Performance Metrics

The project reports:

- cumulative return
- annualized return
- annualized volatility
- Sharpe Ratio
- Max Drawdown
- win rate
- number of trades
- profit factor
- benchmark cumulative return
- excess return

Win rate is calculated from positive daily portfolio returns.

## 13. Ablation Study

The dashboard compares:

- DQN with Sentiment Feature
- DQN without Sentiment Feature
- Buy-and-Hold Benchmark

The purpose is to evaluate whether the sentiment-enhanced RL state improves the risk-adjusted profile in the selected evaluation window.

## 14. Installation

```bash
pip install -r requirements.txt
```

## 15. Running the App

From this project directory:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## 16. Demo Reliability

The application handles:

- AkShare connection failure
- empty market data
- missing columns
- insufficient indicator history
- SQLite connection checks
- NaN and division-by-zero cases
- chart rendering fallback

The goal is to keep the dashboard accessible for presentation even when an external dependency is unavailable.
