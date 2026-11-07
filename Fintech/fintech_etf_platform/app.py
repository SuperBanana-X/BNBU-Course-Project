"""Flask entry point for the NLP-RL Quantitative Trading Platform."""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, redirect, render_template, request, session, url_for

from services.market_data import START_DATE, TARGET_SYMBOL, fetch_market_data
from services.portfolio import build_ablation_results, buy_and_hold_metrics, buy_and_hold_series, run_backtest
from services.rl_decision import generate_rl_outputs
from services.sentiment_features import generate_sentiment_features
from services.storage import StorageService
from services.system_health import build_pipeline_health
from services.technical_features import calculate_technical_features


LOGIN_NAME = "SuperBanana"
LOGIN_PASSWORD = "5201314"
PROJECT_ROOT = Path(__file__).resolve().parent
TARGET_NAME = "Southern Semiconductor ETF"

_CACHE: dict[str, Any] = {"payload": None, "expires_at": datetime.min}


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.getenv("FLASK_SECRET_KEY", "superbanana-local-session-key")
    storage = StorageService(PROJECT_ROOT)

    @app.get("/")
    def index():
        return redirect(url_for("login"))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        error = ""
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "").strip()
            if username == LOGIN_NAME and password == LOGIN_PASSWORD:
                session["authenticated"] = True
                session["username"] = username
                return redirect(url_for("loading"))
            error = "Invalid username or password."
        return render_template("login.html", error=error)

    @app.get("/loading")
    def loading():
        if not session.get("authenticated"):
            return redirect(url_for("login"))
        return render_template("loading.html")

    @app.get("/dashboard")
    def dashboard():
        if not session.get("authenticated"):
            return redirect(url_for("login"))
        payload = get_dashboard_payload(storage)
        return render_template("dashboard.html", **payload)

    @app.get("/logout")
    def logout():
        session.clear()
        return render_template("farewell.html")

    return app


def get_dashboard_payload(storage: StorageService) -> dict[str, Any]:
    now = datetime.now()
    if _CACHE["payload"] is not None and now < _CACHE["expires_at"]:
        return _CACHE["payload"]
    payload = run_pipeline(storage)
    _CACHE["payload"] = payload
    _CACHE["expires_at"] = now + timedelta(seconds=90)
    return payload


def run_pipeline(storage: StorageService) -> dict[str, Any]:
    errors: list[str] = []
    try:
        market_result = fetch_market_data(cache_path=storage.raw_market_dir / "etf_159325_akshare_raw.csv")
        market = market_result.frame
        storage.save_market_data(market)
        storage.log_system("market_data", market_result.source_status, market_result.message, len(market))
    except Exception as exc:
        market_result = fetch_market_data(cache_path=storage.raw_market_dir / "etf_159325_akshare_raw.csv")
        market = market_result.frame
        errors.append(f"Market data service recovered with continuity data: {exc}")

    features = calculate_technical_features(market)
    sentiment = generate_sentiment_features(features)
    states_with, decisions_with = generate_rl_outputs(features, sentiment, with_sentiment=True)
    states_without, decisions_without = generate_rl_outputs(features, sentiment, with_sentiment=False)
    portfolio_with, metrics_with = run_backtest(features, decisions_with, with_sentiment=True)
    portfolio_without, metrics_without = run_backtest(features, decisions_without, with_sentiment=False)
    benchmark = buy_and_hold_series(features)
    metrics_benchmark = buy_and_hold_metrics(features)
    ablation = build_ablation_results(metrics_with, metrics_without, metrics_benchmark)

    states = pd.concat([states_with, states_without], ignore_index=True)
    decisions = pd.concat([decisions_with, decisions_without], ignore_index=True)
    portfolios = pd.concat([portfolio_with, portfolio_without], ignore_index=True)
    metrics = pd.DataFrame([metrics_with, metrics_without, metrics_benchmark])

    _persist_outputs(storage, features, sentiment, states, decisions, portfolios, metrics, ablation)

    record_counts = {
        "market_data": len(market),
        "technical_features": len(features),
        "sentiment_features": len(sentiment),
        "rl_states": len(states),
        "agent_decisions": len(decisions),
        "portfolio_snapshots": len(portfolios),
        "performance_metrics": len(metrics),
        "ablation_results": len(ablation),
    }
    health = build_pipeline_health(
        market_status=market_result.source_status,
        database_status=storage.database_status(),
        record_counts=record_counts,
    )
    latest = _latest_context(features, sentiment, states_with, decisions_with, portfolio_with, metrics_with)
    latest["market_data_mode"] = market_result.source_status
    latest["trading_calendar"] = "A-share trading dates"
    latest["latest_trading_day"] = str(market["date"].iloc[-1])
    latest["dqn_sentiment_return"] = _fmt_pct(metrics_with["cumulative_return"])
    latest["dqn_technical_return"] = _fmt_pct(metrics_without["cumulative_return"])
    latest["benchmark_return"] = _fmt_pct(metrics_benchmark["cumulative_return"])
    charts = _build_charts(features, sentiment, decisions_with, portfolio_with, portfolio_without, benchmark, ablation)

    return {
        "platform_title": "NLP-RL Quantitative Trading Platform",
        "subtitle": "Southern Semiconductor ETF Strategy Workstation",
        "target_asset": f"{TARGET_NAME} {TARGET_SYMBOL}",
        "date_range": f"{START_DATE} to latest trading day {market['date'].iloc[-1]}",
        "market_status": f"Market data source status: {market_result.source_status}",
        "market_status_class": "ok" if market_result.source_status == "AkShare connected" else "warn",
        "data_source_detail": market_result.message,
        "database_status": f"Database: {storage.database_status()}",
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cards": _overview_cards(latest),
        "overview_primary": _overview_primary(latest),
        "overview_flow": _overview_flow(latest),
        "overview_watchlist": _overview_watchlist(latest),
        "latest": latest,
        "charts": charts,
        "recent_ohlcv": _records(market, ["date", "open", "high", "low", "close", "volume", "amount"]),
        "technical_rows": _metric_rows(
            features.tail(1).iloc[0],
            ["MA5", "MA20", "MA50", "MA200", "RSI", "MACD", "MACD_signal", "MACD_histogram", "volatility_5d", "volatility_20d", "volume_ratio_5d", "volume_ratio_20d"],
        ),
        "sentiment_rows": _metric_rows(
            sentiment.tail(1).iloc[0],
            ["positive_probability", "neutral_probability", "negative_probability", "sentiment_score", "sentiment_label", "confidence", "signal_strength", "sentiment_regime", "sentiment_score_lag1", "sentiment_score_lag3_mean", "sentiment_score_lag5_mean"],
        ),
        "state_rows": _metric_rows(
            states_with.tail(1).iloc[0],
            ["price", "ma5", "ma20", "rsi", "macd", "position", "cash", "sentiment_score"],
        ),
        "rl_state_records": _state_records(states_with),
        "selected_state_date": states_with.tail(1).iloc[0]["date"],
        "decision_rows": _records(decisions_with.tail(10), ["date", "action", "q_buy", "q_sell", "q_hold", "position", "cash", "target_position", "order_quantity", "epsilon"]),
        "performance_rows": _records(metrics, ["run_name", "cumulative_return", "annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown", "win_rate", "n_trades", "profit_factor", "benchmark_cumulative_return", "excess_return"]),
        "ablation_rows": _records(ablation, ["model_name", "cumulative_return", "annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown", "win_rate", "n_trades", "profit_factor"]),
        "health_rows": health,
        "ablation_conclusion": "In this 2026 evaluation window, buy-and-hold leads the return ranking, while the sentiment-enhanced DQN underperforms the technical-only DQN, suggesting that sentiment signals may add noise during a structural adjustment phase.",
        "system_errors": errors,
    }


def _persist_outputs(
    storage: StorageService,
    features: pd.DataFrame,
    sentiment: pd.DataFrame,
    states: pd.DataFrame,
    decisions: pd.DataFrame,
    portfolios: pd.DataFrame,
    metrics: pd.DataFrame,
    ablation: pd.DataFrame,
) -> None:
    storage.save_technical_features(features)
    storage.log_system("technical_features", "Generated", "Technical feature layer generated.", len(features))
    storage.save_sentiment_features(sentiment)
    storage.log_system("sentiment_features", "Generated", "FinBERT-compatible sentiment feature layer generated.", len(sentiment))
    storage.save_rl_states(states)
    storage.log_system("rl_states", "Generated", "Sentiment-enhanced RL state vectors generated.", len(states))
    storage.save_agent_decisions(decisions)
    storage.log_system("agent_decisions", "Stored", "DQN decision records stored.", len(decisions))
    storage.save_portfolio_snapshots(portfolios)
    storage.log_system("portfolio_snapshots", "Stored", "Portfolio backtest snapshots stored.", len(portfolios))
    storage.save_performance_metrics(metrics)
    storage.log_system("performance_metrics", "Stored", "Strategy performance metrics stored.", len(metrics))
    storage.save_ablation_results(ablation)
    storage.log_system("ablation_results", "Stored", "Ablation study results stored.", len(ablation))


def _latest_context(
    features: pd.DataFrame,
    sentiment: pd.DataFrame,
    states: pd.DataFrame,
    decisions: pd.DataFrame,
    portfolio: pd.DataFrame,
    metrics: dict,
) -> dict[str, Any]:
    f = features.iloc[-1]
    s = sentiment.iloc[-1]
    state = states.iloc[-1]
    decision = decisions.iloc[-1]
    p = portfolio.iloc[-1]
    return {
        "close": _fmt_number(f["close"], 4),
        "daily_return": _fmt_pct(f["daily_return"]),
        "daily_return_raw": float(f["daily_return"]),
        "cumulative_return": _fmt_pct(f["cumulative_return"]),
        "sentiment_score": _fmt_number(s["sentiment_score"], 3),
        "sentiment_regime": s["sentiment_regime"],
        "sentiment_label": s["sentiment_label"],
        "rl_action": decision["action"],
        "portfolio_value": _fmt_currency(p["portfolio_value"]),
        "sharpe_ratio": _fmt_number(metrics["sharpe_ratio"], 2),
        "max_drawdown": _fmt_pct(metrics["max_drawdown"]),
        "system_status": "Operational",
        "epsilon": _fmt_number(decision["epsilon"], 3),
        "position": _fmt_number(state["position"], 0),
        "cash": _fmt_currency(state["cash"]),
    }


def _overview_cards(latest: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"label": "Market Data Mode", "value": latest["market_data_mode"], "tone": "positive" if latest["market_data_mode"] == "AkShare connected" else "negative"},
        {"label": "Trading Calendar", "value": latest["trading_calendar"], "tone": "info"},
        {"label": "Latest Trading Day", "value": latest["latest_trading_day"], "tone": "neutral"},
        {"label": "Latest Close Price", "value": latest["close"], "tone": "neutral"},
        {"label": "Daily Return", "value": latest["daily_return"], "tone": "positive" if latest["daily_return_raw"] >= 0 else "negative"},
        {"label": "ETF Cumulative Return", "value": latest["cumulative_return"], "tone": "neutral"},
        {"label": "DQN Sentiment Return", "value": latest["dqn_sentiment_return"], "tone": "negative" if latest["dqn_sentiment_return"].startswith("-") else "positive"},
        {"label": "DQN Technical Return", "value": latest["dqn_technical_return"], "tone": "negative" if latest["dqn_technical_return"].startswith("-") else "positive"},
        {"label": "Latest Sentiment Score", "value": latest["sentiment_score"], "tone": "neutral"},
        {"label": "Latest Sentiment Regime", "value": latest["sentiment_regime"], "tone": "neutral"},
        {"label": "Latest RL Action", "value": latest["rl_action"], "tone": "info"},
        {"label": "Portfolio Value", "value": latest["portfolio_value"], "tone": "neutral"},
        {"label": "Sharpe Ratio", "value": latest["sharpe_ratio"], "tone": "neutral"},
        {"label": "Max Drawdown", "value": latest["max_drawdown"], "tone": "negative"},
        {"label": "System Status", "value": latest["system_status"], "tone": "positive"},
    ]


def _overview_primary(latest: dict[str, Any]) -> dict[str, str]:
    return {
        "mode": latest["market_data_mode"],
        "date": latest["latest_trading_day"],
        "close": latest["close"],
        "action": latest["rl_action"],
        "portfolio": latest["portfolio_value"],
        "etf_return": latest["cumulative_return"],
        "strategy_return": latest["dqn_sentiment_return"],
    }


def _overview_flow(latest: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"label": "Bronze Layer", "title": "Market Data", "value": latest["market_data_mode"]},
        {"label": "Silver Layer", "title": "Technical Features", "value": latest["close"]},
        {"label": "Gold Layer", "title": "RL Decision", "value": latest["rl_action"]},
        {"label": "Portfolio Layer", "title": "Backtest Value", "value": latest["portfolio_value"]},
    ]


def _overview_watchlist(latest: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"label": "Daily Return", "value": latest["daily_return"]},
        {"label": "Sentiment Score", "value": latest["sentiment_score"]},
        {"label": "Sentiment Regime", "value": latest["sentiment_regime"]},
        {"label": "Sharpe Ratio", "value": latest["sharpe_ratio"]},
        {"label": "Max Drawdown", "value": latest["max_drawdown"]},
        {"label": "System Status", "value": latest["system_status"]},
    ]


def _build_charts(
    features: pd.DataFrame,
    sentiment: pd.DataFrame,
    decisions: pd.DataFrame,
    portfolio_with: pd.DataFrame,
    portfolio_without: pd.DataFrame,
    benchmark: pd.DataFrame,
    ablation: pd.DataFrame,
) -> dict[str, str]:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        charts: dict[str, str] = {}
        first = True

        price = go.Figure()
        price.add_trace(go.Scatter(x=features["date"], y=features["close"], name="Close", line={"color": "#E5E7EB", "width": 2}))
        price.add_trace(go.Scatter(x=features["date"], y=features["MA5"], name="MA5", line={"color": "#22C55E", "width": 1.5}))
        price.add_trace(go.Scatter(x=features["date"], y=features["MA20"], name="MA20", line={"color": "#60A5FA", "width": 1.5}))
        charts["price"] = _fig_html(price, "Close Price with MA5 / MA20", first)
        first = False

        volume = go.Figure(go.Bar(x=features["date"], y=features["volume"], name="Volume", marker_color="#64748B"))
        charts["volume"] = _fig_html(volume, "ETF Volume", first)

        rsi = go.Figure()
        rsi.add_trace(go.Scatter(x=features["date"], y=features["RSI"], name="RSI14", line={"color": "#F59E0B"}))
        rsi.add_hline(y=70, line_dash="dot", line_color="#EF4444")
        rsi.add_hline(y=30, line_dash="dot", line_color="#22C55E")
        charts["rsi"] = _fig_html(rsi, "RSI", first)

        macd = go.Figure()
        macd.add_trace(go.Scatter(x=features["date"], y=features["MACD"], name="MACD", line={"color": "#60A5FA"}))
        macd.add_trace(go.Scatter(x=features["date"], y=features["MACD_signal"], name="Signal", line={"color": "#F59E0B"}))
        macd.add_trace(go.Bar(x=features["date"], y=features["MACD_histogram"], name="Histogram", marker_color="#64748B"))
        charts["macd"] = _fig_html(macd, "MACD", first)

        sent = go.Figure()
        sent.add_trace(go.Scatter(x=sentiment["date"], y=sentiment["sentiment_score"], name="Daily Sentiment Score", line={"color": "#22C55E"}))
        sent.add_hline(y=0.25, line_dash="dot", line_color="#22C55E")
        sent.add_hline(y=-0.25, line_dash="dot", line_color="#EF4444")
        charts["sentiment"] = _fig_html(sent, "FinBERT-Compatible Sentiment Signal", first)

        action_colors = decisions["action"].map({"Buy": "#22C55E", "Sell": "#EF4444", "Hold": "#9CA3AF"}).fillna("#9CA3AF")
        actions = go.Figure(go.Scatter(x=decisions["date"], y=decisions["action_id"], mode="markers", name="Action", marker={"color": action_colors, "size": 8}))
        actions.update_yaxes(tickmode="array", tickvals=[0, 1, 2], ticktext=["Hold", "Buy", "Sell"])
        charts["actions"] = _fig_html(actions, "Buy / Sell / Hold Decision Timeline", first, div_id="rl-action-timeline")

        port = go.Figure()
        port.add_trace(go.Scatter(x=portfolio_with["date"], y=portfolio_with["portfolio_value"], name="DQN with Sentiment Feature", line={"color": "#22C55E"}))
        port.add_trace(go.Scatter(x=portfolio_without["date"], y=portfolio_without["portfolio_value"], name="DQN without Sentiment Feature", line={"color": "#60A5FA"}))
        port.add_trace(go.Scatter(x=benchmark["date"], y=benchmark["portfolio_value"], name="Buy-and-Hold Benchmark", line={"color": "#9CA3AF"}))
        charts["portfolio"] = _fig_html(port, "Portfolio Value vs Benchmark", first)

        dd = go.Figure()
        dd.add_trace(go.Scatter(x=portfolio_with["date"], y=portfolio_with["current_drawdown"], name="Drawdown", fill="tozeroy", line={"color": "#EF4444"}))
        charts["drawdown"] = _fig_html(dd, "Drawdown", first)

        ab = make_subplots(rows=1, cols=3, subplot_titles=("Cumulative Return", "Sharpe Ratio", "Max Drawdown"))
        ab.add_trace(go.Bar(x=ablation["model_name"], y=ablation["cumulative_return"], marker_color="#22C55E", showlegend=False), row=1, col=1)
        ab.add_trace(go.Bar(x=ablation["model_name"], y=ablation["sharpe_ratio"], marker_color="#60A5FA", showlegend=False), row=1, col=2)
        ab.add_trace(go.Bar(x=ablation["model_name"], y=ablation["max_drawdown"], marker_color="#EF4444", showlegend=False), row=1, col=3)
        charts["ablation"] = _fig_html(ab, "Ablation Comparison", first)

        cash_value = go.Figure()
        cash_value.add_trace(go.Scatter(x=portfolio_with["date"], y=portfolio_with["cash"], name="Cash", line={"color": "#F59E0B"}))
        cash_value.add_trace(go.Scatter(x=portfolio_with["date"], y=portfolio_with["market_value"], name="Market Value", line={"color": "#22C55E"}))
        cash_value.add_trace(go.Scatter(x=portfolio_with["date"], y=portfolio_with["position_weight"], name="Position Weight", yaxis="y2", line={"color": "#60A5FA"}))
        cash_value.update_layout(yaxis2={"overlaying": "y", "side": "right", "tickformat": ".0%"})
        charts["cash_value"] = _fig_html(cash_value, "Cash, Market Value, and Position Weight", first)
        return charts
    except Exception as exc:
        return {"price": f"<div class='chart-error'>Chart rendering unavailable: {exc}</div>"}


def _fig_html(fig: Any, title: str, include_plotlyjs: bool, div_id: str | None = None) -> str:
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#182230",
        plot_bgcolor="#111827",
        font={"color": "#E5E7EB", "family": "Inter, IBM Plex Sans, system-ui, sans-serif", "size": 11},
        margin={"l": 44, "r": 24, "t": 44, "b": 36},
        height=320,
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    fig.update_xaxes(gridcolor="#374151", zerolinecolor="#374151")
    fig.update_yaxes(gridcolor="#374151", zerolinecolor="#374151")
    kwargs = {"div_id": div_id} if div_id else {}
    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs, config={"displayModeBar": False, "responsive": True}, **kwargs)


def _records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    rows = []
    for _, row in frame[columns].iterrows():
        item = {}
        for col in columns:
            value = row[col]
            item[col] = _fmt_cell(value)
        rows.append(item)
    return rows


def _metric_rows(row: pd.Series, columns: list[str]) -> list[dict[str, str]]:
    return [{"metric": col.replace("_", " "), "value": _fmt_cell(row[col])} for col in columns if col in row.index]


def _state_records(states: pd.DataFrame) -> dict[str, dict[str, str]]:
    columns = ["price", "ma5", "ma20", "rsi", "macd", "position", "cash", "sentiment_score"]
    records: dict[str, dict[str, str]] = {}
    for _, row in states.iterrows():
        records[str(row["date"])] = {col: _fmt_cell(row[col]) for col in columns}
    return records


def _fmt_cell(value: Any) -> str:
    if isinstance(value, float):
        if abs(value) < 0.0001 and value != 0:
            return f"{value:.2e}"
        return f"{value:,.4f}"
    return str(value)


def _fmt_number(value: Any, decimals: int = 2) -> str:
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:,.2f}%"
    except (TypeError, ValueError):
        return "0.00%"


def _fmt_currency(value: Any) -> str:
    try:
        return f"CNY {float(value):,.2f}"
    except (TypeError, ValueError):
        return "CNY 0.00"


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=os.getenv("FLASK_DEBUG", "0") == "1")
