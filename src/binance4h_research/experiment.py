from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .analytics import long_short_contribution, pressure_test, summarize_artifacts, yearly_breakdown
from .backtest import funding_returns_from_events, run_backtest
from .config import ExperimentConfig
from .data import load_symbol_funding, load_symbol_klines
from .portfolio import build_market_neutral_weights, rebalance_on_schedule
from .signals import build_close_matrix, build_open_matrix, cross_sectional_momentum_signal
from .universe import build_universe_membership, save_universe_membership


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _unlink_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def run_experiment(config: ExperimentConfig) -> dict[str, Path]:
    config.experiment_dir.mkdir(parents=True, exist_ok=True)

    klines = load_symbol_klines(config.data_dir)
    if not klines:
        raise FileNotFoundError(f"No kline data found in {config.data_dir / 'klines'}")
    funding_raw = load_symbol_funding(config.data_dir)

    membership = build_universe_membership(klines, config)
    close_matrix = build_close_matrix(klines)
    open_matrix = build_open_matrix(klines)
    signal = cross_sectional_momentum_signal(close_matrix, config.signal.lookback_bars)
    weights = build_market_neutral_weights(signal, membership, config.portfolio)
    weights = rebalance_on_schedule(weights, config.backtest.rebalance_every_bars)
    funding = funding_returns_from_events(funding_raw, open_matrix.index)
    funding = funding.reindex(index=open_matrix.index, columns=open_matrix.columns, fill_value=0.0)

    artifacts = run_backtest(weights=weights, opens=open_matrix, funding=funding, costs=config.costs)
    summary = summarize_artifacts(artifacts, config.backtest)
    yearly = yearly_breakdown(artifacts)
    l_s = long_short_contribution(artifacts)
    pressure = pressure_test(artifacts=artifacts, config=config)

    paths = {
        "universe": config.processed_dir / f"{config.name}_universe.csv",
        "weights": config.experiment_dir / "weights.csv",
        "pnl": config.experiment_dir / "pnl.csv",
        "summary": config.experiment_dir / "summary.json",
        "yearly": config.experiment_dir / "yearly.csv",
        "long_short": config.experiment_dir / "long_short.csv",
        "pressure": config.experiment_dir / "pressure.csv",
    }

    if config.outputs.write_universe:
        save_universe_membership(membership, config.processed_dir, config.name)
    else:
        _unlink_if_exists(paths["universe"])

    if config.outputs.write_weights:
        artifacts.weights.to_csv(paths["weights"], index_label="timestamp")
    else:
        _unlink_if_exists(paths["weights"])

    if config.outputs.write_pnl:
        artifacts.pnl.to_csv(paths["pnl"], index_label="timestamp")
    else:
        _unlink_if_exists(paths["pnl"])

    if not yearly.empty:
        yearly.to_csv(paths["yearly"], index_label="year")
    else:
        pd.DataFrame().to_csv(paths["yearly"], index=False)
    l_s.to_csv(paths["long_short"], index_label="timestamp")
    if config.outputs.write_pressure:
        pressure.to_csv(paths["pressure"], index=False)
    else:
        _unlink_if_exists(paths["pressure"])
    _write_json(paths["summary"], summary)
    return paths


def render_report(results_dir: Path) -> str:
    summary_path = results_dir / "summary.json"
    pnl_path = results_dir / "pnl.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    lines = [
        f"Experiment: {results_dir.name}",
        f"Annual Return: {summary.get('annual_return', 0.0):.2%}",
        f"Sharpe: {summary.get('sharpe', 0.0):.3f}",
        f"Max Drawdown: {summary.get('max_drawdown', 0.0):.2%}",
        f"Average Turnover: {summary.get('avg_turnover', 0.0):.3f}",
        f"Bars: {int(summary.get('bars', 0))}",
    ]
    if pnl_path.exists():
        pnl = pd.read_csv(pnl_path)
        lines.append(f"Net bars with PnL rows: {len(pnl)}")
    return "\n".join(lines)
