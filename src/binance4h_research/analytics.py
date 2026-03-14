from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .config import BacktestConfig, CostConfig, ExperimentConfig
from .backtest import BacktestArtifacts, reprice_pnl


def _safe_stat(value: float) -> float:
    if pd.isna(value) or math.isinf(value):
        return 0.0
    return float(value)


def summarize_artifacts(artifacts: BacktestArtifacts, backtest: BacktestConfig) -> dict[str, float]:
    pnl = artifacts.pnl["net_return"].dropna()
    if pnl.empty:
        return {
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "avg_turnover": 0.0,
            "bars": 0.0,
        }

    periods_per_year = int((24 / backtest.bar_hours) * 365)
    equity = artifacts.equity_curve.dropna()
    total_return = equity.iloc[-1] - 1.0
    years = max(len(pnl) / periods_per_year, 1 / periods_per_year)
    annual_return = (1.0 + total_return) ** (1.0 / years) - 1.0 if equity.iloc[-1] > 0 else -1.0
    annual_vol = pnl.std(ddof=0) * math.sqrt(periods_per_year)
    sharpe = pnl.mean() / pnl.std(ddof=0) * math.sqrt(periods_per_year) if pnl.std(ddof=0) > 0 else 0.0
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = drawdown.min()
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
    return {
        "annual_return": _safe_stat(annual_return),
        "annual_volatility": _safe_stat(annual_vol),
        "sharpe": _safe_stat(sharpe),
        "max_drawdown": _safe_stat(max_drawdown),
        "calmar": _safe_stat(calmar),
        "avg_turnover": _safe_stat(artifacts.pnl["turnover"].mean()),
        "gross_total_return": _safe_stat((1.0 + artifacts.pnl["gross_return"].fillna(0.0)).prod() - 1.0),
        "net_total_return": _safe_stat(total_return),
        "bars": float(len(pnl)),
    }


def yearly_breakdown(artifacts: BacktestArtifacts) -> pd.DataFrame:
    frame = artifacts.pnl.copy()
    if frame.empty:
        return pd.DataFrame()
    frame["year"] = frame.index.year
    grouped = frame.groupby("year")[["gross_return", "trading_cost", "funding_return", "net_return"]].sum()
    grouped["avg_turnover"] = frame.groupby("year")["turnover"].mean()
    return grouped


def long_short_contribution(artifacts: BacktestArtifacts) -> pd.DataFrame:
    price = artifacts.interval_returns.reindex_like(artifacts.weights).fillna(0.0)
    long_weights = artifacts.weights.clip(lower=0.0)
    short_weights = artifacts.weights.clip(upper=0.0)
    long_contrib = (long_weights * price).sum(axis=1)
    short_contrib = (short_weights * price).sum(axis=1)
    return pd.DataFrame({"long_price_return": long_contrib, "short_price_return": short_contrib}, index=artifacts.weights.index)


def pressure_test(
    artifacts: BacktestArtifacts,
    config: ExperimentConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fee_mult in config.report.pressure_fee_multipliers:
        for slip_mult in config.report.pressure_slippage_multipliers:
            costs = CostConfig(
                fee_bps=config.costs.fee_bps * fee_mult,
                slippage_bps=config.costs.slippage_bps * slip_mult,
            )
            pnl, equity_curve = reprice_pnl(
                price_component=artifacts.pnl["price_return"],
                funding_component=artifacts.pnl["funding_return"],
                traded_notional=artifacts.pnl["turnover"],
                costs=costs,
            )
            repriced = BacktestArtifacts(
                summary=pnl.copy(),
                weights=artifacts.weights,
                interval_returns=artifacts.interval_returns,
                pnl=pnl,
                equity_curve=equity_curve,
            )
            stats = summarize_artifacts(repriced, config.backtest)
            stats["fee_mult"] = fee_mult
            stats["slippage_mult"] = slip_mult
            rows.append(stats)
    return pd.DataFrame(rows)
