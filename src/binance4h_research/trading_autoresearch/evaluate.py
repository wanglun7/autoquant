from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import math
from pathlib import Path

import pandas as pd

from ..analytics import summarize_artifacts, yearly_breakdown
from ..backtest import BacktestArtifacts, run_backtest
from ..config import BacktestConfig, CostConfig
from .prepare_market import build_context
from .program import TradingAutoResearchProgram


@dataclass(slots=True)
class TradingEvaluation:
    strategy_id: str
    family: str
    summary: dict[str, float | int | str | list[str]]
    splits: dict[str, dict[str, float]]
    walk_forward: list[dict[str, float | str]]
    primary_score: float
    status: str
    reason_tags: list[str]


def _annualized_stats(net_returns: pd.Series, bar_hours: int) -> dict[str, float]:
    pnl = net_returns.dropna()
    if pnl.empty:
        return {"annual_return": 0.0, "sharpe": 0.0, "net_total_return": 0.0, "max_drawdown": 0.0}
    periods_per_year = int((24 / bar_hours) * 365)
    equity = (1.0 + pnl.fillna(0.0)).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    years = max(len(pnl) / periods_per_year, 1 / periods_per_year)
    annual_return = (1.0 + total_return) ** (1.0 / years) - 1.0 if equity.iloc[-1] > 0 else -1.0
    volatility = pnl.std(ddof=0)
    sharpe = pnl.mean() / volatility * math.sqrt(periods_per_year) if volatility > 0 else 0.0
    max_drawdown = float((equity / equity.cummax() - 1.0).min())
    return {
        "annual_return": float(annual_return),
        "sharpe": float(sharpe),
        "net_total_return": total_return,
        "max_drawdown": max_drawdown,
    }


def _split_ranges(index: pd.DatetimeIndex, program: TradingAutoResearchProgram) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    n = len(index)
    train_end = max(1, int(n * program.split.train_frac))
    val_end = max(train_end + 1, int(n * (program.split.train_frac + program.split.validation_frac)))
    return {
        "train": (index[0], index[train_end - 1]),
        "validation": (index[train_end], index[val_end - 1]),
        "test": (index[val_end], index[-1]),
    }


def _slice_pnl(artifacts: BacktestArtifacts, start: pd.Timestamp, end: pd.Timestamp, bar_hours: int) -> dict[str, float]:
    pnl = artifacts.pnl.loc[(artifacts.pnl.index >= start) & (artifacts.pnl.index <= end), "net_return"]
    return _annualized_stats(pnl, bar_hours)


def _walk_forward_rows(artifacts: BacktestArtifacts, program: TradingAutoResearchProgram) -> list[dict[str, float | str]]:
    index = artifacts.pnl.index
    if len(index) < 12:
        return []
    fold = len(index) // 4
    rows: list[dict[str, float | str]] = []
    for idx in range(3):
        start = index[idx * fold]
        end = index[min(len(index) - 1, (idx + 1) * fold + fold - 1)]
        stats = _slice_pnl(artifacts, start, end, program.bar_hours)
        stats["fold"] = f"wf_{idx + 1}"
        rows.append(stats)
    return rows


def _reason_tags(summary: dict[str, float], splits: dict[str, dict[str, float]], stress_2x: dict[str, float]) -> list[str]:
    tags: list[str] = []
    if splits["test"]["net_total_return"] <= 0.0 and summary["gross_total_return"] > 0.0:
        tags.append("too_costly")
    if splits["train"]["net_total_return"] > 0.0 and splits["test"]["net_total_return"] <= 0.0:
        tags.append("overfit")
    if float(stress_2x["net_total_return"]) <= 0.0:
        tags.append("unstable_under_2x_cost")
    if splits["test"]["max_drawdown"] < -0.60:
        tags.append("high_drawdown")
    if splits["test"]["net_total_return"] <= 0.0:
        tags.append("insufficient_oos")
    return tags


def _load_strategy_module():
    from . import strategy as strategy_module

    return importlib.reload(strategy_module)


def evaluate_current_strategy(program: TradingAutoResearchProgram) -> TradingEvaluation:
    context = build_context(program)
    strategy_module = _load_strategy_module()
    weights, metadata = strategy_module.build_weights(context)
    artifacts = run_backtest(weights=weights, opens=context.opens, funding=context.funding, costs=program.costs)
    summary = summarize_artifacts(artifacts, BacktestConfig(bar_hours=program.bar_hours, rebalance_every_bars=1))
    split_ranges = _split_ranges(artifacts.pnl.index, program)
    splits = {name: _slice_pnl(artifacts, start, end, program.bar_hours) for name, (start, end) in split_ranges.items()}
    walk_forward = _walk_forward_rows(artifacts, program)
    stress_artifacts = run_backtest(
        weights=weights,
        opens=context.opens,
        funding=context.funding,
        costs=CostConfig(fee_bps=program.costs.fee_bps * 2, slippage_bps=program.costs.slippage_bps * 2),
    )
    stress_2x = summarize_artifacts(stress_artifacts, BacktestConfig(bar_hours=program.bar_hours, rebalance_every_bars=1))
    yearly = yearly_breakdown(artifacts)
    reason_tags = _reason_tags(summary, splits, stress_2x)
    primary_score = float(splits["test"]["sharpe"])
    status = "keep"
    if (
        splits["test"]["net_total_return"] <= program.constraints.min_test_net_return
        or splits["test"]["sharpe"] < program.constraints.min_test_sharpe
        or splits["test"]["max_drawdown"] < -program.constraints.max_drawdown_limit
        or (program.constraints.require_positive_under_2x_cost and float(stress_2x["net_total_return"]) <= 0.0)
    ):
        status = "reject"
    payload_summary = {
        **summary,
        "stress_2x_net_total_return": float(stress_2x["net_total_return"]),
        "stress_2x_sharpe": float(stress_2x["sharpe"]),
        "yearly_positive_count": int((yearly["net_return"] > 0).sum()) if not yearly.empty else 0,
        "reason_tags": reason_tags,
    }
    return TradingEvaluation(
        strategy_id=str(metadata.get("strategy_id", "unknown_strategy")),
        family=str(metadata.get("family", "unknown_family")),
        summary=payload_summary,
        splits=splits,
        walk_forward=walk_forward,
        primary_score=primary_score,
        status=status,
        reason_tags=reason_tags,
    )
