from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd

from ..analytics import summarize_artifacts, yearly_breakdown
from ..backtest import BacktestArtifacts, run_backtest
from ..config import BacktestConfig
from ..portfolio import build_market_neutral_weights, rebalance_on_schedule
from .context import ResearchContext
from .program import AutoEvolveProgram
from .spec import StrategySpec


@dataclass(slots=True)
class CandidateEvaluation:
    spec: StrategySpec
    spec_hash: str
    summary: dict[str, float | str | list[str]]
    splits: dict[str, dict[str, float]]
    walk_forward: list[dict[str, float | str]]
    status: str
    reason_tags: list[str]
    primary_score: float


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


def _split_ranges(index: pd.DatetimeIndex, program: AutoEvolveProgram) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    n = len(index)
    train_end = max(1, int(n * program.split.train_frac))
    val_end = max(train_end + 1, int(n * (program.split.train_frac + program.split.validation_frac)))
    train = (index[0], index[train_end - 1])
    validation = (index[train_end], index[val_end - 1])
    test = (index[val_end], index[-1])
    return {"train": train, "validation": validation, "test": test}


def _slice_pnl(artifacts: BacktestArtifacts, start: pd.Timestamp, end: pd.Timestamp, bar_hours: int) -> dict[str, float]:
    pnl = artifacts.pnl.loc[(artifacts.pnl.index >= start) & (artifacts.pnl.index <= end), "net_return"]
    return _annualized_stats(pnl, bar_hours)


def _walk_forward_rows(artifacts: BacktestArtifacts, program: AutoEvolveProgram) -> list[dict[str, float | str]]:
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


def _rebalance_every_bars(freq: str, bar_hours: int) -> int:
    mapping = {"4h": 1, "1d": int(24 / bar_hours), "1w": int((24 / bar_hours) * 7)}
    return mapping.get(freq, 1)


def _btc_regime_gate(context: ResearchContext, lookback_days: int = 90) -> pd.Series:
    bars = lookback_days * 6
    btc = context.closes[context.btc_symbol]
    return btc.shift(1).div(btc.shift(1 + bars)).sub(1.0).gt(0).astype(float).reindex(context.closes.index).fillna(0.0)


def _apply_filters(weights: pd.DataFrame, spec: StrategySpec, context: ResearchContext) -> pd.DataFrame:
    filtered = weights.copy()
    if spec.filters.regime == "btc_positive_trend":
        gate = _btc_regime_gate(context)
        filtered = filtered.mul(gate, axis=0)
    if spec.filters.funding_cap is not None:
        long_mask = filtered > 0
        short_mask = filtered < 0
        filtered = filtered.mask(long_mask & context.funding.gt(spec.filters.funding_cap), 0.0)
        filtered = filtered.mask(short_mask & context.funding.lt(-spec.filters.funding_cap), 0.0)
    if spec.filters.volatility_cap is not None:
        vol = context.closes.pct_change(fill_method=None).rolling(30).std().shift(1)
        filtered = filtered.mask(vol.gt(spec.filters.volatility_cap), 0.0)
    if spec.filters.liquidity_bucket == "large_liquid_only":
        top = context.liquidity.rank(axis=1, ascending=False, method="first").le(20)
        filtered = filtered.where(top, 0.0)
    return filtered


def _cross_sectional_weights(spec: StrategySpec, context: ResearchContext, bar_hours: int) -> pd.DataFrame:
    lookback_days = int(spec.params.get("lookback_days", 120))
    skip_days = int(spec.params.get("skip_days", 0))
    top_n = int(spec.params.get("top_n", 50))
    bars_per_day = int(24 / bar_hours)
    lookback_bars = lookback_days * bars_per_day
    skip_bars = skip_days * bars_per_day
    signal = context.closes.shift(1 + skip_bars).div(context.closes.shift(1 + skip_bars + lookback_bars)).sub(1.0)
    if spec.params.get("rank_mode") == "vol_adjusted":
        vol = context.closes.pct_change(fill_method=None).rolling(lookback_bars).std().shift(1)
        signal = signal.div(vol.replace(0.0, pd.NA))
    eligible = context.liquidity.rank(axis=1, ascending=False, method="first").le(top_n)
    weights = build_market_neutral_weights(
        signal,
        eligible.fillna(False),
        type("Port", (), {
            "long_quantile": spec.portfolio.long_quantile,
            "short_quantile": spec.portfolio.short_quantile,
            "gross_exposure": spec.portfolio.gross,
        })(),
    )
    return rebalance_on_schedule(weights, _rebalance_every_bars(spec.portfolio.rebalance, bar_hours))


def _btc_time_series_weights(spec: StrategySpec, context: ResearchContext, bar_hours: int) -> pd.DataFrame:
    close = context.closes[context.btc_symbol]
    weights = pd.DataFrame(0.0, index=context.closes.index, columns=context.closes.columns)
    lookback_days = int(spec.params.get("lookback_days", 60))
    bars_per_day = int(24 / bar_hours)
    bars = lookback_days * bars_per_day
    if spec.model == "momentum":
        score = close.shift(1).div(close.shift(1 + bars)).sub(1.0)
        signal = score.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
    elif spec.model == "breakout":
        rolling_max = close.shift(1).rolling(bars).max()
        rolling_min = close.shift(1).rolling(bars).min()
        signal = pd.Series(0.0, index=close.index)
        signal = signal.mask(close.shift(1) > rolling_max, 1.0)
        signal = signal.mask(close.shift(1) < rolling_min, -1.0)
        signal = signal.fillna(0.0)
    elif spec.model == "mean_reversion":
        mean = close.shift(1).rolling(bars).mean()
        std = close.shift(1).rolling(bars).std().replace(0.0, pd.NA)
        z = close.shift(1).sub(mean).div(std)
        signal = z.apply(lambda x: -1.0 if x > 1 else (1.0 if x < -1 else 0.0))
    else:
        score = close.shift(1).div(close.shift(1 + bars)).sub(1.0)
        vol = close.pct_change(fill_method=None).rolling(bars).std().shift(1).replace(0.0, pd.NA)
        target = pd.Series(1.0, index=close.index).div(vol).clip(upper=1.0).fillna(0.0)
        signal = score.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)).mul(target)
    if spec.portfolio.direction_mode == "long_only":
        signal = signal.clip(lower=0.0)
    weights[context.btc_symbol] = signal * spec.portfolio.gross
    return rebalance_on_schedule(weights, _rebalance_every_bars(spec.portfolio.rebalance, bar_hours))


def _relative_value_weights(spec: StrategySpec, context: ResearchContext, bar_hours: int) -> pd.DataFrame:
    left = str(spec.params["left"])
    right = str(spec.params["right"])
    lookback_days = int(spec.params.get("lookback_days", 60))
    bars_per_day = int(24 / bar_hours)
    bars = lookback_days * bars_per_day
    weights = pd.DataFrame(0.0, index=context.closes.index, columns=context.closes.columns)
    spread = (context.closes[left].apply(math.log) - context.closes[right].apply(math.log)).replace([math.inf, -math.inf], pd.NA)
    if spec.model == "spread_momentum":
        signal = spread.shift(1).sub(spread.shift(1 + bars))
        direction = signal.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
    else:
        mean = spread.shift(1).rolling(bars).mean()
        std = spread.shift(1).rolling(bars).std().replace(0.0, pd.NA)
        zscore = spread.shift(1).sub(mean).div(std)
        entry = float(spec.params.get("z_entry", 1.0))
        direction = zscore.apply(lambda x: -1.0 if x > entry else (1.0 if x < -entry else 0.0))
    weights[left] = direction * (spec.portfolio.gross / 2.0)
    weights[right] = -direction * (spec.portfolio.gross / 2.0)
    return rebalance_on_schedule(weights, _rebalance_every_bars(spec.portfolio.rebalance, bar_hours))


def build_weights_for_spec(spec: StrategySpec, context: ResearchContext, program: AutoEvolveProgram) -> pd.DataFrame:
    if spec.family == "cross_sectional":
        weights = _cross_sectional_weights(spec, context, program.bar_hours)
    elif spec.family == "btc_time_series":
        weights = _btc_time_series_weights(spec, context, program.bar_hours)
    elif spec.family == "relative_value":
        weights = _relative_value_weights(spec, context, program.bar_hours)
    else:
        raise ValueError(f"Unsupported family: {spec.family}")
    return _apply_filters(weights, spec, context)


def _reason_tags(
    spec: StrategySpec,
    summary: dict[str, float],
    test_stats: dict[str, float],
    train_stats: dict[str, float],
    stress_2x: dict[str, float],
) -> list[str]:
    tags: list[str] = []
    if test_stats["net_total_return"] <= 0.0 and summary["gross_total_return"] > 0.0:
        tags.append("too_costly")
    if train_stats["net_total_return"] > 0.0 and test_stats["net_total_return"] <= 0.0:
        tags.append("overfit")
    if stress_2x["net_total_return"] <= 0.0:
        tags.append("unstable_under_2x_cost")
    if test_stats["max_drawdown"] < -0.60:
        tags.append("high_drawdown")
    if spec.complexity() > 5 and test_stats["sharpe"] < 0.1:
        tags.append("too_complex_for_gain")
    if test_stats["net_total_return"] <= 0.0:
        tags.append("insufficient_oos")
    return tags


def evaluate_spec(spec: StrategySpec, context: ResearchContext, program: AutoEvolveProgram) -> CandidateEvaluation:
    weights = build_weights_for_spec(spec, context, program)
    artifacts = run_backtest(weights=weights, opens=context.opens, funding=context.funding, costs=program.costs)
    summary = summarize_artifacts(artifacts, BacktestConfig(bar_hours=program.bar_hours, rebalance_every_bars=1))
    split_ranges = _split_ranges(artifacts.pnl.index, program)
    splits = {name: _slice_pnl(artifacts, start, end, program.bar_hours) for name, (start, end) in split_ranges.items()}
    walk_forward = _walk_forward_rows(artifacts, program)
    stress_artifacts = run_backtest(weights=weights, opens=context.opens, funding=context.funding, costs=type(program.costs)(fee_bps=program.costs.fee_bps * 2, slippage_bps=program.costs.slippage_bps * 2))
    stress_2x = summarize_artifacts(stress_artifacts, BacktestConfig(bar_hours=program.bar_hours, rebalance_every_bars=1))
    primary_score = float(splits["test"]["sharpe"] - program.constraints.complexity_penalty * spec.complexity())
    reason_tags = _reason_tags(spec, summary, splits["test"], splits["train"], stress_2x)
    status = "keep"
    if (
        splits["test"]["net_total_return"] <= program.constraints.min_test_net_return
        or splits["test"]["sharpe"] < program.constraints.min_test_sharpe
        or splits["test"]["max_drawdown"] < -program.constraints.max_drawdown_limit
        or float(stress_2x["net_total_return"]) <= 0.0
    ):
        status = "reject"
    payload_summary = {
        **summary,
        "stress_2x_net_total_return": float(stress_2x["net_total_return"]),
        "stress_2x_sharpe": float(stress_2x["sharpe"]),
        "yearly_positive_count": int((yearly_breakdown(artifacts)["net_return"] > 0).sum()) if not yearly_breakdown(artifacts).empty else 0,
        "reason_tags": reason_tags,
    }
    return CandidateEvaluation(
        spec=spec,
        spec_hash=spec.spec_hash(),
        summary=payload_summary,
        splits=splits,
        walk_forward=walk_forward,
        status=status,
        reason_tags=reason_tags,
        primary_score=primary_score,
    )
