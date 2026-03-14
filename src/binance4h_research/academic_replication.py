from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .academic_config import AcademicExperimentConfig, load_academic_config
from .academic_data import load_coin_series
from .academic_panel import build_weekly_panel, load_weekly_panel, save_weekly_panel


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _group_labels(size: int, scheme: Iterable[float]) -> np.ndarray:
    proportions = list(scheme)
    if size == 0:
        return np.array([], dtype=int)
    if all(abs(p - proportions[0]) < 1e-12 for p in proportions):
        buckets = np.array_split(np.arange(size), len(proportions))
    else:
        counts = [int(math.floor(size * value)) for value in proportions[:-1]]
        counts.append(size - sum(counts))
        buckets = []
        start = 0
        for count in counts:
            end = start + max(count, 0)
            buckets.append(np.arange(start, min(end, size)))
            start = end
        if start < size:
            buckets[-1] = np.concatenate([buckets[-1], np.arange(start, size)])
    labels = np.zeros(size, dtype=int)
    for index, bucket in enumerate(buckets, start=1):
        labels[bucket] = index
    return labels


def _weighted_return(frame: pd.DataFrame, return_col: str, weight_col: str | None) -> float:
    valid = frame.dropna(subset=[return_col])
    if valid.empty:
        return float("nan")
    if weight_col is None:
        return float(valid[return_col].mean())
    weights = valid[weight_col].astype(float)
    if weights.isna().all() or weights.sum() == 0:
        return float("nan")
    normalized = weights / weights.sum()
    return float((valid[return_col].astype(float) * normalized).sum())


def _summarize_factor_returns(returns: pd.Series) -> dict[str, float]:
    clean = returns.dropna()
    if clean.empty:
        return {
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "cumulative_return": 0.0,
            "weeks": 0.0,
        }
    periods_per_year = 52
    equity = (1.0 + clean).cumprod()
    total_return = equity.iloc[-1] - 1.0
    years = max(len(clean) / periods_per_year, 1 / periods_per_year)
    annual_return = (1.0 + total_return) ** (1.0 / years) - 1.0 if equity.iloc[-1] > 0 else -1.0
    annual_vol = clean.std(ddof=0) * math.sqrt(periods_per_year)
    sharpe = clean.mean() / clean.std(ddof=0) * math.sqrt(periods_per_year) if clean.std(ddof=0) > 0 else 0.0
    drawdown = equity / equity.cummax() - 1.0
    return {
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "cumulative_return": float(total_return),
        "weeks": float(len(clean)),
    }


def _yearly_breakdown(factor_df: pd.DataFrame) -> pd.DataFrame:
    if factor_df.empty:
        return pd.DataFrame()
    temp = factor_df.copy()
    temp["year"] = temp["week_end"].dt.year
    return temp.groupby(["variant", "year"], as_index=False)["factor_return"].sum()


def _portfolio_sort(
    cross_section: pd.DataFrame,
    signal_col: str,
    next_return_col: str,
    scheme: Iterable[float],
    weight_col: str | None,
) -> tuple[float, int]:
    ranked = cross_section.dropna(subset=[signal_col, next_return_col]).sort_values(signal_col)
    if ranked.empty:
        return float("nan"), 0
    labels = _group_labels(len(ranked), scheme)
    ranked = ranked.assign(group=labels)
    long_frame = ranked[ranked["group"] == ranked["group"].max()]
    short_frame = ranked[ranked["group"] == ranked["group"].min()]
    long_return = _weighted_return(long_frame, next_return_col, weight_col)
    short_return = _weighted_return(short_frame, next_return_col, weight_col)
    return long_return - short_return, len(ranked)


def run_ltw_replication(panel: pd.DataFrame, config: AcademicExperimentConfig) -> pd.DataFrame:
    temp = panel.copy()
    temp = temp[~temp["coin_id"].isin(config.exclude_coin_ids)]
    temp = temp[temp["market_cap"] > config.minimum_market_cap_usd]
    temp = temp.sort_values(["coin_id", "week_end"]).reset_index(drop=True)
    temp["signal"] = temp.groupby("coin_id")["close"].pct_change(3)
    temp["next_return"] = temp.groupby("coin_id")["weekly_simple_return"].shift(-1)

    rows: list[dict[str, object]] = []
    for week_end, cross_section in temp.groupby("week_end"):
        factor_return, count = _portfolio_sort(
            cross_section,
            signal_col="signal",
            next_return_col="next_return",
            scheme=[0.3, 0.4, 0.3],
            weight_col="market_cap",
        )
        if pd.isna(factor_return):
            continue
        rows.append({"week_end": week_end, "variant": "cmom_3w", "factor_return": factor_return, "n_assets": count})
    return pd.DataFrame(rows)


def _last_sunday_of_previous_year(year: int) -> pd.Timestamp:
    date = pd.Timestamp(f"{int(year) - 1}-12-31", tz="UTC")
    while date.weekday() != 6:
        date -= pd.Timedelta(days=1)
    return date.normalize()


def _annual_top_n_universe(
    daily_by_coin: dict[str, pd.DataFrame],
    years: Iterable[int],
    top_n: int,
    exclude_coin_ids: Iterable[str],
) -> dict[int, set[str]]:
    excluded = set(exclude_coin_ids)
    universe: dict[int, set[str]] = {}
    for year in years:
        snapshot = _last_sunday_of_previous_year(year)
        candidates: list[tuple[str, float]] = []
        for coin_id, frame in daily_by_coin.items():
            if coin_id in excluded:
                continue
            temp = frame.copy()
            temp["date"] = pd.to_datetime(temp["date"], utc=True, format="mixed")
            temp = temp[temp["date"] <= snapshot]
            if temp.empty:
                continue
            latest = temp.iloc[-1]
            market_cap = latest.get("market_cap")
            if pd.notna(market_cap):
                candidates.append((coin_id, float(market_cap)))
        candidates.sort(key=lambda item: item[1], reverse=True)
        universe[year] = {coin_id for coin_id, _ in candidates[:top_n]}
    return universe


def _paper_weeks_from_panel(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[["week_start", "week_end", "paper_year", "paper_week"]].drop_duplicates().sort_values("week_start").reset_index(drop=True)


def _close_on_or_before(frame: pd.DataFrame, date: pd.Timestamp) -> float | None:
    temp = frame[frame["date"] <= date]
    if temp.empty:
        return None
    price = temp.iloc[-1]["price"]
    return None if pd.isna(price) else float(price)


def run_grobys_replication(
    panel: pd.DataFrame,
    daily_by_coin: dict[str, pd.DataFrame],
    config: AcademicExperimentConfig,
) -> pd.DataFrame:
    weeks = _paper_weeks_from_panel(panel)
    annual_universe = _annual_top_n_universe(
        daily_by_coin=daily_by_coin,
        years=weeks["paper_year"].unique().tolist(),
        top_n=config.top_n,
        exclude_coin_ids=config.exclude_coin_ids,
    )
    rows: list[dict[str, object]] = []
    for row in weeks.itertuples(index=False):
        week_start = row.week_start
        week_end = row.week_end
        year = row.paper_year
        eligible = annual_universe.get(year, set())
        cross_rows = []
        for coin_id in eligible:
            frame = daily_by_coin.get(coin_id)
            if frame is None:
                continue
            frame = frame.copy()
            frame["date"] = pd.to_datetime(frame["date"], utc=True, format="mixed")
            end_signal_date = week_start - pd.Timedelta(days=config.skip_days)
            start_signal_date = end_signal_date - pd.Timedelta(days=config.formation_days - 1)
            formation_end = _close_on_or_before(frame, end_signal_date)
            formation_start = _close_on_or_before(frame, start_signal_date)
            hold_end = _close_on_or_before(frame, week_end)
            if formation_end is None or formation_start is None or hold_end is None or formation_start <= 0 or formation_end <= 0:
                continue
            signal = formation_end / formation_start - 1.0
            next_return = hold_end / formation_end - 1.0
            cross_rows.append({"coin_id": coin_id, "signal": signal, "next_return": next_return})
        cross_section = pd.DataFrame(cross_rows)
        factor_return, count = _portfolio_sort(
            cross_section,
            signal_col="signal",
            next_return_col="next_return",
            scheme=[0.2] * 5,
            weight_col=None,
        )
        if pd.isna(factor_return):
            continue
        rows.append({"week_end": week_end, "variant": "plain_momentum_1m", "factor_return": factor_return, "n_assets": count})
    return pd.DataFrame(rows)


def run_ficura_replication(panel: pd.DataFrame, config: AcademicExperimentConfig) -> pd.DataFrame:
    temp = panel.copy()
    temp = temp[~temp["coin_id"].isin(config.exclude_coin_ids)]
    temp = temp.sort_values(["coin_id", "week_end"]).reset_index(drop=True)
    temp["next_return"] = temp.groupby("coin_id")["weekly_simple_return"].shift(-1)
    temp["next_return"] = temp["next_return"].clip(upper=config.winsor_return_cap)
    temp["segment"] = np.where(
        (temp["market_cap"] >= config.large_market_cap_usd)
        & (temp["dollar_volume"] >= config.large_volume_usd)
        & (temp["close"] >= config.price_floor_usd),
        "large_liquid",
        np.where(
            (temp["market_cap"] >= config.minimum_market_cap_usd)
            & (temp["close"] >= config.price_floor_usd),
            "small_or_illiquid",
            pd.NA,
        ),
    )
    for horizon in config.formation_weeks:
        temp[f"signal_{horizon}w"] = temp.groupby("coin_id")["weekly_log_return"].transform(
            lambda series: series.rolling(horizon, min_periods=horizon).sum()
        )

    rows: list[dict[str, object]] = []
    for horizon in config.formation_weeks:
        signal_col = f"signal_{horizon}w"
        for (week_end, segment), cross_section in temp.groupby(["week_end", "segment"]):
            if pd.isna(segment):
                continue
            factor_return, count = _portfolio_sort(
                cross_section,
                signal_col=signal_col,
                next_return_col="next_return",
                scheme=[0.2] * 5,
                weight_col=None,
            )
            if pd.isna(factor_return):
                continue
            rows.append(
                {
                    "week_end": week_end,
                    "variant": f"{segment}_mom_{horizon}w",
                    "factor_return": factor_return,
                    "n_assets": count,
                }
            )
    return pd.DataFrame(rows)


def _load_or_build_panel(config: AcademicExperimentConfig) -> pd.DataFrame:
    if config.weekly_panel_path.exists():
        return load_weekly_panel(config.weekly_panel_path)
    panel = build_weekly_panel(config)
    save_weekly_panel(panel, config.weekly_panel_path)
    return panel


def run_paper_replication(config: AcademicExperimentConfig) -> dict[str, Path]:
    config.experiment_dir.mkdir(parents=True, exist_ok=True)
    panel = _load_or_build_panel(config)
    if panel.empty:
        raise FileNotFoundError(f"No academic weekly panel available for {config.name}")

    if config.paper_id == "ltw":
        factor_df = run_ltw_replication(panel, config)
    elif config.paper_id == "grobys":
        factor_df = run_grobys_replication(panel, load_coin_series(config.raw_data_dir), config)
    elif config.paper_id == "ficura":
        factor_df = run_ficura_replication(panel, config)
    else:
        raise ValueError(f"Unsupported paper_id: {config.paper_id}")

    summary_rows = []
    for variant, variant_frame in factor_df.groupby("variant"):
        stats = _summarize_factor_returns(variant_frame["factor_return"])
        stats["variant"] = variant
        stats["mean_assets"] = float(variant_frame["n_assets"].mean()) if not variant_frame.empty else 0.0
        summary_rows.append(stats)
    summary_frame = pd.DataFrame(summary_rows)
    yearly = _yearly_breakdown(factor_df)

    outputs = {
        "factor_returns": config.experiment_dir / "factor_returns.csv",
        "summary": config.experiment_dir / "summary.csv",
        "summary_json": config.experiment_dir / "summary.json",
        "yearly": config.experiment_dir / "yearly.csv",
        "panel": config.weekly_panel_path,
    }
    factor_df.to_csv(outputs["factor_returns"], index=False)
    summary_frame.to_csv(outputs["summary"], index=False)
    _write_json(outputs["summary_json"], {"paper_id": config.paper_id, "variants": summary_rows})
    yearly.to_csv(outputs["yearly"], index=False)
    return outputs


def compare_replications(config_paths: Iterable[str | Path]) -> Path:
    rows: list[pd.DataFrame] = []
    first_config: AcademicExperimentConfig | None = None
    for path in config_paths:
        config = load_academic_config(path)
        if first_config is None:
            first_config = config
        summary_path = config.experiment_dir / "summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary for {config.name}: {summary_path}")
        frame = pd.read_csv(summary_path)
        frame.insert(0, "paper_id", config.paper_id)
        frame.insert(0, "experiment", config.name)
        rows.append(frame)
    if first_config is None:
        raise ValueError("No configs supplied for comparison")
    combined = pd.concat(rows, ignore_index=True)
    output_path = first_config.results_dir / "comparison_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return output_path
