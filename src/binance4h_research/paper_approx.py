from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .analytics import summarize_artifacts
from .backtest import funding_returns_from_events, run_backtest
from .data import load_symbol_funding, load_symbol_klines
from .paper_approx_config import PaperApproxConfig, load_paper_approx_config
from .signals import build_close_matrix, build_open_matrix


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def base_asset(symbol: str) -> str:
    return symbol[:-4] if symbol.endswith("USDT") else symbol


def eligible_symbols(symbols: Iterable[str], config: PaperApproxConfig) -> list[str]:
    excluded_bases = {item.upper() for item in config.universe.exclude_bases}
    excluded_symbols = {item.upper() for item in config.universe.exclude_symbols}
    filtered: list[str] = []
    for symbol in symbols:
        upper = symbol.upper()
        base = base_asset(upper)
        if upper in excluded_symbols or base in excluded_bases:
            continue
        if any(base.endswith(suffix) for suffix in ("UP", "DOWN", "BULL", "BEAR")):
            continue
        filtered.append(symbol)
    return sorted(filtered)


def build_quote_volume_matrix(klines: dict[str, pd.DataFrame]) -> pd.DataFrame:
    series_map: dict[str, pd.Series] = {}
    for symbol, frame in klines.items():
        series_map[symbol] = frame.set_index("open_time")["quote_volume"].astype(float)
    if not series_map:
        return pd.DataFrame()
    return pd.concat(series_map, axis=1).sort_index()


def build_weekly_rebalance_index(index: pd.DatetimeIndex, weekday: int = 0, hour_utc: int = 0) -> pd.DatetimeIndex:
    mask = (index.weekday == weekday) & (index.hour == hour_utc)
    return index[mask]


def compute_ltw_signal(closes: pd.DataFrame, lookback_weeks: int, bars_per_week: int) -> pd.DataFrame:
    lookback_bars = lookback_weeks * bars_per_week
    return closes.shift(1).div(closes.shift(1 + lookback_bars)).sub(1.0)


def compute_grobys_signal(closes: pd.DataFrame, formation_days: int, skip_days: int, bars_per_day: int) -> pd.DataFrame:
    formation_bars = formation_days * bars_per_day
    skip_bars = skip_days * bars_per_day
    return closes.shift(1 + skip_bars).div(closes.shift(1 + skip_bars + formation_bars)).sub(1.0)


def compute_ficura_signals(closes: pd.DataFrame, formation_weeks: Iterable[int], bars_per_week: int) -> dict[int, pd.DataFrame]:
    signals: dict[int, pd.DataFrame] = {}
    for weeks in formation_weeks:
        lookback_bars = weeks * bars_per_week
        signals[weeks] = closes.shift(1).div(closes.shift(1 + lookback_bars)).sub(1.0)
    return signals


def _select_top(series: pd.Series, top_n: int) -> pd.Index:
    clean = series.dropna().sort_values(ascending=False)
    return clean.iloc[: min(top_n, len(clean))].index


def _bucket_labels(size: int, scheme: list[float]) -> np.ndarray:
    if size == 0:
        return np.array([], dtype=int)
    counts = [int(math.floor(size * value)) for value in scheme[:-1]]
    counts.append(size - sum(counts))
    labels = np.zeros(size, dtype=int)
    start = 0
    for idx, count in enumerate(counts, start=1):
        end = min(size, start + max(count, 0))
        labels[start:end] = idx
        start = end
    if start < size:
        labels[start:] = len(scheme)
    return labels


def _group_weights(
    ranked: pd.DataFrame,
    long_group: int,
    short_group: int,
    gross_exposure: float,
    weighting: str,
    weight_col: str,
) -> pd.Series:
    weights = pd.Series(0.0, index=ranked.index, dtype=float)
    long_frame = ranked[ranked["group"] == long_group]
    short_frame = ranked[ranked["group"] == short_group]
    if weighting == "equal":
        if not long_frame.empty:
            weights.loc[long_frame.index] = (gross_exposure / 2.0) / len(long_frame)
        if not short_frame.empty:
            weights.loc[short_frame.index] = -(gross_exposure / 2.0) / len(short_frame)
        return weights

    long_weights = long_frame[weight_col].clip(lower=0.0)
    short_weights = short_frame[weight_col].clip(lower=0.0)
    if not long_frame.empty and long_weights.sum() > 0:
        weights.loc[long_frame.index] = (gross_exposure / 2.0) * (long_weights / long_weights.sum())
    if not short_frame.empty and short_weights.sum() > 0:
        weights.loc[short_frame.index] = -(gross_exposure / 2.0) * (short_weights / short_weights.sum())
    return weights


def _rank_cross_section(
    signal_row: pd.Series,
    liquidity_row: pd.Series,
    eligible: pd.Index,
    scheme: list[float],
    gross_exposure: float,
    weighting: str,
) -> tuple[pd.Series, dict[str, int]]:
    frame = pd.DataFrame({"signal": signal_row, "weight_proxy": liquidity_row}).loc[eligible]
    frame = frame.dropna(subset=["signal"]).sort_values("signal")
    if frame.empty:
        return pd.Series(0.0, index=signal_row.index, dtype=float), {"n_assets": 0, "long_count": 0, "short_count": 0}
    frame["group"] = _bucket_labels(len(frame), scheme)
    long_group = int(frame["group"].max())
    short_group = int(frame["group"].min())
    weights = pd.Series(0.0, index=signal_row.index, dtype=float)
    weights.loc[frame.index] = _group_weights(
        frame,
        long_group=long_group,
        short_group=short_group,
        gross_exposure=gross_exposure,
        weighting=weighting,
        weight_col="weight_proxy",
    )
    return weights, {
        "n_assets": int(len(frame)),
        "long_count": int((frame["group"] == long_group).sum()),
        "short_count": int((frame["group"] == short_group).sum()),
    }


def build_variant_weights(
    klines: dict[str, pd.DataFrame],
    config: PaperApproxConfig,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    symbols = eligible_symbols(klines.keys(), config)
    if not symbols:
        return {}, pd.DataFrame(), pd.DataFrame()
    closes = build_close_matrix({symbol: klines[symbol] for symbol in symbols})
    quote_volume = build_quote_volume_matrix({symbol: klines[symbol] for symbol in symbols})
    liquidity_min_periods = min(config.universe.min_history_bars, config.universe.liquidity_lookback_bars)
    liquidity_proxy = quote_volume.shift(1).rolling(
        config.universe.liquidity_lookback_bars,
        min_periods=liquidity_min_periods,
    ).mean()
    weight_proxy = quote_volume.shift(1).rolling(config.universe.weight_lookback_bars, min_periods=max(30, config.universe.weight_lookback_bars // 2)).mean()

    bars_per_day = 24 // config.bar_hours
    bars_per_week = 7 * bars_per_day
    rebalance_index = build_weekly_rebalance_index(
        closes.index,
        weekday=config.paper.rebalance_weekday,
        hour_utc=config.paper.rebalance_hour_utc,
    )
    variants: dict[str, pd.DataFrame] = {}
    group_rows: list[dict[str, object]] = []

    if config.paper_id == "ltw":
        signal = compute_ltw_signal(closes, config.paper.ltw_lookback_weeks, bars_per_week)
        weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
        for timestamp in rebalance_index:
            liquidity_row = liquidity_proxy.loc[timestamp]
            eligible = _select_top(liquidity_row, config.universe.top_n)
            row_weights, stats = _rank_cross_section(
                signal_row=signal.loc[timestamp],
                liquidity_row=weight_proxy.loc[timestamp],
                eligible=eligible,
                scheme=[0.3, 0.4, 0.3],
                gross_exposure=config.portfolio.gross_exposure,
                weighting="volume",
            )
            weights.loc[timestamp] = row_weights
            group_rows.append({"timestamp": timestamp, "variant": "ltw_cmom_3w", **stats})
        variants["ltw_cmom_3w"] = weights.ffill().fillna(0.0)

    elif config.paper_id == "grobys":
        signal = compute_grobys_signal(closes, config.paper.grobys_formation_days, config.paper.grobys_skip_days, bars_per_day)
        weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
        for timestamp in rebalance_index:
            eligible = _select_top(liquidity_proxy.loc[timestamp], config.universe.top_n)
            row_weights, stats = _rank_cross_section(
                signal_row=signal.loc[timestamp],
                liquidity_row=weight_proxy.loc[timestamp],
                eligible=eligible,
                scheme=[0.2] * 5,
                gross_exposure=config.portfolio.gross_exposure,
                weighting="equal",
            )
            weights.loc[timestamp] = row_weights
            group_rows.append({"timestamp": timestamp, "variant": "grobys_plain_1m_skip1d", **stats})
        variants["grobys_plain_1m_skip1d"] = weights.ffill().fillna(0.0)

    elif config.paper_id == "ficura":
        signals = compute_ficura_signals(closes, config.paper.ficura_formation_weeks, bars_per_week)
        for weeks, signal in signals.items():
            large_weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
            other_weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
            for timestamp in rebalance_index:
                base_universe = _select_top(liquidity_proxy.loc[timestamp], config.paper.ficura_total_universe)
                if len(base_universe) < 5:
                    continue
                liquid_scores = liquidity_proxy.loc[timestamp, base_universe].dropna().sort_values(ascending=False)
                large_count = max(5, int(math.floor(len(liquid_scores) * config.paper.ficura_large_liquid_fraction)))
                large_universe = liquid_scores.iloc[:large_count].index
                other_universe = liquid_scores.iloc[large_count:].index
                row_weights, stats = _rank_cross_section(
                    signal_row=signal.loc[timestamp],
                    liquidity_row=weight_proxy.loc[timestamp],
                    eligible=large_universe,
                    scheme=[0.2] * 5,
                    gross_exposure=config.portfolio.gross_exposure,
                    weighting="equal",
                )
                large_weights.loc[timestamp] = row_weights
                group_rows.append({"timestamp": timestamp, "variant": f"ficura_large_liquid_mom_{weeks}w", **stats})
                row_weights, stats = _rank_cross_section(
                    signal_row=signal.loc[timestamp],
                    liquidity_row=weight_proxy.loc[timestamp],
                    eligible=other_universe,
                    scheme=[0.2] * 5,
                    gross_exposure=config.portfolio.gross_exposure,
                    weighting="equal",
                )
                other_weights.loc[timestamp] = row_weights
                group_rows.append({"timestamp": timestamp, "variant": f"ficura_others_mom_{weeks}w", **stats})
            variants[f"ficura_large_liquid_mom_{weeks}w"] = large_weights.ffill().fillna(0.0)
            variants[f"ficura_others_mom_{weeks}w"] = other_weights.ffill().fillna(0.0)
    else:
        raise ValueError(f"Unsupported paper_id: {config.paper_id}")

    return variants, closes, build_open_matrix({symbol: klines[symbol] for symbol in symbols}), pd.DataFrame(group_rows)


def _yearly_from_pnl(variant: str, pnl: pd.DataFrame) -> pd.DataFrame:
    frame = pnl.copy()
    frame["year"] = frame.index.year
    grouped = frame.groupby("year")[["gross_return", "trading_cost", "funding_return", "net_return"]].sum().reset_index()
    grouped.insert(0, "variant", variant)
    return grouped


def _pressure_rows(
    variant: str,
    weights: pd.DataFrame,
    opens: pd.DataFrame,
    funding: pd.DataFrame,
    config: PaperApproxConfig,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for fee_mult in config.report.pressure_fee_multipliers:
        for slip_mult in config.report.pressure_slippage_multipliers:
            artifacts = run_backtest(
                weights=weights,
                opens=opens,
                funding=funding,
                costs=type(config.costs)(
                    fee_bps=config.costs.fee_bps * fee_mult,
                    slippage_bps=config.costs.slippage_bps * slip_mult,
                ),
            )
            stats = summarize_artifacts(artifacts, type("BT", (), {"bar_hours": config.bar_hours})())
            stats["variant"] = variant
            stats["fee_mult"] = fee_mult
            stats["slippage_mult"] = slip_mult
            rows.append(stats)
    return pd.DataFrame(rows)


def run_paper_approx(config: PaperApproxConfig) -> dict[str, Path]:
    config.experiment_dir.mkdir(parents=True, exist_ok=True)
    klines = load_symbol_klines(config.data_dir)
    funding_raw = load_symbol_funding(config.data_dir)
    variants, closes, opens, group_stats = build_variant_weights(klines, config)
    if not variants:
        raise FileNotFoundError(f"No eligible variants built for {config.name}")
    funding = funding_returns_from_events(funding_raw, opens.index).reindex(index=opens.index, columns=opens.columns, fill_value=0.0)

    summary_rows: list[dict[str, float | str]] = []
    yearly_rows: list[pd.DataFrame] = []
    pressure_rows: list[pd.DataFrame] = []
    pnl_frames: list[pd.DataFrame] = []
    weights_dir = config.experiment_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    for variant, weights in variants.items():
        artifacts = run_backtest(weights=weights, opens=opens, funding=funding, costs=config.costs)
        stats = summarize_artifacts(artifacts, type("BT", (), {"bar_hours": config.bar_hours})())
        stats["variant"] = variant
        stats["paper_id"] = config.paper_id
        if not group_stats.empty:
            subset = group_stats[group_stats["variant"] == variant]
            stats["mean_assets"] = float(subset["n_assets"].mean()) if not subset.empty else 0.0
        summary_rows.append(stats)
        yearly_rows.append(_yearly_from_pnl(variant, artifacts.pnl))
        pressure_rows.append(_pressure_rows(variant, weights, opens, funding, config))
        pnl = artifacts.pnl.copy()
        pnl.insert(0, "variant", variant)
        pnl_frames.append(pnl.reset_index(names="timestamp"))
        weights.to_csv(weights_dir / f"{variant}.csv", index_label="timestamp")

    summary = pd.DataFrame(summary_rows)
    yearly = pd.concat(yearly_rows, ignore_index=True) if yearly_rows else pd.DataFrame()
    pressure = pd.concat(pressure_rows, ignore_index=True) if pressure_rows else pd.DataFrame()
    pnl = pd.concat(pnl_frames, ignore_index=True) if pnl_frames else pd.DataFrame()

    paths = {
        "summary": config.experiment_dir / "summary.csv",
        "summary_json": config.experiment_dir / "summary.json",
        "yearly": config.experiment_dir / "yearly.csv",
        "pressure": config.experiment_dir / "pressure.csv",
        "group_stats": config.experiment_dir / "group_stats.csv",
        "pnl": config.experiment_dir / "pnl.csv",
    }
    summary.to_csv(paths["summary"], index=False)
    _write_json(paths["summary_json"], {"paper_id": config.paper_id, "variants": summary_rows})
    yearly.to_csv(paths["yearly"], index=False)
    pressure.to_csv(paths["pressure"], index=False)
    group_stats.to_csv(paths["group_stats"], index=False)
    pnl.to_csv(paths["pnl"], index=False)
    return paths


def compare_paper_approx(config_paths: Iterable[str | Path]) -> Path:
    frames: list[pd.DataFrame] = []
    base_results_dir: Path | None = None
    for path in config_paths:
        config = load_paper_approx_config(path)
        if base_results_dir is None:
            base_results_dir = config.results_dir
        summary_path = config.experiment_dir / "summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing paper approx summary: {summary_path}")
        frame = pd.read_csv(summary_path)
        frame.insert(0, "experiment", config.name)
        frames.append(frame)
    if base_results_dir is None:
        raise ValueError("No configs provided")
    combined = pd.concat(frames, ignore_index=True)
    output_path = base_results_dir / "comparison_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return output_path
