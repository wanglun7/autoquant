from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_symbol_funding, load_symbol_klines
from .market_data import combine_field_matrix, funding_returns_from_events
from .signals import build_close_matrix, build_open_matrix


LAGS = (1, 3, 6, 12)
FORWARD_HORIZONS = (1, 3, 6)
QUANTILES = 5
VARIANTS = ("winsorized", "raw")


@dataclass(slots=True)
class StructureScanConfig:
    data_dir: Path = Path("data/raw")
    output_dir: Path = Path("results/structure_scan/scan_v1")
    top_n: int = 100
    liquidity_lookback_bars: int = 180
    min_history_bars: int = 180


@dataclass(slots=True)
class FeatureSpec:
    object_name: str
    feature_name: str
    display_name: str


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _flatten_stats(frame: pd.DataFrame) -> dict[str, float]:
    values = frame.to_numpy(dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "sample_size": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "skew": 0.0,
            "kurtosis": 0.0,
            "q05": 0.0,
            "q25": 0.0,
            "q50": 0.0,
            "q75": 0.0,
            "q95": 0.0,
        }
    series = pd.Series(values)
    return {
        "sample_size": float(values.size),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=0)),
        "skew": float(series.skew()),
        "kurtosis": float(series.kurt()),
        "q05": float(series.quantile(0.05)),
        "q25": float(series.quantile(0.25)),
        "q50": float(series.quantile(0.50)),
        "q75": float(series.quantile(0.75)),
        "q95": float(series.quantile(0.95)),
    }


def _dynamic_membership(
    quote_volume: pd.DataFrame,
    top_n: int,
    liquidity_lookback_bars: int,
    min_history_bars: int,
) -> pd.DataFrame:
    rolling_liquidity = quote_volume.shift(1).rolling(
        window=liquidity_lookback_bars,
        min_periods=min(liquidity_lookback_bars, min_history_bars),
    ).mean()
    history_count = quote_volume.notna().cumsum()
    candidates = rolling_liquidity.notna() & history_count.ge(min_history_bars)
    scores = rolling_liquidity.where(candidates)
    rank = scores.rank(axis=1, method="first", ascending=False)
    return rank.le(top_n) & scores.notna()


def _winsorize_cross_section(frame: pd.DataFrame, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.DataFrame:
    lower = frame.quantile(lower_q, axis=1, interpolation="linear")
    upper = frame.quantile(upper_q, axis=1, interpolation="linear")
    return frame.clip(lower=lower, upper=upper, axis=0)


def _safe_autocorr(series: pd.Series, lag: int) -> float:
    clean = series.dropna()
    if len(clean) <= lag:
        return 0.0
    current = clean.iloc[lag:].to_numpy(dtype=float)
    shifted = clean.iloc[:-lag].to_numpy(dtype=float)
    if current.size == 0 or shifted.size == 0:
        return 0.0
    current = current - current.mean()
    shifted = shifted - shifted.mean()
    denom = np.sqrt(np.dot(current, current) * np.dot(shifted, shifted))
    if denom == 0.0:
        return 0.0
    return float(np.dot(current, shifted) / denom)


def _safe_partial_autocorr(series: pd.Series, lag: int) -> float:
    clean = series.dropna().to_numpy(dtype=float)
    if clean.size <= lag + 5:
        return 0.0
    y = clean[lag:]
    cols = [clean[lag - idx - 1 : -(idx + 1)] for idx in range(lag)]
    x = np.column_stack(cols)
    x = np.column_stack([np.ones(len(x)), x])
    try:
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0
    value = float(coef[-1])
    if not np.isfinite(value):
        return 0.0
    return value


def _scope_series(frame: pd.DataFrame, scope: str, btc_symbol: str) -> pd.Series:
    if scope == "btc":
        if btc_symbol not in frame.columns:
            return pd.Series(dtype=float)
        return frame[btc_symbol]
    if scope == "market_median":
        return frame.median(axis=1, skipna=True)
    raise ValueError(f"Unsupported scope: {scope}")


def _period_labels(index: pd.DatetimeIndex) -> pd.Series:
    labels = pd.Series(index=index, dtype="object")
    labels.loc[(index >= pd.Timestamp("2021-03-13", tz="UTC")) & (index <= pd.Timestamp("2022-12-31 23:59:59", tz="UTC"))] = "2021-2022"
    labels.loc[(index >= pd.Timestamp("2023-01-01", tz="UTC")) & (index <= pd.Timestamp("2023-12-31 23:59:59", tz="UTC"))] = "2023"
    labels.loc[(index >= pd.Timestamp("2024-01-01", tz="UTC")) & (index <= pd.Timestamp("2024-12-31 23:59:59", tz="UTC"))] = "2024"
    labels.loc[(index >= pd.Timestamp("2025-01-01", tz="UTC")) & (index <= pd.Timestamp("2026-03-13 23:59:59", tz="UTC"))] = "2025-2026"
    return labels


def _per_symbol_dependence(frame: pd.DataFrame, lag: int) -> tuple[float, float]:
    acf_values: list[float] = []
    pacf_values: list[float] = []
    for column in frame.columns:
        series = frame[column].dropna()
        if len(series) <= lag + 5:
            continue
        acf_values.append(_safe_autocorr(series, lag))
        pacf_values.append(_safe_partial_autocorr(series, lag))
    if not acf_values:
        return 0.0, 0.0
    return float(np.median(acf_values)), float(np.median(pacf_values))


def _time_dependence_rows(
    feature: pd.DataFrame,
    raw_feature: pd.DataFrame,
    spec: FeatureSpec,
    btc_symbol: str,
    variant: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    scopes = ("btc", "market_median")
    for lag in LAGS:
        per_symbol_acf, per_symbol_pacf = _per_symbol_dependence(feature, lag)
        rows.append(
            {
                "object_name": spec.object_name,
                "feature_name": spec.feature_name,
                "display_name": spec.display_name,
                "variant": variant,
                "scope": "symbol_median",
                "lag": lag,
                "acf": per_symbol_acf,
                "pacf": per_symbol_pacf,
                "sample_size": int(raw_feature.count().sum()),
            }
        )
        for scope in scopes:
            series = _scope_series(feature, scope, btc_symbol)
            rows.append(
                {
                    "object_name": spec.object_name,
                    "feature_name": spec.feature_name,
                    "display_name": spec.display_name,
                    "variant": variant,
                    "scope": scope,
                    "lag": lag,
                    "acf": _safe_autocorr(series, lag),
                    "pacf": _safe_partial_autocorr(series, lag),
                    "sample_size": int(series.dropna().shape[0]),
                }
            )
    return rows


def _rowwise_corr(feature_rank: np.ndarray, future_rank: np.ndarray, valid: np.ndarray) -> np.ndarray:
    feature_masked = np.where(valid, feature_rank, 0.0)
    future_masked = np.where(valid, future_rank, 0.0)
    counts = valid.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        feature_mean = np.divide(feature_masked.sum(axis=1), counts, out=np.zeros(feature_masked.shape[0]), where=counts > 0)
        future_mean = np.divide(future_masked.sum(axis=1), counts, out=np.zeros(future_masked.shape[0]), where=counts > 0)
    feature_centered = np.where(valid, feature_masked - feature_mean[:, None], 0.0)
    future_centered = np.where(valid, future_masked - future_mean[:, None], 0.0)
    numerator = (feature_centered * future_centered).sum(axis=1)
    denominator = np.sqrt((feature_centered * feature_centered).sum(axis=1) * (future_centered * future_centered).sum(axis=1))
    corr = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    corr[counts <= 1] = 0.0
    return corr


def _quantile_means(bucket: np.ndarray, future: np.ndarray, valid: np.ndarray, row_mask: np.ndarray) -> list[float]:
    means: list[float] = []
    subset = valid & row_mask[:, None]
    for quantile in range(1, QUANTILES + 1):
        mask = subset & (bucket == quantile)
        count = int(mask.sum())
        means.append(float(future[mask].mean()) if count else 0.0)
    return means


def _sorting_metrics_from_arrays(
    bucket: np.ndarray,
    future: np.ndarray,
    valid: np.ndarray,
    per_timestamp_ic: np.ndarray,
    row_mask: np.ndarray,
) -> dict[str, object]:
    if not row_mask.any():
        return {
            "q1_mean": 0.0,
            "q2_mean": 0.0,
            "q3_mean": 0.0,
            "q4_mean": 0.0,
            "q5_mean": 0.0,
            "spread_q5_q1": 0.0,
            "rank_ic_mean": 0.0,
            "rank_ic_std": 0.0,
            "monotonic_direction": "none",
            "timestamp_count": 0,
            "sample_size": 0,
        }
    means = _quantile_means(bucket, future, valid, row_mask)
    row_counts = (valid & row_mask[:, None]).sum(axis=1)
    timestamp_mask = row_mask & (row_counts > 1)
    rank_ic_mean = float(per_timestamp_ic[timestamp_mask].mean()) if timestamp_mask.any() else 0.0
    rank_ic_std = float(per_timestamp_ic[timestamp_mask].std(ddof=0)) if int(timestamp_mask.sum()) > 1 else 0.0
    monotonic_direction = "none"
    if means == sorted(means):
        monotonic_direction = "increasing"
    elif means == sorted(means, reverse=True):
        monotonic_direction = "decreasing"
    return {
        "q1_mean": means[0],
        "q2_mean": means[1],
        "q3_mean": means[2],
        "q4_mean": means[3],
        "q5_mean": means[4],
        "spread_q5_q1": means[4] - means[0],
        "rank_ic_mean": rank_ic_mean,
        "rank_ic_std": rank_ic_std,
        "monotonic_direction": monotonic_direction,
        "timestamp_count": int(timestamp_mask.sum()),
        "sample_size": int((valid & row_mask[:, None]).sum()),
    }


def _conditional_stats(values: np.ndarray, valid: np.ndarray, row_mask: np.ndarray) -> tuple[float, float, int]:
    mask = valid & row_mask[:, None]
    count = int(mask.sum())
    if count == 0:
        return 0.0, 0.0, 0
    subset = values[mask]
    return float(subset.mean()), float(subset.var()), count


def _cross_section_rows(
    spec: FeatureSpec,
    variant: str,
    bucket: np.ndarray,
    forward_returns: dict[int, np.ndarray],
    forward_valids: dict[int, np.ndarray],
    per_timestamp_ics: dict[int, np.ndarray],
    period_masks: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for horizon, future in forward_returns.items():
        valid = forward_valids[horizon]
        ic = per_timestamp_ics[horizon]
        for period_name, row_mask in period_masks.items():
            metrics = _sorting_metrics_from_arrays(bucket, future, valid, ic, row_mask)
            rows.append(
                {
                    "object_name": spec.object_name,
                    "feature_name": spec.feature_name,
                    "display_name": spec.display_name,
                    "variant": variant,
                    "period": period_name,
                    "horizon_bars": horizon,
                    **metrics,
                }
            )
    return rows


def _state_rows(
    spec: FeatureSpec,
    variant: str,
    feature_values: np.ndarray,
    feature_valid: np.ndarray,
    bucket: np.ndarray,
    forward_returns: dict[int, np.ndarray],
    forward_valids: dict[int, np.ndarray],
    per_timestamp_ics: dict[int, np.ndarray],
    period_masks: dict[str, np.ndarray],
    state_masks: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for horizon, future in forward_returns.items():
        valid = forward_valids[horizon]
        ic = per_timestamp_ics[horizon]
        for period_name, period_mask in period_masks.items():
            for state_name, state_mask in state_masks.items():
                on_row_mask = period_mask & state_mask
                off_row_mask = period_mask & ~state_mask
                on_metrics = _sorting_metrics_from_arrays(bucket, future, valid, ic, on_row_mask)
                off_metrics = _sorting_metrics_from_arrays(bucket, future, valid, ic, off_row_mask)
                feature_mean_on, feature_var_on, feature_count_on = _conditional_stats(feature_values, feature_valid, on_row_mask)
                feature_mean_off, feature_var_off, feature_count_off = _conditional_stats(feature_values, feature_valid, off_row_mask)
                future_mean_on, future_var_on, future_count_on = _conditional_stats(future, valid, on_row_mask)
                future_mean_off, future_var_off, future_count_off = _conditional_stats(future, valid, off_row_mask)
                rows.append(
                    {
                        "object_name": spec.object_name,
                        "feature_name": spec.feature_name,
                        "display_name": spec.display_name,
                        "variant": variant,
                        "period": period_name,
                        "state_name": state_name,
                        "horizon_bars": horizon,
                        "feature_mean_on": feature_mean_on,
                        "feature_var_on": feature_var_on,
                        "feature_count_on": feature_count_on,
                        "feature_mean_off": feature_mean_off,
                        "feature_var_off": feature_var_off,
                        "feature_count_off": feature_count_off,
                        "future_mean_on": future_mean_on,
                        "future_var_on": future_var_on,
                        "future_count_on": future_count_on,
                        "future_mean_off": future_mean_off,
                        "future_var_off": future_var_off,
                        "future_count_off": future_count_off,
                        "state_on_spread_q5_q1": on_metrics["spread_q5_q1"],
                        "state_off_spread_q5_q1": off_metrics["spread_q5_q1"],
                        "state_gap": float(on_metrics["spread_q5_q1"]) - float(off_metrics["spread_q5_q1"]),
                        "state_on_rank_ic_mean": on_metrics["rank_ic_mean"],
                        "state_off_rank_ic_mean": off_metrics["rank_ic_mean"],
                        "state_on_timestamp_count": on_metrics["timestamp_count"],
                        "state_off_timestamp_count": off_metrics["timestamp_count"],
                    }
                )
    return rows


def _relative_forward_returns(forward_return: pd.DataFrame) -> pd.DataFrame:
    return forward_return.sub(forward_return.mean(axis=1), axis=0)


def _rolling_corr(frame: pd.DataFrame, target: pd.Series, window: int = 180, min_periods: int = 90) -> pd.DataFrame:
    values: dict[str, pd.Series] = {}
    for column in frame.columns:
        values[column] = frame[column].rolling(window=window, min_periods=min_periods).corr(target)
    return pd.DataFrame(values, index=frame.index)


def _feature_definitions(base: dict[str, pd.DataFrame], btc_symbol: str) -> list[tuple[FeatureSpec, pd.DataFrame]]:
    returns = base["returns"]
    market_return = returns.median(axis=1, skipna=True)
    corr_btc = _rolling_corr(returns, returns[btc_symbol]) if btc_symbol in returns.columns else pd.DataFrame(index=returns.index, columns=returns.columns)
    corr_market = _rolling_corr(returns, market_return)
    return [
        (FeatureSpec("returns", "ret_1b", "1-bar return"), returns.shift(1)),
        (FeatureSpec("returns", "ret_3b", "3-bar return"), base["closes"].pct_change(3, fill_method=None).shift(1)),
        (FeatureSpec("returns", "ret_6b", "6-bar return"), base["closes"].pct_change(6, fill_method=None).shift(1)),
        (FeatureSpec("volatility", "abs_ret_1b", "abs 1-bar return"), returns.abs().shift(1)),
        (FeatureSpec("volatility", "ret_sq_1b", "1-bar squared return"), returns.pow(2).shift(1)),
        (FeatureSpec("volatility", "range_vol", "intrabar range volatility"), ((base["highs"] - base["lows"]) / base["opens"]).shift(1)),
        (FeatureSpec("volatility", "realized_vol_6b", "6-bar realized volatility"), returns.rolling(6, min_periods=3).std().shift(1)),
        (FeatureSpec("volume", "quote_volume", "quote volume"), base["quote_volume"].shift(1)),
        (FeatureSpec("volume", "volume", "base volume"), base["volume"].shift(1)),
        (FeatureSpec("volume", "trade_count", "trade count"), base["trade_count"].shift(1)),
        (FeatureSpec("volume", "quote_volume_chg_3b", "3-bar quote volume change"), base["quote_volume"].pct_change(3, fill_method=None).shift(1)),
        (FeatureSpec("liquidity", "liquidity_30d", "30-day rolling quote volume"), base["quote_volume"].shift(1).rolling(180, min_periods=90).mean()),
        (FeatureSpec("liquidity", "amihud", "Amihud proxy"), (returns.abs() / base["quote_volume"].replace(0.0, np.nan)).shift(1)),
        (FeatureSpec("liquidity", "avg_trade_size", "average trade size"), (base["quote_volume"] / base["trade_count"].replace(0.0, np.nan)).shift(1)),
        (FeatureSpec("order_flow", "buy_ratio", "taker buy ratio"), (base["taker_buy_quote_volume"] / base["quote_volume"].replace(0.0, np.nan)).shift(1)),
        (FeatureSpec("order_flow", "imbalance", "buy imbalance"), ((2.0 * base["taker_buy_quote_volume"] / base["quote_volume"].replace(0.0, np.nan)) - 1.0).shift(1)),
        (
            FeatureSpec("order_flow", "imbalance_3b_mean", "3-bar buy imbalance mean"),
            ((2.0 * base["taker_buy_quote_volume"] / base["quote_volume"].replace(0.0, np.nan)) - 1.0).rolling(3, min_periods=2).mean().shift(1),
        ),
        (FeatureSpec("carry", "funding_1b", "1-bar funding"), base["funding"].shift(1)),
        (FeatureSpec("carry", "funding_1d_sum", "1-day funding sum"), base["funding"].rolling(6, min_periods=3).sum().shift(1)),
        (FeatureSpec("carry", "funding_3d_sum", "3-day funding sum"), base["funding"].rolling(18, min_periods=9).sum().shift(1)),
        (FeatureSpec("carry", "abs_funding", "abs funding"), base["funding"].abs().shift(1)),
        (FeatureSpec("correlation", "corr_btc_30d", "30-day correlation to BTC"), corr_btc.shift(1)),
        (FeatureSpec("correlation", "corr_market_30d", "30-day correlation to market median"), corr_market.shift(1)),
    ]


def _state_series(base: dict[str, pd.DataFrame], btc_symbol: str) -> dict[str, pd.Series]:
    btc_returns = base["returns"][btc_symbol] if btc_symbol in base["returns"].columns else base["returns"].median(axis=1, skipna=True)
    btc_vol_30d = btc_returns.rolling(180, min_periods=90).std().shift(1)
    btc_ret_30d = base["closes"][btc_symbol].pct_change(180, fill_method=None).shift(1) if btc_symbol in base["closes"].columns else base["returns"].median(axis=1, skipna=True).rolling(180, min_periods=90).sum()
    market_funding = base["funding"].rolling(6, min_periods=3).sum().shift(1).median(axis=1, skipna=True)
    market_liquidity = base["quote_volume"].shift(1).rolling(180, min_periods=90).mean().median(axis=1, skipna=True)
    return {
        "btc_vol_high": btc_vol_30d > btc_vol_30d.median(skipna=True),
        "btc_return_positive": btc_ret_30d > 0.0,
        "market_funding_positive": market_funding > 0.0,
        "market_liquidity_high": market_liquidity > market_liquidity.median(skipna=True),
    }


def _object_summary(
    object_name: str,
    time_rows: pd.DataFrame,
    cross_rows: pd.DataFrame,
    state_rows: pd.DataFrame,
) -> dict[str, object]:
    object_time = time_rows[(time_rows["object_name"] == object_name) & (time_rows["variant"] == "winsorized")]
    object_time_raw = time_rows[(time_rows["object_name"] == object_name) & (time_rows["variant"] == "raw")]
    object_cross = cross_rows[(cross_rows["object_name"] == object_name) & (cross_rows["period"] == "full") & (cross_rows["variant"] == "winsorized")]
    object_cross_raw = cross_rows[(cross_rows["object_name"] == object_name) & (cross_rows["period"] == "full") & (cross_rows["variant"] == "raw")]
    object_state = state_rows[(state_rows["object_name"] == object_name) & (state_rows["period"] == "full") & (state_rows["variant"] == "winsorized")]
    object_state_raw = state_rows[(state_rows["object_name"] == object_name) & (state_rows["period"] == "full") & (state_rows["variant"] == "raw")]

    time_signal = False
    if not object_time.empty:
        grouped = object_time.groupby(["feature_name", "lag"])
        for _, group in grouped:
            strong_scopes = (group["acf"].abs() >= 0.05) | (group["pacf"].abs() >= 0.05)
            raw_group = object_time_raw[(object_time_raw["feature_name"] == group["feature_name"].iloc[0]) & (object_time_raw["lag"] == group["lag"].iloc[0])]
            raw_strong = (raw_group["acf"].abs() >= 0.05) | (raw_group["pacf"].abs() >= 0.05)
            if int(strong_scopes.sum()) >= 2 and int(raw_strong.sum()) >= 2:
                time_signal = True
                break

    cross_signal = False
    period_consistency = 0
    strongest_cross_feature = ""
    strongest_cross_score = 0.0
    if not object_cross.empty:
        for feature_name, group in object_cross.groupby("feature_name"):
            raw_group = object_cross_raw[object_cross_raw["feature_name"] == feature_name]
            if raw_group.empty:
                continue
            passing_horizons = 0
            local_period_consistency = 0
            for horizon in FORWARD_HORIZONS:
                win_row = group[group["horizon_bars"] == horizon]
                raw_row = raw_group[raw_group["horizon_bars"] == horizon]
                if win_row.empty or raw_row.empty:
                    continue
                win_spread = float(win_row["spread_q5_q1"].iloc[0])
                win_ic = float(win_row["rank_ic_mean"].iloc[0])
                raw_spread = float(raw_row["spread_q5_q1"].iloc[0])
                raw_ic = float(raw_row["rank_ic_mean"].iloc[0])
                if np.sign(win_spread) == 0 or np.sign(win_spread) != np.sign(win_ic):
                    continue
                if np.sign(raw_spread) != np.sign(win_spread) or np.sign(raw_ic) != np.sign(win_ic):
                    continue
                period_group = cross_rows[
                    (cross_rows["object_name"] == object_name)
                    & (cross_rows["feature_name"] == feature_name)
                    & (cross_rows["variant"] == "winsorized")
                    & (cross_rows["period"] != "full")
                    & (cross_rows["horizon_bars"] == horizon)
                ]
                consistency = int((np.sign(period_group["spread_q5_q1"]) == np.sign(win_spread)).sum())
                if consistency >= 3:
                    passing_horizons += 1
                    local_period_consistency = max(local_period_consistency, consistency)
            score = float(group["spread_q5_q1"].abs().mean() + group["rank_ic_mean"].abs().mean())
            if score > strongest_cross_score:
                strongest_cross_score = score
                strongest_cross_feature = feature_name
                period_consistency = local_period_consistency
            if passing_horizons >= 2:
                cross_signal = True
                break

    state_signal = False
    strongest_state_feature = ""
    strongest_state_gap = 0.0
    if not object_state.empty:
        period_state = state_rows[
            (state_rows["object_name"] == object_name)
            & (state_rows["period"] != "full")
            & (state_rows["variant"] == "winsorized")
        ]
        for feature_name, group in object_state.groupby("feature_name"):
            raw_group = object_state_raw[object_state_raw["feature_name"] == feature_name]
            if raw_group.empty:
                continue
            score = float(group["state_gap"].abs().mean())
            if score > strongest_state_gap:
                strongest_state_gap = score
                strongest_state_feature = feature_name
            stable_count = 0
            for (state_name, horizon), subgroup in group.groupby(["state_name", "horizon_bars"]):
                raw_subgroup = raw_group[(raw_group["state_name"] == state_name) & (raw_group["horizon_bars"] == horizon)]
                if raw_subgroup.empty:
                    continue
                win_gap = float(subgroup["state_gap"].iloc[0])
                raw_gap = float(raw_subgroup["state_gap"].iloc[0])
                if np.sign(win_gap) == 0 or np.sign(win_gap) != np.sign(raw_gap):
                    continue
                annual = period_state[
                    (period_state["feature_name"] == feature_name)
                    & (period_state["state_name"] == state_name)
                    & (period_state["horizon_bars"] == horizon)
                ]
                if int((np.sign(annual["state_gap"]) == np.sign(win_gap)).sum()) >= 3:
                    stable_count += 1
            if stable_count >= 2:
                state_signal = True
                break

    signal_count = int(time_signal) + int(cross_signal) + int(state_signal)
    if signal_count >= 2:
        verdict = "存在较强统计结构"
    elif signal_count == 1:
        verdict = "存在弱但可疑的统计结构"
    else:
        verdict = "当前数据下未发现稳健结构"

    strongest_feature = strongest_cross_feature or strongest_state_feature
    return {
        "object_name": object_name,
        "verdict": verdict,
        "time_signal": time_signal,
        "cross_signal": cross_signal,
        "state_signal": state_signal,
        "strongest_feature": strongest_feature,
        "period_consistency_count": period_consistency,
        "strongest_state_gap": strongest_state_gap,
    }


def _summary_markdown(
    config: StructureScanConfig,
    overview: pd.DataFrame,
    object_summaries: pd.DataFrame,
) -> str:
    lines = [
        "# 统计结构扫描结论",
        "",
        "## 样本概况",
        f"- 数据目录: `{config.data_dir}`",
        f"- 输出目录: `{config.output_dir}`",
        f"- 横截面主样本池: 每期过去 30 天平均 quote volume 前 {config.top_n}",
        f"- 最小历史门槛: {config.min_history_bars} 根 4h bar",
        f"- 描述统计对象数: {overview['object_name'].nunique()}",
        "",
        "## 结论摘要",
    ]
    for row in object_summaries.itertuples(index=False):
        reason_parts: list[str] = []
        if row.time_signal:
            reason_parts.append("有时间惯性")
        if row.cross_signal:
            reason_parts.append("有横截面排序")
        if row.state_signal:
            reason_parts.append("有状态依赖")
        if not reason_parts:
            reason_parts.append("三类检验都偏弱")
        strongest = f"，最强特征是 `{row.strongest_feature}`" if row.strongest_feature else ""
        lines.append(f"- `{row.object_name}`: {row.verdict}，{'、'.join(reason_parts)}{strongest}")
    strong = object_summaries[object_summaries["verdict"] == "存在较强统计结构"]["object_name"].tolist()
    weak = object_summaries[object_summaries["verdict"] == "存在弱但可疑的统计结构"]["object_name"].tolist()
    none = object_summaries[object_summaries["verdict"] == "当前数据下未发现稳健结构"]["object_name"].tolist()
    lines.extend(
        [
            "",
            "## 有没有统计结构",
            f"- 较强: {', '.join(strong) if strong else '无'}",
            f"- 较弱但可疑: {', '.join(weak) if weak else '无'}",
            f"- 未发现稳健结构: {', '.join(none) if none else '无'}",
        ]
    )
    return "\n".join(lines) + "\n"


def run_structure_scan(
    config: StructureScanConfig | None = None,
    *,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    top_n: int | None = None,
    liquidity_lookback_bars: int | None = None,
    min_history_bars: int | None = None,
) -> dict[str, Path]:
    if config is None:
        config = StructureScanConfig()
    if data_dir is not None:
        config.data_dir = Path(data_dir)
    if output_dir is not None:
        config.output_dir = Path(output_dir)
    if top_n is not None:
        config.top_n = top_n
    if liquidity_lookback_bars is not None:
        config.liquidity_lookback_bars = liquidity_lookback_bars
    if min_history_bars is not None:
        config.min_history_bars = min_history_bars

    config.output_dir.mkdir(parents=True, exist_ok=True)

    klines = load_symbol_klines(config.data_dir)
    if not klines:
        raise FileNotFoundError(f"No kline data found in {config.data_dir / 'klines'}")
    funding_raw = load_symbol_funding(config.data_dir)

    closes = build_close_matrix(klines)
    opens = build_open_matrix(klines)
    highs = combine_field_matrix(klines, "high")
    lows = combine_field_matrix(klines, "low")
    volume = combine_field_matrix(klines, "volume")
    quote_volume = combine_field_matrix(klines, "quote_volume")
    trade_count = combine_field_matrix(klines, "trade_count")
    taker_buy_quote_volume = combine_field_matrix(klines, "taker_buy_quote_volume")
    funding = funding_returns_from_events(funding_raw, opens.index).reindex(index=opens.index, columns=opens.columns, fill_value=0.0)

    membership = _dynamic_membership(
        quote_volume=quote_volume,
        top_n=config.top_n,
        liquidity_lookback_bars=config.liquidity_lookback_bars,
        min_history_bars=config.min_history_bars,
    )

    btc_symbol = "BTCUSDT" if "BTCUSDT" in closes.columns else closes.columns[0]
    base = {
        "closes": closes,
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "volume": volume,
        "quote_volume": quote_volume,
        "trade_count": trade_count,
        "taker_buy_quote_volume": taker_buy_quote_volume,
        "funding": funding,
        "returns": closes.pct_change(fill_method=None),
    }

    forward_returns = {
        horizon: _relative_forward_returns(opens.shift(-(horizon + 1)).div(opens.shift(-1)).sub(1.0))
        for horizon in FORWARD_HORIZONS
    }
    forward_arrays = {horizon: frame.to_numpy(dtype=np.float64) for horizon, frame in forward_returns.items()}
    forward_ranks = {
        horizon: frame.rank(axis=1, method="average", pct=True).to_numpy(dtype=np.float64)
        for horizon, frame in forward_returns.items()
    }
    periods = _period_labels(opens.index)
    states = _state_series(base, btc_symbol)
    period_masks = {
        "full": np.ones(len(opens.index), dtype=bool),
        "2021-2022": (periods == "2021-2022").to_numpy(dtype=bool),
        "2023": (periods == "2023").to_numpy(dtype=bool),
        "2024": (periods == "2024").to_numpy(dtype=bool),
        "2025-2026": (periods == "2025-2026").to_numpy(dtype=bool),
    }
    state_masks = {name: series.fillna(False).to_numpy(dtype=bool) for name, series in states.items()}

    overview_rows: list[dict[str, object]] = []
    time_rows: list[dict[str, object]] = []
    cross_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []

    for spec, raw_feature in _feature_definitions(base, btc_symbol):
        masked_raw = raw_feature.where(membership)
        variants = {
            "winsorized": _winsorize_cross_section(masked_raw),
            "raw": masked_raw,
        }
        for variant, feature_frame in variants.items():
            stats = _flatten_stats(feature_frame)
            overview_rows.append(
                {
                    "object_name": spec.object_name,
                    "feature_name": spec.feature_name,
                    "display_name": spec.display_name,
                    "variant": variant,
                    "coverage_ratio": float(masked_raw.count().sum() / membership.sum().sum()) if membership.sum().sum() else 0.0,
                    **stats,
                }
            )
            time_rows.extend(_time_dependence_rows(feature_frame, masked_raw, spec, btc_symbol, variant))

            feature_values = feature_frame.to_numpy(dtype=np.float64)
            feature_valid = np.isfinite(feature_values)
            feature_ranks = feature_frame.rank(axis=1, method="average", pct=True).to_numpy(dtype=np.float64)
            bucket = np.ceil(np.nan_to_num(feature_ranks, nan=0.0) * QUANTILES).astype(np.int8)
            bucket[~feature_valid] = 0

            horizon_valids: dict[int, np.ndarray] = {}
            per_timestamp_ics: dict[int, np.ndarray] = {}
            for horizon in FORWARD_HORIZONS:
                valid = feature_valid & np.isfinite(forward_arrays[horizon])
                horizon_valids[horizon] = valid
                per_timestamp_ics[horizon] = _rowwise_corr(feature_ranks, forward_ranks[horizon], valid)

            cross_rows.extend(
                _cross_section_rows(
                    spec=spec,
                    variant=variant,
                    bucket=bucket,
                    forward_returns=forward_arrays,
                    forward_valids=horizon_valids,
                    per_timestamp_ics=per_timestamp_ics,
                    period_masks=period_masks,
                )
            )
            state_rows.extend(
                _state_rows(
                    spec=spec,
                    variant=variant,
                    feature_values=feature_values,
                    feature_valid=feature_valid,
                    bucket=bucket,
                    forward_returns=forward_arrays,
                    forward_valids=horizon_valids,
                    per_timestamp_ics=per_timestamp_ics,
                    period_masks=period_masks,
                    state_masks=state_masks,
                )
            )

    overview_df = pd.DataFrame(overview_rows).sort_values(["object_name", "feature_name", "variant"]).reset_index(drop=True)
    time_df = pd.DataFrame(time_rows).sort_values(["object_name", "feature_name", "variant", "scope", "lag"]).reset_index(drop=True)
    cross_df = pd.DataFrame(cross_rows).sort_values(["object_name", "feature_name", "variant", "period", "horizon_bars"]).reset_index(drop=True)
    state_df = pd.DataFrame(state_rows).sort_values(["object_name", "feature_name", "variant", "period", "state_name", "horizon_bars"]).reset_index(drop=True)

    object_names = list(overview_df["object_name"].drop_duplicates())
    object_summary_rows = [_object_summary(object_name, time_df, cross_df, state_df) for object_name in object_names]
    object_summary_df = pd.DataFrame(object_summary_rows).sort_values("object_name").reset_index(drop=True)

    paths = {
        "object_overview": config.output_dir / "object_overview.csv",
        "time_dependence": config.output_dir / "time_dependence.csv",
        "cross_sectional_sorting": config.output_dir / "cross_sectional_sorting.csv",
        "state_dependence": config.output_dir / "state_dependence.csv",
        "summary": config.output_dir / "summary.md",
        "artifacts": config.output_dir / "artifacts.json",
    }

    overview_df.to_csv(paths["object_overview"], index=False)
    time_df.to_csv(paths["time_dependence"], index=False)
    cross_df.to_csv(paths["cross_sectional_sorting"], index=False)
    state_df.to_csv(paths["state_dependence"], index=False)
    paths["summary"].write_text(_summary_markdown(config, overview_df, object_summary_df), encoding="utf-8")
    _write_json(
        paths["artifacts"],
        {
            "config": {
                "data_dir": str(config.data_dir),
                "output_dir": str(config.output_dir),
                "top_n": config.top_n,
                "liquidity_lookback_bars": config.liquidity_lookback_bars,
                "min_history_bars": config.min_history_bars,
            },
            "dataset": {
                "symbols": int(closes.shape[1]),
                "bars": int(closes.shape[0]),
                "start": str(opens.index.min()),
                "end": str(opens.index.max()),
                "btc_symbol": btc_symbol,
            },
            "objects": object_summary_rows,
        },
    )
    return paths
