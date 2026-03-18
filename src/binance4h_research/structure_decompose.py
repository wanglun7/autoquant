from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .market_data import combine_field_matrix, funding_returns_from_events
from .data import load_symbol_funding, load_symbol_klines
from .signals import build_close_matrix, build_open_matrix
from .structure_scan import (
    FORWARD_HORIZONS,
    VARIANTS,
    _dynamic_membership,
    _feature_definitions,
    _flatten_stats,
    _period_labels,
    _relative_forward_returns,
    _rolling_corr,
    _rowwise_corr,
    _sorting_metrics_from_arrays,
    _winsorize_cross_section,
)

CORE_FEATURES = (
    "ret_6b",
    "realized_vol_6b",
    "range_vol",
    "abs_funding",
    "corr_btc_30d",
    "imbalance_3b_mean",
)
CONTROL_FEATURES = ("beta_btc_30d", "quote_volume", "trade_count")
ANALYSIS_HORIZONS = (3, 6)
TERCILES = 3


@dataclass(slots=True)
class StructureDecomposeConfig:
    data_dir: Path = Path("data/raw")
    output_dir: Path = Path("results/structure_decompose/decompose_v1")
    top_n: int = 100
    liquidity_lookback_bars: int = 180
    min_history_bars: int = 180


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _bucket_from_frame(frame: pd.DataFrame, quantiles: int) -> np.ndarray:
    ranks = frame.rank(axis=1, method="average", pct=True).to_numpy(dtype=np.float64)
    valid = np.isfinite(frame.to_numpy(dtype=np.float64))
    bucket = np.ceil(np.nan_to_num(ranks, nan=0.0) * quantiles).astype(np.int8)
    bucket[~valid] = 0
    return bucket


def _single_control_residual(target: np.ndarray, control: np.ndarray) -> np.ndarray:
    valid = np.isfinite(target) & np.isfinite(control)
    target_masked = np.where(valid, target, 0.0)
    control_masked = np.where(valid, control, 0.0)
    counts = valid.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        target_mean = np.divide(target_masked.sum(axis=1), counts, out=np.zeros(target.shape[0]), where=counts > 0)
        control_mean = np.divide(control_masked.sum(axis=1), counts, out=np.zeros(target.shape[0]), where=counts > 0)
    target_centered = np.where(valid, target_masked - target_mean[:, None], 0.0)
    control_centered = np.where(valid, control_masked - control_mean[:, None], 0.0)
    denom = (control_centered * control_centered).sum(axis=1)
    beta = np.divide(
        (target_centered * control_centered).sum(axis=1),
        denom,
        out=np.zeros(target.shape[0]),
        where=denom > 0,
    )
    alpha = target_mean - beta * control_mean
    predicted = alpha[:, None] + beta[:, None] * control
    residual = np.where(valid, target - predicted, np.nan)
    only_target = np.isfinite(target) & ~np.isfinite(control)
    residual[only_target] = np.nan
    return residual


def _row_mask_to_period_masks(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    period_labels = _period_labels(index)
    return {
        "full": np.ones(len(index), dtype=bool),
        "2021-2022": (period_labels == "2021-2022").to_numpy(dtype=bool),
        "2023": (period_labels == "2023").to_numpy(dtype=bool),
        "2024": (period_labels == "2024").to_numpy(dtype=bool),
        "2025-2026": (period_labels == "2025-2026").to_numpy(dtype=bool),
    }


def _period_consistency_count(period_rows: pd.DataFrame, reference_sign: float, metric_col: str) -> int:
    if reference_sign == 0.0 or period_rows.empty:
        return 0
    return int((np.sign(period_rows[metric_col]) == reference_sign).sum())


def _mean_rowwise_spearman(
    left_rank: np.ndarray,
    right_rank: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    row_mask: np.ndarray,
) -> tuple[float, float, int, int]:
    valid = left_valid & right_valid
    corr = _rowwise_corr(left_rank, right_rank, valid)
    usable_rows = row_mask & (valid.sum(axis=1) > 1)
    if not usable_rows.any():
        return 0.0, 0.0, 0, 0
    corr_subset = corr[usable_rows]
    return (
        float(corr_subset.mean()),
        float(np.median(corr_subset)),
        int(usable_rows.sum()),
        int((valid & row_mask[:, None]).sum()),
    )


def _sorting_table(
    feature_values: np.ndarray,
    future_values: np.ndarray,
    row_mask: np.ndarray,
    quantiles: int = 5,
) -> dict[str, object]:
    valid = np.isfinite(feature_values) & np.isfinite(future_values)
    if not valid.any():
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
    feature_frame = pd.DataFrame(feature_values).where(valid)
    future_frame = pd.DataFrame(future_values).where(valid)
    feature_rank = feature_frame.rank(axis=1, method="average", pct=True).to_numpy(dtype=np.float64)
    future_rank = future_frame.rank(axis=1, method="average", pct=True).to_numpy(dtype=np.float64)
    bucket = np.ceil(np.nan_to_num(feature_rank, nan=0.0) * quantiles).astype(np.int8)
    bucket[~valid] = 0
    ic = _rowwise_corr(feature_rank, future_rank, valid)
    return _sorting_metrics_from_arrays(bucket, future_values, valid, ic, row_mask)


def _bivariate_rows(
    primary_bucket: np.ndarray,
    secondary_bucket: np.ndarray,
    future_values: np.ndarray,
    future_valid: np.ndarray,
    row_mask: np.ndarray,
    primary_feature: str,
    secondary_feature: str,
    variant: str,
    period: str,
    horizon: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base_mask = future_valid & row_mask[:, None]
    for primary_group in range(1, TERCILES + 1):
        for secondary_group in range(1, TERCILES + 1):
            mask = base_mask & (primary_bucket == primary_group) & (secondary_bucket == secondary_group)
            sample_count = int(mask.sum())
            rows.append(
                {
                    "primary_feature": primary_feature,
                    "secondary_feature": secondary_feature,
                    "variant": variant,
                    "period": period,
                    "horizon_bars": horizon,
                    "primary_bucket": primary_group,
                    "secondary_bucket": secondary_group,
                    "future_mean": float(future_values[mask].mean()) if sample_count else 0.0,
                    "sample_count": sample_count,
                }
            )
    return rows


def _build_condition_mask(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    bucket = _bucket_from_frame(frame, TERCILES)
    if mode == "high":
        return pd.DataFrame(bucket == 3, index=frame.index, columns=frame.columns)
    if mode == "low":
        return pd.DataFrame(bucket == 1, index=frame.index, columns=frame.columns)
    if mode == "extreme":
        return pd.DataFrame(bucket == 3, index=frame.index, columns=frame.columns)
    if mode == "non_extreme":
        return pd.DataFrame(bucket == 1, index=frame.index, columns=frame.columns)
    raise ValueError(f"Unsupported condition mode: {mode}")


def _summary_markdown(
    config: StructureDecomposeConfig,
    role_df: pd.DataFrame,
    residual_df: pd.DataFrame,
    conditional_df: pd.DataFrame,
) -> str:
    def _avg_retention(feature: str, controls: tuple[str, ...]) -> float:
        rows = residual_df[
            (residual_df["target_feature"] == feature)
            & (residual_df["control_feature"].isin(controls))
            & (residual_df["variant"] == "winsorized")
            & (residual_df["horizon_bars"].isin(ANALYSIS_HORIZONS))
        ]
        if rows.empty:
            return 0.0
        return float(rows["ic_retention_ratio"].mean())

    def _role(feature: str) -> str:
        row = role_df[role_df["feature_name"] == feature]
        return str(row["role"].iloc[0]) if not row.empty else "Unknown"

    crowding_candidates = role_df[role_df["feature_name"].isin(["ret_6b", "realized_vol_6b", "abs_funding"])]
    if crowding_candidates.empty:
        crowding_core = "未识别"
    else:
        crowding_core = str(crowding_candidates.sort_values("base_strength", ascending=False)["feature_name"].iloc[0])

    ret_retention = _avg_retention("ret_6b", ("realized_vol_6b", "range_vol"))
    funding_retention = _avg_retention("abs_funding", ("realized_vol_6b", "range_vol"))
    corr_rows = residual_df[
        (residual_df["target_feature"] == "corr_btc_30d")
        & (residual_df["control_feature"].isin(["beta_btc_30d", "quote_volume", "realized_vol_6b"]))
        & (residual_df["variant"] == "winsorized")
        & (residual_df["horizon_bars"].isin(ANALYSIS_HORIZONS))
    ]
    corr_retention = float(corr_rows["ic_retention_ratio"].mean()) if not corr_rows.empty else 0.0
    imbalance_gain_rows = conditional_df[
        (conditional_df["signal_feature"] == "realized_vol_6b")
        & (conditional_df["condition_feature"] == "imbalance_3b_mean_abs")
        & (conditional_df["variant"] == "winsorized")
        & (conditional_df["horizon_bars"].isin(ANALYSIS_HORIZONS))
    ]
    imbalance_gain = float((imbalance_gain_rows["rank_ic_mean"].abs() - imbalance_gain_rows["baseline_rank_ic_mean"].abs()).max()) if not imbalance_gain_rows.empty else 0.0

    lines = [
        "# 结构去重验证结论",
        "",
        "## 样本概况",
        f"- 数据目录: `{config.data_dir}`",
        f"- 输出目录: `{config.output_dir}`",
        f"- 样本池: 每期过去 30 天平均 quote volume 前 {config.top_n}",
        "",
        "## 五个核心问题",
        f"- 拥挤-反转家族当前最核心的变量是 `{crowding_core}`。",
        f"- `ret_6b` 控制波动率后的平均 IC 保留比例约为 `{ret_retention:.2f}`，{'仍有独立信息' if ret_retention >= 0.60 else '独立信息有限'}。",
        f"- `abs_funding` 控制波动率后的平均 IC 保留比例约为 `{funding_retention:.2f}`，{'仍有独立信息' if funding_retention >= 0.60 else '更像波动率/拥挤代理'}。",
        f"- `corr_btc_30d` 在控制 beta、活跃度和波动率后的平均 IC 保留比例约为 `{corr_retention:.2f}`，{'更像独立结构' if corr_retention >= 0.60 else '更像 risk-on / beta 条件变量'}。",
        f"- `imbalance_3b_mean` 当前角色是 `{_role('imbalance_3b_mean')}`，条件化后最大 IC 增量约为 `{imbalance_gain:.4f}`。",
        "",
        "## 角色分工",
    ]
    for row in role_df.sort_values(["role", "feature_name"]).itertuples(index=False):
        lines.append(
            f"- `{row.feature_name}`: {row.role}，base_strength={row.base_strength:.4f}，min_retention={row.min_retention:.2f}，conditional_gain={row.conditional_gain:.4f}"
        )
    return "\n".join(lines) + "\n"


def run_structure_decompose(
    config: StructureDecomposeConfig | None = None,
    *,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    top_n: int | None = None,
    liquidity_lookback_bars: int | None = None,
    min_history_bars: int | None = None,
) -> dict[str, Path]:
    if config is None:
        config = StructureDecomposeConfig()
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
    symbol_order = list(closes.columns)
    row_index = closes.index

    def _align(frame: pd.DataFrame, *, fill_value: float = np.nan) -> pd.DataFrame:
        return frame.reindex(index=row_index, columns=symbol_order, fill_value=fill_value)

    closes = _align(closes)
    opens = _align(build_open_matrix(klines))
    highs = _align(combine_field_matrix(klines, "high"))
    lows = _align(combine_field_matrix(klines, "low"))
    quote_volume = _align(combine_field_matrix(klines, "quote_volume"))
    trade_count = _align(combine_field_matrix(klines, "trade_count"))
    taker_buy_quote_volume = _align(combine_field_matrix(klines, "taker_buy_quote_volume"))
    volume = _align(combine_field_matrix(klines, "volume"))
    funding = _align(
        funding_returns_from_events(funding_raw, opens.index),
        fill_value=0.0,
    )

    membership = _dynamic_membership(
        quote_volume=quote_volume,
        top_n=config.top_n,
        liquidity_lookback_bars=config.liquidity_lookback_bars,
        min_history_bars=config.min_history_bars,
    )
    btc_symbol = "BTCUSDT" if "BTCUSDT" in closes.columns else closes.columns[0]
    returns = closes.pct_change(fill_method=None)
    feature_map = {
        spec.feature_name: frame
        for spec, frame in _feature_definitions(
            {
                "closes": closes,
                "opens": opens,
                "highs": highs,
                "lows": lows,
                "volume": volume,
                "quote_volume": quote_volume,
                "trade_count": trade_count,
                "taker_buy_quote_volume": taker_buy_quote_volume,
                "funding": funding,
                "returns": returns,
            },
            btc_symbol,
        )
    }
    corr_btc_raw = _rolling_corr(returns, returns[btc_symbol]) if btc_symbol in returns.columns else pd.DataFrame(index=returns.index, columns=returns.columns)
    rolling_std = returns.rolling(180, min_periods=90).std()
    btc_std = returns[btc_symbol].rolling(180, min_periods=90).std() if btc_symbol in returns.columns else rolling_std.median(axis=1, skipna=True)
    beta_btc = corr_btc_raw.mul(rolling_std, axis=0).div(btc_std.replace(0.0, np.nan), axis=0).shift(1)
    feature_map["beta_btc_30d"] = beta_btc
    feature_map["quote_volume"] = quote_volume.shift(1)
    feature_map["trade_count"] = trade_count.shift(1)

    selected_features = set(CORE_FEATURES) | set(CONTROL_FEATURES)
    selected_feature_frames = {name: feature_map[name].where(membership) for name in selected_features}
    variant_frames = {
        variant: {
            name: (_winsorize_cross_section(frame) if variant == "winsorized" else frame)
            for name, frame in selected_feature_frames.items()
        }
        for variant in VARIANTS
    }

    forward_return_frames = {
        horizon: _relative_forward_returns(opens.shift(-(horizon + 1)).div(opens.shift(-1)).sub(1.0))
        for horizon in FORWARD_HORIZONS
    }
    forward_arrays = {horizon: frame.to_numpy(dtype=np.float64) for horizon, frame in forward_return_frames.items()}
    forward_valids = {horizon: np.isfinite(array) for horizon, array in forward_arrays.items()}
    period_masks = _row_mask_to_period_masks(opens.index)

    feature_arrays: dict[str, dict[str, dict[str, object]]] = {}
    for variant in VARIANTS:
        feature_arrays[variant] = {}
        for feature_name, frame in variant_frames[variant].items():
            values = frame.to_numpy(dtype=np.float64)
            valid = np.isfinite(values)
            ranks = frame.rank(axis=1, method="average", pct=True).to_numpy(dtype=np.float64)
            bucket5 = _bucket_from_frame(frame, 5)
            bucket3 = _bucket_from_frame(frame, TERCILES)
            feature_arrays[variant][feature_name] = {
                "frame": frame,
                "values": values,
                "valid": valid,
                "ranks": ranks,
                "bucket5": bucket5,
                "bucket3": bucket3,
            }
        abs_imbalance_frame = variant_frames[variant]["imbalance_3b_mean"].abs()
        feature_arrays[variant]["imbalance_3b_mean_abs"] = {
            "frame": abs_imbalance_frame,
            "values": abs_imbalance_frame.to_numpy(dtype=np.float64),
            "valid": np.isfinite(abs_imbalance_frame.to_numpy(dtype=np.float64)),
            "ranks": abs_imbalance_frame.rank(axis=1, method="average", pct=True).to_numpy(dtype=np.float64),
            "bucket5": _bucket_from_frame(abs_imbalance_frame, 5),
            "bucket3": _bucket_from_frame(abs_imbalance_frame, TERCILES),
        }

    corr_full_rows: list[dict[str, object]] = []
    corr_yearly_rows: list[dict[str, object]] = []
    overview_rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []
    bivariate_rows: list[dict[str, object]] = []
    conditional_rows: list[dict[str, object]] = []

    base_metrics: dict[tuple[str, str, int, str], dict[str, object]] = {}
    for variant in VARIANTS:
        for feature_name in CORE_FEATURES:
            frame = variant_frames[variant][feature_name]
            stats = _flatten_stats(frame)
            row: dict[str, object] = {
                "feature_name": feature_name,
                "variant": variant,
                "coverage_ratio": float(selected_feature_frames[feature_name].count().sum() / membership.sum().sum()) if membership.sum().sum() else 0.0,
                **stats,
            }
            bundle = feature_arrays[variant][feature_name]
            for horizon in FORWARD_HORIZONS:
                valid = bundle["valid"] & forward_valids[horizon]
                ic = _rowwise_corr(bundle["ranks"], forward_return_frames[horizon].rank(axis=1, method="average", pct=True).to_numpy(dtype=np.float64), valid)
                full_metrics = _sorting_metrics_from_arrays(bundle["bucket5"], forward_arrays[horizon], valid, ic, period_masks["full"])
                row[f"rank_ic_{horizon}b"] = full_metrics["rank_ic_mean"]
                row[f"spread_q5_q1_{horizon}b"] = full_metrics["spread_q5_q1"]
                for period_name, row_mask in period_masks.items():
                    base_metrics[(feature_name, variant, horizon, period_name)] = _sorting_metrics_from_arrays(
                        bundle["bucket5"],
                        forward_arrays[horizon],
                        valid,
                        ic,
                        row_mask,
                    )
            overview_rows.append(row)

    core_pairs = [(x, y) for idx, x in enumerate(CORE_FEATURES) for y in CORE_FEATURES[idx + 1 :]]
    for variant in VARIANTS:
        for left, right in core_pairs:
            left_bundle = feature_arrays[variant][left]
            right_bundle = feature_arrays[variant][right]
            full_mean, full_median, ts_count, sample_size = _mean_rowwise_spearman(
                left_bundle["ranks"],
                right_bundle["ranks"],
                left_bundle["valid"],
                right_bundle["valid"],
                period_masks["full"],
            )
            corr_full_rows.append(
                {
                    "variant": variant,
                    "left_feature": left,
                    "right_feature": right,
                    "mean_spearman": full_mean,
                    "median_spearman": full_median,
                    "timestamp_count": ts_count,
                    "sample_size": sample_size,
                }
            )
            for period_name, row_mask in period_masks.items():
                if period_name == "full":
                    continue
                mean_corr, median_corr, p_ts_count, p_sample_size = _mean_rowwise_spearman(
                    left_bundle["ranks"],
                    right_bundle["ranks"],
                    left_bundle["valid"],
                    right_bundle["valid"],
                    row_mask,
                )
                corr_yearly_rows.append(
                    {
                        "variant": variant,
                        "period": period_name,
                        "left_feature": left,
                        "right_feature": right,
                        "mean_spearman": mean_corr,
                        "median_spearman": median_corr,
                        "timestamp_count": p_ts_count,
                        "sample_size": p_sample_size,
                    }
                )

    residual_specs = [
        ("ret_6b", "realized_vol_6b"),
        ("ret_6b", "range_vol"),
        ("abs_funding", "realized_vol_6b"),
        ("abs_funding", "range_vol"),
        ("realized_vol_6b", "ret_6b"),
        ("realized_vol_6b", "abs_funding"),
        ("range_vol", "realized_vol_6b"),
        ("corr_btc_30d", "beta_btc_30d"),
        ("corr_btc_30d", "realized_vol_6b"),
        ("corr_btc_30d", "quote_volume"),
        ("imbalance_3b_mean", "ret_6b"),
        ("imbalance_3b_mean", "realized_vol_6b"),
    ]
    for variant in VARIANTS:
        for target_feature, control_feature in residual_specs:
            target_values = feature_arrays[variant][target_feature]["values"]
            control_values = feature_arrays[variant][control_feature]["values"]
            residual_values = _single_control_residual(target_values, control_values)
            residual_frame = pd.DataFrame(
                residual_values,
                index=variant_frames[variant][target_feature].index,
                columns=variant_frames[variant][target_feature].columns,
            )
            residual_valid = np.isfinite(residual_values)
            residual_ranks = residual_frame.rank(axis=1, method="average", pct=True).to_numpy(dtype=np.float64)
            residual_bucket5 = _bucket_from_frame(residual_frame, 5)
            target_bundle = feature_arrays[variant][target_feature]
            for horizon in FORWARD_HORIZONS:
                original_valid = target_bundle["valid"] & forward_valids[horizon]
                original_ic = _rowwise_corr(
                    target_bundle["ranks"],
                    forward_return_frames[horizon].rank(axis=1, method="average", pct=True).to_numpy(dtype=np.float64),
                    original_valid,
                )
                residual_future_valid = residual_valid & forward_valids[horizon]
                residual_ic = _rowwise_corr(
                    residual_ranks,
                    forward_return_frames[horizon].rank(axis=1, method="average", pct=True).to_numpy(dtype=np.float64),
                    residual_future_valid,
                )
                original_full = _sorting_metrics_from_arrays(
                    target_bundle["bucket5"],
                    forward_arrays[horizon],
                    original_valid,
                    original_ic,
                    period_masks["full"],
                )
                residual_full = _sorting_metrics_from_arrays(
                    residual_bucket5,
                    forward_arrays[horizon],
                    residual_future_valid,
                    residual_ic,
                    period_masks["full"],
                )
                residual_period_rows = []
                for period_name, row_mask in period_masks.items():
                    if period_name == "full":
                        continue
                    residual_period_rows.append(
                        {
                            "period": period_name,
                            **_sorting_metrics_from_arrays(
                                residual_bucket5,
                                forward_arrays[horizon],
                                residual_future_valid,
                                residual_ic,
                                row_mask,
                            ),
                        }
                    )
                residual_period_df = pd.DataFrame(residual_period_rows)
                residual_sign = float(np.sign(residual_full["rank_ic_mean"]))
                original_sign = float(np.sign(original_full["rank_ic_mean"]))
                residual_rows.append(
                    {
                        "target_feature": target_feature,
                        "control_feature": control_feature,
                        "variant": variant,
                        "horizon_bars": horizon,
                        "original_rank_ic_mean": original_full["rank_ic_mean"],
                        "original_spread_q5_q1": original_full["spread_q5_q1"],
                        "residual_rank_ic_mean": residual_full["rank_ic_mean"],
                        "residual_spread_q5_q1": residual_full["spread_q5_q1"],
                        "ic_retention_ratio": float(abs(residual_full["rank_ic_mean"]) / abs(original_full["rank_ic_mean"])) if original_full["rank_ic_mean"] else 0.0,
                        "spread_retention_ratio": float(abs(residual_full["spread_q5_q1"]) / abs(original_full["spread_q5_q1"])) if original_full["spread_q5_q1"] else 0.0,
                        "original_period_consistency_count": _period_consistency_count(
                            pd.DataFrame(
                                [
                                    {
                                        "period": period_name,
                                        **_sorting_metrics_from_arrays(
                                            target_bundle["bucket5"],
                                            forward_arrays[horizon],
                                            original_valid,
                                            original_ic,
                                            row_mask,
                                        ),
                                    }
                                    for period_name, row_mask in period_masks.items()
                                    if period_name != "full"
                                ]
                            ),
                            original_sign,
                            "rank_ic_mean",
                        ),
                        "residual_period_consistency_count": _period_consistency_count(
                            residual_period_df,
                            residual_sign,
                            "rank_ic_mean",
                        ),
                    }
                )

    bivariate_specs = [
        ("realized_vol_6b", "abs_funding"),
        ("ret_6b", "realized_vol_6b"),
        ("corr_btc_30d", "realized_vol_6b"),
        ("ret_6b", "imbalance_3b_mean"),
    ]
    for variant in VARIANTS:
        for primary_feature, secondary_feature in bivariate_specs:
            primary_bucket = feature_arrays[variant][primary_feature]["bucket3"]
            secondary_bucket = feature_arrays[variant][secondary_feature]["bucket3"]
            for horizon in ANALYSIS_HORIZONS:
                for period_name, row_mask in period_masks.items():
                    bivariate_rows.extend(
                        _bivariate_rows(
                            primary_bucket=primary_bucket,
                            secondary_bucket=secondary_bucket,
                            future_values=forward_arrays[horizon],
                            future_valid=forward_valids[horizon],
                            row_mask=row_mask,
                            primary_feature=primary_feature,
                            secondary_feature=secondary_feature,
                            variant=variant,
                            period=period_name,
                            horizon=horizon,
                        )
                    )

    conditional_specs = [
        ("ret_6b", "corr_btc_30d", "high"),
        ("ret_6b", "corr_btc_30d", "low"),
        ("realized_vol_6b", "imbalance_3b_mean_abs", "extreme"),
        ("realized_vol_6b", "imbalance_3b_mean_abs", "non_extreme"),
        ("abs_funding", "corr_btc_30d", "high"),
        ("abs_funding", "corr_btc_30d", "low"),
    ]
    for variant in VARIANTS:
        for signal_feature, condition_feature, mode in conditional_specs:
            condition_frame = feature_arrays[variant][condition_feature]["frame"]
            condition_mask = _build_condition_mask(condition_frame, mode)
            signal_frame = feature_arrays[variant][signal_feature]["frame"].where(condition_mask)
            signal_values = signal_frame.to_numpy(dtype=np.float64)
            for horizon in ANALYSIS_HORIZONS:
                future_frame = forward_return_frames[horizon].where(condition_mask)
                future_values = future_frame.to_numpy(dtype=np.float64)
                metrics_by_period: dict[str, dict[str, object]] = {}
                for period_name, row_mask in period_masks.items():
                    metrics_by_period[period_name] = _sorting_table(signal_values, future_values, row_mask, quantiles=5)
                    conditional_rows.append(
                        {
                            "signal_feature": signal_feature,
                            "condition_feature": condition_feature,
                            "condition_mode": mode,
                            "variant": variant,
                            "period": period_name,
                            "horizon_bars": horizon,
                            **metrics_by_period[period_name],
                            "baseline_rank_ic_mean": base_metrics[(signal_feature, variant, horizon, period_name)]["rank_ic_mean"],
                            "baseline_spread_q5_q1": base_metrics[(signal_feature, variant, horizon, period_name)]["spread_q5_q1"],
                        }
                    )

    overview_df = pd.DataFrame(overview_rows).sort_values(["feature_name", "variant"]).reset_index(drop=True)
    corr_full_df = pd.DataFrame(corr_full_rows).sort_values(["variant", "left_feature", "right_feature"]).reset_index(drop=True)
    corr_yearly_df = pd.DataFrame(corr_yearly_rows).sort_values(["variant", "period", "left_feature", "right_feature"]).reset_index(drop=True)
    residual_df = pd.DataFrame(residual_rows).sort_values(["target_feature", "control_feature", "variant", "horizon_bars"]).reset_index(drop=True)
    bivariate_df = pd.DataFrame(bivariate_rows).sort_values(
        ["primary_feature", "secondary_feature", "variant", "period", "horizon_bars", "primary_bucket", "secondary_bucket"]
    ).reset_index(drop=True)
    conditional_df = pd.DataFrame(conditional_rows).sort_values(
        ["signal_feature", "condition_feature", "condition_mode", "variant", "period", "horizon_bars"]
    ).reset_index(drop=True)

    role_rows: list[dict[str, object]] = []
    for feature_name in CORE_FEATURES:
        win_row = overview_df[(overview_df["feature_name"] == feature_name) & (overview_df["variant"] == "winsorized")]
        raw_row = overview_df[(overview_df["feature_name"] == feature_name) & (overview_df["variant"] == "raw")]
        base_strength = float(
            win_row[[f"rank_ic_{h}b" for h in ANALYSIS_HORIZONS]].abs().to_numpy(dtype=float).mean()
        ) if not win_row.empty else 0.0
        raw_strength = float(
            raw_row[[f"rank_ic_{h}b" for h in ANALYSIS_HORIZONS]].abs().to_numpy(dtype=float).mean()
        ) if not raw_row.empty else 0.0
        win_residuals = residual_df[(residual_df["target_feature"] == feature_name) & (residual_df["variant"] == "winsorized") & (residual_df["horizon_bars"].isin(ANALYSIS_HORIZONS))]
        raw_residuals = residual_df[(residual_df["target_feature"] == feature_name) & (residual_df["variant"] == "raw") & (residual_df["horizon_bars"].isin(ANALYSIS_HORIZONS))]
        min_retention = float(win_residuals["ic_retention_ratio"].min()) if not win_residuals.empty else 0.0
        raw_min_retention = float(raw_residuals["ic_retention_ratio"].min()) if not raw_residuals.empty else 0.0
        win_conditionals = conditional_df[(conditional_df["signal_feature"] == feature_name) & (conditional_df["variant"] == "winsorized") & (conditional_df["period"] == "full")]
        conditional_gain = float((win_conditionals["rank_ic_mean"].abs() - win_conditionals["baseline_rank_ic_mean"].abs()).max()) if not win_conditionals.empty else 0.0

        role = "Redundant proxy"
        rationale = "baseline weak or mostly explained by stronger variables"
        if feature_name == "range_vol":
            if min_retention >= 0.60 and raw_min_retention >= 0.60 and base_strength >= 0.02:
                role = "Secondary signal"
                rationale = "retains information after realized volatility control"
            else:
                role = "Redundant proxy"
                rationale = "mostly overlaps with realized_vol_6b"
        elif feature_name in {"corr_btc_30d", "imbalance_3b_mean"}:
            if min_retention >= 0.60 and raw_min_retention >= 0.60 and base_strength >= 0.015:
                role = "Secondary signal"
                rationale = "keeps meaningful information after controls"
            elif conditional_gain > 0.002:
                role = "Condition/filter"
                rationale = "works better as a condition than as a standalone sorter"
            else:
                role = "Redundant proxy"
                rationale = "standalone and conditional value both limited"
        else:
            if min_retention >= 0.60 and raw_min_retention >= 0.60 and base_strength >= 0.02 and raw_strength >= 0.015:
                role = "Primary signal"
                rationale = "strong base IC and stable residual information"
            elif min_retention >= 0.50 and raw_min_retention >= 0.50 and base_strength >= 0.012:
                role = "Secondary signal"
                rationale = "keeps partial independent information"
            elif conditional_gain > 0.002:
                role = "Condition/filter"
                rationale = "conditional use dominates standalone use"
        role_rows.append(
            {
                "feature_name": feature_name,
                "base_strength": base_strength,
                "raw_base_strength": raw_strength,
                "min_retention": min_retention,
                "raw_min_retention": raw_min_retention,
                "conditional_gain": conditional_gain,
                "role": role,
                "rationale": rationale,
            }
        )

    role_df = pd.DataFrame(role_rows).sort_values(["role", "feature_name"]).reset_index(drop=True)

    paths = {
        "core_feature_overview": config.output_dir / "core_feature_overview.csv",
        "feature_corr_full": config.output_dir / "feature_corr_full.csv",
        "feature_corr_yearly": config.output_dir / "feature_corr_yearly.csv",
        "residual_signal_tests": config.output_dir / "residual_signal_tests.csv",
        "bivariate_sorts": config.output_dir / "bivariate_sorts.csv",
        "conditional_sorts": config.output_dir / "conditional_sorts.csv",
        "factor_role_summary": config.output_dir / "factor_role_summary.csv",
        "summary": config.output_dir / "summary.md",
        "artifacts": config.output_dir / "artifacts.json",
    }

    overview_df.to_csv(paths["core_feature_overview"], index=False)
    corr_full_df.to_csv(paths["feature_corr_full"], index=False)
    corr_yearly_df.to_csv(paths["feature_corr_yearly"], index=False)
    residual_df.to_csv(paths["residual_signal_tests"], index=False)
    bivariate_df.to_csv(paths["bivariate_sorts"], index=False)
    conditional_df.to_csv(paths["conditional_sorts"], index=False)
    role_df.to_csv(paths["factor_role_summary"], index=False)
    paths["summary"].write_text(_summary_markdown(config, role_df, residual_df, conditional_df), encoding="utf-8")
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
            "roles": role_rows,
        },
    )
    return paths
