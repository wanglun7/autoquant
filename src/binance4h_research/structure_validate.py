from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .market_data import combine_field_matrix, funding_returns_from_events
from .data import load_symbol_funding, load_symbol_klines
from .signals import build_close_matrix, build_open_matrix
from .structure_decompose import _bucket_from_frame
from .structure_scan import (
    VARIANTS,
    _dynamic_membership,
    _feature_definitions,
    _period_labels,
    _relative_forward_returns,
    _rowwise_corr,
    _sorting_metrics_from_arrays,
    _winsorize_cross_section,
)

CORE_FEATURES = ("realized_vol_6b", "corr_btc_30d", "ret_6b", "abs_funding")
ANALYSIS_HORIZONS = (3, 6)
TERCILES = 3


@dataclass(slots=True)
class StructureValidateConfig:
    data_dir: Path = Path("data/raw")
    output_dir: Path = Path("results/structure_validate/validate_v1")
    top_n: int = 100
    liquidity_lookback_bars: int = 180
    min_history_bars: int = 180


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _row_period_masks(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    period_labels = _period_labels(index)
    return {
        "full": np.ones(len(index), dtype=bool),
        "2021-2022": (period_labels == "2021-2022").to_numpy(dtype=bool),
        "2023": (period_labels == "2023").to_numpy(dtype=bool),
        "2024": (period_labels == "2024").to_numpy(dtype=bool),
        "2025-2026": (period_labels == "2025-2026").to_numpy(dtype=bool),
    }


def _median_condition_mask(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    median = frame.median(axis=1, skipna=True)
    if mode == "high":
        return frame.ge(median, axis=0) & frame.notna()
    if mode == "low":
        return frame.lt(median, axis=0) & frame.notna()
    raise ValueError(f"Unsupported condition mode: {mode}")


def _rowwise_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.to_numpy(dtype=np.float64)
    valid = np.isfinite(values)
    masked = np.where(valid, values, 0.0)
    counts = valid.sum(axis=1)
    mean = np.divide(masked.sum(axis=1), counts, out=np.zeros(len(frame.index)), where=counts > 0)
    centered = np.where(valid, values - mean[:, None], 0.0)
    denom = np.sqrt(np.divide((centered * centered).sum(axis=1), counts, out=np.zeros(len(frame.index)), where=counts > 0))
    zscore = np.divide(centered, denom[:, None], out=np.full_like(values, np.nan), where=(valid & (denom[:, None] > 0)))
    zscore[counts <= 1] = np.nan
    return pd.DataFrame(zscore, index=frame.index, columns=frame.columns)


def _sorting_from_frames(feature_frame: pd.DataFrame, future_frame: pd.DataFrame, row_mask: np.ndarray, quantiles: int = 5) -> dict[str, object]:
    feature_values = feature_frame.to_numpy(dtype=np.float64)
    future_values = future_frame.to_numpy(dtype=np.float64)
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
    feature_rank = feature_frame.rank(axis=1, method="average", pct=True).to_numpy(dtype=np.float64)
    future_rank = future_frame.rank(axis=1, method="average", pct=True).to_numpy(dtype=np.float64)
    bucket = np.ceil(np.nan_to_num(feature_rank, nan=0.0) * quantiles).astype(np.int8)
    bucket[~valid] = 0
    ic = _rowwise_corr(feature_rank, future_rank, valid)
    return _sorting_metrics_from_arrays(bucket, future_values, valid, ic, row_mask)


def _period_consistency_count(rows: pd.DataFrame, metric_col: str) -> int:
    full = rows[rows["period"] == "full"]
    if full.empty:
        return 0
    sign = float(np.sign(full[metric_col].iloc[0]))
    if sign == 0.0:
        return 0
    annual = rows[rows["period"] != "full"]
    return int((np.sign(annual[metric_col]) == sign).sum())


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


def _ols_rows(
    model_name: str,
    feature_names: tuple[str, ...],
    feature_frames: dict[str, pd.DataFrame],
    future_frame: pd.DataFrame,
    period_masks: dict[str, np.ndarray],
    variant: str,
) -> list[dict[str, object]]:
    zscored = {name: _rowwise_zscore(feature_frames[name]).to_numpy(dtype=np.float64) for name in feature_names}
    y = future_frame.to_numpy(dtype=np.float64)
    period_by_row = pd.Series(_period_labels(future_frame.index).to_numpy(), index=future_frame.index)
    row_outputs: list[dict[str, object]] = []
    for row_idx, timestamp in enumerate(future_frame.index):
        x_cols = [zscored[name][row_idx] for name in feature_names]
        valid = np.isfinite(y[row_idx])
        for column in x_cols:
            valid &= np.isfinite(column)
        if int(valid.sum()) < len(feature_names) + 1:
            continue
        design = np.column_stack([np.ones(int(valid.sum()))] + [column[valid] for column in x_cols])
        target = y[row_idx, valid]
        try:
            beta, *_ = np.linalg.lstsq(design, target, rcond=None)
        except np.linalg.LinAlgError:
            continue
        fitted = design @ beta
        residual = target - fitted
        sst = float(np.sum((target - target.mean()) ** 2))
        sse = float(np.sum(residual**2))
        r2 = 0.0 if sst <= 0.0 else float(max(0.0, 1.0 - (sse / sst)))
        row = {
            "timestamp": timestamp,
            "period": str(period_by_row.loc[timestamp]),
            "r2": r2,
        }
        for idx, feature_name in enumerate(feature_names, start=1):
            row[f"coef_{feature_name}"] = float(beta[idx])
        row_outputs.append(row)
    row_df = pd.DataFrame(row_outputs)
    if row_df.empty:
        return []
    result_rows: list[dict[str, object]] = []
    for period_name, row_mask in period_masks.items():
        if period_name == "full":
            subset = row_df
        else:
            subset = row_df[row_df["period"] == period_name]
        timestamp_count = int(len(subset))
        mean_r2 = float(subset["r2"].mean()) if timestamp_count else 0.0
        for feature_name in feature_names:
            coef_col = f"coef_{feature_name}"
            mean_coef = float(subset[coef_col].mean()) if timestamp_count else 0.0
            std_coef = float(subset[coef_col].std(ddof=0)) if timestamp_count > 1 else 0.0
            t_stat = 0.0
            if timestamp_count > 1 and std_coef > 0.0:
                t_stat = float(mean_coef / (std_coef / np.sqrt(timestamp_count)))
            result_rows.append(
                {
                    "model_name": model_name,
                    "variant": variant,
                    "period": period_name,
                    "feature_name": feature_name,
                    "coef_mean": mean_coef,
                    "coef_std": std_coef,
                    "coef_tstat": t_stat,
                    "mean_r2": mean_r2,
                    "timestamp_count": timestamp_count,
                }
            )
    return result_rows


def _role_summary(
    univariate_df: pd.DataFrame,
    conditional_df: pd.DataFrame,
    regression_df: pd.DataFrame,
    bivariate_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def _uni(feature: str, variant: str = "winsorized", horizon: int = 6) -> pd.Series:
        subset = univariate_df[
            (univariate_df["feature_name"] == feature)
            & (univariate_df["variant"] == variant)
            & (univariate_df["period"] == "full")
            & (univariate_df["horizon_bars"] == horizon)
        ]
        return subset.iloc[0] if not subset.empty else pd.Series(dtype=float)

    def _reg(model_name: str, feature: str, variant: str = "winsorized", period: str = "full") -> pd.Series:
        subset = regression_df[
            (regression_df["model_name"] == model_name)
            & (regression_df["feature_name"] == feature)
            & (regression_df["variant"] == variant)
            & (regression_df["period"] == period)
        ]
        return subset.iloc[0] if not subset.empty else pd.Series(dtype=float)

    def _cond_gain(signal_feature: str, metric_col: str = "rank_ic_mean") -> float:
        rows_local = conditional_df[
            (conditional_df["signal_feature"] == signal_feature)
            & (conditional_df["variant"] == "winsorized")
            & (conditional_df["period"] == "full")
            & (conditional_df["horizon_bars"] == 6)
        ]
        if rows_local.empty:
            return 0.0
        return float((rows_local[metric_col].abs() - rows_local[f"baseline_{metric_col}"].abs()).max())

    for feature_name in CORE_FEATURES:
        win_row = _uni(feature_name, "winsorized", 6)
        raw_row = _uni(feature_name, "raw", 6)
        win_spread = float(win_row.get("spread_q5_q1", 0.0))
        raw_spread = float(raw_row.get("spread_q5_q1", 0.0))
        win_ic = float(win_row.get("rank_ic_mean", 0.0))
        raw_ic = float(raw_row.get("rank_ic_mean", 0.0))
        win_consistency = _period_consistency_count(
            univariate_df[
                (univariate_df["feature_name"] == feature_name)
                & (univariate_df["variant"] == "winsorized")
                & (univariate_df["horizon_bars"] == 6)
            ],
            "rank_ic_mean",
        )
        conditional_gain = _cond_gain(feature_name)
        model3 = _reg("model_3", feature_name, "winsorized", "full")
        coef_mean = float(model3.get("coef_mean", 0.0))
        coef_tstat = float(model3.get("coef_tstat", 0.0))

        role = "redundant"
        rationale = "standalone and conditional evidence both limited"
        if feature_name == "realized_vol_6b":
            if win_ic < 0.0 and raw_ic < 0.0 and win_consistency >= 3 and (abs(win_ic) >= 0.06 or conditional_gain >= 0.01):
                role = "Primary signal"
                rationale = "strong negative rank structure remains, with cleaner use under ret_6b filtering"
            else:
                role = "downgraded"
                rationale = "failed to preserve expected negative structure"
        elif feature_name == "corr_btc_30d":
            if win_ic > 0.0 and raw_ic > 0.0 and win_consistency >= 3 and coef_mean > 0.0:
                role = "Secondary signal"
                rationale = "positive relative-strength structure remains after joint validation"
                if abs(coef_tstat) >= 2.0 and abs(win_ic) >= 0.05:
                    role = "Primary signal"
                    rationale = "positive standalone and regression evidence are both strong"
            else:
                role = "filter"
                rationale = "more useful as a regime condition than a direct sorter"
        elif feature_name == "ret_6b":
            if conditional_gain >= 0.005 and abs(coef_mean) < max(abs(win_ic), 1e-6):
                role = "filter"
                rationale = "conditional gain dominates direct explanatory power"
            elif win_ic < 0.0 and raw_ic < 0.0 and win_consistency >= 3 and coef_mean < 0.0:
                role = "Secondary signal"
                rationale = "negative reversal remains visible after joint validation"
            else:
                role = "redundant"
                rationale = "weak direct signal and weak conditional gain"
        elif feature_name == "abs_funding":
            high_vol_rows = bivariate_df[
                (bivariate_df["primary_feature"] == "realized_vol_6b")
                & (bivariate_df["secondary_feature"] == "abs_funding")
                & (bivariate_df["variant"] == "winsorized")
                & (bivariate_df["period"] == "full")
                & (bivariate_df["horizon_bars"] == 6)
            ]
            high_vol_high_funding = high_vol_rows[
                (high_vol_rows["primary_bucket"] == 3) & (high_vol_rows["secondary_bucket"] == 3)
            ]["future_mean"]
            high_vol_low_funding = high_vol_rows[
                (high_vol_rows["primary_bucket"] == 3) & (high_vol_rows["secondary_bucket"] == 1)
            ]["future_mean"]
            if not high_vol_high_funding.empty and not high_vol_low_funding.empty and float(high_vol_high_funding.iloc[0]) < float(high_vol_low_funding.iloc[0]):
                role = "confirmation proxy"
                rationale = "mainly deepens the bad tail inside high-volatility names"
            elif win_ic < 0.0 and raw_ic < 0.0 and win_consistency >= 3:
                role = "Secondary signal"
                rationale = "negative structure remains visible as a standalone sorter"
        rows.append(
            {
                "feature_name": feature_name,
                "winsorized_rank_ic_6b": win_ic,
                "raw_rank_ic_6b": raw_ic,
                "winsorized_spread_6b": win_spread,
                "raw_spread_6b": raw_spread,
                "annual_consistency_6b": win_consistency,
                "conditional_gain_6b": conditional_gain,
                "model3_coef_mean": coef_mean,
                "model3_coef_tstat": coef_tstat,
                "role": role,
                "rationale": rationale,
            }
        )
    return pd.DataFrame(rows).sort_values("feature_name").reset_index(drop=True)


def _summary_markdown(
    config: StructureValidateConfig,
    role_df: pd.DataFrame,
    univariate_df: pd.DataFrame,
    conditional_df: pd.DataFrame,
    regression_df: pd.DataFrame,
) -> str:
    role_map = {row.feature_name: row for row in role_df.itertuples(index=False)}
    model1 = regression_df[(regression_df["model_name"] == "model_1") & (regression_df["variant"] == "winsorized") & (regression_df["period"] == "full")]
    model2 = regression_df[(regression_df["model_name"] == "model_2") & (regression_df["variant"] == "winsorized") & (regression_df["period"] == "full")]
    model3 = regression_df[(regression_df["model_name"] == "model_3") & (regression_df["variant"] == "winsorized") & (regression_df["period"] == "full")]
    model1_r2 = float(model1["mean_r2"].iloc[0]) if not model1.empty else 0.0
    model2_r2 = float(model2["mean_r2"].iloc[0]) if not model2.empty else 0.0
    model3_r2 = float(model3["mean_r2"].iloc[0]) if not model3.empty else 0.0
    model2_delta = float(model2["delta_r2_vs_prev"].iloc[0]) if not model2.empty else 0.0
    model3_delta = float(model3["delta_r2_vs_prev"].iloc[0]) if not model3.empty else 0.0

    ret_rows = conditional_df[
        (conditional_df["signal_feature"] == "realized_vol_6b")
        & (conditional_df["condition_feature"] == "ret_6b")
        & (conditional_df["variant"] == "winsorized")
        & (conditional_df["period"] == "full")
        & (conditional_df["horizon_bars"] == 6)
    ]
    corr_rows = conditional_df[
        (conditional_df["signal_feature"] == "realized_vol_6b")
        & (conditional_df["condition_feature"] == "corr_btc_30d")
        & (conditional_df["variant"] == "winsorized")
        & (conditional_df["period"] == "full")
        & (conditional_df["horizon_bars"] == 6)
    ]

    realized_role = role_map.get("realized_vol_6b")
    corr_role = role_map.get("corr_btc_30d")
    ret_role = role_map.get("ret_6b")
    funding_role = role_map.get("abs_funding")
    keepers = [
        feature
        for feature in ("realized_vol_6b", "corr_btc_30d", "ret_6b")
        if role_map.get(feature) and role_map[feature].role in {"Primary signal", "Secondary signal", "filter"}
    ]

    lines = [
        "# 低维统计验证结论",
        "",
        "## 样本概况",
        f"- 数据目录: `{config.data_dir}`",
        f"- 输出目录: `{config.output_dir}`",
        f"- 样本池: 每期过去 30 天平均 quote volume 前 {config.top_n}",
        "",
        "## 四个核心问题",
        f"- 当前最小低维结构{'是' if model1_r2 >= (model3_r2 * 0.85 if model3_r2 else 0.0) else '不是'} `realized_vol_6b + corr_btc_30d`。model_1 mean R^2={model1_r2:.4f}，model_3 mean R^2={model3_r2:.4f}。",
        f"- `ret_6b` 当前角色: `{ret_role.role if ret_role else 'unknown'}`。加入 `ret_6b` 后 model_2 的增量 R^2 约为 `{model2_delta:.4f}`，且 `realized_vol_6b` 在高 `ret_6b` 条件下的最大 IC 增量约为 `{float((ret_rows['rank_ic_mean'].abs() - ret_rows['baseline_rank_ic_mean'].abs()).max()) if not ret_rows.empty else 0.0:.4f}`。",
        f"- `abs_funding` 当前角色: `{funding_role.role if funding_role else 'unknown'}`。它更像 `{funding_role.rationale if funding_role else 'unknown'}`。",
        f"- 下一步规则原型应优先保留: `{', '.join(keepers[:3]) if keepers else 'realized_vol_6b, corr_btc_30d'}`。",
        "",
        "## 变量分工",
        f"- `realized_vol_6b`: {realized_role.role if realized_role else 'unknown'}，6bar Rank IC={realized_role.winsorized_rank_ic_6b:.4f}，model_3 coef={realized_role.model3_coef_mean:.6f}。" if realized_role else "- `realized_vol_6b`: unknown",
        f"- `corr_btc_30d`: {corr_role.role if corr_role else 'unknown'}，6bar Rank IC={corr_role.winsorized_rank_ic_6b:.4f}，model_3 coef={corr_role.model3_coef_mean:.6f}。" if corr_role else "- `corr_btc_30d`: unknown",
        f"- `ret_6b`: {ret_role.role if ret_role else 'unknown'}，6bar Rank IC={ret_role.winsorized_rank_ic_6b:.4f}，conditional gain={ret_role.conditional_gain_6b:.4f}。" if ret_role else "- `ret_6b`: unknown",
        f"- `abs_funding`: {funding_role.role if funding_role else 'unknown'}，6bar Rank IC={funding_role.winsorized_rank_ic_6b:.4f}，model_3 coef={funding_role.model3_coef_mean:.6f}。" if funding_role else "- `abs_funding`: unknown",
        "",
        "## 回归增量",
        f"- model_1: `future_ret ~ realized_vol_6b + corr_btc_30d`，mean R^2={model1_r2:.4f}",
        f"- model_2: `+ ret_6b`，mean R^2={model2_r2:.4f}",
        f"- model_3: `+ abs_funding`，mean R^2={model3_r2:.4f}",
        f"- 增量 R^2: `ret_6b` 约 `{model2_delta:.4f}`，`abs_funding` 约 `{model3_delta:.4f}`。",
        f"- `realized_vol_6b` 在高/低相关条件下的最大 IC 变化约为 `{float((corr_rows['rank_ic_mean'].abs() - corr_rows['baseline_rank_ic_mean'].abs()).max()) if not corr_rows.empty else 0.0:.4f}`。",
    ]
    return "\n".join(lines) + "\n"


def run_structure_validate(
    config: StructureValidateConfig | None = None,
    *,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    top_n: int | None = None,
    liquidity_lookback_bars: int | None = None,
    min_history_bars: int | None = None,
) -> dict[str, Path]:
    if config is None:
        config = StructureValidateConfig()
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
    volume = _align(combine_field_matrix(klines, "volume"))
    quote_volume = _align(combine_field_matrix(klines, "quote_volume"))
    trade_count = _align(combine_field_matrix(klines, "trade_count"))
    taker_buy_quote_volume = _align(combine_field_matrix(klines, "taker_buy_quote_volume"))
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
    feature_map = {
        spec.feature_name: frame.where(membership)
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
                "returns": closes.pct_change(fill_method=None),
            },
            btc_symbol,
        )
        if spec.feature_name in CORE_FEATURES
    }
    variant_frames = {
        variant: {
            name: (_winsorize_cross_section(frame) if variant == "winsorized" else frame)
            for name, frame in feature_map.items()
        }
        for variant in VARIANTS
    }

    forward_frames = {
        horizon: _relative_forward_returns(opens.shift(-(horizon + 1)).div(opens.shift(-1)).sub(1.0))
        for horizon in ANALYSIS_HORIZONS
    }
    period_masks = _row_period_masks(opens.index)

    univariate_rows: list[dict[str, object]] = []
    bivariate_rows: list[dict[str, object]] = []
    conditional_rows: list[dict[str, object]] = []
    regression_rows: list[dict[str, object]] = []

    bivariate_specs = [
        ("realized_vol_6b", "corr_btc_30d"),
        ("realized_vol_6b", "ret_6b"),
        ("realized_vol_6b", "abs_funding"),
    ]
    conditional_specs = [
        ("realized_vol_6b", "corr_btc_30d"),
        ("realized_vol_6b", "ret_6b"),
        ("corr_btc_30d", "realized_vol_6b"),
    ]
    regression_specs = {
        "model_1": ("realized_vol_6b", "corr_btc_30d"),
        "model_2": ("realized_vol_6b", "corr_btc_30d", "ret_6b"),
        "model_3": ("realized_vol_6b", "corr_btc_30d", "ret_6b", "abs_funding"),
    }

    for variant in VARIANTS:
        feature_bundle = {
            name: {
                "frame": frame,
                "bucket3": _bucket_from_frame(frame, TERCILES),
            }
            for name, frame in variant_frames[variant].items()
        }
        for feature_name, feature_frame in variant_frames[variant].items():
            for horizon, future_frame in forward_frames.items():
                for period_name, row_mask in period_masks.items():
                    metrics = _sorting_from_frames(feature_frame, future_frame, row_mask, quantiles=5)
                    univariate_rows.append(
                        {
                            "feature_name": feature_name,
                            "variant": variant,
                            "period": period_name,
                            "horizon_bars": horizon,
                            **metrics,
                        }
                    )

        future_arrays = {horizon: frame.to_numpy(dtype=np.float64) for horizon, frame in forward_frames.items()}
        future_valids = {horizon: np.isfinite(array) for horizon, array in future_arrays.items()}
        for primary_feature, secondary_feature in bivariate_specs:
            primary_bucket = feature_bundle[primary_feature]["bucket3"]
            secondary_bucket = feature_bundle[secondary_feature]["bucket3"]
            for horizon in ANALYSIS_HORIZONS:
                for period_name, row_mask in period_masks.items():
                    bivariate_rows.extend(
                        _bivariate_rows(
                            primary_bucket=primary_bucket,
                            secondary_bucket=secondary_bucket,
                            future_values=future_arrays[horizon],
                            future_valid=future_valids[horizon],
                            row_mask=row_mask,
                            primary_feature=primary_feature,
                            secondary_feature=secondary_feature,
                            variant=variant,
                            period=period_name,
                            horizon=horizon,
                        )
                    )

        for signal_feature, condition_feature in conditional_specs:
            baseline_lookup = {
                (row["period"], row["horizon_bars"]): row
                for row in univariate_rows
                if row["feature_name"] == signal_feature and row["variant"] == variant
            }
            for mode in ("high", "low"):
                condition_mask = _median_condition_mask(variant_frames[variant][condition_feature], mode)
                signal_frame = variant_frames[variant][signal_feature].where(condition_mask)
                for horizon, future_frame in forward_frames.items():
                    masked_future = future_frame.where(condition_mask)
                    for period_name, row_mask in period_masks.items():
                        metrics = _sorting_from_frames(signal_frame, masked_future, row_mask, quantiles=5)
                        baseline = baseline_lookup[(period_name, horizon)]
                        conditional_rows.append(
                            {
                                "signal_feature": signal_feature,
                                "condition_feature": condition_feature,
                                "condition_mode": mode,
                                "variant": variant,
                                "period": period_name,
                                "horizon_bars": horizon,
                                **metrics,
                                "baseline_rank_ic_mean": baseline["rank_ic_mean"],
                                "baseline_spread_q5_q1": baseline["spread_q5_q1"],
                            }
                        )

        for model_name, feature_names in regression_specs.items():
            regression_rows.extend(
                _ols_rows(
                    model_name=model_name,
                    feature_names=feature_names,
                    feature_frames=variant_frames[variant],
                    future_frame=forward_frames[6],
                    period_masks=period_masks,
                    variant=variant,
                )
            )

    univariate_df = pd.DataFrame(univariate_rows).sort_values(["feature_name", "variant", "period", "horizon_bars"]).reset_index(drop=True)
    bivariate_df = pd.DataFrame(bivariate_rows).sort_values(
        ["primary_feature", "secondary_feature", "variant", "period", "horizon_bars", "primary_bucket", "secondary_bucket"]
    ).reset_index(drop=True)
    conditional_df = pd.DataFrame(conditional_rows).sort_values(
        ["signal_feature", "condition_feature", "condition_mode", "variant", "period", "horizon_bars"]
    ).reset_index(drop=True)
    regression_df = pd.DataFrame(regression_rows).sort_values(
        ["model_name", "variant", "period", "feature_name"]
    ).reset_index(drop=True)

    if not regression_df.empty:
        model_r2 = (
            regression_df[["model_name", "variant", "period", "mean_r2"]]
            .drop_duplicates()
            .sort_values(["variant", "period", "model_name"])
            .reset_index(drop=True)
        )
        prev_lookup: dict[tuple[str, str, str], float] = {}
        for row in model_r2.itertuples(index=False):
            prev_model = None
            if row.model_name == "model_2":
                prev_model = "model_1"
            elif row.model_name == "model_3":
                prev_model = "model_2"
            delta = row.mean_r2 - prev_lookup.get((prev_model, row.variant, row.period), 0.0) if prev_model else row.mean_r2
            regression_df.loc[
                (regression_df["model_name"] == row.model_name)
                & (regression_df["variant"] == row.variant)
                & (regression_df["period"] == row.period),
                "delta_r2_vs_prev",
            ] = float(delta)
            prev_lookup[(row.model_name, row.variant, row.period)] = float(row.mean_r2)

    role_df = _role_summary(univariate_df, conditional_df, regression_df, bivariate_df)

    paths = {
        "univariate_validation": config.output_dir / "univariate_validation.csv",
        "bivariate_validation": config.output_dir / "bivariate_validation.csv",
        "conditional_validation": config.output_dir / "conditional_validation.csv",
        "cross_section_regression": config.output_dir / "cross_section_regression.csv",
        "role_decision": config.output_dir / "role_decision.csv",
        "summary": config.output_dir / "summary.md",
        "artifacts": config.output_dir / "artifacts.json",
    }

    univariate_df.to_csv(paths["univariate_validation"], index=False)
    bivariate_df.to_csv(paths["bivariate_validation"], index=False)
    conditional_df.to_csv(paths["conditional_validation"], index=False)
    regression_df.to_csv(paths["cross_section_regression"], index=False)
    role_df.to_csv(paths["role_decision"], index=False)
    paths["summary"].write_text(_summary_markdown(config, role_df, univariate_df, conditional_df, regression_df), encoding="utf-8")
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
            "core_features": list(CORE_FEATURES),
            "analysis_horizons": list(ANALYSIS_HORIZONS),
            "roles": role_df.to_dict(orient="records"),
        },
    )
    return paths
