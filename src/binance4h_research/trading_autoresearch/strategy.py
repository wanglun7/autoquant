from __future__ import annotations

import math

import pandas as pd


STRATEGY_METADATA = {
    "strategy_id": "cross_sectional_20d_top6_weekly",
    "family": "cross_sectional",
    "description": "Concentrated 20-day cross-sectional momentum over the top-6 liquid symbols with weekly rebalancing.",
}


def _rebalance_every_bars(freq: str, bar_hours: int) -> int:
    mapping = {"4h": 1, "1d": int(24 / bar_hours), "1w": int((24 / bar_hours) * 7)}
    return mapping.get(freq, 1)


def _apply_rebalance(weights: pd.DataFrame, freq: str, bar_hours: int) -> pd.DataFrame:
    every_bars = _rebalance_every_bars(freq, bar_hours)
    if every_bars <= 1:
        return weights
    kept = weights.iloc[::every_bars].copy()
    return kept.reindex(weights.index).ffill().fillna(0.0)


def build_cross_sectional_weights(
    context,
    *,
    lookback_days: int = 20,
    top_n: int = 6,
    rebalance: str = "1w",
    long_quantile: float = 0.34,
    short_quantile: float = 0.34,
    gross: float = 1.0,
) -> pd.DataFrame:
    bars_per_day = 24 // 4
    lookback_bars = lookback_days * bars_per_day
    signal = context.closes.shift(1).div(context.closes.shift(1 + lookback_bars)).sub(1.0)
    eligible = context.liquidity.rank(axis=1, ascending=False, method="first").le(top_n).fillna(False)
    weights = pd.DataFrame(0.0, index=context.closes.index, columns=context.closes.columns)
    for timestamp in signal.index:
        row = signal.loc[timestamp].where(eligible.loc[timestamp]).dropna().sort_values()
        if row.empty:
            continue
        long_count = max(1, math.floor(len(row) * long_quantile))
        short_count = max(1, math.floor(len(row) * short_quantile))
        longs = row.iloc[-long_count:].index
        shorts = row.iloc[:short_count].index
        weights.loc[timestamp, longs] = (gross / 2.0) / len(longs)
        weights.loc[timestamp, shorts] = -(gross / 2.0) / len(shorts)
    return _apply_rebalance(weights, rebalance, 4)


def build_btc_time_series_weights(
    context,
    *,
    model: str = "momentum",
    lookback_days: int = 60,
    rebalance: str = "1d",
    gross: float = 1.0,
    direction_mode: str = "long_short",
) -> pd.DataFrame:
    close = context.closes[context.btc_symbol]
    bars_per_day = 24 // 4
    bars = lookback_days * bars_per_day
    if model == "breakout":
        rolling_max = close.shift(1).rolling(bars).max()
        rolling_min = close.shift(1).rolling(bars).min()
        signal = pd.Series(0.0, index=close.index)
        signal = signal.mask(close.shift(1) > rolling_max, 1.0)
        signal = signal.mask(close.shift(1) < rolling_min, -1.0)
        signal = signal.fillna(0.0)
    elif model == "mean_reversion":
        mean = close.shift(1).rolling(bars).mean()
        std = close.shift(1).rolling(bars).std().replace(0.0, pd.NA)
        zscore = close.shift(1).sub(mean).div(std)
        signal = zscore.apply(lambda x: -1.0 if x > 1 else (1.0 if x < -1 else 0.0))
    else:
        score = close.shift(1).div(close.shift(1 + bars)).sub(1.0)
        signal = score.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
    if direction_mode == "long_only":
        signal = signal.clip(lower=0.0)
    weights = pd.DataFrame(0.0, index=context.closes.index, columns=context.closes.columns)
    weights[context.btc_symbol] = signal * gross
    return _apply_rebalance(weights, rebalance, 4)


def build_relative_value_weights(
    context,
    *,
    left: str = "ETHUSDT",
    right: str = "SOLUSDT",
    lookback_days: int = 60,
    rebalance: str = "1d",
    gross: float = 1.0,
    z_entry: float = 1.0,
) -> pd.DataFrame:
    bars_per_day = 24 // 4
    bars = lookback_days * bars_per_day
    spread = (context.closes[left].map(math.log) - context.closes[right].map(math.log)).replace([math.inf, -math.inf], pd.NA)
    mean = spread.shift(1).rolling(bars).mean()
    std = spread.shift(1).rolling(bars).std().replace(0.0, pd.NA)
    zscore = spread.shift(1).sub(mean).div(std)
    direction = zscore.apply(lambda x: -1.0 if x > z_entry else (1.0 if x < -z_entry else 0.0))
    weights = pd.DataFrame(0.0, index=context.closes.index, columns=context.closes.columns)
    weights[left] = direction * (gross / 2.0)
    weights[right] = -direction * (gross / 2.0)
    return _apply_rebalance(weights, rebalance, 4)


def build_weights(context):
    family = STRATEGY_METADATA["family"]
    if family == "cross_sectional":
        weights = build_cross_sectional_weights(context)
    elif family == "btc_time_series":
        weights = build_btc_time_series_weights(context)
    elif family == "relative_value":
        weights = build_relative_value_weights(context)
    else:
        raise ValueError(f"Unsupported strategy family: {family}")
    return weights, STRATEGY_METADATA
