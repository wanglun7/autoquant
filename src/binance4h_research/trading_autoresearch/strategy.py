from __future__ import annotations

import math

import pandas as pd


STRATEGY_METADATA = {
    "strategy_id": "btc_state_breakout_volume_ls_55d",
    "family": "btc_time_series",
    "description": "BTC long-short breakout with volume confirmation, volatility targeting, and funding-aware risk gating.",
}


def _bars_per_day(bar_hours: int) -> int:
    return int(24 / bar_hours)


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
    bars_per_day = _bars_per_day(4)
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
    entry_lookback_days: int = 55,
    exit_lookback_days: int = 20,
    momentum_lookback_days: int = 20,
    volume_fast_days: int = 5,
    volume_slow_days: int = 20,
    funding_lookback_days: int = 7,
    vol_lookback_days: int = 20,
    volume_ratio_threshold: float = 1.05,
    momentum_exit_threshold: float = 0.01,
    funding_soft_cap: float = 0.00005,
    funding_hard_cap: float = 0.00020,
    vol_target: float = 0.010,
    rebalance: str = "1d",
    gross: float = 1.0,
) -> pd.DataFrame:
    bars_per_day = _bars_per_day(4)
    btc_symbol = context.btc_symbol
    close = context.closes[btc_symbol]
    volume = context.quote_volume[btc_symbol]
    funding = context.funding[btc_symbol]
    prior_close = close.shift(1)
    entry_bars = entry_lookback_days * bars_per_day
    exit_bars = exit_lookback_days * bars_per_day
    momentum_bars = momentum_lookback_days * bars_per_day
    volume_fast_bars = volume_fast_days * bars_per_day
    volume_slow_bars = volume_slow_days * bars_per_day
    funding_bars = funding_lookback_days * bars_per_day
    vol_bars = vol_lookback_days * bars_per_day

    entry_high = close.shift(2).rolling(entry_bars).max()
    entry_low = close.shift(2).rolling(entry_bars).min()
    exit_high = close.shift(2).rolling(exit_bars).max()
    exit_low = close.shift(2).rolling(exit_bars).min()
    momentum = prior_close.div(close.shift(1 + momentum_bars)).sub(1.0)
    volume_fast = volume.shift(1).rolling(volume_fast_bars).mean()
    volume_slow = volume.shift(1).rolling(volume_slow_bars).mean()
    volume_ratio = volume_fast.div(volume_slow.replace(0.0, pd.NA))
    realized_vol = close.pct_change().shift(1).rolling(vol_bars).std()
    funding_mean = funding.shift(1).rolling(funding_bars).mean()

    enter_long = (prior_close > entry_high) & (momentum > 0.0) & (volume_ratio > volume_ratio_threshold)
    exit_long = (prior_close < exit_low) | (momentum < -momentum_exit_threshold)
    enter_short = (prior_close < entry_low) & (momentum < 0.0) & (volume_ratio > volume_ratio_threshold)
    exit_short = (prior_close > exit_high) | (momentum > momentum_exit_threshold)

    vol_scale = vol_target / realized_vol.replace(0.0, pd.NA)
    vol_scale = vol_scale.clip(lower=0.0, upper=1.0).fillna(0.0)
    long_funding_scale = (funding_hard_cap - funding_mean) / (funding_hard_cap - funding_soft_cap)
    long_funding_scale = long_funding_scale.clip(lower=0.0, upper=1.0).fillna(1.0)
    short_funding_scale = (funding_mean + funding_hard_cap) / (funding_hard_cap - funding_soft_cap)
    short_funding_scale = short_funding_scale.clip(lower=0.0, upper=1.0).fillna(1.0)

    signal = pd.Series(0.0, index=close.index)
    state = 0
    for timestamp in close.index:
        if state > 0 and bool(exit_long.fillna(False).loc[timestamp]):
            state = 0
        elif state < 0 and bool(exit_short.fillna(False).loc[timestamp]):
            state = 0
        if state == 0:
            if bool(enter_long.fillna(False).loc[timestamp]):
                state = 1
            elif bool(enter_short.fillna(False).loc[timestamp]):
                state = -1
        if state > 0:
            scale = min(float(vol_scale.loc[timestamp]), float(long_funding_scale.loc[timestamp]))
            signal.loc[timestamp] = gross * scale
        elif state < 0:
            scale = min(float(vol_scale.loc[timestamp]), float(short_funding_scale.loc[timestamp]))
            signal.loc[timestamp] = -gross * scale

    weights = pd.DataFrame(0.0, index=context.closes.index, columns=context.closes.columns)
    weights[btc_symbol] = signal
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
    bars_per_day = _bars_per_day(4)
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
