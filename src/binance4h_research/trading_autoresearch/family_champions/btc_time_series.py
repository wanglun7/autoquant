from __future__ import annotations

import math

import pandas as pd


STRATEGY_METADATA = {
    "strategy_id": "btc_trend_pullback_osc_risk_switch_90d",
    "family": "btc_time_series",
    "description": "BTC long-short pullback strategy using trend regime, oscillator entries, and a volatility risk switch.",
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
    fast_ma_days: int = 20,
    slow_ma_days: int = 90,
    long_momentum_days: int = 90,
    oscillator_days: int = 10,
    funding_lookback_days: int = 7,
    vol_fast_days: int = 5,
    vol_slow_days: int = 30,
    vol_target_days: int = 20,
    long_momentum_floor: float = 0.03,
    ma_spread_floor: float = 0.01,
    oscillator_entry_z: float = 1.2,
    oscillator_exit_z: float = 0.25,
    vol_ratio_cap: float = 1.2,
    funding_soft_cap: float = 0.00005,
    funding_hard_cap: float = 0.00020,
    vol_target: float = 0.010,
    rebalance: str = "1d",
    gross: float = 1.0,
) -> pd.DataFrame:
    bars_per_day = _bars_per_day(4)
    btc_symbol = context.btc_symbol
    close = context.closes[btc_symbol]
    funding = context.funding[btc_symbol]
    prior_close = close.shift(1)
    fast_ma_bars = fast_ma_days * bars_per_day
    slow_ma_bars = slow_ma_days * bars_per_day
    long_momentum_bars = long_momentum_days * bars_per_day
    oscillator_bars = oscillator_days * bars_per_day
    vol_fast_bars = vol_fast_days * bars_per_day
    vol_slow_bars = vol_slow_days * bars_per_day
    funding_bars = funding_lookback_days * bars_per_day
    vol_target_bars = vol_target_days * bars_per_day

    fast_ma = close.shift(1).rolling(fast_ma_bars).mean()
    slow_ma = close.shift(1).rolling(slow_ma_bars).mean()
    ma_spread = fast_ma.div(slow_ma.replace(0.0, pd.NA)).sub(1.0)
    long_momentum = prior_close.div(close.shift(1 + long_momentum_bars)).sub(1.0)
    osc_mean = close.shift(1).rolling(oscillator_bars).mean()
    osc_std = close.shift(1).rolling(oscillator_bars).std().replace(0.0, pd.NA)
    oscillator_z = prior_close.sub(osc_mean).div(osc_std)
    realized_returns = close.pct_change().shift(1)
    vol_fast = realized_returns.rolling(vol_fast_bars).std()
    vol_slow = realized_returns.rolling(vol_slow_bars).std()
    vol_ratio = vol_fast.div(vol_slow.replace(0.0, pd.NA))
    realized_vol = realized_returns.rolling(vol_target_bars).std()
    funding_mean = funding.shift(1).rolling(funding_bars).mean()
    risk_on = vol_ratio.lt(vol_ratio_cap).fillna(False)
    trend_up = (ma_spread > ma_spread_floor) & (long_momentum > long_momentum_floor)
    trend_down = (ma_spread < -ma_spread_floor) & (long_momentum < -long_momentum_floor)

    enter_long = (
        trend_up
        & risk_on
        & (oscillator_z < -oscillator_entry_z)
    )
    exit_long = (
        (~trend_up)
        | (~risk_on)
        | (oscillator_z > -oscillator_exit_z)
    )
    enter_short = (
        trend_down
        & risk_on
        & (oscillator_z > oscillator_entry_z)
    )
    exit_short = (
        (~trend_down)
        | (~risk_on)
        | (oscillator_z < oscillator_exit_z)
    )

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
