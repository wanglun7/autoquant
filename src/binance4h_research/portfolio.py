from __future__ import annotations

import math

import pandas as pd

from .config import PortfolioConfig


def _select_count(size: int, quantile: float) -> int:
    return max(1, math.floor(size * quantile))


def build_market_neutral_weights(
    signal: pd.DataFrame,
    membership: pd.DataFrame,
    config: PortfolioConfig,
) -> pd.DataFrame:
    signal, membership = signal.align(membership, join="inner", axis=0)
    signal, membership = signal.align(membership, join="inner", axis=1)
    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)

    for timestamp in signal.index:
        eligible = membership.loc[timestamp]
        row = signal.loc[timestamp].where(eligible).dropna().sort_values()
        if row.empty:
            continue
        long_count = _select_count(len(row), config.long_quantile)
        short_count = _select_count(len(row), config.short_quantile)
        longs = row.iloc[-long_count:].index
        shorts = row.iloc[:short_count].index
        if len(longs) > 0:
            weights.loc[timestamp, longs] = (config.gross_exposure / 2.0) / len(longs)
        if len(shorts) > 0:
            weights.loc[timestamp, shorts] = -(config.gross_exposure / 2.0) / len(shorts)
    return weights


def rebalance_on_schedule(weights: pd.DataFrame, every_bars: int) -> pd.DataFrame:
    if every_bars <= 1:
        return weights
    kept = weights.iloc[::every_bars].copy()
    resampled = kept.reindex(weights.index).ffill().fillna(0.0)
    return resampled


def turnover(weights: pd.DataFrame) -> pd.Series:
    changes = weights.diff()
    if not changes.empty:
        changes.iloc[0] = weights.iloc[0]
    return changes.abs().sum(axis=1)
