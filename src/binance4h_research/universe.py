from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ExperimentConfig


def _combine_field(klines: dict[str, pd.DataFrame], field: str) -> pd.DataFrame:
    series_map: dict[str, pd.Series] = {}
    for symbol, frame in klines.items():
        temp = frame[["open_time", field]].copy()
        temp = temp.dropna()
        series_map[symbol] = temp.set_index("open_time")[field]
    if not series_map:
        return pd.DataFrame()
    combined = pd.concat(series_map, axis=1).sort_index()
    combined.index = pd.DatetimeIndex(combined.index, tz="UTC")
    return combined


def build_universe_membership(klines: dict[str, pd.DataFrame], config: ExperimentConfig) -> pd.DataFrame:
    quote_volume = _combine_field(klines, config.universe.quote_volume_field)
    if quote_volume.empty:
        return pd.DataFrame()

    liquidity_min_periods = min(config.universe.liquidity_lookback_bars, config.universe.min_history_bars)
    rolling_liquidity = (
        quote_volume.shift(1)
        .rolling(window=config.universe.liquidity_lookback_bars, min_periods=liquidity_min_periods)
        .mean()
    )
    history_count = quote_volume.notna().cumsum()
    candidates = rolling_liquidity.notna() & history_count.ge(config.universe.min_history_bars)

    snapshots: list[pd.Series] = []
    for day, day_frame in rolling_liquidity.groupby(rolling_liquidity.index.normalize()):
        snapshot_time = day_frame.index.min()
        scores = day_frame.loc[snapshot_time].where(candidates.loc[snapshot_time])
        top = scores.dropna().nlargest(config.universe.top_n).index
        membership = pd.Series(False, index=rolling_liquidity.columns, dtype=bool)
        membership.loc[list(top)] = True
        membership.name = pd.Timestamp(day)
        snapshots.append(membership)

    if not snapshots:
        return pd.DataFrame(index=rolling_liquidity.index, columns=rolling_liquidity.columns, data=False)

    daily = pd.DataFrame(snapshots)
    daily.index.name = "date"

    membership = pd.DataFrame(index=rolling_liquidity.index, columns=rolling_liquidity.columns, dtype=bool)
    for timestamp in membership.index:
        date_key = pd.Timestamp(timestamp.normalize())
        if date_key in daily.index:
            membership.loc[timestamp] = daily.loc[date_key]
        else:
            membership.loc[timestamp] = False
    return membership.fillna(False).astype(bool)


def save_universe_membership(membership: pd.DataFrame, output_dir: Path, experiment_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{experiment_name}_universe.csv"
    membership.to_csv(path, index_label="timestamp")
    return path
