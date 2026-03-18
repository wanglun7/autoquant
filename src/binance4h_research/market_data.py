from __future__ import annotations

import numpy as np
import pandas as pd


def combine_field_matrix(klines: dict[str, pd.DataFrame], field: str) -> pd.DataFrame:
    series_map: dict[str, pd.Series] = {}
    for symbol, frame in klines.items():
        temp = frame[["open_time", field]].copy()
        temp = temp.dropna()
        series_map[symbol] = temp.set_index("open_time")[field]
    if not series_map:
        return pd.DataFrame()
    combined = pd.concat(series_map, axis=1, sort=False).sort_index()
    combined.index = pd.DatetimeIndex(combined.index, tz="UTC")
    combined = combined.reindex(sorted(combined.columns), axis=1)
    return combined


def funding_returns_from_events(
    funding_by_symbol: dict[str, pd.DataFrame],
    interval_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    funding = pd.DataFrame(0.0, index=interval_index, columns=sorted(funding_by_symbol))
    if funding.empty:
        return funding

    index_values = funding.index.view("int64")
    for symbol, frame in funding_by_symbol.items():
        if symbol not in funding.columns or frame.empty:
            continue
        temp = frame.copy()
        temp["fundingTime"] = pd.to_datetime(temp["fundingTime"], utc=True)
        temp["fundingRate"] = pd.to_numeric(temp["fundingRate"], errors="coerce").fillna(0.0)
        funding_times = temp["fundingTime"].astype("int64").to_numpy()
        funding_rates = temp["fundingRate"].to_numpy(dtype=float)
        positions = np.searchsorted(index_values, funding_times, side="left")
        valid = (positions > 0) & (positions < len(index_values))
        if not valid.any():
            continue
        grouped = pd.Series(funding_rates[valid]).groupby(positions[valid] - 1).sum()
        funding.iloc[grouped.index.to_numpy(dtype=int), funding.columns.get_loc(symbol)] = grouped.to_numpy(dtype=float)
    return funding.fillna(0.0)
