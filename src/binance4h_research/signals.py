from __future__ import annotations

import pandas as pd


def build_close_matrix(klines: dict[str, pd.DataFrame]) -> pd.DataFrame:
    series_map: dict[str, pd.Series] = {}
    for symbol, frame in klines.items():
        series_map[symbol] = frame.set_index("open_time")["close"].astype(float)
    if not series_map:
        return pd.DataFrame()
    return pd.concat(series_map, axis=1).sort_index()


def build_open_matrix(klines: dict[str, pd.DataFrame]) -> pd.DataFrame:
    series_map: dict[str, pd.Series] = {}
    for symbol, frame in klines.items():
        series_map[symbol] = frame.set_index("open_time")["open"].astype(float)
    if not series_map:
        return pd.DataFrame()
    return pd.concat(series_map, axis=1).sort_index()


def cross_sectional_momentum_signal(closes: pd.DataFrame, lookback_bars: int) -> pd.DataFrame:
    # Signal at t is computed only from bars up to t-1.
    return closes.pct_change(lookback_bars, fill_method=None).shift(1)
