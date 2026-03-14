from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CostConfig
from .portfolio import turnover


@dataclass(slots=True)
class BacktestArtifacts:
    summary: pd.DataFrame
    weights: pd.DataFrame
    interval_returns: pd.DataFrame
    pnl: pd.DataFrame
    equity_curve: pd.Series


def compute_open_to_open_returns(opens: pd.DataFrame) -> pd.DataFrame:
    return opens.shift(-1).div(opens).sub(1.0)


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


def reprice_pnl(
    price_component: pd.Series,
    funding_component: pd.Series,
    traded_notional: pd.Series,
    costs: CostConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    trading_cost = traded_notional * ((costs.fee_bps + costs.slippage_bps) / 10000.0)
    gross_return = price_component + funding_component
    net_return = gross_return - trading_cost
    pnl = pd.DataFrame(
        {
            "price_return": price_component,
            "funding_return": funding_component,
            "gross_return": gross_return,
            "trading_cost": trading_cost,
            "net_return": net_return,
            "turnover": traded_notional,
        }
    )
    equity_curve = (1.0 + pnl["net_return"].fillna(0.0)).cumprod()
    return pnl, equity_curve


def run_backtest(
    weights: pd.DataFrame,
    opens: pd.DataFrame,
    funding: pd.DataFrame,
    costs: CostConfig,
) -> BacktestArtifacts:
    price_returns = compute_open_to_open_returns(opens)
    weights, price_returns = weights.align(price_returns, join="inner", axis=0)
    weights, price_returns = weights.align(price_returns, join="inner", axis=1)
    funding = funding.reindex(index=weights.index, columns=weights.columns, fill_value=0.0)

    traded_notional = turnover(weights)
    price_component = (weights * price_returns).sum(axis=1)
    funding_component = -(weights * funding).sum(axis=1)
    pnl, equity_curve = reprice_pnl(price_component, funding_component, traded_notional, costs)
    summary = pnl.copy()
    return BacktestArtifacts(summary=summary, weights=weights, interval_returns=price_returns, pnl=pnl, equity_curve=equity_curve)
