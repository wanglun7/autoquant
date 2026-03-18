from __future__ import annotations

from pathlib import Path

import pandas as pd

from binance4h_research.data import (
    fetch_funding_range,
    fetch_klines_range,
    load_symbol_funding,
    load_symbol_klines,
)


class _PagedClient:
    def __init__(self) -> None:
        self.kline_calls = 0
        self.funding_calls = 0

    def klines(
        self,
        symbol: str,
        interval: str = "4h",
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 1500,
    ) -> pd.DataFrame:
        self.kline_calls += 1
        if self.kline_calls == 1:
            return pd.DataFrame(
                {
                    "open_time": pd.to_datetime([0, 14_400_000], unit="ms", utc=True),
                    "open": [1.0, 2.0],
                    "high": [1.0, 2.0],
                    "low": [1.0, 2.0],
                    "close": [1.0, 2.0],
                    "volume": [1.0, 1.0],
                    "close_time": pd.to_datetime([14_399_999, 28_799_999], unit="ms", utc=True),
                    "quote_volume": [1.0, 1.0],
                    "trade_count": [1, 1],
                    "taker_buy_base_volume": [1.0, 1.0],
                    "taker_buy_quote_volume": [1.0, 1.0],
                }
            )
        if self.kline_calls == 2:
            assert start_time == 28_800_000
            return pd.DataFrame(
                {
                    "open_time": pd.to_datetime([28_800_000], unit="ms", utc=True),
                    "open": [3.0],
                    "high": [3.0],
                    "low": [3.0],
                    "close": [3.0],
                    "volume": [1.0],
                    "close_time": pd.to_datetime([43_199_999], unit="ms", utc=True),
                    "quote_volume": [1.0],
                    "trade_count": [1],
                    "taker_buy_base_volume": [1.0],
                    "taker_buy_quote_volume": [1.0],
                }
            )
        return pd.DataFrame()

    def funding_rates(
        self,
        symbol: str,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        self.funding_calls += 1
        if self.funding_calls == 1:
            return pd.DataFrame(
                {
                    "symbol": [symbol, symbol],
                    "fundingTime": pd.to_datetime([0, 28_800_000], unit="ms", utc=True),
                    "fundingRate": [0.001, 0.002],
                    "markPrice": [100.0, 101.0],
                }
            )
        if self.funding_calls == 2:
            assert start_time == 28_800_001
            return pd.DataFrame(
                {
                    "symbol": [symbol],
                    "fundingTime": pd.to_datetime([57_600_000], unit="ms", utc=True),
                    "fundingRate": [0.003],
                    "markPrice": [102.0],
                }
            )
        return pd.DataFrame()


def test_range_fetch_paginates_without_duplicates() -> None:
    client = _PagedClient()
    klines = fetch_klines_range(
        client,
        symbol="AAAUSDT",
        interval="4h",
        start_time=0,
        end_time=57_600_000,
        limit=2,
    )
    funding = fetch_funding_range(
        client,
        symbol="AAAUSDT",
        start_time=0,
        end_time=57_600_000,
        limit=2,
    )

    assert len(klines) == 3
    assert klines["open_time"].is_unique
    assert len(funding) == 3
    assert funding["fundingTime"].is_unique


def test_loaders_roundtrip_cached_csvs(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    (data_dir / "klines").mkdir(parents=True)
    (data_dir / "funding").mkdir(parents=True)

    pd.DataFrame(
        {
            "open_time": pd.to_datetime(["2024-01-01 00:00:00+00:00"]),
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1.0],
            "close_time": pd.to_datetime(["2024-01-01 03:59:59.999000+00:00"]),
            "quote_volume": [2.0],
            "trade_count": [3],
            "taker_buy_base_volume": [0.5],
            "taker_buy_quote_volume": [1.0],
        }
    ).to_csv(data_dir / "klines" / "BTCUSDT_4h.csv", index=False)
    pd.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "fundingTime": pd.to_datetime(["2024-01-01 08:00:00+00:00"]),
            "fundingRate": [0.0001],
            "markPrice": [100.0],
        }
    ).to_csv(data_dir / "funding" / "BTCUSDT.csv", index=False)

    assert "BTCUSDT" in load_symbol_klines(data_dir)
    assert "BTCUSDT" in load_symbol_funding(data_dir)
