from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pandas import DatetimeTZDtype

from binance4h_research.data import (
    fetch_funding_range,
    fetch_klines_range,
    load_symbol_funding,
    load_symbol_klines,
)


class _OverlappingKlineClient:
    def __init__(self) -> None:
        self.calls = 0

    def klines(
        self,
        symbol: str,
        interval: str = "4h",
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 1500,
    ) -> pd.DataFrame:
        self.calls += 1
        if self.calls == 1:
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
        if self.calls == 2:
            assert start_time == 28_800_000
            assert end_time == 57_600_000
            return pd.DataFrame(
                {
                    "open_time": pd.to_datetime([14_400_000, 28_800_000], unit="ms", utc=True),
                    "open": [2.5, 3.0],
                    "high": [2.0, 3.0],
                    "low": [2.0, 3.0],
                    "close": [2.0, 3.0],
                    "volume": [1.0, 1.0],
                    "close_time": pd.to_datetime([28_799_999, 43_199_999], unit="ms", utc=True),
                    "quote_volume": [1.0, 1.0],
                    "trade_count": [1, 1],
                    "taker_buy_base_volume": [1.0, 1.0],
                    "taker_buy_quote_volume": [1.0, 1.0],
                }
            )
        return pd.DataFrame()


class _OverlappingFundingClient:
    def __init__(self) -> None:
        self.calls = 0

    def funding_rates(
        self,
        symbol: str,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        self.calls += 1
        if self.calls == 1:
            return pd.DataFrame(
                {
                    "symbol": [symbol, symbol],
                    "fundingTime": pd.to_datetime([0, 28_800_000], unit="ms", utc=True),
                    "fundingRate": [0.001, 0.002],
                    "markPrice": [100.0, 101.0],
                }
            )
        if self.calls == 2:
            assert start_time == 28_800_001
            assert end_time == 57_600_000
            return pd.DataFrame(
                {
                    "symbol": [symbol, symbol],
                    "fundingTime": pd.to_datetime([28_800_000, 57_600_000], unit="ms", utc=True),
                    "fundingRate": [0.005, 0.003],
                    "markPrice": [101.0, 102.0],
                }
            )
        return pd.DataFrame()


def test_fetch_klines_range_deduplicates_overlapping_pages() -> None:
    client = _OverlappingKlineClient()
    klines = fetch_klines_range(
        client,
        symbol="AAAUSDT",
        interval="4h",
        start_time=0,
        end_time=57_600_000,
        limit=2,
    )

    expected_timestamps = [pd.Timestamp(0, tz="UTC"), pd.Timestamp(14_400_000, unit="ms", tz="UTC"), pd.Timestamp(28_800_000, unit="ms", tz="UTC")]
    assert len(klines) == len(expected_timestamps)
    assert klines["open_time"].is_unique
    assert klines["open_time"].tolist() == expected_timestamps
    assert klines["close_time"].iloc[0].tzinfo == pd.Timestamp(0, tz="UTC").tzinfo
    duplicate_row = klines.loc[klines["open_time"] == expected_timestamps[1]].iloc[0]
    assert duplicate_row["open"] == pytest.approx(2.0)
    assert duplicate_row["close"] == pytest.approx(2.0)


def test_fetch_funding_range_deduplicates_overlapping_pages() -> None:
    client = _OverlappingFundingClient()
    funding = fetch_funding_range(
        client,
        symbol="AAAUSDT",
        start_time=0,
        end_time=57_600_000,
        limit=2,
    )

    expected_times = [pd.Timestamp(0, tz="UTC"), pd.Timestamp(28_800_000, unit="ms", tz="UTC"), pd.Timestamp(57_600_000, unit="ms", tz="UTC")]
    assert len(funding) == len(expected_times)
    assert funding["fundingTime"].is_unique
    assert funding["fundingTime"].tolist() == expected_times
    assert funding["fundingTime"].is_monotonic_increasing
    duplicate_funding = funding.loc[funding["fundingTime"] == expected_times[1]].iloc[0]
    assert duplicate_funding["fundingRate"] == pytest.approx(0.002)


def test_loaders_roundtrip_cached_csvs(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    (data_dir / "klines").mkdir(parents=True)
    (data_dir / "funding").mkdir(parents=True)

    pd.DataFrame(
        {
            "open_time": pd.to_datetime([
                "2024-01-02 04:00:00+00:00",
                "2024-01-01 00:00:00+00:00",
            ]),
            "open": [101.0, 100.0],
            "high": [102.0, 101.0],
            "low": [100.0, 99.0],
            "close": [101.5, 100.5],
            "volume": [2.0, 1.0],
            "close_time": pd.to_datetime([
                "2024-01-02 07:59:59.999000+00:00",
                "2024-01-01 03:59:59.999000+00:00",
            ]),
            "quote_volume": [4.0, 2.0],
            "trade_count": [5, 3],
            "taker_buy_base_volume": [1.0, 0.5],
            "taker_buy_quote_volume": [2.0, 1.0],
        }
    ).to_csv(data_dir / "klines" / "BTCUSDT_4h.csv", index=False)
    pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "fundingTime": pd.to_datetime([
                "2024-01-02 08:00:00+00:00",
                "2024-01-01 08:00:00+00:00",
            ]),
            "fundingRate": [0.00015, 0.0001],
            "markPrice": [101.0, 100.0],
        }
    ).to_csv(data_dir / "funding" / "BTCUSDT.csv", index=False)

    klines_map = load_symbol_klines(data_dir)
    funding_map = load_symbol_funding(data_dir)

    assert "BTCUSDT" in klines_map
    assert "BTCUSDT" in funding_map

    klines_df = klines_map["BTCUSDT"].copy()
    expected_utc = pd.Timestamp(0, tz="UTC").tz
    assert isinstance(klines_df["open_time"].dtype, DatetimeTZDtype)
    assert isinstance(klines_df["close_time"].dtype, DatetimeTZDtype)
    assert klines_df["open_time"].dt.tz == expected_utc
    assert klines_df["close_time"].dt.tz == expected_utc
    assert klines_df["open_time"].is_monotonic_increasing
    assert klines_df["close_time"].is_monotonic_increasing
    assert klines_df["open_time"].tolist() == sorted(klines_df["open_time"].tolist())
    assert klines_df["open_time"].iloc[0] == pd.Timestamp("2024-01-01 00:00:00+00:00")
    assert klines_df["open"].iloc[0] == pytest.approx(100.0)
    assert klines_df["high"].iloc[0] == pytest.approx(101.0)
    assert klines_df["volume"].iloc[0] == pytest.approx(1.0)

    funding_df = funding_map["BTCUSDT"].copy()
    assert isinstance(funding_df["fundingTime"].dtype, DatetimeTZDtype)
    assert funding_df["fundingTime"].dt.tz == expected_utc
    assert funding_df["fundingTime"].is_monotonic_increasing
    assert funding_df["fundingTime"].tolist() == sorted(funding_df["fundingTime"].tolist())
    assert funding_df["fundingTime"].iloc[0] == pd.Timestamp("2024-01-01 08:00:00+00:00")
    assert funding_df["fundingRate"].iloc[0] == pytest.approx(0.0001)
    assert funding_df["markPrice"].iloc[0] == pytest.approx(100.0)
