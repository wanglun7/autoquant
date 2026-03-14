from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from binance4h_research.analytics import pressure_test, summarize_artifacts
from binance4h_research.backtest import funding_returns_from_events, run_backtest
from binance4h_research.config import BacktestConfig, CostConfig, ExperimentConfig, OutputConfig, PortfolioConfig, SignalConfig, UniverseConfig
from binance4h_research.data import fetch_funding_range, fetch_klines_range
from binance4h_research.experiment import run_experiment
from binance4h_research.portfolio import build_market_neutral_weights
from binance4h_research.signals import build_close_matrix, build_open_matrix, cross_sectional_momentum_signal
from binance4h_research.universe import build_universe_membership


def _symbol_frame(symbol: str, closes: list[float], quote_volume: float = 1_000_000.0) -> pd.DataFrame:
    opens = [closes[0], *closes[:-1]]
    times = pd.date_range("2024-01-01", periods=len(closes), freq="4h", tz="UTC")
    return pd.DataFrame(
        {
            "open_time": times,
            "open": opens,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000.0] * len(closes),
            "close_time": times + pd.Timedelta(hours=4) - pd.Timedelta(milliseconds=1),
            "quote_volume": [quote_volume] * len(closes),
            "trade_count": [100] * len(closes),
            "taker_buy_base_volume": [500.0] * len(closes),
            "taker_buy_quote_volume": [quote_volume / 2] * len(closes),
        }
    )


def test_signal_is_shifted_before_fill() -> None:
    klines = {
        "AAAUSDT": _symbol_frame("AAAUSDT", [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], quote_volume=2_000_000),
        "BBBUSDT": _symbol_frame("BBBUSDT", [10, 9, 8, 7, 6, 5, 4, 3, 2, 2, 2, 2], quote_volume=1_800_000),
        "CCCUSDT": _symbol_frame("CCCUSDT", [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], quote_volume=1_600_000),
    }
    config = ExperimentConfig(
        name="toy",
        universe=UniverseConfig(top_n=3, liquidity_lookback_bars=2, min_history_bars=2),
        signal=SignalConfig(lookback_bars=2),
        portfolio=PortfolioConfig(long_quantile=1 / 3, short_quantile=1 / 3),
    )
    membership = build_universe_membership(klines, config)
    closes = build_close_matrix(klines)
    signal = cross_sectional_momentum_signal(closes, config.signal.lookback_bars)
    weights = build_market_neutral_weights(signal, membership, config.portfolio)

    ts = pd.Timestamp("2024-01-02 00:00:00+00:00")
    assert weights.loc[ts, "AAAUSDT"] > 0
    assert weights.loc[ts, "BBBUSDT"] < 0
    assert weights.loc[ts, "CCCUSDT"] == 0


def test_backtest_charges_costs_and_funding() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="4h", tz="UTC")
    opens = pd.DataFrame({"AAAUSDT": [100.0, 100.0, 100.0], "BBBUSDT": [100.0, 100.0, 100.0]}, index=idx)
    weights = pd.DataFrame({"AAAUSDT": [0.5, 0.0, 0.0], "BBBUSDT": [-0.5, 0.0, 0.0]}, index=idx)
    funding = pd.DataFrame({"AAAUSDT": [0.001, 0.0, 0.0], "BBBUSDT": [-0.001, 0.0, 0.0]}, index=idx)
    artifacts = run_backtest(weights, opens, funding, CostConfig(fee_bps=4.0, slippage_bps=2.0))

    first_row = artifacts.pnl.iloc[0]
    assert first_row["price_return"] == 0.0
    assert round(first_row["funding_return"], 6) == -0.001
    assert round(first_row["trading_cost"], 6) == 0.0006
    assert round(first_row["net_return"], 6) == -0.0016


def test_funding_events_are_bucketed_like_interval_windows() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="4h", tz="UTC")
    funding = funding_returns_from_events(
        {
            "AAAUSDT": pd.DataFrame(
                {
                    "fundingTime": pd.to_datetime(
                        [
                            "2024-01-01 04:00:00+00:00",
                            "2024-01-01 05:00:00+00:00",
                            "2024-01-01 08:00:00+00:00",
                            "2024-01-01 13:00:00+00:00",
                        ]
                    ),
                    "fundingRate": [0.001, 0.002, 0.003, 0.004],
                }
            )
        },
        idx,
    )
    assert round(float(funding.loc[idx[0], "AAAUSDT"]), 6) == 0.001
    assert round(float(funding.loc[idx[1], "AAAUSDT"]), 6) == 0.005
    assert round(float(funding.loc[idx[2], "AAAUSDT"]), 6) == 0.0
    assert round(float(funding.loc[idx[3], "AAAUSDT"]), 6) == 0.0


def test_pressure_test_reprices_without_rerunning_backtest() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="4h", tz="UTC")
    opens = pd.DataFrame({"AAAUSDT": [100.0, 101.0, 102.0], "BBBUSDT": [100.0, 99.0, 98.0]}, index=idx)
    weights = pd.DataFrame({"AAAUSDT": [0.5, 0.5, 0.0], "BBBUSDT": [-0.5, -0.5, 0.0]}, index=idx)
    funding = pd.DataFrame({"AAAUSDT": [0.0, 0.001, 0.0], "BBBUSDT": [0.0, -0.001, 0.0]}, index=idx)
    artifacts = run_backtest(weights, opens, funding, CostConfig(fee_bps=4.0, slippage_bps=2.0))
    config = ExperimentConfig(name="toy")

    pressure = pressure_test(artifacts, config)

    base = pressure[(pressure["fee_mult"] == 1.0) & (pressure["slippage_mult"] == 1.0)].iloc[0]
    assert round(float(base["net_total_return"]), 8) == round(float(summarize_artifacts(artifacts, config.backtest)["net_total_return"]), 8)


def test_end_to_end_experiment(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    (data_dir / "klines").mkdir(parents=True)
    (data_dir / "funding").mkdir(parents=True)
    symbols = {
        "AAAUSDT": _symbol_frame("AAAUSDT", [100, 101, 102, 103, 104, 105, 106, 107], quote_volume=5_000_000),
        "BBBUSDT": _symbol_frame("BBBUSDT", [100, 99, 98, 97, 96, 95, 94, 93], quote_volume=4_500_000),
        "CCCUSDT": _symbol_frame("CCCUSDT", [100, 100, 100, 100, 100, 100, 100, 100], quote_volume=4_000_000),
    }
    for symbol, frame in symbols.items():
        frame.to_csv(data_dir / "klines" / f"{symbol}_4h.csv", index=False)
        pd.DataFrame({"fundingTime": [], "fundingRate": []}).to_csv(data_dir / "funding" / f"{symbol}.csv", index=False)

    config = ExperimentConfig(
        name="integration",
        data_dir=data_dir,
        processed_dir=tmp_path / "processed",
        results_dir=tmp_path / "results",
        universe=UniverseConfig(top_n=3, liquidity_lookback_bars=2, min_history_bars=2),
        signal=SignalConfig(lookback_bars=2),
        portfolio=PortfolioConfig(long_quantile=1 / 3, short_quantile=1 / 3),
        backtest=BacktestConfig(bar_hours=4, rebalance_every_bars=1),
    )
    paths = run_experiment(config)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))

    assert paths["pnl"].exists()
    assert paths["weights"].exists()
    assert "annual_return" in summary
    assert "gross_total_return" in summary


def test_experiment_can_skip_large_output_files(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    (data_dir / "klines").mkdir(parents=True)
    (data_dir / "funding").mkdir(parents=True)
    for symbol, frame in {
        "AAAUSDT": _symbol_frame("AAAUSDT", [100 + i for i in range(12)], quote_volume=5_000_000),
        "BBBUSDT": _symbol_frame("BBBUSDT", [120 - i for i in range(12)], quote_volume=4_500_000),
        "CCCUSDT": _symbol_frame("CCCUSDT", [100] * 12, quote_volume=4_000_000),
    }.items():
        frame.to_csv(data_dir / "klines" / f"{symbol}_4h.csv", index=False)
        pd.DataFrame({"fundingTime": [], "fundingRate": []}).to_csv(data_dir / "funding" / f"{symbol}.csv", index=False)

    config = ExperimentConfig(
        name="compact",
        data_dir=data_dir,
        processed_dir=tmp_path / "processed",
        results_dir=tmp_path / "results",
        universe=UniverseConfig(top_n=3, liquidity_lookback_bars=2, min_history_bars=2),
        signal=SignalConfig(lookback_bars=2),
        portfolio=PortfolioConfig(long_quantile=1 / 3, short_quantile=1 / 3),
        outputs=OutputConfig(write_weights=False, write_pnl=False, write_universe=False, write_pressure=False),
    )
    paths = run_experiment(config)

    assert paths["summary"].exists()
    assert not paths["weights"].exists()
    assert not paths["pnl"].exists()
    assert not paths["universe"].exists()
    assert not paths["pressure"].exists()


def test_rebalance_every_six_bars_changes_weights_daily() -> None:
    idx = pd.date_range("2024-01-01", periods=12, freq="4h", tz="UTC")
    signal = pd.DataFrame(
        {
            "AAAUSDT": range(12),
            "BBBUSDT": range(12, 0, -1),
            "CCCUSDT": [0] * 12,
        },
        index=idx,
    )
    membership = pd.DataFrame(True, index=idx, columns=signal.columns)
    weights = build_market_neutral_weights(signal, membership, PortfolioConfig(long_quantile=1 / 3, short_quantile=1 / 3))
    daily = weights.iloc[::6].reindex(weights.index).ffill().fillna(0.0)
    assert daily.iloc[0].equals(daily.iloc[5])
    assert daily.iloc[6].equals(daily.iloc[11])


class _PagedClient:
    def __init__(self) -> None:
        self.kline_calls = 0
        self.funding_calls = 0

    def klines(self, symbol: str, interval: str = "4h", start_time: int | None = None, end_time: int | None = None, limit: int = 1500) -> pd.DataFrame:
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

    def funding_rates(self, symbol: str, start_time: int | None = None, end_time: int | None = None, limit: int = 1000) -> pd.DataFrame:
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
    klines = fetch_klines_range(client, symbol="AAAUSDT", interval="4h", start_time=0, end_time=57_600_000, limit=2)
    funding = fetch_funding_range(client, symbol="AAAUSDT", start_time=0, end_time=57_600_000, limit=2)

    assert len(klines) == 3
    assert klines["open_time"].is_unique
    assert len(funding) == 3
    assert funding["fundingTime"].is_unique
