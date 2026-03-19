from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from binance4h_research.structure_scan import run_structure_scan


def _symbol_frame(base: float, step: float, bars: int, quote_volume: float, imbalance: float) -> pd.DataFrame:
    closes = [base + step * idx for idx in range(bars)]
    opens = [closes[0], *closes[:-1]]
    times = pd.date_range("2024-01-01", periods=bars, freq="4h", tz="UTC")
    taker_buy_quote_volume = quote_volume * imbalance
    return pd.DataFrame(
        {
            "open_time": times,
            "open": opens,
            "high": [value * 1.01 for value in closes],
            "low": [value * 0.99 for value in closes],
            "close": closes,
            "volume": [1000.0] * bars,
            "close_time": times + pd.Timedelta(hours=4) - pd.Timedelta(milliseconds=1),
            "quote_volume": [quote_volume] * bars,
            "trade_count": [100 + idx % 5 for idx in range(bars)],
            "taker_buy_base_volume": [500.0] * bars,
            "taker_buy_quote_volume": [taker_buy_quote_volume] * bars,
        }
    )


def _seed_market(tmp_path: Path) -> Path:
    data_dir = tmp_path / "raw"
    (data_dir / "klines").mkdir(parents=True)
    (data_dir / "funding").mkdir(parents=True)
    symbols = {
        "BTCUSDT": _symbol_frame(100.0, 0.2, 260, 5_000_000.0, 0.60),
        "ETHUSDT": _symbol_frame(80.0, 0.15, 260, 4_000_000.0, 0.58),
        "SOLUSDT": _symbol_frame(60.0, -0.08, 260, 3_000_000.0, 0.42),
        "XRPUSDT": _symbol_frame(40.0, -0.05, 260, 2_500_000.0, 0.40),
    }
    funding_times = pd.date_range("2024-01-01 08:00:00", periods=130, freq="8h", tz="UTC")
    for symbol, frame in symbols.items():
        frame.to_csv(data_dir / "klines" / f"{symbol}_4h.csv", index=False)
        funding = pd.DataFrame(
            {
                "symbol": [symbol] * len(funding_times),
                "fundingTime": funding_times,
                "fundingRate": [0.0001 if "USDT" in symbol else 0.0] * len(funding_times),
                "markPrice": [frame["close"].iloc[min(idx * 2, len(frame) - 1)] for idx in range(len(funding_times))],
            }
        )
        funding.to_csv(data_dir / "funding" / f"{symbol}.csv", index=False)
    return data_dir


def test_run_structure_scan_writes_expected_outputs(tmp_path: Path) -> None:
    data_dir = _seed_market(tmp_path)
    output_dir = tmp_path / "results" / "structure_scan"

    paths = run_structure_scan(
        data_dir=data_dir,
        output_dir=output_dir,
        top_n=3,
        liquidity_lookback_bars=20,
        min_history_bars=20,
    )

    assert paths["object_overview"].exists()
    assert paths["time_dependence"].exists()
    assert paths["cross_sectional_sorting"].exists()
    assert paths["state_dependence"].exists()
    assert paths["summary"].exists()
    assert paths["artifacts"].exists()

    overview = pd.read_csv(paths["object_overview"])
    time_dep = pd.read_csv(paths["time_dependence"])
    cross = pd.read_csv(paths["cross_sectional_sorting"])
    state = pd.read_csv(paths["state_dependence"])
    artifacts = json.loads(paths["artifacts"].read_text(encoding="utf-8"))
    summary = paths["summary"].read_text(encoding="utf-8")

    assert set(overview["object_name"]) == {
        "returns",
        "volatility",
        "volume",
        "liquidity",
        "order_flow",
        "carry",
        "correlation",
    }
    assert {"btc", "market_median", "symbol_median"} <= set(time_dep["scope"])
    assert {1, 3, 6} <= set(cross["horizon_bars"])
    assert "btc_vol_high" in set(state["state_name"])
    assert artifacts["config"]["top_n"] == 3
    assert "## 有没有统计结构" in summary
