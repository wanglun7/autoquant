from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from binance4h_research.structure_validate import run_structure_validate


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
            "volume": [1_000.0] * bars,
            "close_time": times + pd.Timedelta(hours=4) - pd.Timedelta(milliseconds=1),
            "quote_volume": [quote_volume] * bars,
            "trade_count": [100 + idx % 7 for idx in range(bars)],
            "taker_buy_base_volume": [500.0] * bars,
            "taker_buy_quote_volume": [taker_buy_quote_volume] * bars,
        }
    )


def _seed_market(tmp_path: Path) -> Path:
    data_dir = tmp_path / "raw"
    (data_dir / "klines").mkdir(parents=True)
    (data_dir / "funding").mkdir(parents=True)
    symbols = {
        "BTCUSDT": _symbol_frame(100.0, 0.20, 320, 5_000_000.0, 0.60),
        "ETHUSDT": _symbol_frame(82.0, 0.11, 320, 4_000_000.0, 0.56),
        "SOLUSDT": _symbol_frame(60.0, -0.09, 320, 3_000_000.0, 0.42),
        "XRPUSDT": _symbol_frame(40.0, -0.04, 320, 2_600_000.0, 0.39),
        "DOGEUSDT": _symbol_frame(20.0, 0.06, 320, 2_300_000.0, 0.57),
    }
    funding_times = pd.date_range("2024-01-01 08:00:00", periods=160, freq="8h", tz="UTC")
    for idx, (symbol, frame) in enumerate(symbols.items()):
        frame.to_csv(data_dir / "klines" / f"{symbol}_4h.csv", index=False)
        funding = pd.DataFrame(
            {
                "symbol": [symbol] * len(funding_times),
                "fundingTime": funding_times,
                "fundingRate": [0.0001 * (idx + 1)] * len(funding_times),
                "markPrice": [frame["close"].iloc[min(bar * 2, len(frame) - 1)] for bar in range(len(funding_times))],
            }
        )
        funding.to_csv(data_dir / "funding" / f"{symbol}.csv", index=False)
    return data_dir


def test_run_structure_validate_writes_expected_outputs(tmp_path: Path) -> None:
    data_dir = _seed_market(tmp_path)
    output_dir = tmp_path / "results" / "structure_validate"

    paths = run_structure_validate(
        data_dir=data_dir,
        output_dir=output_dir,
        top_n=5,
        liquidity_lookback_bars=20,
        min_history_bars=20,
    )

    for key in (
        "univariate_validation",
        "bivariate_validation",
        "conditional_validation",
        "cross_section_regression",
        "role_decision",
        "summary",
        "artifacts",
    ):
        assert paths[key].exists()

    univariate = pd.read_csv(paths["univariate_validation"])
    bivariate = pd.read_csv(paths["bivariate_validation"])
    conditional = pd.read_csv(paths["conditional_validation"])
    regression = pd.read_csv(paths["cross_section_regression"])
    role = pd.read_csv(paths["role_decision"])
    artifacts = json.loads(paths["artifacts"].read_text(encoding="utf-8"))
    summary = paths["summary"].read_text(encoding="utf-8")

    assert set(univariate["feature_name"]) == {"realized_vol_6b", "corr_btc_30d", "ret_6b", "abs_funding"}
    assert {"winsorized", "raw"} <= set(univariate["variant"])
    assert {3, 6} <= set(univariate["horizon_bars"])
    assert {"corr_btc_30d", "ret_6b", "abs_funding"} <= set(bivariate["secondary_feature"])
    assert {"high", "low"} <= set(conditional["condition_mode"])
    assert {"model_1", "model_2", "model_3"} <= set(regression["model_name"])
    assert {"Primary signal", "Secondary signal", "filter", "confirmation proxy", "redundant", "downgraded"} >= set(role["role"])
    assert artifacts["config"]["top_n"] == 5
    assert artifacts["core_features"] == ["realized_vol_6b", "corr_btc_30d", "ret_6b", "abs_funding"]
    assert "## 四个核心问题" in summary
