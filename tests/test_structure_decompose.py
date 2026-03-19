from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from binance4h_research.structure_decompose import run_structure_decompose


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
        "BTCUSDT": _symbol_frame(100.0, 0.2, 320, 5_000_000.0, 0.60),
        "ETHUSDT": _symbol_frame(80.0, 0.15, 320, 4_000_000.0, 0.58),
        "SOLUSDT": _symbol_frame(60.0, -0.08, 320, 3_000_000.0, 0.42),
        "XRPUSDT": _symbol_frame(40.0, -0.05, 320, 2_500_000.0, 0.40),
        "DOGEUSDT": _symbol_frame(20.0, 0.04, 320, 2_200_000.0, 0.57),
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


def test_run_structure_decompose_writes_expected_outputs(tmp_path: Path) -> None:
    data_dir = _seed_market(tmp_path)
    output_dir = tmp_path / "results" / "structure_decompose"

    paths = run_structure_decompose(
        data_dir=data_dir,
        output_dir=output_dir,
        top_n=4,
        liquidity_lookback_bars=20,
        min_history_bars=20,
    )

    for key in (
        "core_feature_overview",
        "feature_corr_full",
        "feature_corr_yearly",
        "residual_signal_tests",
        "bivariate_sorts",
        "conditional_sorts",
        "factor_role_summary",
        "summary",
        "artifacts",
    ):
        assert paths[key].exists()

    overview = pd.read_csv(paths["core_feature_overview"])
    residual = pd.read_csv(paths["residual_signal_tests"])
    bivariate = pd.read_csv(paths["bivariate_sorts"])
    conditional = pd.read_csv(paths["conditional_sorts"])
    role = pd.read_csv(paths["factor_role_summary"])
    artifacts = json.loads(paths["artifacts"].read_text(encoding="utf-8"))
    summary = paths["summary"].read_text(encoding="utf-8")

    assert set(overview["feature_name"]) >= {
        "ret_6b",
        "realized_vol_6b",
        "range_vol",
        "abs_funding",
        "corr_btc_30d",
        "imbalance_3b_mean",
    }
    assert {"winsorized", "raw"} <= set(overview["variant"])
    assert {"ret_6b", "abs_funding", "corr_btc_30d"} <= set(residual["target_feature"])
    assert {"realized_vol_6b", "ret_6b", "corr_btc_30d"} <= set(bivariate["primary_feature"])
    assert "abs_funding" in set(bivariate["secondary_feature"])
    assert {"high", "low", "extreme", "non_extreme"} <= set(conditional["condition_mode"])
    assert set(role["role"]) <= {"Primary signal", "Secondary signal", "Condition/filter", "Redundant proxy"}
    assert artifacts["config"]["top_n"] == 4
    assert "## 五个核心问题" in summary
