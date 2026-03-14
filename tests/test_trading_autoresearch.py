from __future__ import annotations

from pathlib import Path

import pandas as pd

from binance4h_research.trading_autoresearch.evaluate import evaluate_current_strategy
from binance4h_research.trading_autoresearch.program import TradingAutoResearchProgram
from binance4h_research.trading_autoresearch.runner import (
    build_trading_context,
    replay_trading_run,
    run_trading_autoresearch_batch,
)


def _symbol_frame(closes: list[float], quote_volume: float) -> pd.DataFrame:
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


def _seed_market(tmp_path: Path) -> Path:
    data_dir = tmp_path / "raw"
    (data_dir / "klines").mkdir(parents=True)
    (data_dir / "funding").mkdir(parents=True)
    symbols = {
        "BTCUSDT": _symbol_frame([100 + i * 0.2 for i in range(900)], 50_000_000),
        "ETHUSDT": _symbol_frame([80 + i * 0.15 for i in range(900)], 40_000_000),
        "SOLUSDT": _symbol_frame([50 + i * 0.12 for i in range(900)], 30_000_000),
        "XRPUSDT": _symbol_frame([30 + i * 0.05 for i in range(900)], 20_000_000),
        "ADAUSDT": _symbol_frame([20 + i * 0.03 for i in range(900)], 18_000_000),
        "DOGEUSDT": _symbol_frame([10 + i * 0.02 for i in range(900)], 16_000_000),
        "BNBUSDT": _symbol_frame([60 + i * 0.1 for i in range(900)], 22_000_000),
        "LINKUSDT": _symbol_frame([25 + i * 0.04 for i in range(900)], 15_000_000),
    }
    for symbol, frame in symbols.items():
        frame.to_csv(data_dir / "klines" / f"{symbol}_4h.csv", index=False)
        funding = pd.DataFrame(
            {
                "fundingTime": pd.date_range("2024-01-01 08:00:00", periods=140, freq="8h", tz="UTC"),
                "fundingRate": [0.0001] * 140,
            }
        )
        funding.to_csv(data_dir / "funding" / f"{symbol}.csv", index=False)
    return data_dir


def test_trading_context_and_evaluation(tmp_path: Path) -> None:
    data_dir = _seed_market(tmp_path)
    program = TradingAutoResearchProgram(
        data_dir=data_dir,
        processed_dir=tmp_path / "processed",
        results_dir=tmp_path / "results",
    )
    outputs = build_trading_context(program)
    assert outputs["context_summary"].exists()

    evaluation = evaluate_current_strategy(program)
    assert evaluation.family == "cross_sectional"
    assert "annual_return" in evaluation.summary
    assert set(evaluation.splits) == {"train", "validation", "test"}


def test_trading_runner_writes_logs_and_replay(tmp_path: Path) -> None:
    data_dir = _seed_market(tmp_path)
    program = TradingAutoResearchProgram(
        data_dir=data_dir,
        processed_dir=tmp_path / "processed",
        results_dir=tmp_path / "results",
    )
    outputs = run_trading_autoresearch_batch(program)
    assert outputs["summary"].exists()
    assert outputs["strategy_snapshot"].exists()
    assert outputs["experiments"].exists()
    assert outputs["results_tsv"].exists()
    assert outputs["champions"].exists()

    rows = [line for line in outputs["experiments"].read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1

    import json

    record = json.loads(rows[0])
    assert "parent_run_id" in record
    replay_path = replay_trading_run(program, record["run_id"])
    assert replay_path.exists()
