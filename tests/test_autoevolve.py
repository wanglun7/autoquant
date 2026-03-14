from __future__ import annotations

from pathlib import Path

import pandas as pd

from binance4h_research.autoevolve.context import build_research_context
from binance4h_research.autoevolve.evaluate import evaluate_spec
from binance4h_research.autoevolve.mutate import propose_specs
from binance4h_research.autoevolve.program import AutoEvolveProgram
from binance4h_research.autoevolve.runner import build_research_context_artifacts, replay_candidate, run_evolution_batch
from binance4h_research.autoevolve.spec import FilterSpec, PortfolioSpec, StrategySpec


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
        "BTCUSDT": _symbol_frame([100 + i * 0.2 for i in range(600)], 50_000_000),
        "ETHUSDT": _symbol_frame([80 + i * 0.15 for i in range(600)], 40_000_000),
        "SOLUSDT": _symbol_frame([50 + i * 0.12 for i in range(600)], 30_000_000),
        "XRPUSDT": _symbol_frame([30 + i * 0.05 for i in range(600)], 20_000_000),
        "ADAUSDT": _symbol_frame([20 + i * 0.03 for i in range(600)], 18_000_000),
        "DOGEUSDT": _symbol_frame([10 + i * 0.02 for i in range(600)], 16_000_000),
        "BNBUSDT": _symbol_frame([60 + i * 0.1 for i in range(600)], 22_000_000),
        "LINKUSDT": _symbol_frame([25 + i * 0.04 for i in range(600)], 15_000_000),
    }
    for symbol, frame in symbols.items():
        frame.to_csv(data_dir / "klines" / f"{symbol}_4h.csv", index=False)
        funding = pd.DataFrame(
            {
                "fundingTime": pd.date_range("2024-01-01 08:00:00", periods=80, freq="8h", tz="UTC"),
                "fundingRate": [0.0001] * 80,
            }
        )
        funding.to_csv(data_dir / "funding" / f"{symbol}.csv", index=False)
    return data_dir


def test_spec_hash_and_mutation_scope(tmp_path: Path) -> None:
    data_dir = _seed_market(tmp_path)
    program = AutoEvolveProgram(data_dir=data_dir, processed_dir=tmp_path / "processed", results_dir=tmp_path / "results", batch_size=3)
    context = build_research_context(program)
    seed = StrategySpec(family="cross_sectional", model="momentum", params={"lookback_days": 120, "skip_days": 0, "top_n": 50, "rank_mode": "raw"})
    proposals = propose_specs("cross_sectional", context, program, {seed.spec_hash()}, seed, 2)
    assert len(proposals) == 2
    assert all(spec.family == "cross_sectional" for spec, _ in proposals)
    assert all(spec.spec_hash() != seed.spec_hash() for spec, _ in proposals)


def test_evaluate_supports_three_families(tmp_path: Path) -> None:
    data_dir = _seed_market(tmp_path)
    program = AutoEvolveProgram(data_dir=data_dir, processed_dir=tmp_path / "processed", results_dir=tmp_path / "results")
    context = build_research_context(program)
    specs = [
        StrategySpec(family="cross_sectional", model="momentum", params={"lookback_days": 60, "skip_days": 0, "top_n": 5, "rank_mode": "raw"}),
        StrategySpec(family="btc_time_series", model="momentum", params={"lookback_days": 20}, portfolio=PortfolioSpec(rebalance="1d", direction_mode="long_short")),
        StrategySpec(family="relative_value", model="zscore_mean_revert", params={"left": "ETHUSDT", "right": "SOLUSDT", "lookback_days": 20, "z_entry": 1.0}),
    ]
    for spec in specs:
        evaluation = evaluate_spec(spec, context, program)
        assert "annual_return" in evaluation.summary
        assert set(evaluation.splits) == {"train", "validation", "test"}


def test_runner_creates_logs_and_replay(tmp_path: Path) -> None:
    data_dir = _seed_market(tmp_path)
    program = AutoEvolveProgram(
        data_dir=data_dir,
        processed_dir=tmp_path / "processed",
        results_dir=tmp_path / "results",
        family_scopes=["cross_sectional", "btc_time_series", "relative_value"],
        batch_size=3,
    )
    context_outputs = build_research_context_artifacts(program)
    assert context_outputs["context_summary"].exists()

    outputs = run_evolution_batch(program, batch_size=3)
    assert outputs["experiments"].exists()
    assert outputs["champions"].exists()
    assert outputs["results_tsv"].exists()

    records = [line for line in outputs["experiments"].read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) >= 3
    champions = Path(outputs["champions"]).read_text(encoding="utf-8")
    assert champions

    import json

    first = json.loads(records[0])
    replay_path = replay_candidate(program, first["run_id"])
    assert replay_path.exists()
