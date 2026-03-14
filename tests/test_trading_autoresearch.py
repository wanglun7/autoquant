from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

from binance4h_research.cli import main as cli_main
from binance4h_research.trading_autoresearch.evaluate import TradingEvaluation, evaluate_current_strategy
from binance4h_research.trading_autoresearch.program import TradingAutoResearchProgram
from binance4h_research.trading_autoresearch.runner import (
    build_trading_context,
    build_trading_family_registry,
    build_trading_research_scorecard,
    evaluate_and_record,
    record_trading_research_turn,
    replay_trading_run,
    run_trading_autoresearch_batch,
    show_trading_family_registry,
    show_trading_research_log,
    show_trading_research_scorecard,
    update_trading_family_registry,
    update_trading_research_scorecard,
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


def _fake_evaluation(
    strategy_id: str,
    family: str,
    score: float,
    *,
    execution_mode: str | None = None,
    family_stage: str | None = None,
) -> TradingEvaluation:
    return TradingEvaluation(
        strategy_id=strategy_id,
        family=family,
        execution_mode=execution_mode or ("cross_sectional" if family == "cross_sectional" else "time_series"),
        family_stage=family_stage or ("formal" if family in {"cross_sectional", "btc_time_series", "relative_value"} else "candidate"),
        summary={"annual_return": score, "net_total_return": score},
        splits={
            "train": {"annual_return": score, "sharpe": score, "net_total_return": score, "max_drawdown": -0.1},
            "validation": {"annual_return": score, "sharpe": score, "net_total_return": score, "max_drawdown": -0.1},
            "test": {"annual_return": score, "sharpe": score, "net_total_return": score, "max_drawdown": -0.1},
        },
        walk_forward=[],
        primary_score=score,
        status="keep",
        reason_tags=[],
    )


def _research_turn_note(
    *,
    run_id: str,
    baseline_run_id: str,
    family: str,
    execution_mode: str,
    family_stage: str,
    turn_action: str = "experiment_within_family",
    info_gain: str = "medium",
    duplicate_of_run_id: str = "",
    mechanism_tag: str = "trend-strength-scaling",
    mechanism_summary: str = "Scale risk by trend strength to reduce weak-regime overexposure.",
    differentiation_note: str = "This family focuses on strength-scaled pullback timing rather than generic BTC trend following.",
) -> dict[str, object]:
    return {
        "timestamp": "2026-03-14T04:30:00Z",
        "family": family,
        "execution_mode": execution_mode,
        "family_stage": family_stage,
        "turn_mode": "explore",
        "turn_action": turn_action,
        "mechanism_tag": mechanism_tag,
        "mechanism_summary": mechanism_summary,
        "differentiation_note": differentiation_note,
        "objective": "Improve walk-forward stability without harming 2x cost resilience.",
        "hypothesis": "Scaling position size by trend strength may reduce weak-trend overexposure.",
        "planned_change": "Scale active BTC position size by normalized trend strength.",
        "success_criteria": "Non-negative deltas in test Sharpe, test net return, and stress_2x net return.",
        "baseline_run_id": baseline_run_id,
        "run_id": run_id,
        "status": "keep",
        "family_champion": False,
        "global_champion": False,
        "failure_mode": "no_info_gain",
        "info_gain": info_gain,
        "duplicate_of_run_id": duplicate_of_run_id,
        "family_state_before_turn": {
            "family_champion_run_id": baseline_run_id,
            "family_champion_score": 0.47,
            "research_turns": 1,
            "experiments": 2,
        },
        "comparison": {
            "test_sharpe_delta": -0.02,
            "test_net_return_delta": -0.01,
            "stress_2x_delta": -0.03,
        },
        "conclusion": "The idea slightly degraded the champion.",
        "next_best_axis": "Entry timing",
    }


def _write_program_yaml(tmp_path: Path, data_dir: Path) -> Path:
    path = tmp_path / "program.yaml"
    path.write_text(
        "\n".join(
            [
                "name: trading_autoresearch_test",
                f"data_dir: {data_dir}",
                f"processed_dir: {tmp_path / 'processed'}",
                f"results_dir: {tmp_path / 'results'}",
                "",
                "costs:",
                "  fee_bps: 4.0",
                "  slippage_bps: 2.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


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
    assert evaluation.family in {"cross_sectional", "btc_time_series", "relative_value"}
    assert evaluation.execution_mode in {"cross_sectional", "time_series", "pair_trade"}
    assert evaluation.family_stage in {"formal", "candidate"}
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
    assert outputs["families"].exists()

    rows = [line for line in outputs["experiments"].read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1

    record = json.loads(rows[0])
    assert "parent_run_id" in record
    assert "family_champion" in record
    assert "global_champion" in record
    assert "champion_scope" in record
    assert "execution_mode" in record
    assert "family_stage" in record
    champions = json.loads(outputs["champions"].read_text(encoding="utf-8"))
    assert set(champions) == {"families", "global"}
    families = json.loads(outputs["families"].read_text(encoding="utf-8"))
    assert "families" in families
    replay_path = replay_trading_run(program, record["run_id"])
    assert replay_path.exists()


def test_trading_runner_distinguishes_family_and_global_champions(tmp_path: Path, monkeypatch) -> None:
    data_dir = _seed_market(tmp_path)
    program = TradingAutoResearchProgram(
        data_dir=data_dir,
        processed_dir=tmp_path / "processed",
        results_dir=tmp_path / "results",
    )
    strategy_path = tmp_path / "strategy.py"
    strategy_path.write_text("print('strategy')\n", encoding="utf-8")

    evaluations = iter(
        [
            _fake_evaluation("cs_a", "cross_sectional", 0.50, execution_mode="cross_sectional", family_stage="formal"),
            _fake_evaluation("btc_a", "btc_time_series", 0.30, execution_mode="time_series", family_stage="formal"),
            _fake_evaluation("btc_b", "btc_time_series", 0.60, execution_mode="time_series", family_stage="formal"),
        ]
    )

    monkeypatch.setattr(
        "binance4h_research.trading_autoresearch.runner.evaluate_current_strategy",
        lambda _: next(evaluations),
    )
    monkeypatch.setattr(
        "binance4h_research.trading_autoresearch.runner._current_strategy_path",
        lambda: strategy_path,
    )
    monkeypatch.setattr(
        "binance4h_research.trading_autoresearch.runner._repo_family_champion_dir",
        lambda: tmp_path / "family_champions",
    )

    outputs = [evaluate_and_record(program) for _ in range(3)]

    records = [
        json.loads(line)
        for line in Path(outputs[-1]["experiments"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [record["champion_scope"] for record in records] == ["global", "family", "global"]
    assert [record["family_champion"] for record in records] == [True, True, True]
    assert [record["global_champion"] for record in records] == [True, False, True]
    assert [record["champion"] for record in records] == [True, True, True]

    champions = json.loads(Path(outputs[-1]["champions"]).read_text(encoding="utf-8"))
    assert champions["families"]["cross_sectional"]["run_id"] == records[0]["run_id"]
    assert champions["families"]["btc_time_series"]["run_id"] == records[2]["run_id"]
    assert Path(champions["families"]["cross_sectional"]["published_source"]).exists()
    assert Path(champions["families"]["btc_time_series"]["published_source"]).exists()
    assert champions["global"]["run_id"] == records[2]["run_id"]


def test_trading_runner_global_mode_uses_global_promotion_alias(tmp_path: Path, monkeypatch) -> None:
    data_dir = _seed_market(tmp_path)
    program = TradingAutoResearchProgram(
        data_dir=data_dir,
        processed_dir=tmp_path / "processed",
        results_dir=tmp_path / "results",
        champion_family_mode="global",
    )
    strategy_path = tmp_path / "strategy.py"
    strategy_path.write_text("print('strategy')\n", encoding="utf-8")

    evaluations = iter(
        [
            _fake_evaluation("cs_a", "cross_sectional", 0.50, execution_mode="cross_sectional", family_stage="formal"),
            _fake_evaluation("btc_a", "btc_time_series", 0.30, execution_mode="time_series", family_stage="formal"),
        ]
    )

    monkeypatch.setattr(
        "binance4h_research.trading_autoresearch.runner.evaluate_current_strategy",
        lambda _: next(evaluations),
    )
    monkeypatch.setattr(
        "binance4h_research.trading_autoresearch.runner._current_strategy_path",
        lambda: strategy_path,
    )
    monkeypatch.setattr(
        "binance4h_research.trading_autoresearch.runner._repo_family_champion_dir",
        lambda: tmp_path / "family_champions",
    )

    outputs = [evaluate_and_record(program) for _ in range(2)]

    records = [
        json.loads(line)
        for line in Path(outputs[-1]["experiments"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records[0]["champion"] is True
    assert records[1]["family_champion"] is True
    assert records[1]["global_champion"] is False
    assert records[1]["champion"] is False


def test_trading_research_log_roundtrip_and_cli(tmp_path: Path, monkeypatch, capsys) -> None:
    data_dir = _seed_market(tmp_path)
    program = TradingAutoResearchProgram(
        name="trading_autoresearch_test",
        data_dir=data_dir,
        processed_dir=tmp_path / "processed",
        results_dir=tmp_path / "results",
    )
    outputs = run_trading_autoresearch_batch(program)
    program_yaml = _write_program_yaml(tmp_path, data_dir)

    log_path = show_trading_research_log(program)
    assert log_path.exists()
    assert log_path.read_text(encoding="utf-8") == ""

    run_record = json.loads(Path(outputs["summary"]).read_text(encoding="utf-8"))
    note = _research_turn_note(
        run_id=run_record["run_id"],
        baseline_run_id=run_record["family_parent_run_id"],
        family=run_record["family"],
        execution_mode=run_record["execution_mode"],
        family_stage=run_record["family_stage"],
    )
    note_path = tmp_path / "note.json"
    note_path.write_text(json.dumps(note), encoding="utf-8")

    recorded_path = record_trading_research_turn(program, note_path)
    assert recorded_path == log_path
    entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(entries) == 1
    assert entries[0]["run_id"] == run_record["run_id"]

    monkeypatch.setattr(sys, "argv", ["binance4h", "show-trading-research-log", "--program", str(program_yaml)])
    cli_main()
    assert capsys.readouterr().out.strip() == str(log_path)

    monkeypatch.setattr(
        sys,
        "argv",
        ["binance4h", "record-trading-research-turn", "--program", str(program_yaml), "--note-file", str(note_path)],
    )
    cli_main()
    assert capsys.readouterr().out.strip() == str(log_path)
    entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(entries) == 2

    family_registry = build_trading_family_registry(program)
    assert family_registry["families"][run_record["family"]]["stage"] == "formal"
    family_registry_path = update_trading_family_registry(program)
    assert family_registry_path.exists()
    shown_family_registry = show_trading_family_registry(program)
    assert shown_family_registry == family_registry_path

    monkeypatch.setattr(
        sys,
        "argv",
        ["binance4h", "update-trading-family-registry", "--program", str(program_yaml)],
    )
    cli_main()
    assert capsys.readouterr().out.strip() == str(family_registry_path)

    monkeypatch.setattr(
        sys,
        "argv",
        ["binance4h", "show-trading-family-registry", "--program", str(program_yaml)],
    )
    cli_main()
    assert capsys.readouterr().out.strip() == str(family_registry_path)

    scorecard = build_trading_research_scorecard(program)
    assert scorecard["summary"]["research_turns"] == 2
    assert scorecard["summary"]["high_quality_experiments"] == 2
    assert scorecard["recommended_next_turn"]["turn_mode"] == "explore"
    assert scorecard["recommended_next_turn"]["action"] == "explore_new_family"
    assert scorecard["recommended_next_turn"]["execution_mode"] == "cross_sectional"

    scorecard_path = update_trading_research_scorecard(program)
    assert scorecard_path.exists()
    shown_path = show_trading_research_scorecard(program)
    assert shown_path == scorecard_path

    monkeypatch.setattr(
        sys,
        "argv",
        ["binance4h", "update-trading-research-scorecard", "--program", str(program_yaml)],
    )
    cli_main()
    assert capsys.readouterr().out.strip() == str(scorecard_path)

    monkeypatch.setattr(
        sys,
        "argv",
        ["binance4h", "show-trading-research-scorecard", "--program", str(program_yaml)],
    )
    cli_main()
    assert capsys.readouterr().out.strip() == str(scorecard_path)


def test_candidate_family_promotes_to_formal_and_publishes_champion(tmp_path: Path, monkeypatch) -> None:
    data_dir = _seed_market(tmp_path)
    program = TradingAutoResearchProgram(
        data_dir=data_dir,
        processed_dir=tmp_path / "processed",
        results_dir=tmp_path / "results",
    )
    strategy_path = tmp_path / "strategy.py"
    strategy_path.write_text("print('strategy')\n", encoding="utf-8")

    evaluations = iter(
        [
            _fake_evaluation("candidate_a", "trend_pullback_state_machine", 0.40, execution_mode="time_series", family_stage="candidate"),
            _fake_evaluation("candidate_b", "trend_pullback_state_machine", 0.55, execution_mode="time_series", family_stage="candidate"),
        ]
    )

    monkeypatch.setattr(
        "binance4h_research.trading_autoresearch.runner.evaluate_current_strategy",
        lambda _: next(evaluations),
    )
    monkeypatch.setattr(
        "binance4h_research.trading_autoresearch.runner._current_strategy_path",
        lambda: strategy_path,
    )
    monkeypatch.setattr(
        "binance4h_research.trading_autoresearch.runner._repo_family_champion_dir",
        lambda: tmp_path / "family_champions",
    )

    outputs = [evaluate_and_record(program) for _ in range(2)]
    run_records = [json.loads(Path(output["summary"]).read_text(encoding="utf-8")) for output in outputs]
    assert all(record["family_champion"] is False for record in run_records)

    note_paths: list[Path] = []
    for record in run_records:
        note = _research_turn_note(
            run_id=record["run_id"],
            baseline_run_id=record["family_parent_run_id"],
            family=record["family"],
            execution_mode=record["execution_mode"],
            family_stage=record["family_stage"],
            turn_action="propose_new_family",
            mechanism_tag="trend-pullback-state-machine",
            mechanism_summary="Use a state-machine pullback entry instead of a generic BTC pullback rule.",
            differentiation_note="This is a standalone state-machine family, not a parameter variant of btc_time_series.",
        )
        note_path = tmp_path / f"{record['run_id']}.json"
        note_path.write_text(json.dumps(note), encoding="utf-8")
        note_paths.append(note_path)

    for note_path in note_paths:
        record_trading_research_turn(program, note_path)

    families = json.loads(program.family_registry_path.read_text(encoding="utf-8"))
    entry = families["families"]["trend_pullback_state_machine"]
    assert entry["stage"] == "formal"
    assert entry["keep_runs"] == 2
    assert entry["non_duplicate_turns"] == 2

    champions = json.loads((program.run_dir / "champions.json").read_text(encoding="utf-8"))
    champion = champions["families"]["trend_pullback_state_machine"]
    assert champion["run_id"] == run_records[-1]["run_id"]
    assert Path(champion["published_source"]).exists()


def test_trading_research_log_rejects_unknown_run_id(tmp_path: Path) -> None:
    data_dir = _seed_market(tmp_path)
    program = TradingAutoResearchProgram(
        name="trading_autoresearch_test",
        data_dir=data_dir,
        processed_dir=tmp_path / "processed",
        results_dir=tmp_path / "results",
    )
    note_path = tmp_path / "note.json"
    note_path.write_text(
        json.dumps(
            _research_turn_note(
                run_id="unknown-run",
                baseline_run_id="",
                family="btc_time_series",
                execution_mode="time_series",
                family_stage="formal",
            )
        ),
        encoding="utf-8",
    )

    import pytest

    with pytest.raises(FileNotFoundError, match="Unknown run_id"):
        record_trading_research_turn(program, note_path)


def test_trading_research_log_rejects_invalid_note(tmp_path: Path) -> None:
    data_dir = _seed_market(tmp_path)
    program = TradingAutoResearchProgram(
        name="trading_autoresearch_test",
        data_dir=data_dir,
        processed_dir=tmp_path / "processed",
        results_dir=tmp_path / "results",
    )
    outputs = run_trading_autoresearch_batch(program)
    run_record = json.loads(Path(outputs["summary"]).read_text(encoding="utf-8"))
    invalid_note = _research_turn_note(
        run_id=run_record["run_id"],
        baseline_run_id=run_record["family_parent_run_id"],
        family=run_record["family"],
        execution_mode=run_record["execution_mode"],
        family_stage=run_record["family_stage"],
    )
    invalid_note["comparison"].pop("stress_2x_delta")
    note_path = tmp_path / "invalid_note.json"
    note_path.write_text(json.dumps(invalid_note), encoding="utf-8")

    import pytest

    with pytest.raises(ValueError, match="Missing comparison fields"):
        record_trading_research_turn(program, note_path)
