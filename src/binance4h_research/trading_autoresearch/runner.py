from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

from .evaluate import evaluate_current_strategy
from .prepare_market import build_context, save_context_summary
from .program import TradingAutoResearchProgram
from .store import append_result, load_champion, load_results, save_champion, snapshot_strategy, write_results_tsv


def build_trading_context(program: TradingAutoResearchProgram) -> dict[str, Path]:
    context = build_context(program)
    summary_path = save_context_summary(context, program)
    return {"context_summary": summary_path}


def _current_strategy_path() -> Path:
    return Path(__file__).with_name("strategy.py")


def evaluate_and_record(program: TradingAutoResearchProgram) -> dict[str, Path]:
    run_dir = program.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluation = evaluate_current_strategy(program)
    run_id = f"{evaluation.family}_{evaluation.strategy_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    run_output_dir = run_dir / "runs" / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)

    champions_path = run_dir / "champions.json"
    experiments_path = run_dir / "experiments.jsonl"
    champions = load_champion(champions_path)
    current_champion = champions.get(evaluation.family)
    champion_score = float(current_champion["primary_score"]) if current_champion else float("-inf")
    parent_run_id = str(current_champion["run_id"]) if current_champion else ""
    is_champion = evaluation.status == "keep" and evaluation.primary_score > champion_score

    record = {
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "family": evaluation.family,
        "strategy_id": evaluation.strategy_id,
        "summary": evaluation.summary,
        "splits": evaluation.splits,
        "walk_forward": evaluation.walk_forward,
        "primary_score": evaluation.primary_score,
        "status": evaluation.status,
        "reason_tags": evaluation.reason_tags,
        "champion": is_champion,
    }
    append_result(experiments_path, record)
    records = load_results(experiments_path)
    results_tsv = write_results_tsv(run_dir / "results.tsv", records)

    strategy_snapshot = snapshot_strategy(_current_strategy_path(), run_output_dir, run_id)
    summary_path = run_output_dir / "summary.json"
    summary_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    if is_champion:
        champions[evaluation.family] = {
            "run_id": run_id,
            "primary_score": evaluation.primary_score,
            "strategy_id": evaluation.strategy_id,
            "summary": evaluation.summary,
            "strategy_snapshot": str(strategy_snapshot),
        }
        save_champion(champions_path, champions)
        snapshot_strategy(_current_strategy_path(), run_dir / "champions" / evaluation.family, run_id)
    elif not champions_path.exists():
        save_champion(champions_path, {})

    return {
        "summary": summary_path,
        "strategy_snapshot": strategy_snapshot,
        "experiments": experiments_path,
        "results_tsv": results_tsv,
        "champions": champions_path,
    }


def run_trading_autoresearch_batch(program: TradingAutoResearchProgram) -> dict[str, Path]:
    return evaluate_and_record(program)


def show_trading_champion(program: TradingAutoResearchProgram) -> Path:
    champions_path = program.run_dir / "champions.json"
    if not champions_path.exists():
        raise FileNotFoundError(f"Missing champions file: {champions_path}")
    return champions_path


def replay_trading_run(program: TradingAutoResearchProgram, run_id: str) -> Path:
    experiments = load_results(program.run_dir / "experiments.jsonl")
    record = next((item for item in experiments if item["run_id"] == run_id), None)
    if record is None:
        raise FileNotFoundError(f"Unknown run_id: {run_id}")
    output = program.run_dir / f"replay_{run_id}.json"
    output.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return output
