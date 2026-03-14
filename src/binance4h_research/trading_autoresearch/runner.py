from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

from .evaluate import evaluate_current_strategy
from .prepare_market import build_context, save_context_summary
from .program import TradingAutoResearchProgram
from .store import (
    append_research_turn,
    append_result,
    ensure_research_log,
    load_champion,
    load_results,
    publish_family_champion,
    save_champion,
    snapshot_strategy,
    validate_research_turn,
    write_results_tsv,
)


def build_trading_context(program: TradingAutoResearchProgram) -> dict[str, Path]:
    context = build_context(program)
    summary_path = save_context_summary(context, program)
    return {"context_summary": summary_path}


def _current_strategy_path() -> Path:
    return Path(__file__).with_name("strategy.py")


def _repo_family_champion_dir() -> Path:
    return Path(__file__).with_name("family_champions")


def _best_record(records: list[dict[str, object]], family: str | None = None) -> dict[str, object] | None:
    candidates = [record for record in records if record.get("status") == "keep"]
    if family is not None:
        candidates = [record for record in candidates if record.get("family") == family]
    if not candidates:
        return None
    return max(candidates, key=lambda record: float(record["primary_score"]))


def _record_run_id(record: dict[str, object] | None) -> str:
    if not record:
        return ""
    return str(record["run_id"])


def _record_score(record: dict[str, object] | None) -> float:
    if not record:
        return float("-inf")
    return float(record["primary_score"])


def _active_champion_flag(
    program: TradingAutoResearchProgram,
    *,
    family_champion: bool,
    global_champion: bool,
) -> bool:
    if program.champion_family_mode == "by_family":
        return family_champion
    if program.champion_family_mode == "global":
        return global_champion
    raise ValueError(f"Unsupported champion_family_mode: {program.champion_family_mode}")


def _champion_scope(*, family_champion: bool, global_champion: bool) -> str:
    if global_champion:
        return "global"
    if family_champion:
        return "family"
    return "none"


def _champion_entry(record: dict[str, object], strategy_snapshot: Path) -> dict[str, object]:
    return {
        "run_id": record["run_id"],
        "primary_score": record["primary_score"],
        "family": record["family"],
        "strategy_id": record["strategy_id"],
        "summary": record["summary"],
        "strategy_snapshot": str(strategy_snapshot),
    }


def evaluate_and_record(program: TradingAutoResearchProgram) -> dict[str, Path]:
    run_dir = program.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluation = evaluate_current_strategy(program)
    run_id = f"{evaluation.family}_{evaluation.strategy_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    run_output_dir = run_dir / "runs" / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)

    champions_path = run_dir / "champions.json"
    experiments_path = run_dir / "experiments.jsonl"
    existing_records = load_results(experiments_path)
    family_parent = _best_record(existing_records, family=evaluation.family)
    global_parent = _best_record(existing_records)
    family_champion = evaluation.status == "keep" and evaluation.primary_score > _record_score(family_parent)
    global_champion = evaluation.status == "keep" and evaluation.primary_score > _record_score(global_parent)
    champion = _active_champion_flag(
        program,
        family_champion=family_champion,
        global_champion=global_champion,
    )
    parent_record = global_parent if program.champion_family_mode == "global" else family_parent

    record = {
        "run_id": run_id,
        "parent_run_id": _record_run_id(parent_record),
        "family_parent_run_id": _record_run_id(family_parent),
        "global_parent_run_id": _record_run_id(global_parent),
        "family": evaluation.family,
        "strategy_id": evaluation.strategy_id,
        "summary": evaluation.summary,
        "splits": evaluation.splits,
        "walk_forward": evaluation.walk_forward,
        "primary_score": evaluation.primary_score,
        "status": evaluation.status,
        "reason_tags": evaluation.reason_tags,
        "family_champion": family_champion,
        "global_champion": global_champion,
        "champion_scope": _champion_scope(
            family_champion=family_champion,
            global_champion=global_champion,
        ),
        "champion": champion,
    }
    append_result(experiments_path, record)
    results_tsv = write_results_tsv(run_dir / "results.tsv", [*existing_records, record])

    strategy_snapshot = snapshot_strategy(_current_strategy_path(), run_output_dir, run_id)
    summary_path = run_output_dir / "summary.json"
    summary_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    champions = load_champion(champions_path)
    families = dict(champions.get("families", {}))
    if family_champion:
        family_snapshot = snapshot_strategy(_current_strategy_path(), run_dir / "champions" / evaluation.family, run_id)
        published_source = publish_family_champion(_current_strategy_path(), _repo_family_champion_dir(), evaluation.family)
        families[evaluation.family] = _champion_entry(record, family_snapshot)
        families[evaluation.family]["published_source"] = str(published_source)
    champions["families"] = families
    if global_champion:
        global_snapshot = snapshot_strategy(_current_strategy_path(), run_dir / "champions" / "global", run_id)
        champions["global"] = _champion_entry(record, global_snapshot)
    elif "global" not in champions:
        champions["global"] = None
    save_champion(champions_path, champions)

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


def record_trading_research_turn(program: TradingAutoResearchProgram, note_path: str | Path) -> Path:
    payload = json.loads(Path(note_path).read_text(encoding="utf-8"))
    note = validate_research_turn(payload)
    known_run_ids = {str(record["run_id"]) for record in load_results(program.run_dir / "experiments.jsonl")}
    if note["run_id"] not in known_run_ids:
        raise FileNotFoundError(f"Unknown run_id: {note['run_id']}")
    baseline_run_id = str(note["baseline_run_id"])
    if baseline_run_id and baseline_run_id not in known_run_ids:
        raise FileNotFoundError(f"Unknown baseline_run_id: {baseline_run_id}")
    return append_research_turn(program.research_log_path, note)


def show_trading_research_log(program: TradingAutoResearchProgram) -> Path:
    return ensure_research_log(program.research_log_path)


def replay_trading_run(program: TradingAutoResearchProgram, run_id: str) -> Path:
    experiments = load_results(program.run_dir / "experiments.jsonl")
    record = next((item for item in experiments if item["run_id"] == run_id), None)
    if record is None:
        raise FileNotFoundError(f"Unknown run_id: {run_id}")
    output = program.run_dir / f"replay_{run_id}.json"
    output.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return output
