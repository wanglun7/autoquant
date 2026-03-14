from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
from math import inf

from .evaluate import evaluate_current_strategy
from .prepare_market import build_context, save_context_summary
from .program import TradingAutoResearchProgram
from .store import (
    append_research_turn,
    append_result,
    ensure_research_log,
    load_champion,
    load_research_turns,
    load_results,
    publish_family_champion,
    save_champion,
    snapshot_strategy,
    validate_research_turn,
    write_results_tsv,
)


TRACKED_FAMILIES = ("cross_sectional", "btc_time_series", "relative_value")


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


def _tracked_families(records: list[dict[str, object]], turns: list[dict[str, object]], champions: dict[str, object]) -> list[str]:
    names = list(TRACKED_FAMILIES)
    for record in records:
        family = str(record["family"])
        if family not in names:
            names.append(family)
    for turn in turns:
        family = str(turn["family"])
        if family not in names:
            names.append(family)
    for family in dict(champions.get("families", {})):
        if family not in names:
            names.append(family)
    return names


def _is_high_quality_turn(turn: dict[str, object]) -> bool:
    return (
        str(turn.get("info_gain", "")) in {"high", "medium"}
        and not str(turn.get("duplicate_of_run_id", ""))
        and bool(str(turn.get("conclusion", "")).strip())
        and bool(str(turn.get("next_best_axis", "")).strip())
    )


def _family_metrics(
    family: str,
    *,
    records: list[dict[str, object]],
    turns: list[dict[str, object]],
    champions: dict[str, object],
) -> dict[str, object]:
    family_records = [record for record in records if record.get("family") == family]
    family_turns = [turn for turn in turns if turn.get("family") == family]
    champion = dict(champions.get("families", {})).get(family)
    duplicate_turns = sum(1 for turn in family_turns if str(turn.get("duplicate_of_run_id", "")))
    high_quality_turns = sum(1 for turn in family_turns if _is_high_quality_turn(turn))
    return {
        "experiments": len(family_records),
        "research_turns": len(family_turns),
        "keep_runs": sum(1 for record in family_records if record.get("status") == "keep"),
        "duplicate_turns": duplicate_turns,
        "high_quality_turns": high_quality_turns,
        "has_family_champion": champion is not None,
        "family_champion_run_id": str(champion["run_id"]) if champion else "",
        "family_champion_score": float(champion["primary_score"]) if champion else None,
        "last_run_id": str(family_records[-1]["run_id"]) if family_records else "",
    }


def _weakness_key(metrics: dict[str, object]) -> tuple[float, float, int, int]:
    has_champion = 1.0 if metrics["has_family_champion"] else 0.0
    champion_score = float(metrics["family_champion_score"]) if metrics["family_champion_score"] is not None else -inf
    return (
        has_champion,
        champion_score,
        int(metrics["research_turns"]),
        int(metrics["experiments"]),
    )


def build_trading_research_scorecard(program: TradingAutoResearchProgram) -> dict[str, object]:
    records = load_results(program.run_dir / "experiments.jsonl")
    turns = load_research_turns(program.research_log_path)
    champions = load_champion(program.run_dir / "champions.json")
    families = _tracked_families(records, turns, champions)
    family_rows = {
        family: _family_metrics(family, records=records, turns=turns, champions=champions)
        for family in families
    }
    duplicate_turns = sum(1 for turn in turns if str(turn.get("duplicate_of_run_id", "")))
    high_quality_turns = sum(1 for turn in turns if _is_high_quality_turn(turn))
    explanation_complete_turns = sum(
        1
        for turn in turns
        if bool(str(turn.get("objective", "")).strip())
        and bool(str(turn.get("hypothesis", "")).strip())
        and bool(str(turn.get("conclusion", "")).strip())
        and bool(str(turn.get("next_best_axis", "")).strip())
    )
    recommended_family = min(family_rows.items(), key=lambda item: _weakness_key(item[1]))[0] if family_rows else ""
    generated_at = datetime.now(timezone.utc).isoformat()
    return {
        "generated_at": generated_at,
        "summary": {
            "total_experiments": len(records),
            "research_turns": len(turns),
            "high_quality_experiments": high_quality_turns,
            "duplicate_rate": (duplicate_turns / len(turns)) if turns else 0.0,
            "explanation_completeness_rate": (explanation_complete_turns / len(turns)) if turns else 0.0,
            "family_asset_count": len(dict(champions.get("families", {}))),
            "knowledge_asset_count": len(turns),
        },
        "recommended_next_turn": {
            "turn_mode": "explore",
            "family": recommended_family,
        },
        "families": family_rows,
    }


def update_trading_research_scorecard(program: TradingAutoResearchProgram) -> Path:
    payload = build_trading_research_scorecard(program)
    path = program.research_scorecard_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def show_trading_research_scorecard(program: TradingAutoResearchProgram) -> Path:
    if not program.research_scorecard_path.exists():
        return update_trading_research_scorecard(program)
    return program.research_scorecard_path


def replay_trading_run(program: TradingAutoResearchProgram, run_id: str) -> Path:
    experiments = load_results(program.run_dir / "experiments.jsonl")
    record = next((item for item in experiments if item["run_id"] == run_id), None)
    if record is None:
        raise FileNotFoundError(f"Unknown run_id: {run_id}")
    output = program.run_dir / f"replay_{run_id}.json"
    output.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return output
