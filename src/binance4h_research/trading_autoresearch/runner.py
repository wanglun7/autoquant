from __future__ import annotations

from datetime import datetime, timezone
import json
from math import inf
from pathlib import Path

from .evaluate import evaluate_current_strategy
from .prepare_market import build_context, save_context_summary
from .program import TradingAutoResearchProgram
from .store import (
    append_research_turn,
    append_result,
    ensure_research_log,
    load_champion,
    load_family_registry,
    load_research_turns,
    load_results,
    publish_family_champion,
    save_champion,
    save_family_registry,
    snapshot_strategy,
    validate_research_turn,
    write_results_tsv,
)


LEGACY_FAMILY_EXECUTION_MODES = {
    "cross_sectional": "cross_sectional",
    "btc_time_series": "time_series",
    "relative_value": "pair_trade",
}
TRACKED_EXECUTION_MODES = ("cross_sectional", "time_series", "pair_trade")


def build_trading_context(program: TradingAutoResearchProgram) -> dict[str, Path]:
    context = build_context(program)
    summary_path = save_context_summary(context, program)
    return {"context_summary": summary_path}


def _current_strategy_path() -> Path:
    return Path(__file__).with_name("strategy.py")


def _repo_family_champion_dir() -> Path:
    return Path(__file__).with_name("family_champions")


def _record_execution_mode(record: dict[str, object]) -> str:
    explicit = str(record.get("execution_mode", "")).strip()
    if explicit:
        return explicit
    return LEGACY_FAMILY_EXECUTION_MODES.get(str(record.get("family", "")), "time_series")


def _record_family_stage(record: dict[str, object]) -> str:
    explicit = str(record.get("family_stage", "")).strip()
    if explicit:
        return explicit
    if str(record.get("family", "")) in LEGACY_FAMILY_EXECUTION_MODES:
        return "formal"
    return "candidate"


def _record_strategy_snapshot(program: TradingAutoResearchProgram, record: dict[str, object]) -> Path:
    explicit = str(record.get("strategy_snapshot", "")).strip()
    if explicit:
        return Path(explicit)
    return program.run_dir / "runs" / str(record["run_id"]) / f"{record['run_id']}_strategy.py"


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


def _champion_entry(
    record: dict[str, object],
    strategy_snapshot: Path,
    *,
    family_stage: str,
    execution_mode: str,
) -> dict[str, object]:
    return {
        "run_id": record["run_id"],
        "primary_score": record["primary_score"],
        "family": record["family"],
        "execution_mode": execution_mode,
        "family_stage": family_stage,
        "strategy_id": record["strategy_id"],
        "summary": record["summary"],
        "strategy_snapshot": str(strategy_snapshot),
    }


def _tracked_families(
    records: list[dict[str, object]],
    turns: list[dict[str, object]],
    champions: dict[str, object],
    registry: dict[str, object],
) -> list[str]:
    names: list[str] = []
    for family in dict(registry.get("families", {})):
        if family not in names:
            names.append(family)
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


def _tracked_execution_modes(
    records: list[dict[str, object]],
    turns: list[dict[str, object]],
    registry: dict[str, object],
) -> list[str]:
    names = list(TRACKED_EXECUTION_MODES)
    for family in dict(registry.get("families", {})).values():
        mode = str(family.get("execution_mode", "")).strip()
        if mode and mode not in names:
            names.append(mode)
    for record in records:
        mode = _record_execution_mode(record)
        if mode not in names:
            names.append(mode)
    for turn in turns:
        mode = str(turn.get("execution_mode", "")).strip()
        if mode and mode not in names:
            names.append(mode)
    return names


def _is_high_quality_turn(turn: dict[str, object]) -> bool:
    return (
        str(turn.get("info_gain", "")) in {"high", "medium"}
        and not str(turn.get("duplicate_of_run_id", ""))
        and bool(str(turn.get("conclusion", "")).strip())
        and bool(str(turn.get("next_best_axis", "")).strip())
    )


def build_trading_family_registry(program: TradingAutoResearchProgram) -> dict[str, object]:
    records = load_results(program.run_dir / "experiments.jsonl")
    turns = load_research_turns(program.research_log_path)
    existing = load_family_registry(program.family_registry_path)
    names = _tracked_families(records, turns, empty_champion_payload(), existing)
    families: dict[str, dict[str, object]] = {}

    for family in names:
        family_records = [record for record in records if str(record.get("family")) == family]
        family_turns = [turn for turn in turns if str(turn.get("family")) == family]
        latest_record = family_records[-1] if family_records else {}
        latest_turn = family_turns[-1] if family_turns else {}
        existing_entry = dict(existing.get("families", {})).get(family, {})

        execution_mode = (
            str(latest_turn.get("execution_mode", "")).strip()
            or str(existing_entry.get("execution_mode", "")).strip()
            or (_record_execution_mode(latest_record) if latest_record else "")
            or LEGACY_FAMILY_EXECUTION_MODES.get(family, "time_series")
        )
        explicit_stage = (
            str(latest_turn.get("family_stage", "")).strip()
            or str(existing_entry.get("stage", "")).strip()
            or (_record_family_stage(latest_record) if latest_record else "")
            or ("formal" if family in LEGACY_FAMILY_EXECUTION_MODES else "candidate")
        )
        non_duplicate_turns = sum(1 for turn in family_turns if not str(turn.get("duplicate_of_run_id", "")).strip())
        keep_runs = sum(1 for record in family_records if record.get("status") == "keep")
        info_gain_count = sum(1 for turn in family_turns if str(turn.get("info_gain", "")) != "low")
        stage = explicit_stage
        if stage != "formal" and non_duplicate_turns >= 2 and keep_runs >= 1 and info_gain_count >= 1:
            stage = "formal"

        best_record = _best_record(records, family=family)
        if stage == "formal" and best_record is not None:
            family_champion_run_id = str(best_record["run_id"])
        else:
            family_champion_run_id = str(existing_entry.get("family_champion_run_id", ""))

        formalized_at = str(existing_entry.get("formalized_at", ""))
        if stage == "formal" and not formalized_at:
            formalized_at = str(latest_turn.get("timestamp", "")) or (
                datetime.now(timezone.utc).isoformat() if family not in LEGACY_FAMILY_EXECUTION_MODES else ""
            )

        families[family] = {
            "family": family,
            "execution_mode": execution_mode,
            "stage": stage,
            "created_run_id": str(family_records[0]["run_id"]) if family_records else str(latest_turn.get("run_id", "")),
            "latest_run_id": str(family_records[-1]["run_id"]) if family_records else str(latest_turn.get("run_id", "")),
            "mechanism_tag": str(latest_turn.get("mechanism_tag", existing_entry.get("mechanism_tag", ""))),
            "mechanism_summary": str(
                latest_turn.get("mechanism_summary", existing_entry.get("mechanism_summary", ""))
            ),
            "differentiation_note": str(
                latest_turn.get("differentiation_note", existing_entry.get("differentiation_note", ""))
            ),
            "non_duplicate_turns": non_duplicate_turns,
            "keep_runs": keep_runs,
            "info_gain_count": info_gain_count,
            "formalized_at": formalized_at,
            "family_champion_run_id": family_champion_run_id,
        }

    return {"families": families}


def update_trading_family_registry(program: TradingAutoResearchProgram) -> Path:
    payload = build_trading_family_registry(program)
    return save_family_registry(program.family_registry_path, payload)


def show_trading_family_registry(program: TradingAutoResearchProgram) -> Path:
    if not program.family_registry_path.exists():
        return update_trading_family_registry(program)
    return program.family_registry_path


def empty_champion_payload() -> dict[str, object]:
    return {"families": {}, "global": None}


def _rebuild_champions(program: TradingAutoResearchProgram) -> Path:
    records = load_results(program.run_dir / "experiments.jsonl")
    registry = build_trading_family_registry(program)
    champions: dict[str, object] = {"families": {}, "global": None}

    for family, entry in dict(registry.get("families", {})).items():
        if str(entry.get("stage", "")) != "formal":
            continue
        champion_record = _best_record(records, family=family)
        if champion_record is None:
            continue
        snapshot = _record_strategy_snapshot(program, champion_record)
        published_source = publish_family_champion(snapshot, _repo_family_champion_dir(), family)
        champion_entry = _champion_entry(
            champion_record,
            snapshot,
            family_stage=str(entry.get("stage", "formal")),
            execution_mode=str(entry.get("execution_mode", _record_execution_mode(champion_record))),
        )
        champion_entry["published_source"] = str(published_source)
        champions["families"][family] = champion_entry

    global_record = _best_record(records)
    if global_record is not None:
        global_entry = _champion_entry(
            global_record,
            _record_strategy_snapshot(program, global_record),
            family_stage=_record_family_stage(global_record),
            execution_mode=_record_execution_mode(global_record),
        )
        champions["global"] = global_entry

    return save_champion(program.run_dir / "champions.json", champions)


def evaluate_and_record(program: TradingAutoResearchProgram) -> dict[str, Path]:
    run_dir = program.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluation = evaluate_current_strategy(program)
    run_id = f"{evaluation.family}_{evaluation.strategy_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    run_output_dir = run_dir / "runs" / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)

    experiments_path = run_dir / "experiments.jsonl"
    existing_records = load_results(experiments_path)
    family_parent = _best_record(existing_records, family=evaluation.family)
    global_parent = _best_record(existing_records)
    family_champion = (
        evaluation.status == "keep"
        and evaluation.family_stage == "formal"
        and evaluation.primary_score > _record_score(family_parent)
    )
    global_champion = evaluation.status == "keep" and evaluation.primary_score > _record_score(global_parent)
    champion = _active_champion_flag(
        program,
        family_champion=family_champion,
        global_champion=global_champion,
    )
    parent_record = global_parent if program.champion_family_mode == "global" else family_parent

    strategy_snapshot = snapshot_strategy(_current_strategy_path(), run_output_dir, run_id)
    record = {
        "run_id": run_id,
        "parent_run_id": _record_run_id(parent_record),
        "family_parent_run_id": _record_run_id(family_parent),
        "global_parent_run_id": _record_run_id(global_parent),
        "family": evaluation.family,
        "execution_mode": evaluation.execution_mode,
        "family_stage": evaluation.family_stage,
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
        "strategy_snapshot": str(strategy_snapshot),
    }
    append_result(experiments_path, record)
    results_tsv = write_results_tsv(run_dir / "results.tsv", [*existing_records, record])

    summary_path = run_output_dir / "summary.json"
    summary_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    family_registry_path = update_trading_family_registry(program)
    champions_path = _rebuild_champions(program)

    return {
        "summary": summary_path,
        "strategy_snapshot": strategy_snapshot,
        "experiments": experiments_path,
        "results_tsv": results_tsv,
        "champions": champions_path,
        "families": family_registry_path,
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
    records = load_results(program.run_dir / "experiments.jsonl")
    known_runs = {str(record["run_id"]): record for record in records}
    if note["run_id"] not in known_runs:
        raise FileNotFoundError(f"Unknown run_id: {note['run_id']}")
    baseline_run_id = str(note["baseline_run_id"])
    if baseline_run_id and baseline_run_id not in known_runs:
        raise FileNotFoundError(f"Unknown baseline_run_id: {baseline_run_id}")

    run_record = known_runs[note["run_id"]]
    if str(note["family"]) != str(run_record["family"]):
        raise ValueError("Research turn family must match the recorded run family")
    if str(note["execution_mode"]) != _record_execution_mode(run_record):
        raise ValueError("Research turn execution_mode must match the recorded run execution_mode")

    path = append_research_turn(program.research_log_path, note)
    update_trading_family_registry(program)
    _rebuild_champions(program)
    return path


def show_trading_research_log(program: TradingAutoResearchProgram) -> Path:
    return ensure_research_log(program.research_log_path)


def _family_metrics(
    family: str,
    *,
    records: list[dict[str, object]],
    turns: list[dict[str, object]],
    champions: dict[str, object],
    registry: dict[str, object],
) -> dict[str, object]:
    family_records = [record for record in records if record.get("family") == family]
    family_turns = [turn for turn in turns if turn.get("family") == family]
    registry_entry = dict(registry.get("families", {})).get(family, {})
    champion = dict(champions.get("families", {})).get(family)
    duplicate_turns = sum(1 for turn in family_turns if str(turn.get("duplicate_of_run_id", "")))
    high_quality_turns = sum(1 for turn in family_turns if _is_high_quality_turn(turn))
    info_gain_count = sum(1 for turn in family_turns if str(turn.get("info_gain", "")) != "low")
    return {
        "execution_mode": str(registry_entry.get("execution_mode", "")) or (
            _record_execution_mode(family_records[-1]) if family_records else LEGACY_FAMILY_EXECUTION_MODES.get(family, "time_series")
        ),
        "stage": str(registry_entry.get("stage", "candidate")),
        "experiments": len(family_records),
        "research_turns": len(family_turns),
        "keep_runs": sum(1 for record in family_records if record.get("status") == "keep"),
        "duplicate_turns": duplicate_turns,
        "high_quality_turns": high_quality_turns,
        "info_gain_count": info_gain_count,
        "has_family_champion": champion is not None,
        "family_champion_run_id": str(champion["run_id"]) if champion else "",
        "family_champion_score": float(champion["primary_score"]) if champion else None,
        "last_run_id": str(family_records[-1]["run_id"]) if family_records else "",
    }


def _execution_mode_metrics(
    execution_mode: str,
    *,
    family_rows: dict[str, dict[str, object]],
) -> dict[str, object]:
    matching = [row for row in family_rows.values() if str(row.get("execution_mode")) == execution_mode]
    return {
        "families": len(matching),
        "formal_families": sum(1 for row in matching if row.get("stage") == "formal"),
        "candidate_families": sum(1 for row in matching if row.get("stage") == "candidate"),
        "experiments": sum(int(row.get("experiments", 0)) for row in matching),
        "research_turns": sum(int(row.get("research_turns", 0)) for row in matching),
        "champions": sum(1 for row in matching if row.get("has_family_champion")),
    }


def _weakness_key(metrics: dict[str, object]) -> tuple[float, int, int]:
    champion_score = float(metrics["family_champion_score"]) if metrics["family_champion_score"] is not None else -inf
    return (
        champion_score,
        int(metrics["research_turns"]),
        int(metrics["experiments"]),
    )


def _recommended_next_turn(
    *,
    family_rows: dict[str, dict[str, object]],
    execution_mode_rows: dict[str, dict[str, object]],
) -> dict[str, str]:
    candidate_rows = [
        (family, row)
        for family, row in family_rows.items()
        if row.get("stage") == "candidate" and int(row.get("info_gain_count", 0)) > 0
    ]
    if candidate_rows:
        family, row = max(
            candidate_rows,
            key=lambda item: (
                int(item[1].get("info_gain_count", 0)),
                int(item[1].get("keep_runs", 0)),
                int(item[1].get("research_turns", 0)),
            ),
        )
        return {
            "action": "deepen_candidate_family",
            "turn_mode": "explore",
            "family": family,
            "execution_mode": str(row.get("execution_mode", "")),
            "reason": "candidate family already shows non-trivial information gain and should be validated before creating another family",
        }

    sparse_modes = [
        (mode, row)
        for mode, row in execution_mode_rows.items()
        if int(row.get("formal_families", 0)) == 0
    ]
    if sparse_modes:
        mode, _ = min(
            sparse_modes,
            key=lambda item: (
                int(item[1].get("formal_families", 0)),
                int(item[1].get("families", 0)),
                int(item[1].get("research_turns", 0)),
            ),
        )
        return {
            "action": "explore_new_family",
            "turn_mode": "explore",
            "family": "",
            "execution_mode": mode,
            "reason": "this execution mode has no formal family coverage yet",
        }

    formal_rows = [
        (family, row)
        for family, row in family_rows.items()
        if row.get("stage") == "formal"
    ]
    if formal_rows:
        family, row = min(formal_rows, key=lambda item: _weakness_key(item[1]))
        return {
            "action": "converge_family",
            "turn_mode": "converge",
            "family": family,
            "execution_mode": str(row.get("execution_mode", "")),
            "reason": "this formal family has the weakest champion quality among the current formal families",
        }

    default_mode = TRACKED_EXECUTION_MODES[0]
    return {
        "action": "explore_new_family",
        "turn_mode": "explore",
        "family": "",
        "execution_mode": default_mode,
        "reason": "no formal family exists yet, so the next turn should open a new family",
    }


def build_trading_research_scorecard(program: TradingAutoResearchProgram) -> dict[str, object]:
    records = load_results(program.run_dir / "experiments.jsonl")
    turns = load_research_turns(program.research_log_path)
    registry = build_trading_family_registry(program)
    champions = load_champion(program.run_dir / "champions.json")
    families = _tracked_families(records, turns, champions, registry)
    family_rows = {
        family: _family_metrics(family, records=records, turns=turns, champions=champions, registry=registry)
        for family in families
    }
    execution_modes = _tracked_execution_modes(records, turns, registry)
    execution_mode_rows = {
        mode: _execution_mode_metrics(mode, family_rows=family_rows)
        for mode in execution_modes
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
        and bool(str(turn.get("mechanism_summary", "")).strip())
        and bool(str(turn.get("differentiation_note", "")).strip())
    )
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
            "formal_family_count": sum(1 for row in family_rows.values() if row.get("stage") == "formal"),
            "candidate_family_count": sum(1 for row in family_rows.values() if row.get("stage") == "candidate"),
        },
        "recommended_next_turn": _recommended_next_turn(
            family_rows=family_rows,
            execution_mode_rows=execution_mode_rows,
        ),
        "families": family_rows,
        "execution_modes": execution_mode_rows,
    }


def update_trading_research_scorecard(program: TradingAutoResearchProgram) -> Path:
    update_trading_family_registry(program)
    _rebuild_champions(program)
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
