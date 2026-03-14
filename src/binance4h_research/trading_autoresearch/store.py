from __future__ import annotations

from pathlib import Path
import json
import shutil

import pandas as pd


RESEARCH_TURN_REQUIRED_FIELDS = (
    "timestamp",
    "family",
    "turn_mode",
    "mechanism_tag",
    "objective",
    "hypothesis",
    "planned_change",
    "success_criteria",
    "baseline_run_id",
    "run_id",
    "status",
    "family_champion",
    "global_champion",
    "failure_mode",
    "info_gain",
    "duplicate_of_run_id",
    "family_state_before_turn",
    "comparison",
    "conclusion",
    "next_best_axis",
)
RESEARCH_TURN_REQUIRED_COMPARISON_FIELDS = (
    "test_sharpe_delta",
    "test_net_return_delta",
    "stress_2x_delta",
)


def empty_champions() -> dict[str, object]:
    return {"families": {}, "global": None}


def load_results(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_result(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_research_turns(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ensure_research_log(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")
    return path


def _require_fields(payload: dict[str, object], required: tuple[str, ...], label: str) -> None:
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"Missing {label} fields: {', '.join(missing)}")


def _require_string(payload: dict[str, object], field: str) -> None:
    value = payload[field]
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")


def _require_bool(payload: dict[str, object], field: str) -> None:
    value = payload[field]
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")


def _require_number(payload: dict[str, object], field: str) -> None:
    value = payload[field]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")


def validate_research_turn(note: dict[str, object]) -> dict[str, object]:
    if not isinstance(note, dict):
        raise ValueError("Research turn note must be a JSON object")
    _require_fields(note, RESEARCH_TURN_REQUIRED_FIELDS, "research turn")

    for field in (
        "timestamp",
        "family",
        "turn_mode",
        "mechanism_tag",
        "objective",
        "hypothesis",
        "planned_change",
        "success_criteria",
        "baseline_run_id",
        "run_id",
        "status",
        "failure_mode",
        "info_gain",
        "duplicate_of_run_id",
        "conclusion",
        "next_best_axis",
    ):
        _require_string(note, field)

    for field in ("family_champion", "global_champion"):
        _require_bool(note, field)

    family_state = note["family_state_before_turn"]
    if not isinstance(family_state, dict):
        raise ValueError("family_state_before_turn must be an object")

    comparison = note["comparison"]
    if not isinstance(comparison, dict):
        raise ValueError("comparison must be an object")
    _require_fields(comparison, RESEARCH_TURN_REQUIRED_COMPARISON_FIELDS, "comparison")
    for field in RESEARCH_TURN_REQUIRED_COMPARISON_FIELDS:
        _require_number(comparison, field)

    return note


def append_research_turn(path: Path, note: dict[str, object]) -> Path:
    ensure_research_log(path)
    validate_research_turn(note)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(note, ensure_ascii=False) + "\n")
    return path


def write_results_tsv(path: Path, records: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        pd.DataFrame().to_csv(path, sep="\t", index=False)
        return path
    rows = []
    for record in records:
        splits = record["splits"]
        rows.append(
            {
                "run_id": record["run_id"],
                "parent_run_id": record.get("parent_run_id", ""),
                "family": record["family"],
                "strategy_id": record["strategy_id"],
                "status": record["status"],
                "primary_score": record["primary_score"],
                "test_sharpe": splits["test"]["sharpe"],
                "test_net_total_return": splits["test"]["net_total_return"],
                "reason_tags": ",".join(record["reason_tags"]),
                "family_champion": record.get("family_champion", False),
                "global_champion": record.get("global_champion", False),
                "champion_scope": record.get("champion_scope", "none"),
                "champion": record.get("champion", False),
            }
        )
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def load_champion(path: Path) -> dict[str, object]:
    if not path.exists():
        return empty_champions()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "families" in raw or "global" in raw:
        return {
            "families": dict(raw.get("families", {})),
            "global": raw.get("global"),
        }
    return {"families": dict(raw), "global": None}


def save_champion(path: Path, champion: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "families": dict(champion.get("families", {})),
        "global": champion.get("global"),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def snapshot_strategy(source: Path, target_dir: Path, run_id: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{run_id}_strategy.py"
    shutil.copy2(source, target)
    return target


def publish_family_champion(source: Path, target_dir: Path, family: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{family}.py"
    shutil.copy2(source, target)
    return target
