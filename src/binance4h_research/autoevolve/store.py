from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import pandas as pd

from .evaluate import CandidateEvaluation


def _json_dumps(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def load_experiments(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_experiment(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_json_dumps(record) + "\n")


def save_results_tsv(path: Path, records: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        pd.DataFrame().to_csv(path, sep="\t", index=False)
        return path
    rows = []
    for record in records:
        rows.append(
            {
                "run_id": record["run_id"],
                "family": record["family"],
                "status": record["status"],
                "primary_score": record["primary_score"],
                "test_sharpe": record["splits"]["test"]["sharpe"],
                "test_net_total_return": record["splits"]["test"]["net_total_return"],
                "reason_tags": ",".join(record["reason_tags"]),
                "champion": record.get("champion", False),
            }
        )
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def load_champions(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_champions(path: Path, champions: dict[str, dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(champions, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def evaluation_record(run_id: str, parent_id: str | None, evaluation: CandidateEvaluation) -> dict[str, object]:
    return {
        "run_id": run_id,
        "parent_id": parent_id,
        "family": evaluation.spec.family,
        "spec_hash": evaluation.spec_hash,
        "spec": evaluation.spec.to_dict(),
        "primary_score": evaluation.primary_score,
        "summary": evaluation.summary,
        "splits": evaluation.splits,
        "walk_forward": evaluation.walk_forward,
        "status": evaluation.status,
        "reason_tags": evaluation.reason_tags,
    }
