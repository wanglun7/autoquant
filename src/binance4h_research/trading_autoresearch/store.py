from __future__ import annotations

from pathlib import Path
import json
import shutil

import pandas as pd


def load_results(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_result(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_results_tsv(path: Path, records: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        pd.DataFrame().to_csv(path, sep="\t", index=False)
        return path
    rows = []
    for record in records:
        summary = record["summary"]
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
                "champion": record.get("champion", False),
            }
        )
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def load_champion(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_champion(path: Path, champion: dict[str, dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(champion, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def snapshot_strategy(source: Path, target_dir: Path, run_id: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{run_id}_strategy.py"
    shutil.copy2(source, target)
    return target
