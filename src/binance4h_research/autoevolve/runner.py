from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json

from .context import build_research_context, save_context_summary
from .evaluate import evaluate_spec
from .mutate import propose_specs
from .program import AutoEvolveProgram
from .spec import StrategySpec
from .store import append_experiment, evaluation_record, load_champions, load_experiments, save_champions, save_results_tsv


def build_research_context_artifacts(program: AutoEvolveProgram) -> dict[str, Path]:
    context = build_research_context(program)
    summary_path = save_context_summary(context, program.context_dir)
    program_path = program.context_dir / "program_snapshot.json"
    program_path.write_text(json.dumps(asdict(program), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return {"context_summary": summary_path, "program_snapshot": program_path}


def _record_to_spec(record: dict[str, object]) -> StrategySpec:
    payload = record["spec"]
    return StrategySpec(
        family=payload["family"],
        model=payload["model"],
        params=payload.get("params", {}),
        filters=StrategySpec.__dataclass_fields__["filters"].type(**payload.get("filters", {})),  # type: ignore[attr-defined]
        portfolio=StrategySpec.__dataclass_fields__["portfolio"].type(**payload.get("portfolio", {})),  # type: ignore[attr-defined]
    )


def _champion_spec_for_family(champions: dict[str, dict[str, object]], family: str) -> StrategySpec | None:
    champion = champions.get(family)
    if not champion:
        return None
    payload = champion["spec"]
    from .spec import FilterSpec, PortfolioSpec

    return StrategySpec(
        family=payload["family"],
        model=payload["model"],
        params=payload.get("params", {}),
        filters=FilterSpec(**payload.get("filters", {})),
        portfolio=PortfolioSpec(**payload.get("portfolio", {})),
    )


def run_evolution_batch(program: AutoEvolveProgram, family_scopes: list[str] | None = None, batch_size: int | None = None) -> dict[str, Path]:
    context = build_research_context(program)
    run_dir = program.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    experiments_path = run_dir / "experiments.jsonl"
    champions_path = run_dir / "champions.json"
    records = load_experiments(experiments_path)
    champions = load_champions(champions_path)
    existing_hashes = {record["spec_hash"] for record in records}
    scopes = family_scopes or program.family_scopes
    per_family = max(1, (batch_size or program.batch_size) // max(1, len(scopes)))

    for family in scopes:
        champion_spec = _champion_spec_for_family(champions, family)
        proposals = propose_specs(family, context, program, existing_hashes, champion_spec, per_family)
        for spec, parent_id in proposals:
            evaluation = evaluate_spec(spec, context, program)
            run_id = f"{family}_{spec.spec_hash()}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
            record = evaluation_record(run_id, parent_id, evaluation)
            current_champion = champions.get(family)
            champion_score = float(current_champion["primary_score"]) if current_champion else float("-inf")
            record["champion"] = evaluation.status == "keep" and evaluation.primary_score > champion_score
            if record["champion"]:
                champions[family] = {
                    "run_id": run_id,
                    "primary_score": evaluation.primary_score,
                    "spec": spec.to_dict(),
                    "summary": evaluation.summary,
                }
            if evaluation.status == "keep" and not record["champion"]:
                record["status"] = "archive"
            append_experiment(experiments_path, record)
            records.append(record)

    save_champions(champions_path, champions)
    results_tsv = save_results_tsv(run_dir / "results.tsv", records)
    batch_summary_path = run_dir / "latest_batch_summary.json"
    batch_summary_path.write_text(
        json.dumps(
            {
                "families": scopes,
                "records_total": len(records),
                "champions": {family: value["run_id"] for family, value in champions.items()},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {"experiments": experiments_path, "champions": champions_path, "results_tsv": results_tsv, "batch_summary": batch_summary_path}


def show_champions(program: AutoEvolveProgram) -> Path:
    champions_path = program.run_dir / "champions.json"
    if not champions_path.exists():
        raise FileNotFoundError(f"Missing champions file: {champions_path}")
    return champions_path


def replay_candidate(program: AutoEvolveProgram, run_id: str) -> Path:
    experiments = load_experiments(program.run_dir / "experiments.jsonl")
    match = next((record for record in experiments if record["run_id"] == run_id), None)
    if match is None:
        raise FileNotFoundError(f"Unknown run_id: {run_id}")
    context = build_research_context(program)
    spec_payload = match["spec"]
    from .spec import FilterSpec, PortfolioSpec

    spec = StrategySpec(
        family=spec_payload["family"],
        model=spec_payload["model"],
        params=spec_payload.get("params", {}),
        filters=FilterSpec(**spec_payload.get("filters", {})),
        portfolio=PortfolioSpec(**spec_payload.get("portfolio", {})),
    )
    evaluation = evaluate_spec(spec, context, program)
    output = program.run_dir / f"replay_{run_id}.json"
    output.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "spec": spec.to_dict(),
                "summary": evaluation.summary,
                "splits": evaluation.splits,
                "walk_forward": evaluation.walk_forward,
                "status": evaluation.status,
                "reason_tags": evaluation.reason_tags,
                "primary_score": evaluation.primary_score,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return output
