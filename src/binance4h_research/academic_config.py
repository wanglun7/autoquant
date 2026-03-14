from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class AcademicExperimentConfig:
    name: str
    paper_id: str
    raw_data_dir: Path = Path("data/academic/raw")
    processed_dir: Path = Path("data/academic/processed")
    results_dir: Path = Path("results/academic")
    start_date: str = "2014-01-01"
    end_date: str = "2018-12-31"
    weekly_calendar: str = "paper_52w"
    minimum_market_cap_usd: float = 1_000_000.0
    top_n: int = 30
    formation_days: int = 30
    skip_days: int = 1
    large_market_cap_usd: float = 50_000_000.0
    large_volume_usd: float = 5_000_000.0
    price_floor_usd: float = 0.001
    formation_weeks: list[int] = field(default_factory=lambda: [1, 2, 4, 12, 26])
    winsor_return_cap: float = 10.0
    exclude_coin_ids: list[str] = field(default_factory=list)

    @property
    def weekly_panel_path(self) -> Path:
        return self.processed_dir / f"{self.name}_weekly_panel.csv"

    @property
    def experiment_dir(self) -> Path:
        return self.results_dir / self.name


def _config_from_dict(raw: dict[str, Any]) -> AcademicExperimentConfig:
    return AcademicExperimentConfig(
        name=raw["name"],
        paper_id=raw["paper_id"],
        raw_data_dir=Path(raw.get("raw_data_dir", "data/academic/raw")),
        processed_dir=Path(raw.get("processed_dir", "data/academic/processed")),
        results_dir=Path(raw.get("results_dir", "results/academic")),
        start_date=raw.get("start_date", "2014-01-01"),
        end_date=raw.get("end_date", "2018-12-31"),
        weekly_calendar=raw.get("weekly_calendar", "paper_52w"),
        minimum_market_cap_usd=float(raw.get("minimum_market_cap_usd", 1_000_000.0)),
        top_n=int(raw.get("top_n", 30)),
        formation_days=int(raw.get("formation_days", 30)),
        skip_days=int(raw.get("skip_days", 1)),
        large_market_cap_usd=float(raw.get("large_market_cap_usd", 50_000_000.0)),
        large_volume_usd=float(raw.get("large_volume_usd", 5_000_000.0)),
        price_floor_usd=float(raw.get("price_floor_usd", 0.001)),
        formation_weeks=[int(value) for value in raw.get("formation_weeks", [1, 2, 4, 12, 26])],
        winsor_return_cap=float(raw.get("winsor_return_cap", 10.0)),
        exclude_coin_ids=list(raw.get("exclude_coin_ids", [])),
    )


def load_academic_config(path: str | Path) -> AcademicExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return _config_from_dict(raw)
