from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..config import CostConfig


@dataclass(slots=True)
class SplitConfig:
    train_frac: float = 0.5
    validation_frac: float = 0.25
    test_frac: float = 0.25


@dataclass(slots=True)
class ConstraintConfig:
    min_test_net_return: float = 0.0
    min_test_sharpe: float = -0.25
    max_drawdown_limit: float = 0.60
    require_positive_under_2x_cost: bool = True


@dataclass(slots=True)
class TradingAutoResearchProgram:
    name: str = "trading_autoresearch_v1"
    data_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed/trading_autoresearch")
    results_dir: Path = Path("results/trading_autoresearch")
    costs: CostConfig = field(default_factory=CostConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    bar_hours: int = 4
    pair_pool_size: int = 8
    champion_family_mode: str = "by_family"

    @property
    def context_dir(self) -> Path:
        return self.processed_dir / self.name

    @property
    def run_dir(self) -> Path:
        return self.results_dir / self.name

    @property
    def research_log_path(self) -> Path:
        return self.run_dir / "research_log.jsonl"

    @property
    def research_scorecard_path(self) -> Path:
        return self.run_dir / "research_scorecard.json"

    @property
    def family_registry_path(self) -> Path:
        return self.run_dir / "families.json"


def _from_dict(cls: type[Any], payload: dict[str, Any]) -> Any:
    return cls(**payload)


def load_trading_autoresearch_program(path: str | Path) -> TradingAutoResearchProgram:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return TradingAutoResearchProgram(
        name=raw.get("name", "trading_autoresearch_v1"),
        data_dir=Path(raw.get("data_dir", "data/raw")),
        processed_dir=Path(raw.get("processed_dir", "data/processed/trading_autoresearch")),
        results_dir=Path(raw.get("results_dir", "results/trading_autoresearch")),
        costs=_from_dict(CostConfig, raw.get("costs", {})),
        split=_from_dict(SplitConfig, raw.get("split", {})),
        constraints=_from_dict(ConstraintConfig, raw.get("constraints", {})),
        bar_hours=int(raw.get("bar_hours", 4)),
        pair_pool_size=int(raw.get("pair_pool_size", 8)),
        champion_family_mode=str(raw.get("champion_family_mode", "by_family")),
    )
