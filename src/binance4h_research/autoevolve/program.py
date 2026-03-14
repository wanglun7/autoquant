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
    max_drawdown_limit: float = 0.60
    min_test_net_return: float = 0.0
    min_test_sharpe: float = -0.25
    complexity_penalty: float = 0.01


@dataclass(slots=True)
class SearchSpaceConfig:
    cross_sectional_lookbacks: list[int] = field(default_factory=lambda: [60, 120, 180])
    cross_sectional_top_n: list[int] = field(default_factory=lambda: [30, 50, 80])
    cross_sectional_rebalance: list[str] = field(default_factory=lambda: ["4h", "1d", "1w"])
    btc_ts_models: list[str] = field(default_factory=lambda: ["momentum", "breakout", "mean_reversion", "vol_trend"])
    btc_ts_lookbacks: list[int] = field(default_factory=lambda: [20, 60, 120])
    pair_pool_size: int = 8
    pair_lookbacks: list[int] = field(default_factory=lambda: [20, 60, 120])
    funding_caps: list[float | None] = field(default_factory=lambda: [None, 0.0005, 0.001])
    regime_filters: list[str | None] = field(default_factory=lambda: [None, "btc_positive_trend"])
    vol_caps: list[float | None] = field(default_factory=lambda: [None, 0.05, 0.08])


@dataclass(slots=True)
class AutoEvolveProgram:
    name: str = "autoevolve_v1"
    data_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed/autoevolve")
    results_dir: Path = Path("results/autoevolve")
    costs: CostConfig = field(default_factory=CostConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    search: SearchSpaceConfig = field(default_factory=SearchSpaceConfig)
    family_scopes: list[str] = field(default_factory=lambda: ["cross_sectional", "btc_time_series", "relative_value"])
    batch_size: int = 6
    random_seed: int = 7
    bar_hours: int = 4

    @property
    def context_dir(self) -> Path:
        return self.processed_dir / self.name

    @property
    def run_dir(self) -> Path:
        return self.results_dir / self.name


def _from_dict(cls: type[Any], payload: dict[str, Any]) -> Any:
    return cls(**payload)


def load_autoevolve_program(path: str | Path) -> AutoEvolveProgram:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return AutoEvolveProgram(
        name=raw.get("name", "autoevolve_v1"),
        data_dir=Path(raw.get("data_dir", "data/raw")),
        processed_dir=Path(raw.get("processed_dir", "data/processed/autoevolve")),
        results_dir=Path(raw.get("results_dir", "results/autoevolve")),
        costs=_from_dict(CostConfig, raw.get("costs", {})),
        split=_from_dict(SplitConfig, raw.get("split", {})),
        constraints=_from_dict(ConstraintConfig, raw.get("constraints", {})),
        search=_from_dict(SearchSpaceConfig, raw.get("search", {})),
        family_scopes=list(raw.get("family_scopes", ["cross_sectional", "btc_time_series", "relative_value"])),
        batch_size=int(raw.get("batch_size", 6)),
        random_seed=int(raw.get("random_seed", 7)),
        bar_hours=int(raw.get("bar_hours", 4)),
    )
