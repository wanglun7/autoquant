from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import CostConfig, ReportConfig


@dataclass(slots=True)
class PaperApproxUniverseConfig:
    liquidity_lookback_bars: int = 360
    weight_lookback_bars: int = 180
    top_n: int = 50
    min_history_bars: int = 252
    exclude_bases: list[str] = field(default_factory=list)
    exclude_symbols: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PaperApproxPaperConfig:
    rebalance_weekday: int = 0
    rebalance_hour_utc: int = 0
    ltw_lookback_weeks: int = 3
    grobys_formation_days: int = 30
    grobys_skip_days: int = 1
    ficura_formation_weeks: list[int] = field(default_factory=lambda: [1, 2, 4, 12, 26])
    ficura_large_liquid_fraction: float = 0.3
    ficura_total_universe: int = 60


@dataclass(slots=True)
class PaperApproxPortfolioConfig:
    gross_exposure: float = 1.0


@dataclass(slots=True)
class PaperApproxConfig:
    name: str
    paper_id: str
    data_dir: Path = Path("data/raw")
    results_dir: Path = Path("results/paper_approx")
    universe: PaperApproxUniverseConfig = field(default_factory=PaperApproxUniverseConfig)
    paper: PaperApproxPaperConfig = field(default_factory=PaperApproxPaperConfig)
    portfolio: PaperApproxPortfolioConfig = field(default_factory=PaperApproxPortfolioConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    bar_hours: int = 4

    @property
    def experiment_dir(self) -> Path:
        return self.results_dir / self.name


def _from_dict(cls: type[Any], payload: dict[str, Any]) -> Any:
    return cls(**payload)


def load_paper_approx_config(path: str | Path) -> PaperApproxConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return PaperApproxConfig(
        name=raw["name"],
        paper_id=raw["paper_id"],
        data_dir=Path(raw.get("data_dir", "data/raw")),
        results_dir=Path(raw.get("results_dir", "results/paper_approx")),
        universe=_from_dict(PaperApproxUniverseConfig, raw.get("universe", {})),
        paper=_from_dict(PaperApproxPaperConfig, raw.get("paper", {})),
        portfolio=_from_dict(PaperApproxPortfolioConfig, raw.get("portfolio", {})),
        costs=_from_dict(CostConfig, raw.get("costs", {})),
        report=_from_dict(ReportConfig, raw.get("report", {})),
        bar_hours=int(raw.get("bar_hours", 4)),
    )
