from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class UniverseConfig:
    top_n: int = 40
    liquidity_lookback_bars: int = 180
    min_history_bars: int = 126
    quote_volume_field: str = "quote_volume"


@dataclass(slots=True)
class SignalConfig:
    kind: str = "cross_sectional_momentum"
    lookback_bars: int = 42


@dataclass(slots=True)
class PortfolioConfig:
    long_quantile: float = 0.2
    short_quantile: float = 0.2
    gross_exposure: float = 1.0


@dataclass(slots=True)
class BacktestConfig:
    bar_hours: int = 4
    rebalance_every_bars: int = 1


@dataclass(slots=True)
class CostConfig:
    fee_bps: float = 4.0
    slippage_bps: float = 2.0


@dataclass(slots=True)
class ReportConfig:
    pressure_fee_multipliers: list[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    pressure_slippage_multipliers: list[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])


@dataclass(slots=True)
class OutputConfig:
    write_weights: bool = True
    write_pnl: bool = True
    write_universe: bool = True
    write_pressure: bool = True


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    data_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    results_dir: Path = Path("results")
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)

    @property
    def experiment_dir(self) -> Path:
        return self.results_dir / self.name


def _dataclass_from_dict(cls: type[Any], payload: dict[str, Any]) -> Any:
    return cls(**payload)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    return ExperimentConfig(
        name=raw["name"],
        data_dir=Path(raw.get("data_dir", "data/raw")),
        processed_dir=Path(raw.get("processed_dir", "data/processed")),
        results_dir=Path(raw.get("results_dir", "results")),
        universe=_dataclass_from_dict(UniverseConfig, raw.get("universe", {})),
        signal=_dataclass_from_dict(SignalConfig, raw.get("signal", {})),
        portfolio=_dataclass_from_dict(PortfolioConfig, raw.get("portfolio", {})),
        backtest=_dataclass_from_dict(BacktestConfig, raw.get("backtest", {})),
        costs=_dataclass_from_dict(CostConfig, raw.get("costs", {})),
        report=_dataclass_from_dict(ReportConfig, raw.get("report", {})),
        outputs=_dataclass_from_dict(OutputConfig, raw.get("outputs", {})),
    )
