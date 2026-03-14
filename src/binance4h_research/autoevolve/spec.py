from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json


@dataclass(slots=True)
class FilterSpec:
    regime: str | None = None
    funding_cap: float | None = None
    volatility_cap: float | None = None
    liquidity_bucket: str = "all"


@dataclass(slots=True)
class PortfolioSpec:
    rebalance: str = "1d"
    gross: float = 1.0
    long_quantile: float = 0.2
    short_quantile: float = 0.2
    weight_mode: str = "equal"
    direction_mode: str = "long_short"


@dataclass(slots=True)
class StrategySpec:
    family: str
    model: str
    params: dict[str, object] = field(default_factory=dict)
    filters: FilterSpec = field(default_factory=FilterSpec)
    portfolio: PortfolioSpec = field(default_factory=PortfolioSpec)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def spec_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def complexity(self) -> int:
        score = 1
        score += len([value for value in self.params.values() if value is not None])
        score += sum(value is not None and value != "all" for value in asdict(self.filters).values())
        score += int(self.portfolio.rebalance != "1d")
        score += int(self.portfolio.weight_mode != "equal")
        return score
