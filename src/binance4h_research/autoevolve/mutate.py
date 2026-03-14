from __future__ import annotations

from copy import deepcopy
import random

from .context import ResearchContext
from .program import AutoEvolveProgram
from .spec import FilterSpec, PortfolioSpec, StrategySpec


def _seed_specs(family: str, context: ResearchContext) -> list[StrategySpec]:
    if family == "cross_sectional":
        return [
            StrategySpec(family=family, model="momentum", params={"lookback_days": 60, "skip_days": 0, "top_n": 50, "rank_mode": "raw"}),
            StrategySpec(family=family, model="momentum", params={"lookback_days": 120, "skip_days": 0, "top_n": 50, "rank_mode": "raw"}),
        ]
    if family == "btc_time_series":
        return [
            StrategySpec(family=family, model="momentum", params={"lookback_days": 60}, portfolio=PortfolioSpec(rebalance="1d", direction_mode="long_short")),
            StrategySpec(family=family, model="breakout", params={"lookback_days": 60}, portfolio=PortfolioSpec(rebalance="1d", direction_mode="long_short")),
        ]
    pair = context.pair_pool[0] if context.pair_pool else ("ETHUSDT", "SOLUSDT")
    return [
        StrategySpec(family=family, model="zscore_mean_revert", params={"left": pair[0], "right": pair[1], "lookback_days": 60, "z_entry": 1.0}),
        StrategySpec(family=family, model="spread_momentum", params={"left": pair[0], "right": pair[1], "lookback_days": 60}),
    ]


def _mutate_cross_sectional(spec: StrategySpec, program: AutoEvolveProgram, rng: random.Random) -> StrategySpec:
    mutated = deepcopy(spec)
    choice = rng.choice(["lookback", "top_n", "rebalance", "filter"])
    if choice == "lookback":
        mutated.params["lookback_days"] = rng.choice(program.search.cross_sectional_lookbacks)
    elif choice == "top_n":
        mutated.params["top_n"] = rng.choice(program.search.cross_sectional_top_n)
    elif choice == "rebalance":
        mutated.portfolio.rebalance = rng.choice(program.search.cross_sectional_rebalance)
    else:
        mutated.filters.funding_cap = rng.choice(program.search.funding_caps)
        mutated.filters.regime = rng.choice(program.search.regime_filters)
    return mutated


def _mutate_btc_ts(spec: StrategySpec, program: AutoEvolveProgram, rng: random.Random) -> StrategySpec:
    mutated = deepcopy(spec)
    choice = rng.choice(["model", "lookback", "rebalance", "direction", "filter"])
    if choice == "model":
        mutated.model = rng.choice(program.search.btc_ts_models)
    elif choice == "lookback":
        mutated.params["lookback_days"] = rng.choice(program.search.btc_ts_lookbacks)
    elif choice == "rebalance":
        mutated.portfolio.rebalance = rng.choice(program.search.cross_sectional_rebalance)
    elif choice == "direction":
        mutated.portfolio.direction_mode = rng.choice(["long_short", "long_only"])
    else:
        mutated.filters.regime = rng.choice(program.search.regime_filters)
        mutated.filters.volatility_cap = rng.choice(program.search.vol_caps)
    return mutated


def _mutate_relative_value(spec: StrategySpec, context: ResearchContext, program: AutoEvolveProgram, rng: random.Random) -> StrategySpec:
    mutated = deepcopy(spec)
    choice = rng.choice(["pair", "lookback", "model", "filter"])
    if choice == "pair" and context.pair_pool:
        left, right = rng.choice(context.pair_pool)
        mutated.params["left"] = left
        mutated.params["right"] = right
    elif choice == "lookback":
        mutated.params["lookback_days"] = rng.choice(program.search.pair_lookbacks)
    elif choice == "model":
        mutated.model = rng.choice(["zscore_mean_revert", "spread_momentum"])
    else:
        mutated.filters.regime = rng.choice(program.search.regime_filters)
    return mutated


def propose_specs(
    family: str,
    context: ResearchContext,
    program: AutoEvolveProgram,
    existing_hashes: set[str],
    champion_spec: StrategySpec | None,
    batch_size: int,
) -> list[tuple[StrategySpec, str | None]]:
    rng = random.Random(program.random_seed + len(existing_hashes))
    proposals: list[tuple[StrategySpec, str | None]] = []
    seeds = _seed_specs(family, context)
    while len(proposals) < batch_size:
        if champion_spec is None and seeds:
            spec = seeds.pop(0)
            parent = None
        else:
            base = champion_spec or rng.choice([item[0] for item in proposals] or _seed_specs(family, context))
            parent = base.spec_hash()
            if family == "cross_sectional":
                spec = _mutate_cross_sectional(base, program, rng)
            elif family == "btc_time_series":
                spec = _mutate_btc_ts(base, program, rng)
            else:
                spec = _mutate_relative_value(base, context, program, rng)
        if spec.spec_hash() in existing_hashes:
            continue
        existing_hashes.add(spec.spec_hash())
        proposals.append((spec, parent))
    return proposals
