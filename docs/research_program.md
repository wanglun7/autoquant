# Research Program

## Goal

Find Binance perpetual strategies that survive fixed costs, fixed sample splits, and fixed stress tests.

## Mutation Surface

The agent may mutate only the structured strategy spec:

- family
- model
- parameters
- filters
- portfolio mapping

The agent may not mutate:

- raw market data
- cost model
- funding model
- sample splits
- primary score
- keep / reject / champion rules

## Families

- `cross_sectional`
- `btc_time_series`
- `relative_value`

## Filters

- `btc_regime`
- `funding_crowding`
- `volatility_cap`
- `liquidity_bucket`

Filters may gate or downweight a strategy, but are not standalone families.

## Keep / Reject Rules

- `keep`
  - test net return > 0
  - test drawdown within limit
  - score improves enough to justify complexity
- `reject`
  - fails hard constraints
  - or becomes more complex without enough gain
- `champion`
  - best kept candidate inside a family

## Research Discipline

- small mutations only
- preserve trajectory history
- do not optimize the evaluator
- prefer simple strategies over fragile improvements
