# Trading Autoresearch Program

## Goal

Find Binance USDT-M perpetual strategies that remain positive after `fee + slippage + funding`, and still survive a `2x` cost stress test.

This program treats the agent as an automated quant researcher. The job is to produce high-quality experiments, avoid repeated low-information turns, explain why an idea worked or failed, and preserve both strategy assets and research knowledge.

## Core Rule

This workflow follows the Karpathy `autoresearch` pattern:

- Human edits this program file.
- The agent reads this file, the latest results, and the current champions.
- The agent may change only `src/binance4h_research/trading_autoresearch/strategy.py`.
- Data preparation, cost model, split logic, and evaluator stay fixed.

## Family Model

`family` is the primary research asset unit. Families are parallel, not nested.

- `family`
  - a stable, semantic strategy family name
  - examples: `cross_sectional`, `btc_time_series`, `trend_pullback_state_machine`
- `execution_mode`
  - the execution shape used by that family
  - current common values: `cross_sectional`, `time_series`, `pair_trade`
- `family_stage`
  - `candidate` or `formal`
  - only formal families get published champion code

Legacy families remain valid:

- `cross_sectional`
- `btc_time_series`
- `relative_value`

They are now just existing formal families, not a privileged parent layer.

## Research Loop

Run this as a hypothesis-driven loop, not a blind parameter search:

1. choose one family or propose a new family
2. restate one objective
3. write one falsifiable hypothesis
4. make one focused code change
5. evaluate on the fixed setup
6. record the conclusion in the research log

If a run is only `keep`, log it but do not preserve it as the active working strategy.

Default to `Explore First` when the user does not specify a direction:

1. read `research_scorecard.json`
2. if a promising candidate family exists, deepen it
3. else if an `execution_mode` has weak coverage, explore a new family there
4. else converge on the weakest formal family

Use first principles before each turn:

- what inputs actually exist in the fixed context
- what alpha mechanism those inputs could express
- what cost, turnover, or drawdown mechanism could break the idea
- why this one change is worth testing now

## Fixed Files

- `src/binance4h_research/trading_autoresearch/prepare_market.py`
- `src/binance4h_research/trading_autoresearch/evaluate.py`
- `src/binance4h_research/trading_autoresearch/runner.py`
- `configs/trading_autoresearch.yaml`

Do not modify these during normal research runs.

## Mutable File

- `src/binance4h_research/trading_autoresearch/strategy.py`

## Published Champion Assets

- Every formal family champion is mirrored into `src/binance4h_research/trading_autoresearch/family_champions/<family>.py`
- Treat these files as published outputs of the research loop, not as the main mutation surface
- Continue editing only `strategy.py` during normal research runs
- When starting a new turn for a formal family, prefer using that family champion as the working baseline
- When starting a new candidate family, prefer the strongest formal family in the same `execution_mode` as the baseline

The agent may invent new signals, filters, ranking formulas, spread logic, and position mapping inside `strategy.py`, as long as it preserves the public interface:

- `build_cross_sectional_weights(context, ...)`
- `build_btc_time_series_weights(context, ...)`
- `build_relative_value_weights(context, ...)`
- `build_weights(context)`

## Evaluation Rules

Every run is evaluated on the same fixed setup:

- local Binance data in `data/raw`
- `fee + slippage + funding`
- fixed train / validation / test split
- fixed walk-forward summary
- fixed `2x` cost stress test

## Keep / Reject / Champion

- `reject`
  - test net return `<= 0`
  - or test Sharpe below threshold
  - or max drawdown exceeds limit
  - or `2x` cost stress turns non-positive
- `keep`
  - passes all hard constraints
- `family_champion`
  - best kept run inside the same formal family
- `global_champion`
  - best kept run across all families

If a family is still `candidate`, it may produce a `keep`, but it does not publish a family champion until it becomes `formal`.

## Family Lifecycle

- `candidate`
  - created when the agent proposes a new family
- `formal`
  - automatically reached after:
    - at least 2 non-duplicate turns
    - at least 1 `keep`
    - at least 1 turn with `info_gain != low`

Candidate families are knowledge assets only. Formal families become strategy assets.

## Complexity Discipline

- Prefer one clear idea per run.
- Do not combine many new concepts at once.
- Prefer changing one family at a time.
- If complexity rises but test performance does not clearly improve, reject it.
- Do not present a parameter tweak as a new family.

## Standard Commands

```bash
PYTHONPATH=src python3 -m binance4h_research build-trading-context --program configs/trading_autoresearch.yaml
PYTHONPATH=src python3 -m binance4h_research run-trading-autoresearch-batch --program configs/trading_autoresearch.yaml
PYTHONPATH=src python3 -m binance4h_research show-trading-champions --program configs/trading_autoresearch.yaml
PYTHONPATH=src python3 -m binance4h_research replay-trading-run --program configs/trading_autoresearch.yaml --run-id <run_id>
PYTHONPATH=src python3 -m binance4h_research show-trading-research-log --program configs/trading_autoresearch.yaml
PYTHONPATH=src python3 -m binance4h_research update-trading-family-registry --program configs/trading_autoresearch.yaml
PYTHONPATH=src python3 -m binance4h_research show-trading-family-registry --program configs/trading_autoresearch.yaml
PYTHONPATH=src python3 -m binance4h_research update-trading-research-scorecard --program configs/trading_autoresearch.yaml
PYTHONPATH=src python3 -m binance4h_research show-trading-research-scorecard --program configs/trading_autoresearch.yaml
```

## Result Files

- `results/trading_autoresearch/<program>/experiments.jsonl`
- `results/trading_autoresearch/<program>/results.tsv`
- `results/trading_autoresearch/<program>/champions.json`
- `results/trading_autoresearch/<program>/families.json`
- `results/trading_autoresearch/<program>/research_log.jsonl`
- `results/trading_autoresearch/<program>/research_scorecard.json`
- `results/trading_autoresearch/<program>/runs/<run_id>/summary.json`
- `results/trading_autoresearch/<program>/champions/global/`
- `src/binance4h_research/trading_autoresearch/family_champions/`
