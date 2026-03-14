# Trading Autoresearch Program

## Goal

Find Binance USDT-M perpetual strategies that remain positive after `fee + slippage + funding`, and still survive a `2x` cost stress test.

This program treats the agent as an automated quant researcher, not as a portfolio manager or execution system. The job is to produce high-quality experiments, avoid low-information repetition, explain why an idea worked or failed, and preserve both strategy assets and research knowledge.

## Research Loop

Run this as a hypothesis-driven loop, not a blind parameter search:

1. choose one family
2. restate one objective
3. write one falsifiable hypothesis
4. make one focused code change
5. evaluate on the fixed setup
6. record the conclusion in the research log

If a run is only `keep`, log it but do not preserve it as the active working strategy.

Default to `Explore First` when the user does not specify a direction:

1. read `research_scorecard.json`
2. choose the weakest family or least-covered family
3. run one explore turn to test a mechanism with high information value
4. after a family produces a real promotion, spend at most a small number of follow-up turns converging on that family

Use first principles before each turn:

- what inputs actually exist in the fixed context
- what alpha mechanism those inputs could express
- what cost, turnover, or drawdown mechanism could break the idea
- why this one change is worth testing now

## Core Rule

This workflow follows the Karpathy `autoresearch` pattern:

- Human edits this program file.
- The agent reads this file, the latest results, and the current champion.
- The agent may change only `src/binance4h_research/trading_autoresearch/strategy.py`.
- Data preparation, cost model, split logic, and evaluator stay fixed.

## Fixed Files

- `src/binance4h_research/trading_autoresearch/prepare_market.py`
- `src/binance4h_research/trading_autoresearch/evaluate.py`
- `src/binance4h_research/trading_autoresearch/runner.py`
- `configs/trading_autoresearch.yaml`

Do not modify these during normal research runs.

## Mutable File

- `src/binance4h_research/trading_autoresearch/strategy.py`

## Published Family Champions

- Every promoted family champion is also mirrored into `src/binance4h_research/trading_autoresearch/family_champions/<family>.py`
- Treat these files as published outputs of the research loop, not as the main mutation surface
- Continue editing only `strategy.py` during normal research runs
- When starting a new turn for a family, prefer using that family's mirrored champion as the working baseline

The agent may invent new signals, filters, ranking formulas, spread logic, and position mapping inside this file, as long as it preserves the public interface:

- `build_cross_sectional_weights(context, ...)`
- `build_btc_time_series_weights(context, ...)`
- `build_relative_value_weights(context, ...)`
- `build_weights(context)`

## Supported Families

- `cross_sectional`
- `btc_time_series`
- `relative_value`

Time series research is restricted to `BTCUSDT`.

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
- `champion`
  - depends on `champion_family_mode`
  - `by_family`: beats the current family champion on primary score
  - `global`: beats the current global champion on primary score

Runs also record two explicit promotion flags:

- `family_champion`
  - beats the current champion inside the same family
- `global_champion`
  - beats the current best kept run across all families

## Complexity Discipline

- Prefer one clear idea per run.
- Do not combine many new concepts at once.
- Prefer changing one family at a time.
- If complexity rises but test performance does not clearly improve, reject it.

## Research Modes

- `explore`
  - default mode when the user does not specify a family
  - prioritizes the weakest family or least-covered family
  - optimizes for information gain, not just immediate promotion
- `converge`
  - used after a family has a viable champion worth refining
  - optimizes for robustness, cost survival, and family champion quality

Treat funding, regime, volatility, and liquidity as internal logic inside `strategy.py`, not as separate evaluators.

## Standard Commands

```bash
PYTHONPATH=src python3 -m binance4h_research build-trading-context --program configs/trading_autoresearch.yaml
PYTHONPATH=src python3 -m binance4h_research run-trading-autoresearch-batch --program configs/trading_autoresearch.yaml
PYTHONPATH=src python3 -m binance4h_research show-trading-champions --program configs/trading_autoresearch.yaml
PYTHONPATH=src python3 -m binance4h_research replay-trading-run --program configs/trading_autoresearch.yaml --run-id <run_id>
PYTHONPATH=src python3 -m binance4h_research show-trading-research-log --program configs/trading_autoresearch.yaml
PYTHONPATH=src python3 -m binance4h_research update-trading-research-scorecard --program configs/trading_autoresearch.yaml
PYTHONPATH=src python3 -m binance4h_research show-trading-research-scorecard --program configs/trading_autoresearch.yaml
```

## Result Files

- `results/trading_autoresearch/<program>/experiments.jsonl`
- `results/trading_autoresearch/<program>/results.tsv`
- `results/trading_autoresearch/<program>/champions.json`
- `results/trading_autoresearch/<program>/research_log.jsonl`
- `results/trading_autoresearch/<program>/research_scorecard.json`
- `results/trading_autoresearch/<program>/runs/<run_id>/summary.json`
- `results/trading_autoresearch/<program>/champions/<family>/`
- `results/trading_autoresearch/<program>/champions/global/`
- `src/binance4h_research/trading_autoresearch/family_champions/`
