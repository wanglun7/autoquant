# Trading Autoresearch Program

## Goal

Find Binance USDT-M perpetual strategies that remain positive after `fee + slippage + funding`, and still survive a `2x` cost stress test.

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

## Suggested Research Order

1. Improve `cross_sectional` first.
2. Use `btc_time_series` as a separate fallback family.
3. Use `relative_value` only on large-liquid pairs.
4. Treat funding, regime, volatility, and liquidity as internal logic inside `strategy.py`, not as separate evaluators.

## Standard Commands

```bash
PYTHONPATH=src python3 -m binance4h_research build-trading-context --program configs/trading_autoresearch.yaml
PYTHONPATH=src python3 -m binance4h_research run-trading-autoresearch-batch --program configs/trading_autoresearch.yaml
PYTHONPATH=src python3 -m binance4h_research show-trading-champions --program configs/trading_autoresearch.yaml
PYTHONPATH=src python3 -m binance4h_research replay-trading-run --program configs/trading_autoresearch.yaml --run-id <run_id>
```

## Result Files

- `results/trading_autoresearch/<program>/experiments.jsonl`
- `results/trading_autoresearch/<program>/results.tsv`
- `results/trading_autoresearch/<program>/champions.json`
- `results/trading_autoresearch/<program>/runs/<run_id>/summary.json`
- `results/trading_autoresearch/<program>/champions/<family>/`
- `results/trading_autoresearch/<program>/champions/global/`
- `src/binance4h_research/trading_autoresearch/family_champions/`
