# Evaluation

The fixed evaluator lives in:

- `src/binance4h_research/trading_autoresearch/prepare_market.py`
- `src/binance4h_research/trading_autoresearch/evaluate.py`
- `configs/trading_autoresearch.yaml`

## Locked Inputs

- local Binance raw data in `data/raw`
- `fee + slippage + funding`
- fixed train / validation / test split
- fixed walk-forward summary
- fixed `2x` cost stress test

## Hard Constraints

- test net return must stay positive
- test Sharpe must exceed the configured floor
- max drawdown must stay within the configured limit
- `2x` cost stress must remain positive

## Primary Score

Champion comparison uses test-set Sharpe as the primary score, after the hard constraints are applied.

- `family_champion`: best kept run within the same family
- `global_champion`: best kept run across all families
- `champion`: the active promotion alias controlled by `champion_family_mode`

## Output Files

- `experiments.jsonl`
- `results.tsv`
- `champions.json`
- `runs/<run_id>/summary.json`
