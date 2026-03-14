# Workflow

## Standard Loop

1. Read `docs/program_trading.md`.
2. Read the latest `results.tsv` and `champions.json`.
3. Inspect the current `strategy.py`.
4. Make one focused modification in `strategy.py`.
5. Run:

```bash
PYTHONPATH=src python3 -m binance4h_research run-trading-autoresearch-batch --program configs/trading_autoresearch.yaml
```

6. Compare the new run against the current family champion.
7. Summarize:
- what changed
- why it might help
- whether it was rejected, kept, or promoted
8. If the run is only `keep`, summarize it and stop. Do not auto-push.
9. If the run is a `family_champion` but not a `global_champion`, stage the relevant code changes, commit them, and push a research branch such as `research/<family>`. Do not add large local `data/` or `results/` artifacts unless the repo already tracks them on purpose.
10. If the run is a `global_champion`, stage the relevant code changes, commit them, and push the main working branch.

## Good Changes

- change one lookback or ranking idea
- add one filter
- simplify a noisy signal
- change one position-mapping rule

## Bad Changes

- rewriting the evaluator
- modifying multiple unrelated families in one run
- changing both data logic and strategy logic together
