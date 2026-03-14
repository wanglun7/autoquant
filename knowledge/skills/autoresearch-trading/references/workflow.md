# Workflow

## Standard Loop

1. Read `docs/program_trading.md`.
2. Read the latest `results.tsv`, `champions.json`, and `research_log.jsonl`.
3. Choose one family and one objective. If the user did not specify them, follow the default family priority from `program_trading.md`.
4. Start from first principles:
- what inputs are actually available in the fixed context
- what market mechanism could produce alpha
- what mechanism could create costs, turnover, or drawdown
- why one specific change could improve the current champion
5. Restore `strategy.py` from `src/binance4h_research/trading_autoresearch/family_champions/<family>.py` when that file exists.
6. Write one short hypothesis and one planned change.
7. Make one focused modification in `strategy.py`.
8. Run:

```bash
PYTHONPATH=src python3 -m binance4h_research run-trading-autoresearch-batch --program configs/trading_autoresearch.yaml
```

9. Compare the new run against the current family champion.
10. Summarize in Chinese by default:
- what changed
- why it might help
- whether it was rejected, kept, or promoted
11. Record the turn in `research_log.jsonl` with `record-trading-research-turn`.
12. If the run is only `keep` or `reject`, restore `strategy.py` to the family baseline, summarize it, and stop. Do not auto-push.
13. If the run is a `family_champion`, make sure the mirrored file in `src/binance4h_research/trading_autoresearch/family_champions/` is preserved, then stage the relevant code changes, commit them, and push the current branch. Do not add large local `data/` or `results/` artifacts unless the repo already tracks them on purpose.
14. If the run is a `global_champion`, stage the relevant code changes, commit them, and push the current branch.

## Good Changes

- change one lookback or ranking idea
- add one filter
- simplify a noisy signal
- change one position-mapping rule

## Bad Changes

- rewriting the evaluator
- modifying multiple unrelated families in one run
- changing both data logic and strategy logic together
