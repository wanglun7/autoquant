# Workflow

## Standard Loop

1. Read `docs/program_trading.md`.
2. Read the latest `results.tsv`, `champions.json`, `families.json`, `research_log.jsonl`, and `research_scorecard.json`.
3. Decide whether the next turn is:
   - `experiment_within_family`
   - `propose_new_family`
4. If the user did not specify a family, follow `Explore First`:
   - deepen a promising candidate family first
   - otherwise open a new family in a weakly covered `execution_mode`
   - otherwise converge on the weakest formal family
5. Start from first principles:
- what inputs are actually available in the fixed context
- what market mechanism could produce alpha
- what mechanism could create costs, turnover, or drawdown
- why one specific change is worth testing now
6. Choose one `family` and one `execution_mode`.
7. If the family is formal, restore `strategy.py` from `src/binance4h_research/trading_autoresearch/family_champions/<family>.py` when that file exists.
8. If the family is a candidate, use the strongest formal family in the same `execution_mode` as the baseline when possible.
9. Write one short objective, one short hypothesis, one mechanism summary, and one differentiation note.
10. Make one focused modification in `strategy.py`.
11. Run:

```bash
PYTHONPATH=src python3 -m binance4h_research run-trading-autoresearch-batch --program configs/trading_autoresearch.yaml
```

12. Compare the new run against the current formal family champion and note whether the turn created high information gain even if it did not promote.
13. Summarize in Chinese by default:
- what changed
- why it might help
- why it might fail
- whether it was rejected, kept, or promoted
- whether it deepened an existing family or proposed a new one
14. Record the turn in `research_log.jsonl` with `record-trading-research-turn`.
15. Update `families.json` with `update-trading-family-registry`.
16. Update `research_scorecard.json` with `update-trading-research-scorecard`.
17. If the run is only `keep` or `reject`, restore `strategy.py` to the family baseline, summarize it, and stop. Do not auto-push.
18. If the run becomes a `formal` family champion, make sure the mirrored file in `src/binance4h_research/trading_autoresearch/family_champions/` is preserved, then stage the relevant code changes, commit them, and push the current branch. Do not add large local `data/` or `results/` artifacts unless the repo already tracks them on purpose.
19. If the run is a `global_champion`, stage the relevant code changes, commit them, and push the current branch.

## Good Changes

- change one lookback or ranking idea
- add one filter
- simplify a noisy signal
- change one position-mapping rule
- propose one clearly differentiated new family with one execution mode

## Bad Changes

- rewriting the evaluator
- modifying multiple unrelated families in one run
- changing both data logic and strategy logic together
- inventing a new family name for a pure parameter tweak
