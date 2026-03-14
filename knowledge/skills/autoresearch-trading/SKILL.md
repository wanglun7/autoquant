---
name: "autoresearch-trading"
description: "Use when the user wants to run or continue the repo-specific trading autoresearch loop in /Users/lun/Desktop/manifex/quant. This skill applies the Karpathy autoresearch pattern for quantitative research: read program_trading.md and recent results, keep data preparation and evaluation fixed, modify only src/binance4h_research/trading_autoresearch/strategy.py, then run the trading-autoresearch commands to evaluate, record, and promote or reject the new strategy."
---

# Autoresearch Trading

Use this skill for repo-specific strategy evolution in this project. The workflow is intentionally narrow:

- Read [docs/program_trading.md](/Users/lun/Desktop/manifex/quant/docs/program_trading.md) first.
- Then read the latest results in `results/trading_autoresearch/...`.
- Only modify `src/binance4h_research/trading_autoresearch/strategy.py`.
- Do not modify `prepare_market.py`, `evaluate.py`, `runner.py`, or the fixed config during a normal research turn.

## Workflow

1. Read [references/repo-map.md](references/repo-map.md) to confirm mutable vs fixed files.
2. Read [references/workflow.md](references/workflow.md) for the standard run loop.
3. If you need the exact evaluation gate, read [references/evaluation.md](references/evaluation.md).
4. Edit only `src/binance4h_research/trading_autoresearch/strategy.py`.
5. Run:

```bash
PYTHONPATH=src python3 -m binance4h_research run-trading-autoresearch-batch --program configs/trading_autoresearch.yaml
```

6. Review:
- `results/trading_autoresearch/trading_autoresearch_v1/results.tsv`
- `results/trading_autoresearch/trading_autoresearch_v1/champions.json`
- `results/trading_autoresearch/trading_autoresearch_v1/runs/<run_id>/summary.json`
7. If the new run is a `family_champion`, commit the code changes and push a research branch such as `research/<family>`.
8. If the new run is a `global_champion`, commit the code changes and push the main working branch.
9. If the run is only `keep`, do not auto-push; summarize the result and stop unless the user asked to persist keeps too.

## Rules

- Treat `strategy.py` as the only mutable research surface.
- Keep each run to one clear idea.
- Prefer improving the current family champion over inventing multiple unrelated concepts in one turn.
- Never change the evaluator to make a strategy look better.
- Do not auto-push plain `keep` runs.
- Treat `family_champion` and `global_champion` differently: family promotions go to a research branch; global promotions may update the main branch.
- If no remote is configured, use the user's provided remote or ask once.
