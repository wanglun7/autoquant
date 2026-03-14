---
name: "autoresearch-trading"
description: "Use when the user wants to run or continue the repo-specific trading autoresearch loop in /Users/lun/Desktop/manifex/quant. This skill applies the Karpathy autoresearch pattern for quantitative research: read program_trading.md and recent results, keep data preparation and evaluation fixed, modify only src/binance4h_research/trading_autoresearch/strategy.py, then run the trading-autoresearch commands to evaluate, record, and promote or reject the new strategy."
---

# Autoresearch Trading

Use this skill for repo-specific strategy evolution in this project. The workflow is intentionally narrow:

- Read [docs/program_trading.md](/Users/lun/Desktop/manifex/quant/docs/program_trading.md) first.
- Then read the latest results in `results/trading_autoresearch/...`, including `research_log.jsonl`.
- Only modify `src/binance4h_research/trading_autoresearch/strategy.py`.
- Do not modify `prepare_market.py`, `evaluate.py`, `runner.py`, or the fixed config during a normal research turn.

## Workflow

1. Read [references/repo-map.md](references/repo-map.md) to confirm mutable vs fixed files.
2. Read [references/workflow.md](references/workflow.md) for the standard run loop.
3. If you need the exact evaluation gate, read [references/evaluation.md](references/evaluation.md).
4. Choose one family and one objective. If the user does not specify them, default to the repo priority order in `program_trading.md`.
5. Restore `src/binance4h_research/trading_autoresearch/strategy.py` from `src/binance4h_research/trading_autoresearch/family_champions/<family>.py` when that family champion exists.
6. Write one short hypothesis before editing code.
7. Edit only `src/binance4h_research/trading_autoresearch/strategy.py`.
8. Run:

```bash
PYTHONPATH=src python3 -m binance4h_research run-trading-autoresearch-batch --program configs/trading_autoresearch.yaml
```

9. Review:
- `results/trading_autoresearch/trading_autoresearch_v1/results.tsv`
- `results/trading_autoresearch/trading_autoresearch_v1/champions.json`
- `results/trading_autoresearch/trading_autoresearch_v1/research_log.jsonl`
- `results/trading_autoresearch/trading_autoresearch_v1/runs/<run_id>/summary.json`
10. Record the turn with:

```bash
PYTHONPATH=src python3 -m binance4h_research record-trading-research-turn --program configs/trading_autoresearch.yaml --note-file <note.json>
```

11. If the run is only `keep` or `reject`, restore `strategy.py` to the family baseline and do not auto-push.
12. If the new run is a `family_champion`, keep the mirrored champion file under `src/binance4h_research/trading_autoresearch/family_champions/`, commit the code changes, and push the current branch.
13. If the new run is a `global_champion`, commit the code changes and push the current branch.

## Rules

- Treat `strategy.py` as the only mutable research surface.
- Treat `src/binance4h_research/trading_autoresearch/family_champions/` as published outputs that preserve one champion per family.
- Treat `research_log.jsonl` as the memory of what was tried and what was learned.
- Keep each run to one clear idea.
- State one objective and one hypothesis before changing code.
- Prefer improving the current family champion over inventing multiple unrelated concepts in one turn.
- Never change the evaluator to make a strategy look better.
- Do not auto-push plain `keep` runs.
- Never create or switch git branches automatically. Push the current branch only.
- If no remote is configured, use the user's provided remote or ask once.
