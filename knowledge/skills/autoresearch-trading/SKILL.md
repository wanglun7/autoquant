---
name: "autoresearch-trading"
description: "Use when the user wants to run or continue the repo-specific trading autoresearch loop in /Users/lun/Desktop/manifex/quant. This skill runs the fixed trading autoresearch workflow: read program_trading.md and recent results, think from first principles, modify only strategy.py, evaluate one turn, record the result, and preserve or reject the family according to the research rules."
---

# Autoresearch Trading

Use this skill for repo-specific strategy research in this project.

- Read [docs/program_trading.md](/Users/lun/Desktop/manifex/quant/docs/program_trading.md) first.
- Then read the latest `results/trading_autoresearch/...`, including `families.json`, `research_log.jsonl`, and `research_scorecard.json`.
- Only modify `src/binance4h_research/trading_autoresearch/strategy.py`.
- Do not modify `prepare_market.py`, `evaluate.py`, `runner.py`, or the fixed config during a normal research turn.
- Default to answering the user in Chinese unless they explicitly ask for another language.
- Treat this skill as an automated quant researcher: maximize experiment quality, reduce repeated low-information attempts, explain why an idea worked or failed, and preserve both family assets and research knowledge.

## Workflow

1. Read [references/repo-map.md](references/repo-map.md) to confirm mutable vs fixed files.
2. Read [references/workflow.md](references/workflow.md) for the standard run loop.
3. If you need the exact evaluation gate, read [references/evaluation.md](references/evaluation.md).
4. Read `families.json` and `research_scorecard.json`.
5. Decide whether this turn is:
   - `experiment_within_family`
   - `propose_new_family`
6. If the user does not specify a family, follow `Explore First`:
   - deepen a promising candidate family first
   - otherwise open a new family in a weakly covered `execution_mode`
   - otherwise converge on the weakest formal family
7. If the target family is formal and already has a champion, restore `strategy.py` from `src/binance4h_research/trading_autoresearch/family_champions/<family>.py`.
8. If the target family is a candidate, prefer the strongest formal family in the same `execution_mode` as the working baseline.
9. Write one short objective, one falsifiable hypothesis, one mechanism summary, and one differentiation note before editing code.
10. Edit only `src/binance4h_research/trading_autoresearch/strategy.py`.
11. Run:

```bash
PYTHONPATH=src python3 -m binance4h_research run-trading-autoresearch-batch --program configs/trading_autoresearch.yaml
```

12. Review:
- `results/trading_autoresearch/trading_autoresearch_v1/results.tsv`
- `results/trading_autoresearch/trading_autoresearch_v1/champions.json`
- `results/trading_autoresearch/trading_autoresearch_v1/families.json`
- `results/trading_autoresearch/trading_autoresearch_v1/research_log.jsonl`
- `results/trading_autoresearch/trading_autoresearch_v1/research_scorecard.json`
- `results/trading_autoresearch/trading_autoresearch_v1/runs/<run_id>/summary.json`
13. Record the turn with:

```bash
PYTHONPATH=src python3 -m binance4h_research record-trading-research-turn --program configs/trading_autoresearch.yaml --note-file <note.json>
```

14. Update the family registry and scorecard with:

```bash
PYTHONPATH=src python3 -m binance4h_research update-trading-family-registry --program configs/trading_autoresearch.yaml
PYTHONPATH=src python3 -m binance4h_research update-trading-research-scorecard --program configs/trading_autoresearch.yaml
```

15. If the run is only `keep` or `reject`, preserve it as knowledge in `research_log.jsonl`, restore `strategy.py` to the family baseline, and do not auto-push.
16. If the run becomes a `formal` family champion, keep the mirrored champion file under `src/binance4h_research/trading_autoresearch/family_champions/`, commit the code changes, and push the current branch.
17. If the run is a `global_champion`, commit the code changes and push the current branch.

## Rules

- Treat `family` as the primary research asset unit. Families are parallel, not nested.
- Treat `execution_mode` as a structural constraint, not as a parent layer.
- Treat `src/binance4h_research/trading_autoresearch/family_champions/` as published outputs that preserve one champion per formal family.
- Treat `families.json` as the source of truth for `candidate` and `formal` family state.
- Treat `research_log.jsonl` as the memory of what was tried and what was learned.
- Treat `research_scorecard.json` as the short-term control panel for choosing the next turn.
- Keep each run to one clear idea.
- State one objective and one hypothesis before changing code.
- Use first-principles thinking: start from the fixed inputs that actually exist in this repo, the mechanism that could create alpha, the mechanism that creates costs and drawdowns, and only then choose one code change.
- Prefer mechanism-first hypotheses over surface-level parameter sweeps. Do not change parameters without stating what market behavior the change is trying to capture or avoid.
- A new family must have a semantic name, a mechanism summary, and a differentiation note. Do not rename an old family just to make a parameter variant look novel.
- Default to `Explore First` when the user does not specify a direction. Use `converge` only after a family has something worth refining.
- Prefer one research axis per turn: signal definition, filter, position mapping, rebalance logic, or risk scaling.
- Explain both the intended edge and the likely failure mode. Do not report metrics without explaining why the idea may have helped or degraded.
- Never change the evaluator to make a strategy look better.
- Do not auto-push plain `keep` runs.
- Never create or switch git branches automatically. Push the current branch only.
- If no remote is configured, use the user's provided remote or ask once.
