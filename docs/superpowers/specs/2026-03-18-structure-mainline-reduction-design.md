# Structure Mainline Reduction Design

Reduce the repository to a single market-structure research mainline:

`fetch-data -> run-structure-scan -> run-structure-decompose -> run-structure-validate`

The goal is to turn the repo into a coherent statistical research base for future quantitative trading work. This phase keeps the scope neutral and limited to market-structure analysis over Binance futures cross-sectional data. Future time-series research should be added later as a new peer research line on top of the same data foundation, not by preserving today's non-mainline systems.

## Objectives

- Keep one user-facing CLI with exactly four subcommands.
- Keep one code path from raw Binance data to structure analysis outputs.
- Remove all academic, paper-approx, experiment, autoevolve, and trading-autoresearch code and narrative.
- Preserve raw data under `data/raw/**` and existing `results/structure_*` directories on disk.
- Allow breaking old commands, old module imports, and old configs. No compatibility layer.

## Non-Goals

- No attempt to preserve old backtesting or portfolio workflows.
- No attempt to keep YAML-driven experiment configuration alive.
- No automatic orchestration that chains scan, decompose, and validate together.
- No time-series research implementation in this change.

## Current State

The current repo exposes multiple unrelated research systems behind one CLI:

- experiment/backtest/config/universe/portfolio path
- paper approximation path
- academic replication path
- autoevolve path
- trading autoresearch path
- structure scan/decompose/validate path

The structure modules are not fully isolated. They still depend on helpers from deleted systems:

- `structure_* -> backtest.funding_returns_from_events`
- `structure_* -> universe._combine_field`

That coupling is the main technical constraint in the reduction.

## Recommended Approach

Use a hard reduction with explicit helper extraction:

1. Introduce a new mainline-only helper module, `src/binance4h_research/market_data.py`.
2. Move the matrix-building helper and funding-event bucketing helper there.
3. Repoint all `structure_*` modules to `market_data.py`.
4. Rewrite the CLI to expose only the four mainline commands.
5. Delete all non-mainline code, configs, docs, tests, and skill assets.
6. Rewrite the README around the statistical mainline only.

This is preferred over a CLI-only cleanup because it removes hidden coupling and makes the repo semantics match the codebase reality.

## Target Architecture

The reduced codebase should keep these modules:

- `src/binance4h_research/data.py`
- `src/binance4h_research/market_data.py`
- `src/binance4h_research/signals.py`
- `src/binance4h_research/structure_scan.py`
- `src/binance4h_research/structure_decompose.py`
- `src/binance4h_research/structure_validate.py`
- `src/binance4h_research/cli.py`
- `src/binance4h_research/__main__.py`
- `src/binance4h_research/__init__.py`

Responsibility boundaries:

- `data.py`: fetch raw Binance futures data, update caches, load raw symbol tables.
- `market_data.py`: transform raw kline/funding tables into aligned analysis matrices.
- `signals.py`: pure signal and return transforms derived from aligned matrices.
- `structure_scan.py`: first-pass descriptive structure statistics.
- `structure_decompose.py`: factor-family decomposition and interaction analysis.
- `structure_validate.py`: low-dimensional validation and role assignment.
- `cli.py`: user-facing command parser and dispatch for the single mainline.
- `__main__.py`: module entrypoint only.

## Mainline Runtime Semantics

The four commands are intentionally independent readers of `data/raw`:

- `fetch-data` writes raw Binance data into `data/raw`.
- `run-structure-scan` reads raw data and writes scan outputs to `results/structure_scan/scan_v1` by default.
- `run-structure-decompose` reads raw data and writes decomposition outputs to `results/structure_decompose/decompose_v1` by default.
- `run-structure-validate` reads raw data and writes validation outputs to `results/structure_validate/validate_v1` by default.

`scan`, `decompose`, and `validate` do not consume each other's files as machine inputs. Their output directories are research artifacts for humans, not pipeline contracts.

This keeps the mainline stable and avoids introducing an unnecessary ETL dependency chain.

## CLI Design

The user-facing CLI keeps a single command surface with exactly four subcommands:

- `fetch-data`
- `run-structure-scan`
- `run-structure-decompose`
- `run-structure-validate`

Invocation forms:

- `python -m binance4h_research ...`
- `binance4h ...`

`binance4h` remains in `pyproject.toml` as a convenience alias to the same `cli.py:main`. This does not create a second maintained interface; it is the same CLI exposed through packaging.

No deprecated aliases, compatibility shims, or hidden legacy commands remain.

## Helper Extraction Rules

Create `src/binance4h_research/market_data.py` with two responsibilities:

- `combine_field_matrix(klines: dict[str, pd.DataFrame], field: str) -> pd.DataFrame`
- `funding_returns_from_events(funding_by_symbol: dict[str, pd.DataFrame], interval_index: pd.DatetimeIndex) -> pd.DataFrame`

Design rules:

- The new module must not depend on `config.py`, `portfolio.py`, `experiment.py`, or any deleted subsystem.
- `combine_field_matrix` replaces `universe._combine_field`.
- `funding_returns_from_events` preserves current bucketing behavior exactly.
- `signals.py` stays focused on signal transforms. Do not turn it into a generic helper bucket.

## Files To Delete

Delete all non-mainline code:

- `src/binance4h_research/academic_panel.py`
- `src/binance4h_research/academic_data.py`
- `src/binance4h_research/academic_replication.py`
- `src/binance4h_research/academic_config.py`
- `src/binance4h_research/paper_approx.py`
- `src/binance4h_research/paper_approx_config.py`
- `src/binance4h_research/experiment.py`
- `src/binance4h_research/analytics.py`
- `src/binance4h_research/portfolio.py`
- `src/binance4h_research/config.py`
- `src/binance4h_research/universe.py`
- `src/binance4h_research/backtest.py`
- `src/binance4h_research/autoevolve/`
- `src/binance4h_research/trading_autoresearch/`

Delete non-mainline configs and narrative assets:

- `configs/`
- `docs/program_trading.md`
- `docs/experiment_template.md`
- `docs/systematic_experiments.md`
- `docs/academic_replication.md`
- `docs/research_program.md`
- `docs/experiments/`
- `knowledge/skills/autoresearch-trading/`

Rewrite:

- `README.md`

Keep on disk, but stop referencing from code/docs/tests:

- `data/raw/**`
- existing `results/structure_*`
- existing non-mainline results directories

## Error Handling

The reduced CLI keeps error behavior simple:

- If no kline data exists under `data/raw/klines`, each structure command raises `FileNotFoundError`.
- Missing funding files are allowed and should behave as zero funding after matrix alignment.
- CLI command handlers should not wrap exceptions in custom layers unless needed to keep argument errors readable.
- `--help` output must contain only the four supported subcommands.

## Testing Strategy

Reshape tests around current capabilities, not around old file names.

Keep or create:

- `tests/test_data.py`
  - `fetch_klines_range` pagination and deduplication
  - `fetch_funding_range` pagination and deduplication
  - `load_symbol_klines`
  - `load_symbol_funding`
- `tests/test_market_data.py`
  - `combine_field_matrix`
  - `funding_returns_from_events`
- `tests/test_structure_scan.py`
- `tests/test_structure_decompose.py`
- `tests/test_structure_validate.py`
- `tests/test_cli.py`
  - root `--help`
  - each subcommand `--help`

Delete:

- `tests/test_pipeline.py` after extracting any still-relevant data-layer and funding-bucketing assertions
- `tests/test_paper_approx.py`
- `tests/test_academic_replication.py`
- `tests/test_autoevolve.py`
- `tests/test_trading_autoresearch.py`

## Documentation Changes

Rewrite `README.md` so the repo presents one clear story:

- what the repo is
- the four supported commands
- expected raw-data layout under `data/raw`
- where structure outputs are written
- the current project boundary: statistical market-structure research only
- future direction: time-series research will be added later as a separate line on the same data foundation

The README must stop mentioning:

- backtests
- experiments
- academic replication
- paper approximation
- trading autoresearch
- autoevolve

## Acceptance Criteria

The change is complete when all of the following are true:

- `python -m binance4h_research --help` shows only four subcommands.
- `binance4h --help` shows the same four subcommands.
- Each supported subcommand has a working `--help`.
- Structure modules import only mainline dependencies.
- No remaining source file imports deleted non-mainline modules.
- No tests reference deleted non-mainline modules.
- README and surviving docs describe only the structure mainline.
- Raw data and existing structure result directories remain untouched.

## Risks And Mitigations

Risk: deleting old modules before helper extraction breaks the structure path.
Mitigation: add `market_data.py` first, repoint imports second, delete old modules last.

Risk: removing `tests/test_pipeline.py` loses coverage for still-needed funding bucketing and pagination behavior.
Mitigation: extract those assertions into `tests/test_data.py` and `tests/test_market_data.py` before deleting the old file.

Risk: the repo still looks multi-purpose after code cleanup because docs and skill assets are left behind.
Mitigation: delete non-mainline docs and `knowledge/skills/autoresearch-trading/`, then rewrite `README.md`.

## Deferred Work

Not part of this change:

- time-series research modules
- backtesting or execution logic on top of structure outputs
- shared planning for a future multi-line research platform

Those can be designed later once the statistical structure foundation is stable.
