# Structure Mainline Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the repo to a single statistical market-structure mainline with exactly four CLI commands and no legacy research systems.

**Architecture:** First isolate the two hidden dependencies that keep the structure path tied to deleted systems, then migrate the structure modules and tests onto that new helper boundary, then remove the legacy source tree, CLI branches, configs, docs, and skill assets. The resulting repo keeps one raw-data layer, one matrix-construction layer, one signal-transform layer, and three independent structure-analysis commands.

**Tech Stack:** Python 3.11, pandas, numpy, requests, pytest, setuptools

---

## File Map

### Create

- `src/binance4h_research/market_data.py`
- `tests/test_market_data.py`
- `tests/test_data.py`
- `tests/test_cli.py`

### Modify

- `src/binance4h_research/structure_scan.py`
- `src/binance4h_research/structure_decompose.py`
- `src/binance4h_research/structure_validate.py`
- `src/binance4h_research/cli.py`
- `src/binance4h_research/__main__.py`
- `src/binance4h_research/signals.py`
- `src/binance4h_research/data.py`
- `pyproject.toml`
- `README.md`

### Delete

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
- `configs/`
- `docs/program_trading.md`
- `docs/experiment_template.md`
- `docs/systematic_experiments.md`
- `docs/academic_replication.md`
- `docs/research_program.md`
- `docs/experiments/`
- `knowledge/skills/autoresearch-trading/`
- `tests/test_pipeline.py`
- `tests/test_paper_approx.py`
- `tests/test_academic_replication.py`
- `tests/test_autoevolve.py`
- `tests/test_trading_autoresearch.py`

## Preconditions

- The current repository has unrelated tracked and untracked changes. Do not implement this plan in the existing working tree.
- Start in an isolated worktree and branch before touching code.
- Do not revert or overwrite unrelated user changes while carrying out this plan.

### Task 0: Create An Isolated Worktree

**Files:**
- Create: none
- Modify: none
- Test: none

- [ ] **Step 1: Create a feature worktree**

Run:

```bash
git worktree add ../quant-structure-mainline-reduction -b feat/structure-mainline-reduction
```

Expected: Git creates a new worktree directory and checks out a new branch.

- [ ] **Step 2: Enter the worktree and confirm it is isolated**

Run:

```bash
cd ../quant-structure-mainline-reduction
git status --short
```

Expected: No unrelated dirty files from the original working tree appear in the new worktree.

- [ ] **Step 3: Install editable dependencies in the worktree if needed**

Run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Expected: Editable install succeeds and `pytest` is available.

- [ ] **Step 4: Commit the worktree bootstrap if any repo files changed**

```bash
git status --short
```

Expected: Usually no repo files changed. If lockfiles or local config files appear, stop and avoid committing them.

### Task 1: Introduce `market_data.py` And Lock Its Behavior With Unit Tests

**Files:**
- Create: `src/binance4h_research/market_data.py`
- Create: `tests/test_market_data.py`
- Test: `tests/test_market_data.py`

- [ ] **Step 1: Write the failing market-data tests**

```python
from __future__ import annotations

import pandas as pd

from binance4h_research.market_data import combine_field_matrix, funding_returns_from_events


def test_combine_field_matrix_aligns_symbols_on_open_time() -> None:
    btc = pd.DataFrame(
        {
            "open_time": pd.to_datetime(["2024-01-01 00:00:00+00:00", "2024-01-01 04:00:00+00:00"]),
            "close": [100.0, 101.0],
        }
    )
    eth = pd.DataFrame(
        {
            "open_time": pd.to_datetime(["2024-01-01 04:00:00+00:00", "2024-01-01 08:00:00+00:00"]),
            "close": [200.0, 201.0],
        }
    )

    frame = combine_field_matrix({"BTCUSDT": btc, "ETHUSDT": eth}, "close")

    assert list(frame.columns) == ["BTCUSDT", "ETHUSDT"]
    assert frame.index.tz is not None
    assert pd.isna(frame.loc[pd.Timestamp("2024-01-01 00:00:00+00:00"), "ETHUSDT"])


def test_funding_returns_from_events_preserves_existing_bucket_rules() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="4h", tz="UTC")
    funding = funding_returns_from_events(
        {
            "AAAUSDT": pd.DataFrame(
                {
                    "fundingTime": pd.to_datetime(
                        [
                            "2024-01-01 04:00:00+00:00",
                            "2024-01-01 05:00:00+00:00",
                            "2024-01-01 08:00:00+00:00",
                            "2024-01-01 13:00:00+00:00",
                        ]
                    ),
                    "fundingRate": [0.001, 0.002, 0.003, 0.004],
                }
            )
        },
        idx,
    )

    assert round(float(funding.loc[idx[0], "AAAUSDT"]), 6) == 0.001
    assert round(float(funding.loc[idx[1], "AAAUSDT"]), 6) == 0.005
    assert round(float(funding.loc[idx[2], "AAAUSDT"]), 6) == 0.0
```

- [ ] **Step 2: Run the market-data tests and confirm they fail**

Run:

```bash
pytest tests/test_market_data.py -q
```

Expected: FAIL with `ModuleNotFoundError` or missing symbol errors for `binance4h_research.market_data`.

- [ ] **Step 3: Implement `market_data.py` with the extracted helpers**

```python
from __future__ import annotations

import numpy as np
import pandas as pd


def combine_field_matrix(klines: dict[str, pd.DataFrame], field: str) -> pd.DataFrame:
    series_map: dict[str, pd.Series] = {}
    for symbol, frame in klines.items():
        temp = frame[["open_time", field]].copy().dropna()
        series_map[symbol] = temp.set_index("open_time")[field].astype(float)
    if not series_map:
        return pd.DataFrame()
    combined = pd.concat(series_map, axis=1).sort_index()
    combined.index = pd.DatetimeIndex(combined.index, tz="UTC")
    return combined


def funding_returns_from_events(
    funding_by_symbol: dict[str, pd.DataFrame],
    interval_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    funding = pd.DataFrame(0.0, index=interval_index, columns=sorted(funding_by_symbol))
    if funding.empty:
        return funding

    index_values = funding.index.view("int64")
    for symbol, frame in funding_by_symbol.items():
        if symbol not in funding.columns or frame.empty:
            continue
        temp = frame.copy()
        temp["fundingTime"] = pd.to_datetime(temp["fundingTime"], utc=True)
        temp["fundingRate"] = pd.to_numeric(temp["fundingRate"], errors="coerce").fillna(0.0)
        funding_times = temp["fundingTime"].astype("int64").to_numpy()
        funding_rates = temp["fundingRate"].to_numpy(dtype=float)
        positions = np.searchsorted(index_values, funding_times, side="left")
        valid = (positions > 0) & (positions < len(index_values))
        if not valid.any():
            continue
        grouped = pd.Series(funding_rates[valid]).groupby(positions[valid] - 1).sum()
        funding.iloc[grouped.index.to_numpy(dtype=int), funding.columns.get_loc(symbol)] = grouped.to_numpy(dtype=float)
    return funding.fillna(0.0)
```

- [ ] **Step 4: Run the market-data tests and confirm they pass**

Run:

```bash
pytest tests/test_market_data.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the helper extraction foundation**

```bash
git add src/binance4h_research/market_data.py tests/test_market_data.py
git commit -m "refactor: extract structure market data helpers"
```

### Task 2: Preserve Data-Layer Coverage Outside The Legacy Pipeline Test

**Files:**
- Create: `tests/test_data.py`
- Modify: `src/binance4h_research/data.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: Write the failing data-layer tests**

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd

from binance4h_research.data import fetch_funding_range, fetch_klines_range, load_symbol_funding, load_symbol_klines


class _PagedClient:
    def __init__(self) -> None:
        self.kline_calls = 0
        self.funding_calls = 0

    def klines(self, symbol: str, interval: str = "4h", start_time: int | None = None, end_time: int | None = None, limit: int = 1500) -> pd.DataFrame:
        self.kline_calls += 1
        if self.kline_calls == 1:
            return pd.DataFrame({"open_time": pd.to_datetime([0, 14_400_000], unit="ms", utc=True), "open": [1.0, 2.0], "high": [1.0, 2.0], "low": [1.0, 2.0], "close": [1.0, 2.0], "volume": [1.0, 1.0], "close_time": pd.to_datetime([14_399_999, 28_799_999], unit="ms", utc=True), "quote_volume": [1.0, 1.0], "trade_count": [1, 1], "taker_buy_base_volume": [1.0, 1.0], "taker_buy_quote_volume": [1.0, 1.0]})
        if self.kline_calls == 2:
            return pd.DataFrame({"open_time": pd.to_datetime([28_800_000], unit="ms", utc=True), "open": [3.0], "high": [3.0], "low": [3.0], "close": [3.0], "volume": [1.0], "close_time": pd.to_datetime([43_199_999], unit="ms", utc=True), "quote_volume": [1.0], "trade_count": [1], "taker_buy_base_volume": [1.0], "taker_buy_quote_volume": [1.0]})
        return pd.DataFrame()

    def funding_rates(self, symbol: str, start_time: int | None = None, end_time: int | None = None, limit: int = 1000) -> pd.DataFrame:
        self.funding_calls += 1
        if self.funding_calls == 1:
            return pd.DataFrame({"symbol": [symbol, symbol], "fundingTime": pd.to_datetime([0, 28_800_000], unit="ms", utc=True), "fundingRate": [0.001, 0.002], "markPrice": [100.0, 101.0]})
        if self.funding_calls == 2:
            return pd.DataFrame({"symbol": [symbol], "fundingTime": pd.to_datetime([57_600_000], unit="ms", utc=True), "fundingRate": [0.003], "markPrice": [102.0]})
        return pd.DataFrame()


def test_range_fetch_paginates_without_duplicates() -> None:
    client = _PagedClient()
    klines = fetch_klines_range(client, symbol="AAAUSDT", interval="4h", start_time=0, end_time=57_600_000, limit=2)
    funding = fetch_funding_range(client, symbol="AAAUSDT", start_time=0, end_time=57_600_000, limit=2)

    assert len(klines) == 3
    assert klines["open_time"].is_unique
    assert len(funding) == 3
    assert funding["fundingTime"].is_unique


def test_loaders_roundtrip_cached_csvs(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    (data_dir / "klines").mkdir(parents=True)
    (data_dir / "funding").mkdir(parents=True)

    pd.DataFrame(
        {
            "open_time": pd.to_datetime(["2024-01-01 00:00:00+00:00"]),
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1.0],
            "close_time": pd.to_datetime(["2024-01-01 03:59:59.999000+00:00"]),
            "quote_volume": [2.0],
            "trade_count": [3],
            "taker_buy_base_volume": [0.5],
            "taker_buy_quote_volume": [1.0],
        }
    ).to_csv(data_dir / "klines" / "BTCUSDT_4h.csv", index=False)
    pd.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "fundingTime": pd.to_datetime(["2024-01-01 08:00:00+00:00"]),
            "fundingRate": [0.0001],
            "markPrice": [100.0],
        }
    ).to_csv(data_dir / "funding" / "BTCUSDT.csv", index=False)

    assert "BTCUSDT" in load_symbol_klines(data_dir)
    assert "BTCUSDT" in load_symbol_funding(data_dir)
```

- [ ] **Step 2: Run the data-layer tests and confirm they pass before cleanup**

Run:

```bash
pytest tests/test_data.py -q
```

Expected: PASS once the test file is created. If loader behavior differs, adjust `data.py` only enough to preserve the existing public behavior.

- [ ] **Step 3: Keep `data.py` mainline-focused**

```python
def load_symbol_klines(data_dir: Path, interval: str = "4h") -> dict[str, pd.DataFrame]:
    folder = data_dir / "klines"
    result: dict[str, pd.DataFrame] = {}
    if not folder.exists():
        return result
    for path in sorted(folder.glob(f"*_{interval}.csv")):
        symbol = path.stem.replace(f"_{interval}", "")
        frame = pd.read_csv(path, parse_dates=["open_time", "close_time"])
        frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
        frame["close_time"] = pd.to_datetime(frame["close_time"], utc=True)
        result[symbol] = frame.sort_values("open_time").reset_index(drop=True)
    return result
```

Implementation note: only make changes here if test extraction surfaces a genuine loader issue. Do not add back any config-driven or experiment-oriented behavior.

- [ ] **Step 4: Run the extracted data-layer tests again**

Run:

```bash
pytest tests/test_data.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the extracted data-layer tests**

```bash
git add tests/test_data.py src/binance4h_research/data.py
git commit -m "test: extract mainline data coverage"
```

### Task 3: Repoint The Structure Modules To The New Mainline Helper

**Files:**
- Modify: `src/binance4h_research/structure_scan.py`
- Modify: `src/binance4h_research/structure_decompose.py`
- Modify: `src/binance4h_research/structure_validate.py`
- Test: `tests/test_structure_scan.py`
- Test: `tests/test_structure_decompose.py`
- Test: `tests/test_structure_validate.py`

- [ ] **Step 1: Change the imports in all three structure modules**

```python
from .data import load_symbol_funding, load_symbol_klines
from .market_data import combine_field_matrix, funding_returns_from_events
from .signals import build_close_matrix, build_open_matrix
```

Replace every `from .backtest import funding_returns_from_events` and `from .universe import _combine_field` import with the new mainline import.

- [ ] **Step 2: Replace `_combine_field(...)` call sites**

```python
highs = combine_field_matrix(klines, "high")
lows = combine_field_matrix(klines, "low")
quote_volume = combine_field_matrix(klines, "quote_volume")
```

Update each structure module so all matrix construction uses `combine_field_matrix(...)`.

- [ ] **Step 3: Run the three structure test files**

Run:

```bash
pytest tests/test_structure_scan.py tests/test_structure_decompose.py tests/test_structure_validate.py -q
```

Expected: PASS. If one file fails because of alignment or dtype drift, fix the structure module using the new helper boundary rather than reintroducing old imports.

- [ ] **Step 4: Grep for forbidden helper imports**

Run:

```bash
rg -n "from \\.backtest import funding_returns_from_events|from \\.universe import _combine_field|_combine_field\\(" src/binance4h_research
```

Expected: No matches in the surviving structure modules.

- [ ] **Step 5: Commit the structure-module migration**

```bash
git add src/binance4h_research/structure_scan.py src/binance4h_research/structure_decompose.py src/binance4h_research/structure_validate.py
git commit -m "refactor: decouple structure modules from legacy helpers"
```

### Task 4: Collapse The CLI And Packaging Surface To Four Commands

**Files:**
- Create: `tests/test_cli.py`
- Modify: `src/binance4h_research/cli.py`
- Modify: `src/binance4h_research/__main__.py`
- Modify: `pyproject.toml`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing CLI smoke tests**

```python
from __future__ import annotations

import subprocess
import sys


def test_root_help_shows_only_mainline_commands() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "binance4h_research", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    for command in ("fetch-data", "run-structure-scan", "run-structure-decompose", "run-structure-validate"):
        assert command in result.stdout
    for legacy in ("run-backtest", "run-paper-approx", "run-trading-autoresearch-batch", "fetch-academic-data"):
        assert legacy not in result.stdout
```

- [ ] **Step 2: Run the CLI smoke tests and confirm they fail against the current parser**

Run:

```bash
pytest tests/test_cli.py -q
```

Expected: FAIL because legacy commands are still present.

- [ ] **Step 3: Rewrite `cli.py` to expose only the four supported commands**

```python
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="binance4h")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch-data", help="Download Binance futures data")
    fetch.add_argument("--data-dir", default="data/raw")
    fetch.add_argument("--symbols", nargs="*", default=None)
    fetch.add_argument("--interval", default="4h")
    fetch.add_argument("--start-date", default=None)
    fetch.add_argument("--end-date", default=None)

    scan = sub.add_parser("run-structure-scan", help="Run descriptive structure scan on current market data")
    scan.add_argument("--data-dir", default="data/raw")
    scan.add_argument("--output-dir", default="results/structure_scan/scan_v1")
    scan.add_argument("--top-n", type=int, default=100)
    scan.add_argument("--liquidity-lookback-bars", type=int, default=180)
    scan.add_argument("--min-history-bars", type=int, default=180)

    # mirror for decompose and validate
    return parser
```

Implementation notes:

- Remove all imports tied to academic, paper, experiment, autoevolve, and trading-autoresearch paths.
- Keep dispatch logic for the surviving four commands wired to `fetch_all`, `run_structure_scan`, `run_structure_decompose`, and `run_structure_validate`.
- Keep `__main__.py` as a thin `from .cli import main`.
- Keep `[project.scripts] binance4h = "binance4h_research.cli:main"` in `pyproject.toml`.
- Remove `PyYAML>=6.0` from `pyproject.toml` if no surviving source file imports `yaml`.

- [ ] **Step 4: Run the CLI smoke tests**

Run:

```bash
pytest tests/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 5: Verify the parser help manually**

Run:

```bash
python -m binance4h_research --help
python -m binance4h_research fetch-data --help
python -m binance4h_research run-structure-scan --help
python -m binance4h_research run-structure-decompose --help
python -m binance4h_research run-structure-validate --help
binance4h --help
```

Expected: Every command returns exit code 0, both root help surfaces show only four subcommands, and `binance4h --help` matches the module entrypoint.

- [ ] **Step 6: Commit the CLI reduction**

```bash
git add src/binance4h_research/cli.py src/binance4h_research/__main__.py pyproject.toml tests/test_cli.py
git commit -m "refactor: reduce cli to structure mainline"
```

### Task 5: Remove The Legacy Source Tree And Legacy Test Suites

**Files:**
- Delete: `src/binance4h_research/academic_panel.py`
- Delete: `src/binance4h_research/academic_data.py`
- Delete: `src/binance4h_research/academic_replication.py`
- Delete: `src/binance4h_research/academic_config.py`
- Delete: `src/binance4h_research/paper_approx.py`
- Delete: `src/binance4h_research/paper_approx_config.py`
- Delete: `src/binance4h_research/experiment.py`
- Delete: `src/binance4h_research/analytics.py`
- Delete: `src/binance4h_research/portfolio.py`
- Delete: `src/binance4h_research/config.py`
- Delete: `src/binance4h_research/universe.py`
- Delete: `src/binance4h_research/backtest.py`
- Delete: `src/binance4h_research/autoevolve/`
- Delete: `src/binance4h_research/trading_autoresearch/`
- Delete: `tests/test_pipeline.py`
- Delete: `tests/test_paper_approx.py`
- Delete: `tests/test_academic_replication.py`
- Delete: `tests/test_autoevolve.py`
- Delete: `tests/test_trading_autoresearch.py`
- Test: `tests/test_market_data.py`
- Test: `tests/test_data.py`
- Test: `tests/test_structure_scan.py`
- Test: `tests/test_structure_decompose.py`
- Test: `tests/test_structure_validate.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Delete the legacy source modules and test files**

Run:

```bash
rm -rf src/binance4h_research/autoevolve src/binance4h_research/trading_autoresearch
rm -f src/binance4h_research/academic_panel.py src/binance4h_research/academic_data.py src/binance4h_research/academic_replication.py src/binance4h_research/academic_config.py
rm -f src/binance4h_research/paper_approx.py src/binance4h_research/paper_approx_config.py src/binance4h_research/experiment.py src/binance4h_research/analytics.py
rm -f src/binance4h_research/portfolio.py src/binance4h_research/config.py src/binance4h_research/universe.py src/binance4h_research/backtest.py
rm -f tests/test_pipeline.py tests/test_paper_approx.py tests/test_academic_replication.py tests/test_autoevolve.py tests/test_trading_autoresearch.py
```

Expected: Only the planned legacy files disappear.

- [ ] **Step 2: Verify there are no remaining imports of deleted legacy modules**

Run:

```bash
rg -n "academic_|paper_approx|run_backtest|build_universe|autoevolve|trading_autoresearch|load_experiment_config|yaml" src tests
```

Expected: No matches in surviving source and test files, except allowed historical text inside the plan/spec documents.

- [ ] **Step 3: Run the surviving automated tests**

Run:

```bash
pytest tests/test_market_data.py tests/test_data.py tests/test_structure_scan.py tests/test_structure_decompose.py tests/test_structure_validate.py tests/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit the code-tree deletion**

```bash
git add -A
git commit -m "refactor: remove legacy research systems"
```

### Task 6: Rewrite The Repo Narrative And Remove Legacy Assets

**Files:**
- Modify: `README.md`
- Delete: `configs/`
- Delete: `docs/program_trading.md`
- Delete: `docs/experiment_template.md`
- Delete: `docs/systematic_experiments.md`
- Delete: `docs/academic_replication.md`
- Delete: `docs/research_program.md`
- Delete: `docs/experiments/`
- Delete: `knowledge/skills/autoresearch-trading/`
- Test: none

- [ ] **Step 1: Rewrite `README.md` around the single mainline**

```md
# Binance 4h Research

Statistical market-structure research toolkit for Binance USDT-M perpetual futures.

## Mainline

```bash
binance4h fetch-data --data-dir data/raw --symbols BTCUSDT ETHUSDT
binance4h run-structure-scan --data-dir data/raw --output-dir results/structure_scan/scan_v1
binance4h run-structure-decompose --data-dir data/raw --output-dir results/structure_decompose/decompose_v1
binance4h run-structure-validate --data-dir data/raw --output-dir results/structure_validate/validate_v1
```

## Scope

- Current scope: statistical market-structure analysis
- Future scope: time-series research on the same raw-data foundation
```

Implementation notes:

- Document `data/raw/klines/*.csv` and `data/raw/funding/*.csv`.
- Remove every mention of backtests, experiments, academic replication, paper approximation, autoevolve, and trading autoresearch.

- [ ] **Step 2: Delete configs, legacy docs, and the repo-local autoresearch skill assets**

Run:

```bash
rm -rf configs docs/experiments knowledge/skills/autoresearch-trading
rm -f docs/program_trading.md docs/experiment_template.md docs/systematic_experiments.md docs/academic_replication.md docs/research_program.md
```

Expected: Only legacy narrative/config assets are removed.

- [ ] **Step 3: Grep for stale user-facing references**

Run:

```bash
rg -n "run-backtest|paper-approx|academic|autoevolve|trading-autoresearch|trading_autoresearch|sample_momentum|build-universe" README.md docs knowledge src tests pyproject.toml
```

Expected: No matches outside historical spec/plan files under `docs/superpowers/`.

- [ ] **Step 4: Commit the narrative cleanup**

```bash
git add README.md docs knowledge
git commit -m "docs: rewrite repo around structure mainline"
```

### Task 7: Final Verification And Release Readiness Check

**Files:**
- Modify: none
- Test: all surviving mainline tests and CLI commands

- [ ] **Step 1: Run the full surviving test suite**

Run:

```bash
pytest -q
```

Expected: PASS with only the mainline test files collected.

- [ ] **Step 2: Verify package help surfaces**

Run:

```bash
python -m binance4h_research --help
python -m binance4h_research fetch-data --help
python -m binance4h_research run-structure-scan --help
python -m binance4h_research run-structure-decompose --help
python -m binance4h_research run-structure-validate --help
binance4h --help
```

Expected: All help commands succeed, `python -m ...` and `binance4h` expose the same four commands, and no legacy subcommands appear.

- [ ] **Step 3: Verify the repo contains only mainline source modules**

Run:

```bash
find src/binance4h_research -maxdepth 2 -type f | sort
```

Expected: Only `__init__.py`, `__main__.py`, `cli.py`, `data.py`, `market_data.py`, `signals.py`, and the three `structure_*` modules remain.

- [ ] **Step 4: Verify no deleted-system references remain in live code/docs**

Run:

```bash
rg -n "academic_|paper_approx|experiment|portfolio|autoevolve|trading_autoresearch|build-universe|run-backtest" src tests README.md pyproject.toml docs knowledge
```

Expected: No matches outside historical documents under `docs/superpowers/`.

- [ ] **Step 5: Create the final integration commit if needed**

```bash
git status --short
```

Expected: Clean working tree. If any tracked files remain uncommitted, stage them and create a final commit such as:

```bash
git add -A
git commit -m "chore: finalize structure mainline reduction"
```
