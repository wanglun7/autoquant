from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


EXPECTED_COMMANDS = (
    "fetch-data",
    "run-structure-scan",
    "run-structure-decompose",
    "run-structure-validate",
)
LEGACY_COMMANDS = (
    "run-backtest",
    "run-paper-approx",
    "run-trading-autoresearch-batch",
    "fetch-academic-data",
)
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"


def _run_help(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    if existing:
        env["PYTHONPATH"] = f"{SRC_PATH}{os.pathsep}{existing}"
    else:
        env["PYTHONPATH"] = str(SRC_PATH)
    return subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)


def _extract_subcommands(help_text: str) -> tuple[str, ...]:
    match = re.search(r"\{([^}]+)\}", help_text)
    if not match:
        raise AssertionError("Failed to locate subcommand list in help output")
    parts = tuple(cmd.strip() for cmd in match.group(1).split(","))
    return parts


def _assert_only_expected_commands(help_text: str) -> None:
    commands = _extract_subcommands(help_text)
    assert len(commands) == len(EXPECTED_COMMANDS)
    assert set(commands) == set(EXPECTED_COMMANDS)
    for legacy in LEGACY_COMMANDS:
        assert legacy not in help_text


def test_root_help_shows_only_mainline_commands() -> None:
    module_result = _run_help([sys.executable, "-m", "binance4h_research", "--help"])
    assert module_result.returncode == 0
    _assert_only_expected_commands(module_result.stdout)

    console_script = Path(sys.executable).parent / "binance4h"
    if not console_script.exists():
        pytest.skip("binance4h console script missing in this environment")
    console_result = _run_help([str(console_script), "--help"])
    assert console_result.returncode == 0
    _assert_only_expected_commands(console_result.stdout)
