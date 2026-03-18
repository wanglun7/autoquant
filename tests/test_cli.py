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
    expected = (
        "fetch-data",
        "run-structure-scan",
        "run-structure-decompose",
        "run-structure-validate",
    )
    legacy = (
        "run-backtest",
        "run-paper-approx",
        "run-trading-autoresearch-batch",
        "fetch-academic-data",
    )

    for command in expected:
        assert command in result.stdout
    for command in legacy:
        assert command not in result.stdout
