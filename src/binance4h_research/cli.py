from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import fetch_all
from .structure_decompose import run_structure_decompose
from .structure_scan import run_structure_scan
from .structure_validate import run_structure_validate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="binance4h")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch-data", help="Download Binance futures data")
    fetch.add_argument("--data-dir", default="data/raw")
    fetch.add_argument("--symbols", nargs="*", default=None)
    fetch.add_argument("--interval", default="4h")
    fetch.add_argument("--start-date", default=None, help="Inclusive UTC date, e.g. 2021-03-13")
    fetch.add_argument("--end-date", default=None, help="Inclusive UTC date, e.g. 2026-03-13")

    scan = sub.add_parser("run-structure-scan", help="Run descriptive structure scan on current market data")
    scan.add_argument("--data-dir", default="data/raw")
    scan.add_argument("--output-dir", default="results/structure_scan/scan_v1")
    scan.add_argument("--top-n", type=int, default=100)
    scan.add_argument("--liquidity-lookback-bars", type=int, default=180)
    scan.add_argument("--min-history-bars", type=int, default=180)

    decompose = sub.add_parser("run-structure-decompose", help="Run structure decompose analysis")
    decompose.add_argument("--data-dir", default="data/raw")
    decompose.add_argument("--output-dir", default="results/structure_decompose/decompose_v1")
    decompose.add_argument("--top-n", type=int, default=100)
    decompose.add_argument("--liquidity-lookback-bars", type=int, default=180)
    decompose.add_argument("--min-history-bars", type=int, default=180)

    validate = sub.add_parser("run-structure-validate", help="Run low-dimensional structure validation")
    validate.add_argument("--data-dir", default="data/raw")
    validate.add_argument("--output-dir", default="results/structure_validate/validate_v1")
    validate.add_argument("--top-n", type=int, default=100)
    validate.add_argument("--liquidity-lookback-bars", type=int, default=180)
    validate.add_argument("--min-history-bars", type=int, default=180)

    return parser


def _parse_timestamp(date_str: str | None) -> int | None:
    if not date_str:
        return None
    return int(pd.Timestamp(date_str, tz="UTC").timestamp() * 1000)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "fetch-data":
        data_dir = Path(args.data_dir)
        start_time = _parse_timestamp(args.start_date)
        end_time = _parse_timestamp(args.end_date)
        paths = fetch_all(
            data_dir=data_dir,
            symbols=args.symbols,
            interval=args.interval,
            start_time=start_time,
            end_time=end_time,
        )
        print(f"Fetched {len(paths)} artifacts into {data_dir}")
        return

    if args.command == "run-structure-scan":
        paths = run_structure_scan(
            data_dir=Path(args.data_dir),
            output_dir=Path(args.output_dir),
            top_n=args.top_n,
            liquidity_lookback_bars=args.liquidity_lookback_bars,
            min_history_bars=args.min_history_bars,
        )
        print(paths["summary"])
        return

    if args.command == "run-structure-decompose":
        paths = run_structure_decompose(
            data_dir=Path(args.data_dir),
            output_dir=Path(args.output_dir),
            top_n=args.top_n,
            liquidity_lookback_bars=args.liquidity_lookback_bars,
            min_history_bars=args.min_history_bars,
        )
        print(paths["summary"])
        return

    if args.command == "run-structure-validate":
        paths = run_structure_validate(
            data_dir=Path(args.data_dir),
            output_dir=Path(args.output_dir),
            top_n=args.top_n,
            liquidity_lookback_bars=args.liquidity_lookback_bars,
            min_history_bars=args.min_history_bars,
        )
        print(paths["summary"])
        return

    raise ValueError(f"Unsupported command: {args.command}")
