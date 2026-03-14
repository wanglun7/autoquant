from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from .academic_config import load_academic_config
from .academic_data import CoinGeckoClient, fetch_academic_data, fetch_coin_list_cache, resolve_coin_ids
from .academic_panel import build_weekly_panel, save_weekly_panel
from .academic_replication import compare_replications, run_paper_replication
from .autoevolve import build_research_context_artifacts, load_autoevolve_program, replay_candidate, run_evolution_batch, show_champions
from .config import load_experiment_config
from .data import fetch_all, load_symbol_klines
from .experiment import render_report, run_experiment
from .paper_approx import compare_paper_approx, run_paper_approx
from .paper_approx_config import load_paper_approx_config
from .trading_autoresearch import (
    build_trading_context,
    build_trading_research_scorecard,
    load_trading_autoresearch_program,
    record_trading_research_turn,
    replay_trading_run,
    run_trading_autoresearch_batch,
    show_trading_champion,
    show_trading_research_log,
    show_trading_research_scorecard,
    update_trading_research_scorecard,
)
from .universe import build_universe_membership, save_universe_membership


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="binance4h")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch-data", help="Download Binance futures data")
    fetch.add_argument("--data-dir", default="data/raw")
    fetch.add_argument("--symbols", nargs="*", default=None)
    fetch.add_argument("--interval", default="4h")
    fetch.add_argument("--start-date", default=None, help="Inclusive UTC date, e.g. 2021-03-13")
    fetch.add_argument("--end-date", default=None, help="Inclusive UTC date, e.g. 2026-03-13")

    build = sub.add_parser("build-universe", help="Build dynamic liquidity universe")
    build.add_argument("--config", required=True)

    backtest = sub.add_parser("run-backtest", help="Run configured experiment")
    backtest.add_argument("--config", required=True)

    report = sub.add_parser("report", help="Render result summary")
    report.add_argument("--results-dir", required=True)

    academic_fetch = sub.add_parser("fetch-academic-data", help="Download academic spot market data from CoinGecko")
    academic_fetch.add_argument("--data-dir", default="data/academic/raw")
    academic_fetch.add_argument("--start-date", required=True)
    academic_fetch.add_argument("--end-date", required=True)
    academic_fetch.add_argument("--coin-ids", nargs="*", default=None)
    academic_fetch.add_argument("--coin-ids-file", default=None)
    academic_fetch.add_argument("--all-listed", action="store_true")
    academic_fetch.add_argument("--include-inactive", action="store_true")

    panel = sub.add_parser("build-academic-panel", help="Build weekly academic panel")
    panel.add_argument("--config", required=True)

    replication = sub.add_parser("run-paper-replication", help="Run academic paper replication")
    replication.add_argument("--config", required=True)

    comparison = sub.add_parser("compare-replications", help="Compare academic replication outputs")
    comparison.add_argument("--configs", nargs="+", required=True)

    paper_approx = sub.add_parser("run-paper-approx", help="Run Binance paper-approx replication")
    paper_approx.add_argument("--config", required=True)

    compare_paper = sub.add_parser("compare-paper-approx", help="Compare Binance paper-approx outputs")
    compare_paper.add_argument("--configs", nargs="+", required=True)

    build_context = sub.add_parser("build-research-context", help="Build autoevolve research context")
    build_context.add_argument("--program", required=True)

    evolve = sub.add_parser("run-evolution-batch", help="Run one autoevolve batch")
    evolve.add_argument("--program", required=True)
    evolve.add_argument("--families", nargs="*", default=None)
    evolve.add_argument("--batch-size", type=int, default=None)

    champions = sub.add_parser("show-champions", help="Show current autoevolve champions")
    champions.add_argument("--program", required=True)

    replay = sub.add_parser("replay-candidate", help="Replay one autoevolve candidate")
    replay.add_argument("--program", required=True)
    replay.add_argument("--run-id", required=True)

    trading_context = sub.add_parser("build-trading-context", help="Build trading-autoresearch context")
    trading_context.add_argument("--program", required=True)

    trading_batch = sub.add_parser("run-trading-autoresearch-batch", help="Evaluate current trading-autoresearch strategy.py and record the run")
    trading_batch.add_argument("--program", required=True)

    trading_champions = sub.add_parser("show-trading-champions", help="Show current trading-autoresearch champions")
    trading_champions.add_argument("--program", required=True)

    trading_replay = sub.add_parser("replay-trading-run", help="Replay one trading-autoresearch run")
    trading_replay.add_argument("--program", required=True)
    trading_replay.add_argument("--run-id", required=True)

    trading_record = sub.add_parser("record-trading-research-turn", help="Append one trading-autoresearch research-log note")
    trading_record.add_argument("--program", required=True)
    trading_record.add_argument("--note-file", required=True)

    trading_log = sub.add_parser("show-trading-research-log", help="Show the trading-autoresearch research-log path")
    trading_log.add_argument("--program", required=True)

    trading_scorecard = sub.add_parser("update-trading-research-scorecard", help="Build the trading-autoresearch research scorecard")
    trading_scorecard.add_argument("--program", required=True)

    trading_show_scorecard = sub.add_parser("show-trading-research-scorecard", help="Show the trading-autoresearch research-scorecard path")
    trading_show_scorecard.add_argument("--program", required=True)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "fetch-data":
        data_dir = Path(args.data_dir)
        symbols = args.symbols
        start_time = int(pd.Timestamp(args.start_date, tz="UTC").timestamp() * 1000) if args.start_date else None
        end_time = int(pd.Timestamp(args.end_date, tz="UTC").timestamp() * 1000) if args.end_date else None
        paths = fetch_all(
            data_dir=data_dir,
            symbols=symbols,
            interval=args.interval,
            start_time=start_time,
            end_time=end_time,
        )
        print(f"Fetched {len(paths)} artifacts into {data_dir}")
        return

    if args.command == "build-universe":
        config = load_experiment_config(args.config)
        klines = load_symbol_klines(config.data_dir)
        membership = build_universe_membership(klines, config)
        path = save_universe_membership(membership, config.processed_dir, config.name)
        print(path)
        return

    if args.command == "run-backtest":
        config = load_experiment_config(args.config)
        paths = run_experiment(config)
        print(paths["summary"])
        return

    if args.command == "report":
        print(render_report(Path(args.results_dir)))
        return

    if args.command == "fetch-academic-data":
        data_dir = Path(args.data_dir)
        fetch_coin_list_cache(CoinGeckoClient(), data_dir, include_inactive=args.include_inactive)
        resolved = resolve_coin_ids(
            data_dir=data_dir,
            coin_ids=args.coin_ids,
            coin_ids_file=args.coin_ids_file,
            all_listed=args.all_listed,
            include_inactive=args.include_inactive,
        )
        outputs = fetch_academic_data(
            data_dir=data_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            coin_ids=resolved,
            include_inactive=args.include_inactive,
        )
        print(f"Fetched {len(outputs)} artifacts into {data_dir}")
        return

    if args.command == "build-academic-panel":
        config = load_academic_config(args.config)
        panel = build_weekly_panel(config)
        path = save_weekly_panel(panel, config.weekly_panel_path)
        print(path)
        return

    if args.command == "run-paper-replication":
        config = load_academic_config(args.config)
        outputs = run_paper_replication(config)
        print(outputs["summary"])
        return

    if args.command == "compare-replications":
        path = compare_replications(args.configs)
        print(path)
        return

    if args.command == "run-paper-approx":
        config = load_paper_approx_config(args.config)
        outputs = run_paper_approx(config)
        print(outputs["summary"])
        return

    if args.command == "compare-paper-approx":
        path = compare_paper_approx(args.configs)
        print(path)
        return

    if args.command == "build-research-context":
        program = load_autoevolve_program(args.program)
        outputs = build_research_context_artifacts(program)
        print(outputs["context_summary"])
        return

    if args.command == "run-evolution-batch":
        program = load_autoevolve_program(args.program)
        outputs = run_evolution_batch(program, family_scopes=args.families, batch_size=args.batch_size)
        print(outputs["batch_summary"])
        return

    if args.command == "show-champions":
        program = load_autoevolve_program(args.program)
        path = show_champions(program)
        print(path)
        return

    if args.command == "replay-candidate":
        program = load_autoevolve_program(args.program)
        path = replay_candidate(program, args.run_id)
        print(path)
        return

    if args.command == "build-trading-context":
        program = load_trading_autoresearch_program(args.program)
        outputs = build_trading_context(program)
        print(outputs["context_summary"])
        return

    if args.command == "run-trading-autoresearch-batch":
        program = load_trading_autoresearch_program(args.program)
        outputs = run_trading_autoresearch_batch(program)
        print(outputs["summary"])
        return

    if args.command == "show-trading-champions":
        program = load_trading_autoresearch_program(args.program)
        path = show_trading_champion(program)
        print(path)
        return

    if args.command == "replay-trading-run":
        program = load_trading_autoresearch_program(args.program)
        path = replay_trading_run(program, args.run_id)
        print(path)
        return

    if args.command == "record-trading-research-turn":
        program = load_trading_autoresearch_program(args.program)
        path = record_trading_research_turn(program, args.note_file)
        print(path)
        return

    if args.command == "show-trading-research-log":
        program = load_trading_autoresearch_program(args.program)
        path = show_trading_research_log(program)
        print(path)
        return

    if args.command == "update-trading-research-scorecard":
        program = load_trading_autoresearch_program(args.program)
        path = update_trading_research_scorecard(program)
        print(path)
        return

    if args.command == "show-trading-research-scorecard":
        program = load_trading_autoresearch_program(args.program)
        path = show_trading_research_scorecard(program)
        print(path)
        return

    raise ValueError(f"Unsupported command: {args.command}")
