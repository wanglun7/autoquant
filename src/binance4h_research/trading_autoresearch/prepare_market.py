from __future__ import annotations

from pathlib import Path
import json

from ..autoevolve.context import ResearchContext
from ..backtest import funding_returns_from_events
from ..data import load_symbol_funding, load_symbol_klines
from ..paper_approx import eligible_symbols
from ..paper_approx_config import PaperApproxConfig, PaperApproxUniverseConfig
from ..signals import build_close_matrix, build_open_matrix
from ..universe import _combine_field
from .program import TradingAutoResearchProgram


def _eligible_market_symbols(klines: dict[str, object]) -> list[str]:
    config = PaperApproxConfig(
        name="trading_autoresearch",
        paper_id="ltw",
        universe=PaperApproxUniverseConfig(exclude_bases=["USDC", "FDUSD", "TUSD", "BUSD", "USDP", "DAI"]),
    )
    return eligible_symbols(klines.keys(), config)


def build_context(program: TradingAutoResearchProgram) -> ResearchContext:
    klines_all = load_symbol_klines(program.data_dir)
    funding_raw = load_symbol_funding(program.data_dir)
    symbols = _eligible_market_symbols(klines_all)
    klines = {symbol: klines_all[symbol] for symbol in symbols}
    closes = build_close_matrix(klines)
    opens = build_open_matrix(klines)
    quote_volume = _combine_field(klines, "quote_volume")
    liquidity = quote_volume.shift(1).rolling(window=360, min_periods=180).mean()
    history_count = quote_volume.notna().cumsum()
    funding = funding_returns_from_events(funding_raw, opens.index).reindex(index=opens.index, columns=opens.columns, fill_value=0.0)
    last_scores = liquidity.iloc[-1].dropna().sort_values(ascending=False)
    major_symbols = [symbol for symbol in last_scores.index if symbol != "BTCUSDT"][: program.pair_pool_size]
    pair_pool: list[tuple[str, str]] = []
    for idx, left in enumerate(major_symbols):
        for right in major_symbols[idx + 1 :]:
            pair_pool.append((left, right))
    btc_symbol = "BTCUSDT" if "BTCUSDT" in closes.columns else closes.columns[0]
    return ResearchContext(
        closes=closes,
        opens=opens,
        funding=funding,
        quote_volume=quote_volume,
        liquidity=liquidity,
        history_count=history_count,
        symbols=symbols,
        major_symbols=major_symbols,
        pair_pool=pair_pool,
        btc_symbol=btc_symbol,
    )


def save_context_summary(context: ResearchContext, program: TradingAutoResearchProgram) -> Path:
    program.context_dir.mkdir(parents=True, exist_ok=True)
    path = program.context_dir / "context_summary.json"
    path.write_text(
        json.dumps(
            {
                "bars": int(len(context.closes)),
                "symbols": int(len(context.symbols)),
                "start": str(context.closes.index.min()),
                "end": str(context.closes.index.max()),
                "btc_symbol": context.btc_symbol,
                "major_symbols": context.major_symbols,
                "pair_pool_size": len(context.pair_pool),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path
