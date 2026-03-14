from __future__ import annotations

from pathlib import Path

import pandas as pd

from binance4h_research.paper_approx import build_variant_weights, compare_paper_approx, compute_grobys_signal, eligible_symbols, run_paper_approx
from binance4h_research.paper_approx_config import PaperApproxConfig, PaperApproxPaperConfig, PaperApproxPortfolioConfig, PaperApproxUniverseConfig


def _symbol_frame(closes: list[float], quote_volume: float) -> pd.DataFrame:
    opens = [closes[0], *closes[:-1]]
    times = pd.date_range("2024-01-01", periods=len(closes), freq="4h", tz="UTC")
    return pd.DataFrame(
        {
            "open_time": times,
            "open": opens,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000.0] * len(closes),
            "close_time": times + pd.Timedelta(hours=4) - pd.Timedelta(milliseconds=1),
            "quote_volume": [quote_volume] * len(closes),
            "trade_count": [100] * len(closes),
            "taker_buy_base_volume": [500.0] * len(closes),
            "taker_buy_quote_volume": [quote_volume / 2] * len(closes),
        }
    )


def test_eligible_symbols_filters_stables() -> None:
    config = PaperApproxConfig(
        name="x",
        paper_id="ltw",
        universe=PaperApproxUniverseConfig(exclude_bases=["USDC", "FDUSD"]),
    )
    filtered = eligible_symbols(["BTCUSDT", "ETHUSDT", "USDCUSDT", "FDUSDUSDT"], config)
    assert filtered == ["BTCUSDT", "ETHUSDT"]


def test_grobys_signal_skips_one_day() -> None:
    idx = pd.date_range("2024-01-01", periods=220, freq="4h", tz="UTC")
    closes = pd.DataFrame({"AAAUSDT": [100.0] * 213 + [200.0] * 7}, index=idx)
    signal = compute_grobys_signal(closes, formation_days=30, skip_days=1, bars_per_day=6)
    ts = pd.Timestamp("2024-02-05 00:00:00+00:00")
    # The jump in the last 1-day gap should not contaminate the signal.
    assert round(float(signal.loc[ts, "AAAUSDT"]), 8) == 0.0


def test_build_variant_weights_for_ltw_and_grobys() -> None:
    klines = {
        f"C{i}USDT": _symbol_frame([100 + i + j for j in range(420)], quote_volume=10_000_000 - i * 100_000)
        for i in range(12)
    }
    ltw_config = PaperApproxConfig(
        name="ltw",
        paper_id="ltw",
        universe=PaperApproxUniverseConfig(top_n=10, min_history_bars=100),
        paper=PaperApproxPaperConfig(rebalance_weekday=0, rebalance_hour_utc=0, ltw_lookback_weeks=3),
        portfolio=PaperApproxPortfolioConfig(gross_exposure=1.0),
    )
    variants, _, _, group_stats = build_variant_weights(klines, ltw_config)
    assert "ltw_cmom_3w" in variants
    assert not group_stats[group_stats["variant"] == "ltw_cmom_3w"].empty

    grobys_config = PaperApproxConfig(
        name="grobys",
        paper_id="grobys",
        universe=PaperApproxUniverseConfig(top_n=10, min_history_bars=100),
        paper=PaperApproxPaperConfig(rebalance_weekday=0, rebalance_hour_utc=0, grobys_formation_days=30, grobys_skip_days=1),
        portfolio=PaperApproxPortfolioConfig(gross_exposure=1.0),
    )
    variants, _, _, group_stats = build_variant_weights(klines, grobys_config)
    assert "grobys_plain_1m_skip1d" in variants
    subset = group_stats[group_stats["variant"] == "grobys_plain_1m_skip1d"]
    assert not subset.empty
    assert subset["n_assets"].max() <= 10


def test_run_and_compare_paper_approx(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    (data_dir / "klines").mkdir(parents=True)
    (data_dir / "funding").mkdir(parents=True)

    for symbol, frame in {
        f"C{i}USDT": _symbol_frame([100 + i + j for j in range(1600)], quote_volume=20_000_000 - i * 300_000)
        for i in range(14)
    }.items():
        frame.to_csv(data_dir / "klines" / f"{symbol}_4h.csv", index=False)
        pd.DataFrame({"fundingTime": [], "fundingRate": []}).to_csv(data_dir / "funding" / f"{symbol}.csv", index=False)

    ltw = PaperApproxConfig(
        name="paper_ltw_fixture",
        paper_id="ltw",
        data_dir=data_dir,
        results_dir=tmp_path / "results",
        universe=PaperApproxUniverseConfig(top_n=12, min_history_bars=100),
    )
    grobys = PaperApproxConfig(
        name="paper_grobys_fixture",
        paper_id="grobys",
        data_dir=data_dir,
        results_dir=tmp_path / "results",
        universe=PaperApproxUniverseConfig(top_n=10, min_history_bars=100),
    )
    ltw_outputs = run_paper_approx(ltw)
    grobys_outputs = run_paper_approx(grobys)
    assert ltw_outputs["summary"].exists()
    assert grobys_outputs["summary"].exists()

    ltw_cfg = tmp_path / "ltw.yaml"
    grobys_cfg = tmp_path / "grobys.yaml"
    ltw_cfg.write_text(
        "\n".join(
            [
                "name: paper_ltw_fixture",
                "paper_id: ltw",
                f"data_dir: {data_dir}",
                f"results_dir: {tmp_path / 'results'}",
                "universe:",
                "  top_n: 12",
                "  min_history_bars: 100",
            ]
        ),
        encoding="utf-8",
    )
    grobys_cfg.write_text(
        "\n".join(
            [
                "name: paper_grobys_fixture",
                "paper_id: grobys",
                f"data_dir: {data_dir}",
                f"results_dir: {tmp_path / 'results'}",
                "universe:",
                "  top_n: 10",
                "  min_history_bars: 100",
            ]
        ),
        encoding="utf-8",
    )
    comparison_path = compare_paper_approx([ltw_cfg, grobys_cfg])
    comparison = pd.read_csv(comparison_path)
    assert not comparison.empty
