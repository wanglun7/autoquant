from __future__ import annotations

from pathlib import Path

import pandas as pd

from binance4h_research.academic_config import AcademicExperimentConfig
from binance4h_research.academic_panel import assign_paper_52w_calendar, build_weekly_panel_from_series
from binance4h_research.academic_replication import compare_replications, run_ficura_replication, run_grobys_replication, run_ltw_replication, run_paper_replication


def test_assign_paper_calendar_has_52_weeks() -> None:
    dates = pd.Series(pd.date_range("2020-01-01", "2020-12-31", freq="D", tz="UTC"))
    calendar = assign_paper_52w_calendar(dates)
    assert calendar["paper_week"].min() == 1
    assert calendar["paper_week"].max() == 52


def _daily_coin(start: str, periods: int, price_start: float, step: float, market_cap: float, volume: float) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="D", tz="UTC")
    prices = [price_start + step * idx for idx in range(periods)]
    return pd.DataFrame(
        {
            "date": dates,
            "price": prices,
            "market_cap": [market_cap] * periods,
            "total_volume": [volume] * periods,
        }
    )


def test_ltw_replication_produces_cmom_series() -> None:
    panel = pd.DataFrame(
        [
            {"week_end": pd.Timestamp("2024-01-07", tz="UTC"), "coin_id": f"c{i}", "close": 100 + i, "market_cap": 1000 + i * 100, "weekly_simple_return": 0.0, "weekly_log_return": 0.0}
            for i in range(10)
        ]
        + [
            {"week_end": pd.Timestamp("2024-01-14", tz="UTC"), "coin_id": f"c{i}", "close": 102 + i, "market_cap": 1000 + i * 100, "weekly_simple_return": 0.0, "weekly_log_return": 0.0}
            for i in range(10)
        ]
        + [
            {"week_end": pd.Timestamp("2024-01-21", tz="UTC"), "coin_id": f"c{i}", "close": 104 + i, "market_cap": 1000 + i * 100, "weekly_simple_return": 0.0, "weekly_log_return": 0.0}
            for i in range(10)
        ]
        + [
            {"week_end": pd.Timestamp("2024-01-28", tz="UTC"), "coin_id": f"c{i}", "close": 106 + i * 10, "market_cap": 1000 + i * 100, "weekly_simple_return": 0.10 * i, "weekly_log_return": 0.0}
            for i in range(10)
        ]
        + [
            {"week_end": pd.Timestamp("2024-02-04", tz="UTC"), "coin_id": f"c{i}", "close": 120 + i, "market_cap": 1000 + i * 100, "weekly_simple_return": 0.01 * i, "weekly_log_return": 0.0}
            for i in range(10)
        ]
    )
    config = AcademicExperimentConfig(name="ltw_test", paper_id="ltw", minimum_market_cap_usd=500.0)
    result = run_ltw_replication(panel, config)
    assert not result.empty
    assert "cmom_3w" in result["variant"].unique()


def test_grobys_replication_uses_top_n_universe() -> None:
    daily_by_coin = {
        f"coin{i}": _daily_coin("2020-12-01", 80, 100 + i, 1 + i * 0.1, 10_000_000 - i * 100_000, 1_000_000)
        for i in range(6)
    }
    config = AcademicExperimentConfig(name="grobys_test", paper_id="grobys", start_date="2021-01-01", end_date="2021-02-28", top_n=5)
    panel = build_weekly_panel_from_series(daily_by_coin, config)
    result = run_grobys_replication(panel, daily_by_coin, config)
    assert not result.empty
    assert result["n_assets"].max() <= 5


def test_ficura_replication_builds_segmented_variants() -> None:
    panel = build_weekly_panel_from_series(
        {
            "large_a": _daily_coin("2021-01-01", 200, 100, 1.5, 60_000_000, 7_000_000),
            "large_b": _daily_coin("2021-01-01", 200, 120, 1.2, 55_000_000, 6_000_000),
            "small_a": _daily_coin("2021-01-01", 200, 20, -0.02, 2_000_000, 100_000),
            "small_b": _daily_coin("2021-01-01", 200, 22, 0.01, 3_000_000, 150_000),
            "small_c": _daily_coin("2021-01-01", 200, 18, 0.03, 4_000_000, 200_000),
        },
        AcademicExperimentConfig(name="ficura_test", paper_id="ficura", start_date="2021-01-01", end_date="2021-06-30"),
    )
    config = AcademicExperimentConfig(name="ficura_test", paper_id="ficura", start_date="2021-01-01", end_date="2021-06-30")
    result = run_ficura_replication(panel, config)
    assert not result.empty
    assert any(value.startswith("large_liquid_mom_") for value in result["variant"].unique())


def test_run_and_compare_replications(tmp_path: Path) -> None:
    raw_dir = tmp_path / "academic_raw" / "daily"
    raw_dir.mkdir(parents=True)
    for coin_id, frame in {
        "btc": _daily_coin("2014-01-01", 500, 100.0, 1.0, 100_000_000, 20_000_000),
        "eth": _daily_coin("2014-01-01", 500, 80.0, 0.8, 80_000_000, 15_000_000),
        "xrp": _daily_coin("2014-01-01", 500, 50.0, 0.5, 60_000_000, 10_000_000),
        "doge": _daily_coin("2014-01-01", 500, 10.0, 0.2, 5_000_000, 2_000_000),
        "ltc": _daily_coin("2014-01-01", 500, 30.0, 0.4, 20_000_000, 4_000_000),
    }.items():
        frame.to_csv(raw_dir / f"{coin_id}.csv", index=False)

    ltw = AcademicExperimentConfig(
        name="academic_ltw_fixture",
        paper_id="ltw",
        raw_data_dir=tmp_path / "academic_raw",
        processed_dir=tmp_path / "processed",
        results_dir=tmp_path / "results",
        start_date="2014-01-01",
        end_date="2015-05-15",
        minimum_market_cap_usd=1_000_000,
    )
    ficura = AcademicExperimentConfig(
        name="academic_ficura_fixture",
        paper_id="ficura",
        raw_data_dir=tmp_path / "academic_raw",
        processed_dir=tmp_path / "processed",
        results_dir=tmp_path / "results",
        start_date="2014-01-01",
        end_date="2015-05-15",
        minimum_market_cap_usd=1_000_000,
    )
    outputs_ltw = run_paper_replication(ltw)
    outputs_ficura = run_paper_replication(ficura)
    assert outputs_ltw["summary"].exists()
    assert outputs_ficura["summary"].exists()

    ltw_config_path = tmp_path / "ltw.yaml"
    ficura_config_path = tmp_path / "ficura.yaml"
    ltw_config_path.write_text(
        "\n".join(
            [
                "name: academic_ltw_fixture",
                "paper_id: ltw",
                f"raw_data_dir: {ltw.raw_data_dir}",
                f"processed_dir: {ltw.processed_dir}",
                f"results_dir: {ltw.results_dir}",
                "start_date: 2014-01-01",
                "end_date: 2015-05-15",
            ]
        ),
        encoding="utf-8",
    )
    ficura_config_path.write_text(
        "\n".join(
            [
                "name: academic_ficura_fixture",
                "paper_id: ficura",
                f"raw_data_dir: {ficura.raw_data_dir}",
                f"processed_dir: {ficura.processed_dir}",
                f"results_dir: {ficura.results_dir}",
                "start_date: 2014-01-01",
                "end_date: 2015-05-15",
            ]
        ),
        encoding="utf-8",
    )
    comparison_path = compare_replications([ltw_config_path, ficura_config_path])
    comparison = pd.read_csv(comparison_path)
    assert not comparison.empty
