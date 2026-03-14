from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .academic_config import AcademicExperimentConfig
from .academic_data import load_coin_series


def assign_paper_52w_calendar(dates: pd.Series) -> pd.DataFrame:
    normalized = pd.to_datetime(dates, utc=True).dt.normalize()
    year_start = pd.to_datetime(normalized.dt.year.astype(str) + "-01-01", utc=True)
    year_end = pd.to_datetime(normalized.dt.year.astype(str) + "-12-31", utc=True)
    day_offset = (normalized - year_start).dt.days
    week_index = (day_offset // 7 + 1).clip(upper=52)
    week_start = year_start + pd.to_timedelta((week_index - 1) * 7, unit="D")
    provisional_week_end = week_start + pd.Timedelta(days=6)
    week_end = provisional_week_end.where(week_index < 52, year_end)
    return pd.DataFrame(
        {
            "paper_year": normalized.dt.year.astype(int),
            "paper_week": week_index.astype(int),
            "week_start": week_start,
            "week_end": week_end,
        }
    )


def build_weekly_panel_from_series(
    series_by_coin: dict[str, pd.DataFrame],
    config: AcademicExperimentConfig,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    sample_start = pd.Timestamp(config.start_date, tz="UTC")
    sample_end = pd.Timestamp(config.end_date, tz="UTC")

    for coin_id, frame in series_by_coin.items():
        if frame.empty:
            continue
        temp = frame.copy()
        temp["date"] = pd.to_datetime(temp["date"], utc=True, format="mixed")
        temp = temp[(temp["date"] >= sample_start) & (temp["date"] <= sample_end)]
        if temp.empty:
            continue
        calendar = assign_paper_52w_calendar(temp["date"])
        temp = pd.concat([temp.reset_index(drop=True), calendar], axis=1)
        weekly = (
            temp.groupby(["paper_year", "paper_week", "week_start", "week_end"], as_index=False)
            .agg(
                close=("price", "last"),
                market_cap=("market_cap", "last"),
                dollar_volume=("total_volume", "sum"),
                daily_rows=("date", "count"),
            )
            .sort_values(["paper_year", "paper_week"])
            .reset_index(drop=True)
        )
        weekly["coin_id"] = coin_id
        weekly["weekly_simple_return"] = weekly["close"].pct_change(fill_method=None)
        ratio = weekly["close"] / weekly["close"].shift(1)
        weekly["weekly_log_return"] = np.where(ratio > 0, np.log(ratio), np.nan)
        rows.append(weekly)

    if not rows:
        return pd.DataFrame()

    panel = pd.concat(rows, ignore_index=True)
    panel["week_label"] = panel["paper_year"].astype(str) + "-W" + panel["paper_week"].astype(str).str.zfill(2)
    return panel.sort_values(["week_end", "coin_id"]).reset_index(drop=True)


def build_weekly_panel(config: AcademicExperimentConfig) -> pd.DataFrame:
    return build_weekly_panel_from_series(load_coin_series(config.raw_data_dir), config)


def save_weekly_panel(panel: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(path, index=False)
    return path


def load_weekly_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in ["week_start", "week_end"]:
        frame[column] = pd.to_datetime(frame[column], utc=True, format="mixed")
    return frame
