from __future__ import annotations

import pandas as pd

from binance4h_research.market_data import combine_field_matrix, funding_returns_from_events


def test_combine_field_matrix_aligns_symbols_on_open_time() -> None:
    btc = pd.DataFrame(
        {
            "open_time": pd.to_datetime(
                ["2024-01-01 00:00:00+00:00", "2024-01-01 04:00:00+00:00"]
            ),
            "close": [100.0, 101.0],
        }
    )
    eth = pd.DataFrame(
        {
            "open_time": pd.to_datetime(
                ["2024-01-01 04:00:00+00:00", "2024-01-01 08:00:00+00:00"]
            ),
            "close": [200.0, 201.0],
        }
    )

    frame = combine_field_matrix({"ETHUSDT": eth, "BTCUSDT": btc}, "close")

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
