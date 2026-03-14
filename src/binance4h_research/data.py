from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Iterable

import pandas as pd
import requests


BINANCE_FAPI_BASE = "https://fapi.binance.com"
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]
INTERVAL_TO_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


@dataclass(slots=True)
class BinanceFuturesClient:
    base_url: str = BINANCE_FAPI_BASE
    timeout: int = 20
    max_retries: int = 3

    def _get(self, path: str, params: dict[str, object] | None = None) -> list[dict] | dict:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(1.5 * attempt)
        if last_error is None:
            raise RuntimeError("Unexpected missing request error state")
        raise last_error

    def exchange_info(self) -> dict:
        return self._get("/fapi/v1/exchangeInfo")  # type: ignore[return-value]

    def active_usdt_perpetual_symbols(self) -> list[str]:
        info = self.exchange_info()
        symbols = []
        for item in info.get("symbols", []):
            if (
                item.get("contractType") == "PERPETUAL"
                and item.get("status") == "TRADING"
                and item.get("quoteAsset") == "USDT"
            ):
                symbols.append(item["symbol"])
        return sorted(symbols)

    def klines(
        self,
        symbol: str,
        interval: str = "4h",
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 1500,
    ) -> pd.DataFrame:
        rows = self._get(
            "/fapi/v1/klines",
            {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_time,
                "endTime": end_time,
                "limit": limit,
            },
        )
        frame = pd.DataFrame(rows, columns=KLINE_COLUMNS)
        if frame.empty:
            return frame
        numeric_cols = [col for col in KLINE_COLUMNS if col not in {"open_time", "close_time"}]
        for col in numeric_cols:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
        frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
        return frame.drop(columns=["ignore"])

    def funding_rates(
        self,
        symbol: str,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        rows = self._get(
            "/fapi/v1/fundingRate",
            {
                "symbol": symbol,
                "startTime": start_time,
                "endTime": end_time,
                "limit": limit,
            },
        )
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        frame["fundingRate"] = pd.to_numeric(frame["fundingRate"], errors="coerce")
        frame["fundingTime"] = pd.to_datetime(frame["fundingTime"], unit="ms", utc=True)
        if "markPrice" in frame.columns:
            frame["markPrice"] = pd.to_numeric(frame["markPrice"], errors="coerce")
        return frame


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _to_millis(timestamp: str | pd.Timestamp | None) -> int | None:
    if timestamp is None:
        return None
    if isinstance(timestamp, pd.Timestamp):
        ts = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    else:
        ts = pd.Timestamp(timestamp)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def _merge_deduplicate(existing: pd.DataFrame, incoming: pd.DataFrame, subset: str) -> pd.DataFrame:
    if existing.empty:
        combined = incoming.copy()
    elif incoming.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, incoming], ignore_index=True)
    if combined.empty:
        return combined
    if subset in {"open_time", "close_time", "fundingTime"}:
        combined[subset] = pd.to_datetime(combined[subset], utc=True, format="mixed")
    return combined.drop_duplicates(subset=[subset]).sort_values(subset).reset_index(drop=True)


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=[0])


def fetch_klines_range(
    client: BinanceFuturesClient,
    symbol: str,
    interval: str = "4h",
    start_time: int | None = None,
    end_time: int | None = None,
    limit: int = 1500,
) -> pd.DataFrame:
    step_ms = INTERVAL_TO_MS[interval]
    cursor = start_time
    chunks: list[pd.DataFrame] = []

    while True:
        frame = client.klines(symbol=symbol, interval=interval, start_time=cursor, end_time=end_time, limit=limit)
        if frame.empty:
            break
        chunks.append(frame)
        if len(frame) < limit:
            break
        last_open = int(frame["open_time"].iloc[-1].timestamp() * 1000)
        next_cursor = last_open + step_ms
        if cursor is not None and next_cursor <= cursor:
            break
        if end_time is not None and next_cursor > end_time:
            break
        cursor = next_cursor

    if not chunks:
        return pd.DataFrame(columns=[col for col in KLINE_COLUMNS if col != "ignore"])
    combined = pd.concat(chunks, ignore_index=True)
    return combined.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)


def fetch_funding_range(
    client: BinanceFuturesClient,
    symbol: str,
    start_time: int | None = None,
    end_time: int | None = None,
    limit: int = 1000,
) -> pd.DataFrame:
    cursor = start_time
    chunks: list[pd.DataFrame] = []

    while True:
        frame = client.funding_rates(symbol=symbol, start_time=cursor, end_time=end_time, limit=limit)
        if frame.empty:
            break
        chunks.append(frame)
        if len(frame) < limit:
            break
        last_time = int(frame["fundingTime"].iloc[-1].timestamp() * 1000)
        next_cursor = last_time + 1
        if cursor is not None and next_cursor <= cursor:
            break
        if end_time is not None and next_cursor > end_time:
            break
        cursor = next_cursor

    if not chunks:
        return pd.DataFrame(columns=["symbol", "fundingTime", "fundingRate", "markPrice"])
    combined = pd.concat(chunks, ignore_index=True)
    return combined.drop_duplicates(subset=["fundingTime"]).sort_values("fundingTime").reset_index(drop=True)


def update_klines_cache(
    client: BinanceFuturesClient,
    data_dir: Path,
    symbol: str,
    interval: str = "4h",
    limit: int = 1500,
    start_time: int | None = None,
    end_time: int | None = None,
) -> Path:
    kline_dir = data_dir / "klines"
    _ensure_dir(kline_dir)
    path = kline_dir / f"{symbol}_{interval}.csv"
    existing = _read_csv_if_exists(path)
    incoming_frames: list[pd.DataFrame] = []

    if start_time is not None or end_time is not None:
        incoming_frames.append(
            fetch_klines_range(
                client=client,
                symbol=symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
            )
        )
    elif not existing.empty:
        latest = pd.to_datetime(existing["open_time"].iloc[-1], utc=True)
        incremental_start = int(latest.timestamp() * 1000) + 1
        incoming_frames.append(
            fetch_klines_range(
                client=client,
                symbol=symbol,
                interval=interval,
                start_time=incremental_start,
                end_time=end_time,
                limit=limit,
            )
        )
    else:
        incoming_frames.append(
            fetch_klines_range(client=client, symbol=symbol, interval=interval, start_time=None, end_time=end_time, limit=limit)
        )

    incoming = pd.concat(incoming_frames, ignore_index=True) if incoming_frames else pd.DataFrame()
    merged = _merge_deduplicate(existing, incoming, "open_time")
    if not merged.empty:
        merged.to_csv(path, index=False)
    return path


def update_funding_cache(
    client: BinanceFuturesClient,
    data_dir: Path,
    symbol: str,
    limit: int = 1000,
    start_time: int | None = None,
    end_time: int | None = None,
) -> Path:
    funding_dir = data_dir / "funding"
    _ensure_dir(funding_dir)
    path = funding_dir / f"{symbol}.csv"
    existing = _read_csv_if_exists(path)
    incoming_frames: list[pd.DataFrame] = []

    if start_time is not None or end_time is not None:
        incoming_frames.append(fetch_funding_range(client=client, symbol=symbol, start_time=start_time, end_time=end_time, limit=limit))
    elif not existing.empty:
        latest = pd.to_datetime(existing["fundingTime"].iloc[-1], utc=True)
        incremental_start = int(latest.timestamp() * 1000) + 1
        incoming_frames.append(
            fetch_funding_range(client=client, symbol=symbol, start_time=incremental_start, end_time=end_time, limit=limit)
        )
    else:
        incoming_frames.append(fetch_funding_range(client=client, symbol=symbol, start_time=None, end_time=end_time, limit=limit))

    incoming = pd.concat(incoming_frames, ignore_index=True) if incoming_frames else pd.DataFrame()
    merged = _merge_deduplicate(existing, incoming, "fundingTime")
    if not merged.empty:
        merged.to_csv(path, index=False)
    return path


def update_exchange_info_cache(client: BinanceFuturesClient, data_dir: Path) -> Path:
    _ensure_dir(data_dir)
    path = data_dir / "exchange_info.json"
    path.write_text(json.dumps(client.exchange_info(), indent=2), encoding="utf-8")
    return path


def fetch_all(
    data_dir: Path,
    symbols: Iterable[str] | None = None,
    interval: str = "4h",
    start_time: int | None = None,
    end_time: int | None = None,
) -> list[Path]:
    client = BinanceFuturesClient()
    symbols = list(symbols or client.active_usdt_perpetual_symbols())
    paths: list[Path] = [update_exchange_info_cache(client, data_dir)]
    for symbol in symbols:
        paths.append(
            update_klines_cache(
                client,
                data_dir,
                symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time,
            )
        )
        paths.append(update_funding_cache(client, data_dir, symbol, start_time=start_time, end_time=end_time))
    return paths


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


def load_symbol_funding(data_dir: Path) -> dict[str, pd.DataFrame]:
    folder = data_dir / "funding"
    result: dict[str, pd.DataFrame] = {}
    if not folder.exists():
        return result
    for path in sorted(folder.glob("*.csv")):
        symbol = path.stem
        frame = pd.read_csv(path)
        frame["fundingTime"] = pd.to_datetime(frame["fundingTime"], utc=True, format="mixed")
        result[symbol] = frame.sort_values("fundingTime").reset_index(drop=True)
    return result
