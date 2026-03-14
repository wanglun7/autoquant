from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Iterable

import pandas as pd
import requests


COINGECKO_API_BASE = "https://api.coingecko.com/api/v3"


@dataclass(slots=True)
class CoinGeckoClient:
    base_url: str = COINGECKO_API_BASE
    timeout: int = 30
    max_retries: int = 3

    def _headers(self) -> dict[str, str]:
        api_key = os.getenv("COINGECKO_API_KEY")
        return {"x-cg-demo-api-key": api_key} if api_key else {}

    def _get(self, path: str, params: dict[str, object] | None = None) -> dict | list:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(
                    f"{self.base_url}{path}",
                    params=params,
                    timeout=self.timeout,
                    headers=self._headers(),
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(1.5 * attempt)
        if last_error is None:
            raise RuntimeError("Unexpected missing CoinGecko error state")
        raise last_error

    def coins_list(self, status: str | None = None) -> list[dict]:
        params: dict[str, object] = {"include_platform": "false"}
        if status:
            params["status"] = status
        payload = self._get("/coins/list", params=params)
        return payload if isinstance(payload, list) else []

    def market_chart_range(
        self,
        coin_id: str,
        from_timestamp: int,
        to_timestamp: int,
        vs_currency: str = "usd",
    ) -> dict:
        payload = self._get(
            f"/coins/{coin_id}/market_chart/range",
            params={"vs_currency": vs_currency, "from": from_timestamp, "to": to_timestamp},
        )
        return payload if isinstance(payload, dict) else {}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _to_unix_seconds(date_like: str) -> int:
    return int(pd.Timestamp(date_like, tz="UTC").timestamp())


def _series_frame(points: list[list[float]], value_name: str) -> pd.DataFrame:
    if not points:
        return pd.DataFrame(columns=["date", value_name])
    frame = pd.DataFrame(points, columns=["timestamp_ms", value_name])
    frame["date"] = pd.to_datetime(frame["timestamp_ms"], unit="ms", utc=True).dt.normalize()
    frame = frame.drop(columns=["timestamp_ms"]).groupby("date", as_index=False).last()
    return frame


def _market_chart_to_frame(payload: dict) -> pd.DataFrame:
    prices = _series_frame(payload.get("prices", []), "price")
    market_caps = _series_frame(payload.get("market_caps", []), "market_cap")
    volumes = _series_frame(payload.get("total_volumes", []), "total_volume")
    return prices.merge(market_caps, on="date", how="outer").merge(volumes, on="date", how="outer").sort_values("date").reset_index(drop=True)


def _merge_daily(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        combined = incoming.copy()
    elif incoming.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, incoming], ignore_index=True)
    if combined.empty:
        return combined
    combined["date"] = pd.to_datetime(combined["date"], utc=True, format="mixed")
    return combined.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)


def _read_daily_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, format="mixed")
    return frame


def fetch_coin_market_chart_cache(
    client: CoinGeckoClient,
    data_dir: Path,
    coin_id: str,
    start_date: str,
    end_date: str,
) -> Path:
    raw_dir = data_dir / "daily"
    _ensure_dir(raw_dir)
    path = raw_dir / f"{coin_id}.csv"
    existing = _read_daily_cache(path)
    payload = client.market_chart_range(
        coin_id=coin_id,
        from_timestamp=_to_unix_seconds(start_date),
        to_timestamp=_to_unix_seconds(end_date),
    )
    incoming = _market_chart_to_frame(payload)
    merged = _merge_daily(existing, incoming)
    if not merged.empty:
        merged.to_csv(path, index=False)
    return path


def fetch_coin_list_cache(
    client: CoinGeckoClient,
    data_dir: Path,
    include_inactive: bool = False,
) -> list[Path]:
    meta_dir = data_dir / "metadata"
    _ensure_dir(meta_dir)
    outputs: list[Path] = []
    active_path = meta_dir / "coins_list_active.json"
    active_path.write_text(json.dumps(client.coins_list(), indent=2), encoding="utf-8")
    outputs.append(active_path)
    if include_inactive:
        inactive_path = meta_dir / "coins_list_inactive.json"
        inactive_path.write_text(json.dumps(client.coins_list(status="inactive"), indent=2), encoding="utf-8")
        outputs.append(inactive_path)
    return outputs


def load_coin_series(data_dir: Path) -> dict[str, pd.DataFrame]:
    raw_dir = data_dir / "daily"
    result: dict[str, pd.DataFrame] = {}
    if not raw_dir.exists():
        return result
    for path in sorted(raw_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        frame["date"] = pd.to_datetime(frame["date"], utc=True, format="mixed")
        result[path.stem] = frame.sort_values("date").reset_index(drop=True)
    return result


def resolve_coin_ids(
    data_dir: Path,
    coin_ids: Iterable[str] | None = None,
    coin_ids_file: str | None = None,
    all_listed: bool = False,
    include_inactive: bool = False,
) -> list[str]:
    resolved: list[str] = []
    if coin_ids:
        resolved.extend(str(value).strip() for value in coin_ids if str(value).strip())
    if coin_ids_file:
        resolved.extend(line.strip() for line in Path(coin_ids_file).read_text(encoding="utf-8").splitlines() if line.strip())
    if all_listed:
        meta_dir = data_dir / "metadata"
        filenames = ["coins_list_active.json", "coins_list_inactive.json"] if include_inactive else ["coins_list_active.json"]
        for filename in filenames:
            path = meta_dir / filename
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                resolved.extend(item["id"] for item in payload if "id" in item)
    deduped = sorted(set(resolved))
    if not deduped:
        raise ValueError("No academic coin ids resolved; provide --coin-ids, --coin-ids-file, or --all-listed")
    return deduped


def fetch_academic_data(
    data_dir: Path,
    start_date: str,
    end_date: str,
    coin_ids: Iterable[str],
    include_inactive: bool = False,
) -> list[Path]:
    client = CoinGeckoClient()
    outputs = fetch_coin_list_cache(client, data_dir, include_inactive=include_inactive)
    for coin_id in coin_ids:
        outputs.append(fetch_coin_market_chart_cache(client, data_dir, coin_id=coin_id, start_date=start_date, end_date=end_date))
    return outputs
