"""Carga y limpieza de datos de tasas de cambio (eltoque.com vía cubanomic).

Reusa la lógica de sample.ipynb (dedup por _id, resample diario ffill).
Salida canónica: DataFrame largo [item_id, timestamp, target] para AutoGluon
TimeSeries.

Multi-moneda (D1): añadir código a CURRENCIES. Nada más cambia.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://api.cubanomic.com/api/v1/x-rates-by-date-range-history"
TOKEN = "CUBANOMIC_TOKEN_REDACTED"
CURRENCIES = ["USD"]  # D1: añadir "ECU" aquí escala a multi-moneda
CACHE_PATH = Path("data/tasas_de_cambio.json")


def fetch_rates(
    currencies: list[str] = CURRENCIES, period: str = "730D"
) -> list[list[dict]]:
    """GET por moneda. Devuelve lista paralela a `currencies` y actualiza cache.

    Lanza excepción si falla requests (no hace fallback aquí).
    """
    all_data: list[list[dict]] = []
    for cur in currencies:
        url = f"{API_URL}?trmi=true&cur={cur}&token={TOKEN}&period={period}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        all_data.append(response.json())
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    return all_data


def load_cache(path: Path = CACHE_PATH) -> list[list[dict]]:
    """Lee cache local. FileNotFoundError si no existe."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_long_dataframe(
    raw: list[list[dict]], currencies: list[str] = CURRENCIES
) -> pd.DataFrame:
    """raw[i] = registros de currencies[i] -> formato largo AutoGluon.

    Dedup por _id, ordena por fecha, resample('D').ffill() del target,
    devuelve solo columnas [item_id, timestamp, target].
    """
    frames: list[pd.DataFrame] = []
    for cur, data in zip(currencies, raw):
        unique = {item["_id"]: item for item in data}
        df = pd.json_normalize(list(unique.values()))
        df["timestamp"] = pd.to_datetime(df["_id"])
        df = df.set_index("timestamp").sort_index()
        # asfreq rellena días faltantes; ffill propaga target.
        # D2: target = median (valor que publica el sitio eltoque.com)
        df = df.asfreq("D").ffill()
        df = df.rename(columns={"median": "target"})
        df["target"] = df["target"].astype(float)
        df["item_id"] = cur
        frames.append(df.reset_index()[["item_id", "timestamp", "target"]])
    out = pd.concat(frames, ignore_index=True)
    return out[["item_id", "timestamp", "target"]]


def load_series(refresh: bool = False) -> pd.DataFrame:
    """Punto de entrada único.

    refresh=True -> fetch + cache; si fetch falla, fallback a cache (warning).
    refresh=False -> cache directo.
    """
    if refresh:
        try:
            raw = fetch_rates()
        except Exception as e:  # noqa: BLE001 - fallback cache
            warnings.warn(f"fetch falló ({e}); usando cache local", stacklevel=2)
            raw = load_cache()
    else:
        raw = load_cache()
    return to_long_dataframe(raw)


if __name__ == "__main__":
    df = load_series()
    assert list(df.columns) == ["item_id", "timestamp", "target"], df.columns
    assert not df["target"].isna().any(), "NaN en target"
    for cur, g in df.groupby("item_id"):
        diffs = g["timestamp"].diff().dropna()
        assert (diffs == pd.Timedelta(days=1)).all(), f"huecos en {cur}"
        print(
            f"{cur}: {len(g)} días | {g['timestamp'].min().date()} → "
            f"{g['timestamp'].max().date()} | target "
            f"min={g['target'].min():.2f} max={g['target'].max():.2f}"
        )
    print("data_loader OK")
