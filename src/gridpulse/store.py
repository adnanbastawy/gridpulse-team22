"""Reference data: the raw hourly series exported by the notebook (targets, operator forecasts, prices, weather)."""
from __future__ import annotations

import pandas as pd

from .artifacts import Bundle

WEATHER_PREFIX = "wx_"


class ReferenceStore:
    def __init__(self, bundle: Bundle):
        df = pd.read_parquet(bundle.path("reference_data.parquet"))
        df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
        self.df = df.sort_index()
        self.start, self.end = self.df.index.min(), self.df.index.max()

    def actual(self, signal: str) -> pd.Series:
        return self.df[f"{signal}_actual"]

    def tso(self, signal: str) -> pd.Series:
        return self.df[f"{signal}_tso"]

    def weather(self) -> pd.DataFrame:
        return self.df[[c for c in self.df.columns if c.startswith(WEATHER_PREFIX)]]

    def split_of(self, ts: pd.Timestamp) -> str:
        return str(self.df["split"].get(ts, "outside"))

    def contains(self, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        return self.start <= start and end <= self.end
