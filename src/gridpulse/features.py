"""Feature builders - identical to notebook 1.8 / 3.2 (forecast) and 4.2 (nowcast)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .quality import solar_elevation_deg


def hourly(s: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Put a series on a complete hourly UTC grid (missing hours become NaN)."""
    if s.index.tz is None:
        raise ValueError("timestamps must be timezone-aware (UTC)")
    s = s.sort_index()
    s.index = s.index.tz_convert("UTC")
    grid = pd.date_range(s.index.min(), s.index.max(), freq="h", tz="UTC")
    return s[~s.index.duplicated(keep="last")].reindex(grid)


def calendar_features(index: pd.DatetimeIndex, site: dict) -> pd.DataFrame:
    """UTC calendar + sun elevation at the target hour (all known in advance)."""
    hour, month, dow = index.hour.to_numpy(), index.month.to_numpy(), index.dayofweek.to_numpy()
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "month_sin": np.sin(2 * np.pi * (month - 1) / 12), "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
        "day_of_week": dow, "is_weekend": (dow >= 5).astype(int),
        "sun_elevation_deg": solar_elevation_deg(index, site["lat"], site["lon"]).round(2).to_numpy(),
    }, index=index)


def forecast_features(usable: pd.Series, targets: pd.DatetimeIndex, lags: list[int], site: dict,
                      order: list[str]) -> pd.DataFrame:
    """Features for target hours: calendar at the target + target values `lag` hours earlier (lag >= 48)."""
    X = calendar_features(targets, site)
    for lag in lags:
        X[f"lag_{lag}h"] = usable.reindex(targets - pd.Timedelta(hours=lag)).to_numpy()
    return X[order]


def nowcast_features(usable: pd.Series, weather: pd.DataFrame, site: dict, lags: list[int],
                     order: list[str]) -> pd.DataFrame:
    """Nowcast inputs for every hour of `usable` (hourly grid): values up to t-1, weather at t, calendar at t."""
    X = pd.DataFrame({f"lag_{L}h": usable.shift(L) for L in lags}, index=usable.index)
    X["diff_1_2"] = X["lag_1h"] - X["lag_2h"]
    X["mean_24h"] = usable.shift(1).rolling(24, min_periods=12).mean()
    X = X.join(weather.reindex(usable.index)).join(calendar_features(usable.index, site))
    return X.reindex(columns=order)
