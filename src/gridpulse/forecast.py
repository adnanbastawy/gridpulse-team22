"""48-hour-ahead forecasting with exact explanations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .artifacts import Bundle
from .features import forecast_features, hourly
from .models import ForecastModel
from .quality import trust_rules
from .store import ReferenceStore

MAX_HORIZON_H = 48


class ForecastError(ValueError):
    """Request cannot be served (bad origin, not enough history, ...)."""


@dataclass
class ForecastResult:
    signal: str
    model: str
    origin: pd.Timestamp
    table: pd.DataFrame                 # index target_time: horizon_h, forecast_mw, actual_mw, tso_mw
    explanations: dict | None
    warnings: list[str]


class ForecastService:
    def __init__(self, bundle: Bundle, store: ReferenceStore):
        cfg = bundle.config
        self.rules, self.site = cfg["trust_rules"], cfg["trust_rules"]["site"]
        self.lags: list[int] = cfg["forecast"]["lags_h"]
        self.min_history_h = int(cfg["forecast"]["min_history_h"])
        self.models = {s: ForecastModel(spec, bundle.root) for s, spec in cfg["forecast"]["models"].items()}
        self.store = store

    def usable(self, raw: pd.Series, signal: str) -> pd.Series:
        """Raw series on an hourly grid with data errors (trust layer) set to NaN."""
        raw = hourly(raw)
        err, _, _ = trust_rules(raw, signal, self.rules)
        return raw.where(~err)

    def predict_hours(self, signal: str, usable: pd.Series, targets: pd.DatetimeIndex) -> pd.Series:
        """Forecast for each target hour using only values at least 48 h older (lags 48-336 h)."""
        model = self.models[signal]
        X = forecast_features(usable, targets, self.lags, self.site, model.features)
        ok = X.notna().all(axis=1).to_numpy()      # same rule as training: complete inputs only
        out = np.full(len(X), np.nan)
        if ok.any():
            out[ok] = model.predict(X[ok])
        return pd.Series(out, index=targets, name="forecast_mw")

    def explain(self, signal: str, usable: pd.Series, targets: pd.DatetimeIndex, top_k: int) -> dict:
        model = self.models[signal]
        X = forecast_features(usable, targets, self.lags, self.site, model.features)
        X = X[X.notna().all(axis=1)]
        if X.empty:
            return {}
        values, base = model.contributions(X)
        out = {}
        for i, ts in enumerate(X.index):
            order = np.argsort(-np.abs(values[i]))[:top_k]
            out[ts] = {"base_value_mw": float(base[i]),
                       "contributions": [{"feature": model.features[j], "value": float(X.iloc[i, j]),
                                          "contribution_mw": float(values[i, j])} for j in order]}
        return out

    def _run(self, signal: str, raw_history: pd.Series, origin: pd.Timestamp, horizon_h: int, explain: bool,
             top_k: int) -> ForecastResult:
        if signal not in self.models:
            raise ForecastError(f"unknown signal '{signal}'")
        if not 1 <= horizon_h <= MAX_HORIZON_H:
            raise ForecastError(f"horizon_hours must be between 1 and {MAX_HORIZON_H}")
        history = raw_history[raw_history.index <= origin]           # nothing after the origin can be used
        if history.empty or (origin - history.index.min()) < pd.Timedelta(hours=self.min_history_h):
            raise ForecastError(f"at least {self.min_history_h} hours of history before the origin are needed")
        usable = self.usable(history, signal)
        targets = pd.date_range(origin + pd.Timedelta(hours=1), periods=horizon_h, freq="h", tz="UTC")
        pred = self.predict_hours(signal, usable, targets)
        table = pd.DataFrame({"horizon_h": np.arange(1, horizon_h + 1), "forecast_mw": pred.to_numpy()}, index=targets)
        warnings = []
        if table["forecast_mw"].isna().any():
            warnings.append(f"{int(table['forecast_mw'].isna().sum())} hour(s) have no forecast: an input value "
                            "48-336 h earlier is missing or flagged as a data error")
        return ForecastResult(signal, self.models[signal].name, origin, table,
                              self.explain(signal, usable, targets, top_k) if explain else None, warnings)

    def forecast(self, signal: str, origin: pd.Timestamp, horizon_h: int = 48, explain: bool = False,
                 top_k: int = 5) -> ForecastResult:
        """Forecast from the reference data, as of `origin` (UTC, whole hour)."""
        if origin < self.store.start or origin > self.store.end:
            raise ForecastError(f"origin must be between {self.store.start} and {self.store.end}")
        res = self._run(signal, self.store.actual(signal), origin, horizon_h, explain, top_k)
        idx = res.table.index
        res.table["actual_mw"] = self.store.actual(signal).reindex(idx).to_numpy()
        res.table["tso_mw"] = self.store.tso(signal).reindex(idx).to_numpy()
        return res

    def forecast_custom(self, signal: str, history: pd.Series, horizon_h: int = 48, explain: bool = False,
                        top_k: int = 5) -> ForecastResult:
        """Forecast the hours after the last timestamp of a user-supplied history."""
        history = hourly(history.dropna() if history.notna().any() else history)
        return self._run(signal, history, history.index.max(), horizon_h, explain, top_k)
