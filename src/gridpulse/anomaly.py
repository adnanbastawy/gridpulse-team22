"""Anomaly detection: trust rules + nowcast band + forecast band (notebook Part 4)."""
from __future__ import annotations

from dataclasses import dataclass, field

import holidays
import numpy as np
import pandas as pd

from .artifacts import Bundle
from .features import hourly, nowcast_features
from .forecast import ForecastService
from .models import NowcastModel
from .quality import reason_text, trust_rules
from .store import ReferenceStore

EXPECTATIONS = ("forecast", "nowcast")
SEVERITY_RANK = {"MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


class AnomalyError(ValueError):
    """Request cannot be served (window outside the data, not enough history, ...)."""


@dataclass
class ScanResult:
    hourly: dict[str, pd.DataFrame]          # per signal: actual, expectations, residuals, flags, reasons
    events: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


class AnomalyService:
    def __init__(self, bundle: Bundle, store: ReferenceStore, forecaster: ForecastService):
        cfg = bundle.config
        a = cfg["anomaly"]
        self.cfg, self.rules, self.store, self.forecaster = a, cfg["trust_rules"], store, forecaster
        self.site = self.rules["site"]
        self.tz = a["local_tz"]
        self.history = pd.Timedelta(days=int(cfg["nowcast"]["history_days"]))
        self.lags = cfg["nowcast"]["lags_h"]
        self.nowcast = {s: NowcastModel(bundle.path(p)) for s, p in cfg["nowcast"]["files"].items()}
        self.bands = {}
        for key, records in a["bands"].items():
            s, e = key.split("/")
            self.bands[(s, e)] = pd.DataFrame(records).set_index("hour")   # local hour of day
        self.tso_far = a["tso_far_threshold_mw"]
        self.holidays = holidays.Spain(years=range(2014, 2031))

    # ------------------------------------------------------------------ building blocks
    def local_hour(self, idx: pd.DatetimeIndex) -> np.ndarray:
        return idx.tz_convert(self.tz).hour.to_numpy()

    def frames(self, signal: str, raw: pd.Series, weather: pd.DataFrame, check: pd.DatetimeIndex) -> dict:
        """Expectation frames on the checked hours. `raw` must include the history before `check`."""
        raw = hourly(raw)
        err, sus, reasons = trust_rules(raw, signal, self.rules)
        usable = raw.where(~err)
        model = self.nowcast[signal]
        X = nowcast_features(usable, weather, self.site, self.lags, model.features)
        now = pd.Series(model.predict(X), index=raw.index)
        fc = self.forecaster.predict_hours(signal, usable, check).where(~err.reindex(check))   # never compare an error value
        hour = self.local_hour(check)
        actual = raw.reindex(check)
        out = {}
        for exp, pred in (("forecast", fc), ("nowcast", now.reindex(check))):
            f = pd.DataFrame({"actual": actual, "pred": pred.to_numpy()}, index=check)
            f["resid"] = f["actual"] - f["pred"]
            f["hour"] = hour
            out[exp] = f
        out["rules"] = (err.reindex(check), sus.reindex(check), reason_text(reasons.reindex(check)))
        return out

    def band(self, signal: str, exp: str, hours: np.ndarray) -> pd.DataFrame:
        return self.bands[(signal, exp)].reindex(hours)

    def flags(self, signal: str, fr: dict) -> pd.DataFrame:
        err, sus, _ = fr["rules"]
        out = pd.DataFrame(index=fr["nowcast"].index)
        out["rules"] = (err | sus).to_numpy()
        for exp in EXPECTATIONS:
            f = fr[exp]
            b = self.band(signal, exp, f["hour"].to_numpy())
            r = f["resid"].to_numpy()
            out[f"band_{exp}"] = ((r < b["lo"].to_numpy()) | (r > b["hi"].to_numpy())) & f["resid"].notna().to_numpy()
        out["combined"] = err.to_numpy() | out["band_nowcast"] | out["band_forecast"]
        return out

    def ratio(self, signal: str, f: pd.DataFrame, exp: str) -> pd.Series:
        b = self.band(signal, exp, f["hour"].to_numpy()).set_index(f.index)
        return f["resid"] / np.where(f["resid"] >= b["center"], b["hi"], -b["lo"].abs())

    def events(self, signal: str, fr: dict, flag: pd.Series) -> pd.DataFrame:
        """Merge flagged hours into events with size, direction and severity (same rules as the notebook)."""
        t = pd.Series(flag.index[flag.to_numpy()])
        if t.empty:
            return pd.DataFrame()
        gid = (t.diff() > pd.Timedelta(hours=self.cfg["merge_gap_h"] + 1)).cumsum()
        ratio = pd.concat([self.ratio(signal, fr[e], e) for e in EXPECTATIONS], axis=1).max(axis=1)
        rows = []
        for _, ts in t.groupby(gid.to_numpy()):
            w = slice(ts.min(), ts.max())
            rn, rf = fr["nowcast"].loc[w, "resid"], fr["forecast"].loc[w, "resid"]
            main = rn if rn.notna().any() else rf
            i = main.abs().idxmax() if main.notna().any() else ts.min()
            rows.append({"signal": signal, "start_utc": ts.min(), "end_utc": ts.max(),
                         "hours": int((ts.max() - ts.min()) / pd.Timedelta(hours=1)) + 1,
                         "direction": "above_expected" if main.get(i, 0) > 0 else "below_expected",
                         "peak_dev_nowcast_MW": round(float(rn.abs().max()), 1) if rn.notna().any() else None,
                         "peak_dev_forecast_MW": round(float(rf.abs().max()), 1) if rf.notna().any() else None,
                         "peak_ratio_to_band": round(float(ratio.loc[w].max()), 2) if ratio.loc[w].notna().any() else None})
        ev = pd.DataFrame(rows)
        crit = self.cfg["critical"]
        long_ = ev["hours"] >= crit["min_hours"]
        far = ev["peak_ratio_to_band"].astype(float) >= crit["min_ratio"]
        ev["severity"] = np.where(long_ & far, "CRITICAL", np.where(long_ | far, "HIGH", "MEDIUM"))
        return ev

    def evidence(self, ev: pd.DataFrame, fr_all: dict, flags_all: dict, tso: dict | None) -> pd.DataFrame:
        """One evidence tag per event, strongest first (same order as the notebook)."""
        if ev.empty:
            return ev
        alert_hours = {s: set(fl.index[fl["combined"]]) for s, fl in flags_all.items()}
        tags, others, reasons = [], [], []
        for _, e in ev.iterrows():
            s, w = e["signal"], slice(e["start_utc"], e["end_utc"])
            err, _, rtext = fr_all[s]["rules"]
            hours = set(pd.date_range(e["start_utc"], e["end_utc"], freq="h"))
            other = ";".join(o for o in alert_hours if o != s and hours & alert_hours[o])
            hol = s == "load" and any(d in self.holidays for d in pd.DatetimeIndex(sorted(hours)).tz_convert(self.tz).date)
            tso_far = False
            if tso is not None and s in tso:
                tso_err = (fr_all[s]["nowcast"].loc[w, "actual"] - tso[s].reindex(fr_all[s]["nowcast"].loc[w].index)).abs()
                tso_far = bool((tso_err > self.tso_far[s]).any())
            if err.loc[w].any():
                tag = "data_quality_rule"
            elif other:
                tag = "simultaneous_alert_other_signal"
            elif hol:
                tag = "public_holiday"
            elif tso_far:
                tag = "operator_forecast_also_far_off"
            else:
                tag = "unexplained_review"
            tags.append(tag)
            others.append(other)
            reasons.append(";".join(sorted({r for x in rtext.loc[w] for r in x.split(";") if r})))
        ev = ev.copy()
        ev["other_signals_alerting"], ev["evidence"], ev["trust_layer_reasons"] = others, tags, reasons
        return ev

    @staticmethod
    def hourly_table(fr: dict, flags: pd.DataFrame) -> pd.DataFrame:
        err, sus, rtext = fr["rules"]
        t = pd.DataFrame({"actual_mw": fr["nowcast"]["actual"],
                          "nowcast_mw": fr["nowcast"]["pred"], "forecast_mw": fr["forecast"]["pred"],
                          "data_error": err, "suspected": sus, "reasons": rtext}, index=flags.index)
        return t.join(flags)

    # ------------------------------------------------------------------ public API
    def scan(self, start: pd.Timestamp, end: pd.Timestamp, signals: tuple[str, ...] = ("load", "solar", "wind"),
             min_severity: str = "MEDIUM") -> ScanResult:
        """Run the detectors on the reference data between start and end (UTC, inclusive)."""
        if end < start:
            raise AnomalyError("end must be after start")
        first = self.store.start + self.history
        if start < first or end > self.store.end:
            raise AnomalyError(f"window must be between {first} and {self.store.end} (35 days of history are needed)")
        check = pd.date_range(start, end, freq="h", tz="UTC")
        weather = self.store.weather()
        all_signals = ("load", "solar", "wind")                  # cross-signal evidence needs all three
        fr_all, flags_all = {}, {}
        for s in all_signals:
            raw = self.store.actual(s).loc[start - self.history: end]
            fr_all[s] = self.frames(s, raw, weather, check)
            flags_all[s] = self.flags(s, fr_all[s])
        evs = [self.events(s, fr_all[s], flags_all[s]["combined"]) for s in signals]
        ev = pd.concat([e for e in evs if not e.empty], ignore_index=True) if any(not e.empty for e in evs) else pd.DataFrame()
        ev = self.evidence(ev, fr_all, flags_all, {s: self.store.tso(s) for s in all_signals})
        if not ev.empty:
            ev = ev[ev["severity"].map(SEVERITY_RANK) >= SEVERITY_RANK[min_severity]].reset_index(drop=True)
        return ScanResult({s: self.hourly_table(fr_all[s], flags_all[s]) for s in signals}, ev)

    def check(self, signal: str, values: pd.Series, weather: pd.DataFrame | None = None) -> ScanResult:
        """Run the detectors on user data. The first 35 days are history; later hours are checked."""
        values = hourly(values)
        warnings = []
        if weather is None or weather.dropna(how="all").empty:
            ref_w = self.store.weather()
            weather = ref_w.reindex(values.index)
            if weather.notna().any().any():
                warnings.append("no weather columns supplied: weather taken from the reference data for matching hours")
            else:
                warnings.append("no weather available for these hours: nowcast is less accurate, expect more alerts")
        start = values.index.min() + self.history
        if start > values.index.max():
            raise AnomalyError("at least 35 days of hourly data are needed (history) plus the hours to check")
        check = pd.date_range(start, values.index.max(), freq="h", tz="UTC")
        fr = self.frames(signal, values, weather, check)
        fl = self.flags(signal, fr)
        ev = self.evidence(self.events(signal, fr, fl["combined"]), {signal: fr}, {signal: fl}, None)
        return ScanResult({signal: self.hourly_table(fr, fl)}, ev, warnings)
