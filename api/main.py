"""GridPulse REST API: 48-hour forecasts with explanations, and anomaly detection."""
from __future__ import annotations

import io
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse

from gridpulse import SIGNALS, __version__
from gridpulse.anomaly import SEVERITY_RANK, AnomalyError, AnomalyService, ScanResult
from gridpulse.artifacts import ArtifactError, load_bundle
from gridpulse.forecast import ForecastError, ForecastResult, ForecastService
from gridpulse.store import ReferenceStore

from .schemas import (AnomalyEvent, AnomalyResponse, CheckRequest, CustomForecastRequest, ForecastPoint,
                      ForecastRequest, ForecastResponse, HealthResponse, HourlyFlag, Observation, ScanRequest,
                      Severity, Signal, SignalSummary, MAX_OBSERVATIONS, MIN_CHECK_HOURS)

log = logging.getLogger("gridpulse.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
WEATHER_COLS = ["wx_temp", "wx_humidity", "wx_wind_speed", "wx_pressure", "wx_clouds_all"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = load_bundle(verify=True)                       # refuses to start on missing / altered artifacts
    store = ReferenceStore(bundle)
    forecaster = ForecastService(bundle, store)
    app.state.bundle, app.state.store, app.state.forecaster = bundle, store, forecaster
    app.state.anomaly = AnomalyService(bundle, store, forecaster)
    app.state.check = bundle.verify()
    alerts = pd.read_csv(bundle.path("alerts_test_2018.csv"), parse_dates=["start_utc", "end_utc"])
    app.state.alerts_2018 = alerts
    log.info("ready: dataset %s, models %s", bundle.manifest["dataset_version"],
             {s: m.name for s, m in forecaster.models.items()})
    yield


app = FastAPI(
    title="GridPulse API",
    version=__version__,
    description="48-hour-ahead forecasts (load, solar, wind) with SHAP explanations, and anomaly detection "
                "(trust rules + nowcast band + forecast band) for the Spanish grid, 2015-2018 data.",
    lifespan=lifespan,
)


@app.middleware("http")
async def timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-ms"] = f"{ms:.1f}"
    log.info("%s %s -> %s in %.0f ms", request.method, request.url.path, response.status_code, ms)
    return response


@app.exception_handler(ForecastError)
@app.exception_handler(AnomalyError)
async def domain_error(_: Request, exc: Exception):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# ------------------------------------------------------------------ helpers
def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _num(x) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x) else float(x)


def _forecast_response(res: ForecastResult) -> ForecastResponse:
    points = []
    for ts, row in res.table.iterrows():
        exp = (res.explanations or {}).get(ts)
        points.append(ForecastPoint(target_time_utc=ts.to_pydatetime(), horizon_h=int(row["horizon_h"]),
                                    forecast_mw=_num(row["forecast_mw"]), actual_mw=_num(row.get("actual_mw")),
                                    tso_mw=_num(row.get("tso_mw")), explanation=exp))
    mae = None
    if "actual_mw" in res.table:
        ok = res.table[["forecast_mw", "actual_mw"]].dropna()
        mae = float((ok["forecast_mw"] - ok["actual_mw"]).abs().mean()) if len(ok) else None
    return ForecastResponse(signal=res.signal, model=res.model, origin_utc=res.origin.to_pydatetime(),
                            points=points, mae_vs_actual_mw=mae, warnings=res.warnings)


def _events(ev: pd.DataFrame) -> list[AnomalyEvent]:
    if ev.empty:
        return []
    ev = ev.replace({np.nan: None})
    return [AnomalyEvent(**{k: (v.to_pydatetime() if isinstance(v, pd.Timestamp) else v) for k, v in r.items()
                            if k in AnomalyEvent.model_fields}) for r in ev.to_dict(orient="records")]


def _anomaly_response(res: ScanResult, start, end, include_hourly: bool) -> AnomalyResponse:
    summary = {}
    for s, h in res.hourly.items():
        counts = res.events[res.events["signal"] == s]["severity"].value_counts().to_dict() if not res.events.empty else {}
        summary[s] = SignalSummary(hours_checked=int(len(h)), flagged_hours=int(h["combined"].sum()),
                                   events={k: int(counts.get(k, 0)) for k in SEVERITY_RANK})
    hourly = None
    if include_hourly:
        hourly = {s: [HourlyFlag(time_utc=ts.to_pydatetime(), actual_mw=_num(r["actual_mw"]), nowcast_mw=_num(r["nowcast_mw"]),
                                 forecast_mw=_num(r["forecast_mw"]), data_error=bool(r["data_error"]),
                                 suspected=bool(r["suspected"]), reasons=str(r["reasons"]), rules=bool(r["rules"]),
                                 band_forecast=bool(r["band_forecast"]), band_nowcast=bool(r["band_nowcast"]),
                                 combined=bool(r["combined"])) for ts, r in h.iterrows()] for s, h in res.hourly.items()}
    return AnomalyResponse(start_utc=start, end_utc=end, events=_events(res.events), summary=summary,
                           hourly=hourly, warnings=res.warnings)


def _observations_frame(obs: list[Observation]) -> tuple[pd.Series, pd.DataFrame | None]:
    df = pd.DataFrame([o.model_dump() for o in obs])
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df = df.drop_duplicates("time_utc", keep="last").set_index("time_utc").sort_index()
    weather = df[WEATHER_COLS] if df[WEATHER_COLS].notna().any().any() else None
    return df["value_mw"].astype(float), weather


# ------------------------------------------------------------------ endpoints
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health(request: Request):
    st = request.app.state
    return HealthResponse(status="ok", version=__version__, dataset_version=st.bundle.manifest["dataset_version"],
                          artifact_files_checked=st.check["files_checked"],
                          models={s: m.name for s, m in st.forecaster.models.items()},
                          data_range_utc=(st.store.start.to_pydatetime(), st.store.end.to_pydatetime()))


@app.get("/info", tags=["service"])
def info(request: Request):
    """Model card: production models, metrics, tuning, explanations, anomaly policy and evaluation."""
    cfg = request.app.state.bundle.config
    return {"dataset_version": cfg["dataset_version"], "provenance": cfg["provenance"], "splits": cfg["splits"],
            "forecast": cfg["forecast"], "production": cfg["metrics"]["production"], "tuning": cfg["metrics"]["tuning"],
            "shap_mean_abs_test": cfg["metrics"]["shap_mean_abs_test"],
            "anomaly_policy": {k: cfg["anomaly"][k] for k in ("policy", "alert_budget", "merge_gap_h", "critical")},
            "anomaly_evaluation": cfg["anomaly_evaluation"], "trained_with": request.app.state.bundle.manifest["trained_with"]}


@app.get("/metrics/models", tags=["service"])
def model_metrics(request: Request, split: str = Query("test", pattern="^(validation|test)$")):
    """Every compared model (baselines, 8 ML models, tuned LightGBM, ensemble) on validation or test."""
    return request.app.state.bundle.config["metrics"][split]


@app.get("/business", tags=["service"])
def business(request: Request):
    """Cost of forecast errors (proxy) and alert workload, from notebook Part 5."""
    return request.app.state.bundle.config["business"]


@app.post("/forecast", response_model=ForecastResponse, tags=["forecast"])
def forecast(req: ForecastRequest, request: Request):
    """48-hour-ahead forecast as of `origin_utc`, from the reference data (2015-2018)."""
    res = request.app.state.forecaster.forecast(req.signal, _utc(req.origin_utc), req.horizon_hours,
                                                req.explain, req.top_k)
    return _forecast_response(res)


@app.post("/forecast/custom", response_model=ForecastResponse, tags=["forecast"])
def forecast_custom(req: CustomForecastRequest, request: Request):
    """Forecast the hours after the last timestamp of your own hourly history (>= 336 hours)."""
    values, _ = _observations_frame(req.history)
    res = request.app.state.forecaster.forecast_custom(req.signal, values, req.horizon_hours, req.explain, req.top_k)
    return _forecast_response(res)


@app.post("/anomaly/scan", response_model=AnomalyResponse, tags=["anomaly"])
def anomaly_scan(req: ScanRequest, request: Request):
    """Run the detectors on the reference data in a window (needs 35 days of data before the window)."""
    res = request.app.state.anomaly.scan(_utc(req.start_utc), _utc(req.end_utc), tuple(req.signals),
                                         req.min_severity)
    return _anomaly_response(res, req.start_utc, req.end_utc, req.include_hourly)


@app.post("/anomaly/check", response_model=AnomalyResponse, tags=["anomaly"])
def anomaly_check(req: CheckRequest, request: Request):
    """Run the detectors on your own hourly data (JSON). The first 35 days are used as history."""
    values, weather = _observations_frame(req.observations)
    res = request.app.state.anomaly.check(req.signal, values, weather)
    h = res.hourly[req.signal]
    return _anomaly_response(res, h.index.min().to_pydatetime(), h.index.max().to_pydatetime(), req.include_hourly)


@app.post("/anomaly/upload", response_model=AnomalyResponse, tags=["anomaly"])
async def anomaly_upload(request: Request, signal: Signal = Form(...), file: UploadFile = File(...),
                         include_hourly: bool = Form(True)):
    """Same as /anomaly/check with a CSV file: columns `time_utc,value_mw` and optional `wx_*` weather columns."""
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "file larger than 5 MB")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001 - any parse error is a client error
        raise HTTPException(400, f"could not read CSV: {exc}") from exc
    missing = {"time_utc", "value_mw"} - set(df.columns)
    if missing:
        raise HTTPException(400, f"missing column(s): {sorted(missing)}")
    if not MIN_CHECK_HOURS <= len(df) <= MAX_OBSERVATIONS:
        raise HTTPException(400, f"the file needs between {MIN_CHECK_HOURS} and {MAX_OBSERVATIONS} rows")
    try:
        obs = [Observation(**{k: (None if pd.isna(v) else v) for k, v in r.items() if k in Observation.model_fields})
               for r in df.to_dict(orient="records")]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, f"invalid rows: {exc}") from exc
    req = CheckRequest(signal=signal, observations=obs, include_hourly=include_hourly)
    return anomaly_check(req, request)


@app.get("/alerts/2018", response_model=list[AnomalyEvent], tags=["anomaly"])
def alerts_2018(request: Request, signal: Signal | None = None, min_severity: Severity = "MEDIUM",
                start_utc: datetime | None = None, end_utc: datetime | None = None, limit: int = Query(500, ge=1, le=5000)):
    """Alerts of the 2018 test year as produced by the notebook (evidence included)."""
    a = request.app.state.alerts_2018
    a = a[a["severity"].map(SEVERITY_RANK) >= SEVERITY_RANK[min_severity]]
    if signal:
        a = a[a["signal"] == signal]
    if start_utc:
        a = a[a["end_utc"] >= _utc(start_utc)]
    if end_utc:
        a = a[a["start_utc"] <= _utc(end_utc)]
    return _events(a.head(limit).fillna({"other_signals_alerting": "", "trust_layer_reasons": ""}))


@app.get("/data/series", tags=["data"])
def series(request: Request, signal: Signal, start_utc: datetime, end_utc: datetime):
    """Actual values and the operator forecast for charts (max 62 days)."""
    st = request.app.state.store
    start, end = _utc(start_utc), _utc(end_utc)
    if end < start or (end - start).days > 62:
        raise HTTPException(400, "window must be 0-62 days")
    idx = st.df.loc[start:end].index
    return {"time_utc": [t.isoformat() for t in idx],
            "actual_mw": [_num(v) for v in st.actual(signal).loc[idx]],
            "tso_mw": [_num(v) for v in st.tso(signal).loc[idx]]}
