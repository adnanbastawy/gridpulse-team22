"""Request/response models (Pydantic v2). Every timestamp is UTC."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Signal = Literal["load", "solar", "wind"]
Severity = Literal["MEDIUM", "HIGH", "CRITICAL"]
MAX_SCAN_DAYS = 400
MIN_CHECK_HOURS = 35 * 24 + 1
MAX_OBSERVATIONS = 20_000


def to_utc(v: datetime) -> datetime:
    return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v.astimezone(timezone.utc)


def whole_hour(v: datetime) -> datetime:
    v = to_utc(v)
    if v.minute or v.second or v.microsecond:
        raise ValueError("timestamps must be whole hours (e.g. 2018-06-04T09:00:00Z)")
    return v


class ForecastRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "signal": "load", "origin_utc": "2018-06-04T09:00:00Z", "horizon_hours": 48, "explain": True, "top_k": 5}})
    signal: Signal
    origin_utc: datetime = Field(description="Forecast is made as of this hour; nothing after it is used.")
    horizon_hours: int = Field(48, ge=1, le=48)
    explain: bool = Field(False, description="Return SHAP contributions per forecast hour.")
    top_k: int = Field(5, ge=1, le=11, description="Number of contributions returned per hour.")

    _hour = field_validator("origin_utc")(whole_hour)


class Observation(BaseModel):
    time_utc: datetime
    value_mw: float | None = Field(None, description="Measured value; null = missing")
    wx_temp: float | None = None
    wx_humidity: float | None = None
    wx_wind_speed: float | None = None
    wx_pressure: float | None = None
    wx_clouds_all: float | None = None

    _hour = field_validator("time_utc")(whole_hour)


class CustomForecastRequest(BaseModel):
    signal: Signal
    history: list[Observation] = Field(min_length=336, max_length=MAX_OBSERVATIONS,
                                       description="Hourly history; at least the last 336 hours before the forecast.")
    horizon_hours: int = Field(48, ge=1, le=48)
    explain: bool = False
    top_k: int = Field(5, ge=1, le=11)


class Contribution(BaseModel):
    feature: str
    value: float
    contribution_mw: float


class Explanation(BaseModel):
    base_value_mw: float
    contributions: list[Contribution]


class ForecastPoint(BaseModel):
    target_time_utc: datetime
    horizon_h: int
    forecast_mw: float | None
    actual_mw: float | None = None
    tso_mw: float | None = None
    explanation: Explanation | None = None


class ForecastResponse(BaseModel):
    signal: Signal
    model: str
    origin_utc: datetime
    points: list[ForecastPoint]
    mae_vs_actual_mw: float | None = Field(None, description="Only when the actual values are known (reference data).")
    warnings: list[str] = []


class ScanRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "start_utc": "2018-05-01T00:00:00Z", "end_utc": "2018-05-31T23:00:00Z", "signals": ["load", "solar", "wind"],
        "min_severity": "MEDIUM", "include_hourly": False}})
    start_utc: datetime
    end_utc: datetime
    signals: list[Signal] = Field(default_factory=lambda: ["load", "solar", "wind"], min_length=1)
    min_severity: Severity = "MEDIUM"
    include_hourly: bool = False

    _hour = field_validator("start_utc", "end_utc")(whole_hour)

    @model_validator(mode="after")
    def _window(self):
        if self.end_utc < self.start_utc:
            raise ValueError("end_utc must not be before start_utc")
        if (self.end_utc - self.start_utc).days > MAX_SCAN_DAYS:
            raise ValueError(f"window longer than {MAX_SCAN_DAYS} days")
        self.signals = list(dict.fromkeys(self.signals))
        return self


class CheckRequest(BaseModel):
    signal: Signal
    observations: list[Observation] = Field(min_length=MIN_CHECK_HOURS, max_length=MAX_OBSERVATIONS,
                                            description="Hourly values; the first 35 days are history, later hours are checked.")
    include_hourly: bool = True


class AnomalyEvent(BaseModel):
    signal: Signal
    start_utc: datetime
    end_utc: datetime
    hours: int
    direction: Literal["above_expected", "below_expected"]
    peak_dev_nowcast_MW: float | None
    peak_dev_forecast_MW: float | None
    peak_ratio_to_band: float | None
    severity: Severity
    evidence: str
    other_signals_alerting: str = ""
    trust_layer_reasons: str = ""


class HourlyFlag(BaseModel):
    time_utc: datetime
    actual_mw: float | None
    nowcast_mw: float | None
    forecast_mw: float | None
    data_error: bool
    suspected: bool
    reasons: str
    rules: bool
    band_forecast: bool
    band_nowcast: bool
    combined: bool


class SignalSummary(BaseModel):
    hours_checked: int
    flagged_hours: int
    events: dict[str, int]


class AnomalyResponse(BaseModel):
    start_utc: datetime
    end_utc: datetime
    events: list[AnomalyEvent]
    summary: dict[str, SignalSummary]
    hourly: dict[str, list[HourlyFlag]] | None = None
    warnings: list[str] = []


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    dataset_version: str
    artifact_files_checked: int
    models: dict[str, str]
    data_range_utc: tuple[datetime, datetime]
