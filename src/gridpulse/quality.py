"""Data-quality rules (trust layer) - identical to notebook section 1.5 / 4.1."""
from __future__ import annotations

import numpy as np
import pandas as pd


def solar_elevation_deg(index: pd.DatetimeIndex, lat: float, lon: float) -> pd.Series:
    """Approximate sun height (degrees) at mid-hour, NOAA formulas. Known in advance, so not leakage."""
    t = index + pd.Timedelta(minutes=30)
    doy = np.asarray(t.dayofyear, dtype=float)
    hour = np.asarray(t.hour, dtype=float) + np.asarray(t.minute, dtype=float) / 60
    g = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    ha = np.deg2rad((hour * 60 + eqtime + 4 * lon) / 4 - 180)
    la = np.deg2rad(lat)
    cos_zenith = np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)
    return pd.Series(90 - np.degrees(np.arccos(np.clip(cos_zenith, -1, 1))), index=index)


def flatline(s: pd.Series, n: int) -> pd.Series:
    """True where the same value repeats for at least n consecutive hours."""
    group = (s != s.shift()).cumsum()
    return (s.groupby(group).transform("size") >= n) & s.notna()


def trust_rules(s: pd.Series, signal: str, rules: dict) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Apply the trust-layer rules to an hourly UTC series.

    Returns (data_error, suspected, reasons) where `reasons` has one boolean column per rule.
    """
    elev = solar_elevation_deg(s.index, rules["site"]["lat"], rules["site"]["lon"])
    lo, hi = rules["ranges_mw"][signal]
    scope = (elev > rules["solar_flatline_min_elevation_deg"]) if signal == "solar" else pd.Series(True, index=s.index)
    errors = {"missing_value": s.isna(), "out_of_range": (s < lo) | (s > hi),
              "flatline": flatline(s, rules["flatline_hours"][signal]) & (s != 0) & scope}
    suspects: dict[str, pd.Series] = {}
    if signal == "solar":
        errors["night_output"] = (elev < rules["solar_night_elevation_deg"]) & (s > rules["solar_night_max_mw"])
        suspects["midday_zero"] = (elev > rules["solar_midday_elevation_deg"]) & (s == 0)
    if signal == "load":
        ref = pd.concat([s.shift(168 * k) for k in (1, 2, 3, 4)], axis=1).median(axis=1)
        suspects["far_from_same_hour_last_4_weeks"] = ((s - ref).abs() / ref) > rules["load_deviation_pct"]
    err_df = pd.DataFrame(errors, index=s.index).fillna(False).astype(bool)
    sus_df = pd.DataFrame(suspects, index=s.index).fillna(False).astype(bool)
    err = err_df.any(axis=1)
    sus = (sus_df.any(axis=1) if len(sus_df.columns) else pd.Series(False, index=s.index)) & ~err
    return err, sus, pd.concat([err_df, sus_df], axis=1)


def reason_text(reasons: pd.DataFrame) -> pd.Series:
    cols = np.array(reasons.columns)
    return pd.Series([";".join(cols[row]) for row in reasons.to_numpy()], index=reasons.index)
