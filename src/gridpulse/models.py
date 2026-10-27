"""Model wrappers for the exported formats: LightGBM text, XGBoost JSON, linear JSON."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


class ForecastModel:
    """One production forecast model with exact per-feature contributions (TreeSHAP or linear)."""

    def __init__(self, spec: dict, root: Path):
        self.spec, self.name, self.format = spec, spec["model"], spec["format"]
        self.features: list[str] = spec["features"]
        path = root / spec["file"]
        if self.format == "lightgbm_txt":
            self._booster = lgb.Booster(model_file=str(path))
        elif self.format == "xgboost_json":
            import xgboost as xgb
            self._xgb = xgb
            self._booster = xgb.Booster(model_file=str(path))
        elif self.format == "linear_json":
            p = json.loads(path.read_text())
            self._mean, self._scale = np.array(p["mean"]), np.array(p["scale"])
            self._coef, self._intercept = np.array(p["coef"]), float(p["intercept"])
        else:
            raise ValueError(f"unknown model format {self.format}")

    @property
    def handles_missing(self) -> bool:
        return self.format != "linear_json"

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.features]
        if self.format == "lightgbm_txt":
            return self._booster.predict(X)
        if self.format == "xgboost_json":
            return self._booster.predict(self._xgb.DMatrix(X))
        return ((X.to_numpy(float) - self._mean) / self._scale) @ self._coef + self._intercept

    def contributions(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(values [n, features] in MW, base value [n]); base + sum(values) = forecast."""
        X = X[self.features]
        if self.format == "lightgbm_txt":
            c = self._booster.predict(X, pred_contrib=True)
            return c[:, :-1], c[:, -1]
        if self.format == "xgboost_json":
            c = self._booster.predict(self._xgb.DMatrix(X), pred_contribs=True)
            return c[:, :-1], c[:, -1]
        values = (X.to_numpy(float) - self._mean) / self._scale * self._coef
        return values, np.full(len(X), self._intercept)


class NowcastModel:
    def __init__(self, path: Path):
        self._booster = lgb.Booster(model_file=str(path))
        self.features = self._booster.feature_name()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._booster.predict(X[self.features])
