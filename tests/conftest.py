import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GRIDPULSE_ARTIFACTS", str(ROOT / "artifacts"))


@pytest.fixture(scope="session")
def bundle():
    from gridpulse.artifacts import load_bundle
    return load_bundle(verify=True)


@pytest.fixture(scope="session")
def services(bundle):
    from gridpulse.anomaly import AnomalyService
    from gridpulse.forecast import ForecastService
    from gridpulse.store import ReferenceStore
    store = ReferenceStore(bundle)
    fc = ForecastService(bundle, store)
    return store, fc, AnomalyService(bundle, store, fc)


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        yield c
