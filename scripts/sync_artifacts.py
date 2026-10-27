"""Copy a fresh serving bundle (and the MLflow store) from a notebook run into this repository, then verify it.

Usage: python scripts/sync_artifacts.py /path/to/gridpulse_outputs
(the folder the notebook writes to: /content/gridpulse_outputs in Colab - download it as a zip first)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gridpulse.artifacts import load_bundle  # noqa: E402


def main(src: Path) -> None:
    bundle_src = src / "artifacts"
    if not (bundle_src / "manifest.json").exists():
        sys.exit(f"{bundle_src} has no manifest.json - run the notebook up to Part 6 first")
    load_bundle(bundle_src, verify=True)
    shutil.rmtree(ROOT / "artifacts", ignore_errors=True)
    shutil.copytree(bundle_src, ROOT / "artifacts")
    if (src / "mlflow.db").exists():
        dst = ROOT / "mlflow"
        shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir()
        shutil.copy2(src / "mlflow.db", dst / "mlflow.db")
        if (src / "mlartifacts").exists():
            shutil.copytree(src / "mlartifacts", dst / "mlartifacts")
    info = load_bundle(ROOT / "artifacts", verify=True).verify()
    print(f"artifacts updated: {info}. Now run `pytest` and rebuild the images.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
