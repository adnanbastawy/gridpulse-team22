"""Load and verify the serving bundle exported by the notebook (Part 6)."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

DEFAULT_DIR = Path(os.environ.get("GRIDPULSE_ARTIFACTS", Path(__file__).resolve().parents[2] / "artifacts"))


class ArtifactError(RuntimeError):
    """The serving bundle is missing, incomplete or altered."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Bundle:
    root: Path

    @cached_property
    def config(self) -> dict:
        return json.loads((self.root / "config.json").read_text())

    @cached_property
    def manifest(self) -> dict:
        return json.loads((self.root / "manifest.json").read_text())

    def path(self, relative: str) -> Path:
        return self.root / relative

    def verify(self) -> dict:
        """Check every file listed in the manifest against its SHA-256. Raises ArtifactError on any mismatch."""
        if not (self.root / "manifest.json").exists():
            raise ArtifactError(f"manifest.json not found in {self.root}")
        bad = {}
        for rel, expected in self.manifest["files_sha256"].items():
            p = self.root / rel
            if not p.exists():
                bad[rel] = "missing"
            elif sha256_file(p) != expected:
                bad[rel] = "hash mismatch"
        if bad:
            raise ArtifactError(f"serving bundle check failed: {bad}")
        return {"files_checked": len(self.manifest["files_sha256"]), "dataset_version": self.manifest["dataset_version"]}


def load_bundle(root: str | Path | None = None, verify: bool = True) -> Bundle:
    bundle = Bundle(Path(root) if root else DEFAULT_DIR)
    if verify:
        bundle.verify()
    return bundle
