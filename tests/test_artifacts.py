import json
import shutil

import pytest

from gridpulse.artifacts import ArtifactError, load_bundle


def test_bundle_hashes_match(bundle):
    info = bundle.verify()
    assert info["files_checked"] >= 8


def test_altered_file_is_rejected(bundle, tmp_path):
    copy = tmp_path / "artifacts"
    shutil.copytree(bundle.root, copy)
    cfg = copy / "config.json"
    cfg.write_text(cfg.read_text().replace("load", "LOAD", 1))
    with pytest.raises(ArtifactError):
        load_bundle(copy, verify=True)


def test_config_has_three_signals(bundle):
    assert set(bundle.config["forecast"]["models"]) == {"load", "solar", "wind"}
    assert json.dumps(bundle.config)  # serialisable
