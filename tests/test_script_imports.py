"""Importing geometry helpers must never rebuild an existing user's scene."""

import runpy
from pathlib import Path


def test_visual_review_helpers_have_no_import_side_effects(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/build_visual_review.py"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["import-helper", "--unrelated-argument"])
    namespace = runpy.run_path(str(script), run_name="geometry_helper_import")
    assert callable(namespace["panel_from_observation"])
    assert list(tmp_path.iterdir()) == []
