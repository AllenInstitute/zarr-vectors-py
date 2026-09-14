"""Attribute gathering on the public ``Level`` read path.

``Level._gather_attributes`` attaches per-vertex attributes for the four
readers that cannot return them.  It can only do that for a whole-level
reading — a level-ordered column has no way to line up with a result the
reader already cut down — and it used to discover that the expensive
way, by decompressing every vertices chunk in the level to count them and
then discarding the answer.  These tests pin that a narrowed read costs
nothing and that a full one still gathers.
"""

from __future__ import annotations

import numpy as np
import pytest

import zarr_vectors as zv
from zarr_vectors.types.polylines import write_polylines


@pytest.fixture
def store_with_vertex_attributes(tmp_path):
    """A polyline level carrying a per-vertex attribute.

    Polylines, because ``read_polylines`` hardcodes
    ``attributes_read=False`` — which is the whole reason the gather
    exists.
    """
    rng = np.random.default_rng(0)
    lines = [
        rng.uniform(0, 200, (12, 3)).astype(np.float32) for _ in range(40)
    ]
    # One array per polyline, matching the writer's per-object shape.
    fa = [rng.uniform(0, 1, len(line)).astype(np.float32) for line in lines]
    path = str(tmp_path / "pl.zv")
    write_polylines(
        path, lines, chunk_shape=(50.0, 50.0, 50.0),
        vertex_attributes={"fa": fa},
    )
    return path


def _forbid_level_scan(monkeypatch):
    """Make the whole-level vertex scan fail loudly if it is reached."""
    import zarr_vectors.spatial.boundary as boundary

    def explode(*a, **k):
        raise AssertionError(
            "the level was scanned to gather attributes for a narrowed read"
        )

    monkeypatch.setattr(boundary, "chunk_local_to_global_offsets", explode)


@pytest.mark.parametrize("narrowing", [
    {"bbox": ([0.0, 0.0, 0.0], [60.0, 60.0, 60.0])},
    {"objects": [0, 1, 2]},
])
def test_a_narrowed_read_does_not_scan_the_level(
    store_with_vertex_attributes, monkeypatch, narrowing,
):
    _forbid_level_scan(monkeypatch)
    level = zv.open(store_with_vertex_attributes).level(0)
    result = level.select(attributes=["fa"], **narrowing).read()
    assert result.vertex_count > 0
    assert result.attributes_read is False


def test_a_full_read_still_gathers_attributes(store_with_vertex_attributes):
    level = zv.open(store_with_vertex_attributes).level(0)
    result = level.select(attributes=["fa"]).read()
    assert result.attributes_read is True
    assert "fa" in result.attributes
    assert len(result.attributes["fa"]) == result.vertex_count


def test_a_full_read_is_unaffected_by_the_guard(store_with_vertex_attributes):
    """The guard keys off the Selection, so an unnarrowed one passes it."""
    from zarr_vectors.api.level import _narrows
    from zarr_vectors.api.select import Selection

    assert _narrows(Selection()) is False
    assert _narrows(Selection(level=0)) is False
    assert _narrows(Selection(bbox=([0.0] * 3, [1.0] * 3))) is True
    assert _narrows(Selection(near=([0.0] * 3, 1.0))) is True
    assert _narrows(Selection(objects=[1])) is True
    assert _narrows(Selection(groups=[1])) is True
    assert _narrows(Selection(where={"a": 1})) is True
    assert _narrows(Selection(cells=[(0, 0, 0)])) is True
