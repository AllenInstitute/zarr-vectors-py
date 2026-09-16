"""Attribute gathering on the public ``Level`` read path.

``Level._gather_attributes`` attaches per-vertex attributes for readers
that cannot return them.  It can only do that for a whole-level reading —
a level-ordered column has no way to line up with a result the reader
already cut down — and it used to discover that the expensive way, by
decompressing every vertices chunk in the level to count them and then
discarding the answer.

These tests pin that a narrowed read never pays for that scan.  They no
longer pin that it comes back *without* attributes: ``read_polylines``
gathers them itself now, in its own by-object order, so the facade has
nothing left to repair and the guard is never consulted.  The cheapness
is the property worth protecting; the emptiness was a symptom of the gap.
"""

from __future__ import annotations

import numpy as np
import pytest

import zarr_vectors as zv
from zarr_vectors.types.polylines import write_polylines


@pytest.fixture
def store_with_vertex_attributes(tmp_path):
    """A polyline level carrying a per-vertex attribute.

    Polylines, because they are the geometry whose reader the gather was
    written for.
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
    # Supplied by the reader, in the order it assembled the result -- so
    # the facade's level-ordered gather is not reached, which is what the
    # monkeypatch above asserts.  This used to be `attributes_read is
    # False`: the read was cheap because it gave up.
    assert result.attributes_read is True
    assert len(result.attributes["fa"]) == result.vertex_count


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


class TestAttributesOnObjectIdReads:
    """``level.objects[i]`` could never carry attributes, for any geometry.

    ``read_points``' object path returned ``vertex_attributes={}``
    unconditionally -- not because a store had none, but because nothing
    there looked -- and ``read_polylines`` returned none on any path.  The
    facade could not repair either: its gather is level-ordered, and a
    by-object assembly cannot be aligned to that, so it declined (and said
    so via ``attributes_read``).  Both readers now gather in their own
    order, where the alignment is free.
    """

    @pytest.fixture
    def points(self, tmp_path):
        from zarr_vectors.types.points import write_points

        rng = np.random.default_rng(5)
        path = tmp_path / "p.zv"
        positions = rng.uniform(0, 400, (60, 3)).astype(np.float32)
        write_points(
            path, positions,
            chunk_shape=(200.0,) * 3, bin_shape=(50.0,) * 3,
            vertex_attributes={
                "a": rng.random(60).astype(np.float32),
                "rgb": rng.random((60, 3)).astype(np.float32),
            },
            object_ids=np.repeat(np.arange(6), 10),
        )
        return path

    def test_indexing_an_object_returns_its_attributes(self, points):
        level = zv.open(points).level(0)
        result = level.objects[2]
        assert result.attributes_read is True
        assert result.attributes.names() == ("a", "rgb")
        assert len(result.attributes["a"]) == result.vertex_count

    def test_multi_channel_attributes_keep_their_width(self, points):
        result = zv.open(points).level(0).objects[2]
        assert result.attributes["rgb"].shape == (result.vertex_count, 3)

    def test_the_values_are_the_objects_own(self, points):
        """Not just the right count -- the right rows."""
        level = zv.open(points).level(0)
        whole = level.read()
        one = level.objects[2]
        # Every value the object read returns must appear in the level's.
        assert set(np.round(one.attributes["a"], 6)) <= set(
            np.round(whole.attributes["a"], 6)
        )

    def test_selecting_objects_returns_attributes(self, points):
        result = zv.open(points).level(0).select(objects=[0, 3]).read()
        assert result.attributes_read is True
        assert len(result.attributes["a"]) == result.vertex_count == 20

    def test_polylines_too(self, store_with_vertex_attributes):
        level = zv.open(store_with_vertex_attributes).level(0)
        result = level.select(objects=[0, 1, 2]).read()
        assert result.attributes_read is True
        assert len(result.attributes["fa"]) == result.vertex_count

    def test_a_polyline_bbox_read_keeps_them_aligned(
        self, store_with_vertex_attributes,
    ):
        level = zv.open(store_with_vertex_attributes).level(0)
        result = level.select(bbox=([0.0] * 3, [60.0] * 3)).read()
        assert result.vertex_count > 0
        assert len(result.attributes["fa"]) == result.vertex_count
