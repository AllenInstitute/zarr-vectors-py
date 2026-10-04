"""Additive pyramid levels (``refinement: "add"``) and ``required_capabilities``.

Invariants under test:

* **A1** ``refinement`` and ``required_capabilities`` round-trip through the
  metadata dataclasses, and are written only when they say something.
* **A2** A store listing a required capability this package does not
  implement is refused on open, by every reader.
* **A3** ``make_levels_additive`` stores each object at one level: level
  ``L`` keeps exactly the objects level ``L + 1`` lacks, byte for byte, and
  counts only its own vertices.
* **A4** Readers return a level's complete content -- the union over its
  chain -- by default, and its stored data with ``own_level_only=True``;
  the facade (``Level.read``, ``count``, ``chain``) agrees.
* **A5** Validation: an additive coarsest level, a missing required
  capability and an object stored at two levels of a chain are errors.
* **A6** Anything that rebuilds levels from one another, or removes a level
  an additive level depends on, refuses an additive store.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import zarr_vectors as zv
from zarr_vectors.constants import CAP_ADDITIVE_LEVELS
from zarr_vectors.core.metadata import LevelMetadata, RootMetadata
from zarr_vectors.core.refinement import (
    level_chain,
    map_cells,
    stored_object_ids,
)
from zarr_vectors.core.store import (
    get_resolution_level,
    open_store,
    read_level_metadata,
    read_root_metadata,
    remove_resolution_level,
    update_level_metadata,
    update_root_metadata,
)
from zarr_vectors.exceptions import (
    MetadataError,
    StoreError,
    UnsupportedCapabilityError,
)
from zarr_vectors.multiresolution.additive import make_levels_additive
from zarr_vectors.multiresolution.coarsen import build_pyramid, coarsen_level
from zarr_vectors.ops.edit import EditSession
from zarr_vectors.ops.refresh import rebuild_pyramid_from_level
from zarr_vectors.types.points import read_points, write_points
from zarr_vectors.types.polylines import read_polylines, write_polylines
from zarr_vectors.validate import validate


def _lines(n=30, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        start = rng.uniform(5, 55, 3)
        steps = rng.normal(0, 2.0, (int(rng.integers(5, 25)), 3))
        out.append((start + np.cumsum(steps, axis=0)).clip(0.5, 63.5).astype("float32"))
    return out


def _by_object(path, level, **kw):
    """``{object id: (positions, attribute)}`` as a reader returns them."""
    r = read_polylines(str(path), level=level, **kw)
    fa = r["vertex_attributes"].get("fa")
    out: dict[int, list] = {}
    pos = 0
    for oid, segments in zip(r["object_ids"], r["polylines"]):
        arr = np.concatenate(segments) if len(segments) else np.zeros((0, 3), np.float32)
        out.setdefault(int(oid), []).append(
            (arr, None if fa is None else fa[pos:pos + len(arr)]),
        )
        pos += len(arr)
    return {
        k: (np.concatenate([a for a, _ in v]),
            None if v[0][1] is None else np.concatenate([b for _, b in v]))
        for k, v in out.items()
    }


def _same(a, b):
    return set(a) == set(b) and all(
        np.array_equal(a[k][0], b[k][0])
        and (a[k][1] is None or np.array_equal(a[k][1], b[k][1]))
        for k in a
    )


@pytest.fixture
def pyramid(tmp_path):
    """A three-level polyline pyramid, before and after conversion."""
    lines = _lines()
    path = tmp_path / "p.zv"
    write_polylines(
        str(path), lines, chunk_shape=(32.0,) * 3, bounds=([0, 0, 0], [64, 64, 64]),
        geometry_type="polyline",
        vertex_attributes={"fa": [np.linspace(0, 1, len(x)).astype("float32") for x in lines]},
    )
    build_pyramid(str(path), factors=[(1.0, 2.0), (1.0, 2.0)])
    before = {lv: _by_object(path, lv) for lv in range(3)}
    vertex_counts = {
        lv: read_level_metadata(open_store(str(path)), lv).vertex_count for lv in range(3)
    }
    reports = make_levels_additive(str(path))
    return path, before, vertex_counts, reports


# ------------------------------------------------------------------ A1


def test_a1_refinement_round_trips_and_is_omitted_when_replace():
    add = LevelMetadata(level=0, vertex_count=1, arrays_present=["vertices"],
                        refinement="add")
    assert add.to_dict()["zarr_vectors_level"]["refinement"] == "add"
    assert LevelMetadata.from_dict(add.to_dict()).refinement == "add"
    plain = LevelMetadata(level=0, vertex_count=1, arrays_present=["vertices"])
    assert "refinement" not in plain.to_dict()["zarr_vectors_level"]
    assert LevelMetadata.from_dict(plain.to_dict()).refinement == "replace"
    with pytest.raises(MetadataError, match="refinement"):
        LevelMetadata(level=0, vertex_count=1, arrays_present=[],
                      refinement="multiply").validate()


def test_a1_required_capabilities_round_trip():
    meta = RootMetadata(
        spatial_index_dims=[{"name": "x"}], chunk_shape=(1.0,), bounds=([0], [1]),
        geometry_types=["point_cloud"], required_capabilities=[CAP_ADDITIVE_LEVELS],
    )
    d = meta.to_dict()
    assert d["zarr_vectors"]["required_capabilities"] == [CAP_ADDITIVE_LEVELS]
    assert RootMetadata.from_dict(d, strict=False).required_capabilities == [
        CAP_ADDITIVE_LEVELS,
    ]
    meta.required_capabilities = []
    assert "required_capabilities" not in meta.to_dict()["zarr_vectors"]


# ------------------------------------------------------------------ A2


def test_a2_an_unknown_required_capability_is_refused(tmp_path):
    path = str(tmp_path / "pts.zv")
    write_points(path, np.random.default_rng(0).uniform(0, 10, (20, 3)).astype("float32"),
                 chunk_shape=(5.0,) * 3, bounds=([0, 0, 0], [10, 10, 10]))
    update_root_metadata(open_store(path, mode="r+"),
                         required_capabilities=[CAP_ADDITIVE_LEVELS])
    assert len(read_points(path)["positions"]) == 20      # known: opens
    update_root_metadata(open_store(path, mode="r+"),
                         required_capabilities=["teleportation"])
    for attempt in (
        lambda: open_store(path),
        lambda: read_points(path),
        lambda: zv.open(path),
        lambda: open_store(path, mode="r+"),
    ):
        with pytest.raises(UnsupportedCapabilityError, match="teleportation"):
            attempt()


# ------------------------------------------------------------------ A3


def test_a3_each_object_is_stored_at_one_level(pyramid):
    path, before, vertex_counts, reports = pyramid
    root = open_store(str(path))
    own = {lv: stored_object_ids(get_resolution_level(root, lv)) for lv in range(3)}
    for lv in (0, 1):
        expected = np.setdiff1d(np.asarray(sorted(before[lv])), np.asarray(sorted(before[lv + 1])))
        np.testing.assert_array_equal(own[lv], expected)
        assert not np.intersect1d(own[lv], own[lv + 1]).size
    np.testing.assert_array_equal(own[2], np.asarray(sorted(before[2])))
    # Byte for byte: what a level keeps is what it held.
    for lv in range(3):
        stored = _by_object(path, lv, own_level_only=True)
        assert set(stored) == set(own[lv].tolist())
        for oid, (positions, fa) in stored.items():
            np.testing.assert_array_equal(positions, before[lv][oid][0])
            np.testing.assert_array_equal(fa, before[lv][oid][1])
    # vertex_count is the level's own (shared fragments can keep rows a
    # dropped object also used, so it need not shrink).
    for lv, report in enumerate(reports):
        meta = read_level_metadata(root, lv)
        assert meta.vertex_count == report["vertices_after"] <= vertex_counts[lv]
        assert meta.refinement == "add"
    assert read_level_metadata(root, 2).refinement == "replace"
    # Level 0 shares no fragments, so it stores strictly less.
    assert read_level_metadata(root, 0).vertex_count < vertex_counts[0]


def test_a3_metadata_on_disk(pyramid):
    path = pyramid[0]
    level0 = json.loads((Path(path) / "0" / "zarr.json").read_text())
    assert level0["attributes"]["zarr_vectors_level"]["refinement"] == "add"
    top = json.loads((Path(path) / "2" / "zarr.json").read_text())
    assert "refinement" not in top["attributes"]["zarr_vectors_level"]
    root = json.loads((Path(path) / "zarr.json").read_text())["attributes"]["zarr_vectors"]
    assert root["required_capabilities"] == [CAP_ADDITIVE_LEVELS]
    assert CAP_ADDITIVE_LEVELS in root["format_capabilities"]
    # Cross-level links join one object at two levels; none survive.
    assert root["cross_level_storage"] == "none" and root["cross_level_depth"] == 0
    assert not any((Path(path) / str(lv) / "links" / "+1").exists() for lv in range(3))


# ------------------------------------------------------------------ A4


def test_a4_readers_return_the_complete_content(pyramid):
    path, before, _, _ = pyramid
    expected = dict(before[2])
    for lv in (2, 1, 0):
        if lv < 2:
            own = set(before[lv]) - set(before[lv + 1])
            expected = {**expected, **{k: before[lv][k] for k in own}}
        assert _same(_by_object(path, lv), expected)
        assert level_chain(str(path), lv) == list(range(lv, 3))


def test_a4_the_facade_reads_the_chain(pyramid):
    path, before, _, _ = pyramid
    ds = zv.open(str(path))
    for lv in range(3):
        level = ds.level(lv)
        assert level.chain == tuple(range(lv, 3))
        full = level.read()
        own = level.read(own_level_only=True)
        assert full.vertex_count == sum(
            ds.level(x).read(own_level_only=True).vertex_count for x in level.chain
        )
        assert own.vertex_count < full.vertex_count or lv == 2
        # The metadata count sums the chain's stored vertex counts.
        assert level.select().count() == sum(ds.level(x).vertex_count for x in level.chain)
        assert set(np.unique(full.part_objects)) == set(before[lv])
        # An attribute survives the union where every level read it.
        parts = [ds.level(x).read(own_level_only=True) for x in level.chain]
        common = set.intersection(*(set(r.attributes) for r in parts if r.vertex_count))
        assert set(full.attributes) == common
    limited = ds.level(0).select(limit=50).read()
    assert limited.vertex_count == 50 and limited.truncated


def test_a4_chunk_filters_map_onto_coarser_grids():
    assert map_cells([(0, 0, 0), (1, 0, 0)], (16.0,) * 3, (32.0,) * 3) == [(0, 0, 0)]
    assert map_cells([(3, 1, 0)], (16.0,) * 3, (32.0,) * 3) == [(1, 0, 0)]
    assert map_cells([(1, 0, 0)], (32.0,) * 3, (16.0,) * 3) == [
        (2, 0, 0), (2, 0, 1), (2, 1, 0), (2, 1, 1),
        (3, 0, 0), (3, 0, 1), (3, 1, 0), (3, 1, 1),
    ]
    assert map_cells([(7, 1, 2, 3)], (16.0,) * 3, (32.0,) * 3) == [(7, 0, 1, 1)]


def test_a4_a_chunk_read_covers_the_chain(pyramid):
    path, before, _, _ = pyramid
    cells = [(0, 0, 0)]
    whole = read_polylines(str(path), level=0, chunks=cells)
    parts = sum(
        read_polylines(str(path), level=lv, chunks=cells, own_level_only=True)["vertex_count"]
        for lv in range(3)
    )
    assert whole["vertex_count"] == parts > 0


# ------------------------------------------------------------------ A5


def test_a5_the_validator_accepts_the_conversion(pyramid):
    result = validate(str(pyramid[0]), level=5)
    assert result.ok, result.errors


def test_a5_an_additive_coarsest_level_is_an_error(pyramid):
    path = pyramid[0]
    update_level_metadata(get_resolution_level(open_store(str(path), mode="r+"), 2),
                          refinement="add")
    result = validate(str(path), level=2)
    assert any("no level 3" in e for e in result.errors)
    with pytest.raises(MetadataError, match="no level 3"):
        level_chain(str(path), 0)


def test_a5_an_additive_level_needs_the_required_capability(pyramid):
    path = pyramid[0]
    update_root_metadata(open_store(str(path), mode="r+"), required_capabilities=[])
    result = validate(str(path), level=2)
    assert any("required_capabilities" in e for e in result.errors)


def test_a5_an_object_at_two_levels_is_an_error(tmp_path):
    path = tmp_path / "dup.zv"
    lines = _lines(12)
    write_polylines(str(path), lines, chunk_shape=(32.0,) * 3,
                    bounds=([0, 0, 0], [64, 64, 64]), geometry_type="polyline")
    build_pyramid(str(path), factors=[(1.0, 2.0)])
    # Declared additive without removing the objects level 1 already has.
    from zarr_vectors.core.refinement import declare_additive_levels

    declare_additive_levels(open_store(str(path), mode="r+"), [0])
    result = validate(str(path), level=3)
    assert any("also in level 1" in e for e in result.errors)


# ------------------------------------------------------------------ A6


def test_a6_rebuilding_or_removing_levels_is_refused(pyramid):
    path = str(pyramid[0])
    root = open_store(path, mode="r+")
    with pytest.raises(StoreError, match="additive"):
        build_pyramid(path, factors=[(1.0, 2.0)])
    with pytest.raises(StoreError, match="additive"):
        coarsen_level(path, 2, 3, sparsity_factor=2.0)
    with pytest.raises(StoreError, match="additive"):
        rebuild_pyramid_from_level(root, 0)
    with pytest.raises(StoreError, match="additive"):
        EditSession(root, refresh_pyramid=True)
    with pytest.raises(StoreError, match="level 1 is additive"):
        remove_resolution_level(root, 2)
    with pytest.raises(Exception, match="already additive"):
        make_levels_additive(path)
    # An edit that leaves the pyramid alone is fine.
    EditSession(root, refresh_pyramid=False)


def test_a6_conversion_needs_a_pyramid(tmp_path):
    path = str(tmp_path / "one.zv")
    write_polylines(path, _lines(3), chunk_shape=(32.0,) * 3,
                    bounds=([0, 0, 0], [64, 64, 64]), geometry_type="polyline")
    with pytest.raises(Exception, match="single-level"):
        make_levels_additive(path)
    assert read_root_metadata(open_store(path)).required_capabilities == []
