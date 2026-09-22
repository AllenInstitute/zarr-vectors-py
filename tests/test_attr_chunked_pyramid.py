"""A pyramid over a level chunked by an attribute keeps the attribute's bins.

``build_pyramid`` over such a level used to fail at the cross-level link
writer: the fine level's chunk keys carry a leading bin axis (rank 4), the
coarse level was written spatial-only (rank 3), and the partitioner has
one rank for both ends. The coarse level now inherits the bins, and a
metavertex never mixes them -- otherwise a categorical read at a coarse
level would return vertices from other categories.
"""

from __future__ import annotations

import numpy as np
import pytest

import zarr_vectors as zv
from tests.test_attr_chunked_allocation import _attr_store
from zarr_vectors.constants import OBJECT_INDEX, XLEVEL_EXPLICIT, XLEVEL_IMPLICIT
from zarr_vectors.core.arrays import list_link_deltas, read_link_arrays
from zarr_vectors.core.store import (
    get_resolution_level,
    open_store,
    read_level_metadata,
    read_root_metadata,
)
from zarr_vectors.multiresolution.coarsen import (
    _write_empty_preserve_level,
    build_pyramid,
)
from zarr_vectors.types.polylines import read_polylines, write_polylines
from zarr_vectors.validate import validate


def _mixed_store(tmp_path):
    """Two polylines; polyline 1 turns from bundle A to bundle B.

    At a coarse bin of 10, polyline 1's A vertices (10.2, 11) and B
    vertices (12, 13) share one spatial bin, so only the attribute bin
    keeps them apart.
    """
    polys = [
        np.array([[60, 60, 60], [61, 61, 61]], dtype="float32"),
        np.array([[10.2] * 3, [11] * 3, [12] * 3, [13] * 3], dtype="float32"),
    ]
    labels = [np.array(["A", "A"]), np.array(["A", "A", "B", "B"])]
    path = str(tmp_path / "mixed.zarrvectors")
    write_polylines(
        path, polys,
        chunk_shape=(50.0, 50.0, 50.0),
        bin_shape=(5.0, 5.0, 5.0),
        vertex_attributes={"bundle": labels},
        chunk_by_attribute="bundle",
    )
    return path


def _level(path, level):
    return get_resolution_level(open_store(path), level)


# --- the coarse level has the fine level's grid ---------------------------


def test_explicit_cross_level_links_over_an_attribute_chunked_level(tmp_path):
    path = _mixed_store(tmp_path)

    build_pyramid(
        path, factors=[(2.0, 1.0)],
        cross_level_depth=1, cross_level_storage=XLEVEL_EXPLICIT,
    )

    root = open_store(path)
    fine, coarse = _level(path, 0), _level(path, 1)
    vertices = coarse._sharded_chunk_array("vertices")
    assert vertices.ndim == 4 and vertices.shape[0] == 2
    assert all(len(k.split(".")) == 4 for k in coarse.list_chunks("vertices"))
    lm0, lm1 = read_level_metadata(root, 0), read_level_metadata(root, 1)
    assert lm1.chunk_dims == lm0.chunk_dims
    assert lm1.chunk_attribute_name == "bundle"
    assert lm1.chunk_attribute_values == ["A", "B"]
    assert coarse.read_array_meta(OBJECT_INDEX)["sid_ndim"] == 4

    chunks, _ = read_link_arrays(fine, delta=1)
    assert chunks.shape == (lm0.vertex_count, 2, 4)
    # Every link joins a vertex to a metavertex of its own bin.
    np.testing.assert_array_equal(chunks[:, 0, 0], chunks[:, 1, 0])
    back, _ = read_link_arrays(coarse, delta=-1)
    assert back.shape[1:] == (2, 4)
    assert validate(path, level=5).ok, validate(path, level=5).summary()


def test_implicit_storage_writes_only_the_fine_side(tmp_path):
    path = _mixed_store(tmp_path)

    build_pyramid(
        path, factors=[(2.0, 1.0)],
        cross_level_depth=1, cross_level_storage=XLEVEL_IMPLICIT,
    )

    assert 1 in list_link_deltas(_level(path, 0))
    assert -1 not in list_link_deltas(_level(path, 1))


def test_a_coarse_metavertex_never_mixes_bins(tmp_path):
    path = _mixed_store(tmp_path)

    build_pyramid(path, factors=[(2.0, 1.0)], cross_level_depth=0)

    b = read_polylines(path, level=1, attribute_filter={"bundle": "B"})
    assert b["object_ids"] == [1]
    np.testing.assert_allclose(np.concatenate(b["polylines"][0]), [[12.5] * 3])
    a = read_polylines(path, level=1, attribute_filter={"bundle": "A"})
    by_id = dict(zip(a["object_ids"], a["polylines"]))
    np.testing.assert_allclose(np.concatenate(by_id[1]), [[10.6] * 3], rtol=1e-6)
    np.testing.assert_allclose(np.concatenate(by_id[0]), [[60.5] * 3])
    # One metavertex per (bin, spatial bin): 60s, A at 10s, B at 10s.
    assert read_level_metadata(open_store(path), 1).vertex_count == 3


def test_depth_two_links_keep_the_bin_axis(tmp_path):
    path = _mixed_store(tmp_path)

    build_pyramid(
        path, factors=[(2.0, 1.0), (2.0, 1.0)],
        cross_level_depth=2, cross_level_storage=XLEVEL_EXPLICIT,
    )

    chunks, _ = read_link_arrays(_level(path, 0), delta=2)
    assert chunks.shape[1:] == (2, 4)
    np.testing.assert_array_equal(chunks[:, 0, 0], chunks[:, 1, 0])
    assert validate(path, level=5).ok


def test_points_pyramid_through_the_public_api_validates(tmp_path):
    path = _attr_store(tmp_path / "g.zarrvectors", n=200)

    zv.open(path, mode="r+").build_pyramid(factors=[(2.0, 1.0)])

    coarse = _level(path, 1)
    assert coarse._sharded_chunk_array("vertices").shape[0] == 3
    assert validate(path, level=5).ok, validate(path, level=5).summary()


def test_a_bin_axis_without_bin_values_is_carried_too(tmp_path):
    """A rechunk made before labels were recorded has the axis, no values."""
    path = _attr_store(tmp_path / "g.zarrvectors")
    root = open_store(path, mode="r+")
    block = dict(root["0"].attrs.to_dict()["zarr_vectors_level"])
    block.pop("chunk_attribute_name", None)
    block.pop("chunk_attribute_values", None)
    root["0"].attrs.update({"zarr_vectors_level": block})

    build_pyramid(path, factors=[(2.0, 1.0)])

    coarse = _level(path, 1)
    assert coarse._sharded_chunk_array("vertices").ndim == 4
    assert coarse._sharded_chunk_array("vertices").shape[0] == 3
    lm1 = read_level_metadata(open_store(path), 1)
    assert lm1.chunk_attribute_values is None
    assert lm1.chunk_dims == block["chunk_dims"]
    assert all(len(k.split(".")) == 4 for k in coarse.list_chunks("vertices"))


def test_an_empty_coarse_level_keeps_the_bins_and_chunk_shape(tmp_path):
    """The no-survivors path writes the level a populated run would have."""
    path = _attr_store(tmp_path / "g.zarrvectors")
    root = open_store(path, mode="r+")
    root_meta = read_root_metadata(root)
    lm0 = read_level_metadata(root, 0)

    _write_empty_preserve_level(
        root, 0, 1,
        base_bin=root_meta.effective_bin_shape,
        root_meta=root_meta,
        coarsen_factor=2.0,
        sparsity_factor=1.0,
        inherited_num_objects=3,
        chunk_shape=(100.0, 100.0, 100.0),
        chunk_shape_override=(100.0, 100.0, 100.0),
        leading=(3,),
        bin_meta={
            "chunk_dims": lm0.chunk_dims,
            "chunk_attribute_name": lm0.chunk_attribute_name,
            "chunk_attribute_values": lm0.chunk_attribute_values,
        },
    )

    lm1 = read_level_metadata(open_store(path), 1)
    assert tuple(lm1.chunk_shape) == (100.0, 100.0, 100.0)
    assert lm1.chunk_attribute_values == lm0.chunk_attribute_values
    coarse = _level(path, 1)
    assert coarse._sharded_chunk_array("vertices").ndim == 4
    assert coarse.read_array_meta(OBJECT_INDEX)["sid_ndim"] == 4


@pytest.mark.parametrize("storage", [XLEVEL_EXPLICIT, XLEVEL_IMPLICIT])
def test_a_spatial_pyramid_is_unchanged(tmp_path, storage):
    """No leading axis: rank-3 keys, and the object index says so."""
    rng = np.random.default_rng(0)
    path = str(tmp_path / "s.zarrvectors")
    write_polylines(
        path, [rng.uniform(0, 100, (12, 3)).astype("f4") for _ in range(6)],
        chunk_shape=(50.0, 50.0, 50.0), bin_shape=(5.0, 5.0, 5.0),
    )

    build_pyramid(
        path, factors=[(2.0, 1.0)],
        cross_level_depth=1, cross_level_storage=storage,
    )

    coarse = _level(path, 1)
    assert coarse._sharded_chunk_array("vertices").ndim == 3
    assert coarse.read_array_meta(OBJECT_INDEX)["sid_ndim"] == 3
    chunks, _ = read_link_arrays(_level(path, 0), delta=1)
    assert chunks.shape[2] == 3


@pytest.mark.parametrize("by", ["attribute:cluster", "group"])
def test_a_rechunked_store_takes_a_pyramid_and_validates(tmp_path, by):
    """Rechunk output is the other producer of a leading bin axis."""
    from zarr_vectors.rechunk import RechunkSpec, rechunk

    rng = np.random.default_rng(2)
    polys = [rng.uniform(0, 100, (8, 3)).astype("f4") for _ in range(12)]
    src = str(tmp_path / "src.zarrvectors")
    write_polylines(
        src, polys, chunk_shape=(50.0, 50.0, 50.0), bin_shape=(5.0, 5.0, 5.0),
        object_attributes={"cluster": np.arange(12) % 3},
        groups={0: list(range(6)), 1: list(range(6, 12))},
    )
    out = str(tmp_path / "out.zarrvectors")
    rechunk(src, RechunkSpec(by=by, categorical=True), output=out)

    build_pyramid(
        out, factors=[(2.0, 1.0), (2.0, 1.0)],
        cross_level_depth=2, cross_level_storage=XLEVEL_EXPLICIT,
    )

    result = validate(out, level=5)
    assert result.ok, result.summary()
    lm0 = read_level_metadata(open_store(out), 0)
    lm2 = read_level_metadata(open_store(out), 2)
    assert lm2.chunk_attribute_values == lm0.chunk_attribute_values
