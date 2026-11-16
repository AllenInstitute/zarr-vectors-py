"""Composite levels: objects made of another store's fragments (``chunk_sources``).

A composite level declares a leading ``"source"`` chunk axis. Source 0 is the
store itself; every other source is another store, whose cells are read
through ``Group.read_bytes`` with their vertices put through the source's
transform. Object manifests are unchanged: one more chunk coordinate.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from zarr_vectors.building import (
    SOURCE_AXIS,
    chunk_sources,
    create_object_index_array,
    create_store,
    create_vertices_array,
    get_resolution_level,
    is_composite,
    open_store,
    open_write_session,
    read_all_object_manifests,
    read_object_vertices,
    set_chunk_sources,
    write_chunk_vertices,
    write_object_index,
)
from zarr_vectors.exceptions import ArrayError, MetadataError
from zarr_vectors.types.polylines import read_polylines, write_polylines

CHUNK = (10.0, 10.0, 10.0)
BOUNDS = ([0.0] * 3, [40.0] * 3)


def _lines(n=2):
    """Polylines crossing chunk faces, so each has several fragments."""
    out = []
    for k in range(n):
        x = np.linspace(1.0, 38.0, 12)
        out.append(np.stack([x, np.full_like(x, 5.0 + 10 * k), np.full_like(x, 15.0)],
                            axis=1).astype("f4"))
    return out


def _transform():
    M = np.eye(4)
    M[:3, :3] *= 2.0
    M[:3, 3] = (100.0, -3.0, 7.0)
    return M


def _apply(M, pts):
    return pts.astype(np.float64) @ M[:3, :3].T + M[:3, 3]


@pytest.fixture
def stores(tmp_path):
    """A source of two polylines, and a composite whose objects are them (through a
    transform), the first followed by one fragment of the composite's own."""
    src = tmp_path / "a" / "source.zarrvectors"
    src.parent.mkdir()
    lines = _lines(2)
    write_polylines(str(src), lines, chunk_shape=CHUNK, bounds=BOUNDS)
    src_lg = get_resolution_level(open_store(str(src)), 0)
    src_manifests = read_all_object_manifests(src_lg)

    comp = tmp_path / "c" / "composite.zarrvectors"
    comp.parent.mkdir()
    root = create_store(comp, bounds=([0.0] * 3, [200.0] * 3), chunk_shape=CHUNK,
                        geometry_types=["polyline"])
    lg = get_resolution_level(root, 0)
    M = _transform()
    table = set_chunk_sources(lg, [
        {"self": True},
        {"ref": {"path": os.path.relpath(src, comp), "id": "world"}, "level": 0,
         "transform": M.tolist()},
    ])
    stub = np.array([[120.0, 7.0, 37.0], [121.0, 7.5, 37.0]], "f4")
    own_cell = (0, 12, 0, 3)
    with open_write_session(lg, bounds=([0.0] * 3, [200.0] * 3), chunk_shape=CHUNK,
                            bin_count=len(table)):
        create_vertices_array(lg)
        create_object_index_array(lg)
        write_chunk_vertices(lg, own_cell, [stub])
        manifests = {
            oid: [((1, *cc), f) for cc, f in blocks] for oid, blocks in enumerate(src_manifests)
        }
        manifests[0] = manifests[0] + [(own_cell, 0)]
        write_object_index(lg, manifests, 4)
    return {"src": src, "comp": comp, "lines": lines, "stub": stub, "M": M,
            "table": table, "src_manifests": src_manifests}


def _composite(stores):
    return get_resolution_level(open_store(str(stores["comp"])), 0)


def test_the_level_declares_its_sources(stores):
    lg = _composite(stores)
    assert is_composite(lg)
    table = chunk_sources(lg)
    assert table[0] == {"self": True} and len(table) == 2
    assert table[1]["fingerprint"]["object_count"] == 2      # recorded when declared
    from zarr_vectors.core.store import read_level_metadata

    meta = read_level_metadata(open_store(str(stores["comp"])), 0)
    assert meta.chunk_dims[0] == SOURCE_AXIS and len(meta.chunk_dims) == 4


def test_an_object_reads_as_its_sources_fragments_through_the_transform(stores):
    lg = _composite(stores)
    M = stores["M"]
    got0 = np.concatenate(read_object_vertices(lg, 0))
    want0 = np.concatenate([_apply(M, stores["lines"][0]), stores["stub"]])
    np.testing.assert_allclose(got0, want0, rtol=0, atol=1e-4)
    got1 = np.concatenate(read_object_vertices(lg, 1))
    np.testing.assert_allclose(got1, _apply(M, stores["lines"][1]), rtol=0, atol=1e-4)


def test_a_rebuilt_source_is_refused_not_misread(stores):
    write_polylines(str(stores["src"]), _lines(3), chunk_shape=CHUNK, bounds=BOUNDS)
    lg = _composite(stores)
    with pytest.raises(ArrayError, match="has changed since this level was written"):
        read_object_vertices(lg, 1)


def test_a_source_is_never_written_through_the_composite(stores):
    lg = get_resolution_level(open_store(str(stores["comp"]), mode="r+"), 0)
    with pytest.raises(ArrayError, match="writes only its own cells"):
        write_chunk_vertices(lg, (1, 0, 0, 0), [np.zeros((2, 3), "f4")])


def test_a_box_query_over_a_composite_is_refused(stores):
    with pytest.raises(ArrayError, match="composite level"):
        read_polylines(str(stores["comp"]), bbox=(np.zeros(3), np.full(3, 50.0)))


def test_an_ordinary_level_is_not_a_composite(stores):
    src_lg = get_resolution_level(open_store(str(stores["src"])), 0)
    assert not is_composite(src_lg) and chunk_sources(src_lg) is None


@pytest.mark.parametrize("table, dims, match", [
    ([{"ref": {"path": "x", "id": "world"}, "transform": np.eye(4).tolist()}],
     ["source", "dim0", "dim1", "dim2"], "must be"),
    ([{"self": True}], ["dim0", "dim1", "dim2"], "lead with"),
    ([{"self": True}, {"ref": {"path": "x"}, "transform": [[1, 0], [0, 1], [0, 0]]}],
     ["source", "dim0", "dim1", "dim2"], "square"),
    ([], ["source"], "non-empty"),
])
def test_a_malformed_table_is_refused(table, dims, match):
    from zarr_vectors.core.sources import validate_chunk_sources

    with pytest.raises(MetadataError, match=match):
        validate_chunk_sources(table, dims)


def test_verify_checks_every_source_at_once(stores):
    from zarr_vectors.building import verify_chunk_sources

    lg = _composite(stores)
    assert verify_chunk_sources(lg) == chunk_sources(lg)
    src_lg = get_resolution_level(open_store(str(stores["src"])), 0)
    assert verify_chunk_sources(src_lg) is None
    write_polylines(str(stores["src"]), _lines(3), chunk_shape=CHUNK, bounds=BOUNDS)
    with pytest.raises(ArrayError, match="has changed since"):
        verify_chunk_sources(_composite(stores))


def test_source_vertices_come_back_in_the_composites_dtype(tmp_path):
    """A float64 source read through a float32 composite reads as float32."""
    src = tmp_path / "s.zarrvectors"
    lines = _lines(1)
    write_polylines(str(src), [ln.astype("f8") for ln in lines], chunk_shape=CHUNK,
                    bounds=BOUNDS, dtype="float64")
    src_lg = get_resolution_level(open_store(str(src)), 0)
    comp = tmp_path / "c.zarrvectors"
    root = create_store(comp, bounds=BOUNDS, chunk_shape=CHUNK, geometry_types=["polyline"])
    lg = get_resolution_level(root, 0)
    table = set_chunk_sources(lg, [{"self": True},
                                   {"ref": {"path": os.path.relpath(src, comp), "id": "world"},
                                    "transform": np.eye(4).tolist()}])
    with open_write_session(lg, bounds=BOUNDS, chunk_shape=CHUNK, bin_count=len(table)):
        create_vertices_array(lg)
        create_object_index_array(lg)
        write_object_index(lg, {0: [((1, *cc), f) for cc, f in read_all_object_manifests(src_lg)[0]]}, 4)
    got = read_object_vertices(_composite({"comp": comp}), 0)
    assert all(np.asarray(g).dtype == np.float32 for g in got)
    np.testing.assert_allclose(np.concatenate(got), lines[0], atol=1e-5)


def test_array_manifests_take_the_composite_rank(tmp_path):
    """write_object_manifests / commit_object_index with no sid_ndim: the source is a coordinate."""
    from zarr_vectors.building import commit_object_index, write_object_manifests

    src = tmp_path / "s.zarrvectors"
    lines = _lines(2)
    write_polylines(str(src), lines, chunk_shape=CHUNK, bounds=BOUNDS)
    src_lg = get_resolution_level(open_store(str(src)), 0)
    mans = read_all_object_manifests(src_lg)
    comp = tmp_path / "c.zarrvectors"
    root = create_store(comp, bounds=BOUNDS, chunk_shape=CHUNK, geometry_types=["polyline"])
    lg = get_resolution_level(root, 0)
    table = set_chunk_sources(lg, [{"self": True},
                                   {"ref": {"path": os.path.relpath(src, comp), "id": "world"},
                                    "transform": np.eye(4).tolist()}])
    cc = np.array([(1, *c) for m in mans for c, _f in m], np.int64)
    fi = np.array([f for m in mans for _c, f in m], np.int64)
    off = np.concatenate([[0], np.cumsum([len(m) for m in mans])]).astype(np.int64)
    with open_write_session(lg, bounds=BOUNDS, chunk_shape=CHUNK, bin_count=len(table)):
        create_vertices_array(lg)
        create_object_index_array(lg)
        write_object_manifests(lg, chunk_coords=cc, fragment_idx=fi, manifest_offsets=off)
        commit_object_index(lg, len(mans))
    lg = _composite({"comp": comp})
    for o, want in enumerate(lines):
        np.testing.assert_allclose(np.concatenate(read_object_vertices(lg, o)), want, atol=1e-5)
