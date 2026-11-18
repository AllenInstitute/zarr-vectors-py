"""``rechunk()`` carries what it can and refuses what it cannot.

It used to copy vertices, manifests and (for ``by="group"`` only)
groupings, and silently drop links, vertex and object attributes, group
attributes and every level above 0; it stamped ``vertex_count: 0``; and
it deleted an existing ``output`` directory without asking.
``RechunkSpec(by="spatial")`` went down the same lossy path.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.core.arrays import (
    read_all_groupings,
    read_chunk_attributes,
    read_chunk_vertices,
    read_link_arrays,
    read_object_attributes,
    read_object_manifest_rows,
)
from zarr_vectors.core.store import (
    get_resolution_level,
    list_resolution_levels,
    open_store,
    read_level_metadata,
)
from zarr_vectors.rechunk import RechunkSpec, rechunk
from zarr_vectors.types.polylines import read_polylines, write_polylines
from zarr_vectors.validate import validate


def _polys():
    rng = np.random.default_rng(5)
    out = []
    for _ in range(12):
        start = rng.uniform(10, 90, 3)
        out.append((start + np.cumsum(rng.normal(0, 6, (15, 3)), 0))
                   .clip(0.5, 99.5).astype(np.float32))
    return out


@pytest.fixture
def lines(tmp_path: Path):
    polys = _polys()
    path = tmp_path / "src.zarr"
    write_polylines(
        path, polys, chunk_shape=(25.0,) * 3, bounds=([0.0] * 3, [100.0] * 3),
        vertex_attributes={"t": [np.arange(len(p), dtype=np.float32) + 100 * i
                                 for i, p in enumerate(polys)]},
        object_attributes={"score": np.arange(12, dtype=np.float32) * 1.5},
        groups={0: [0, 1, 2, 3], 1: [4, 5, 6, 7, 8, 9, 10, 11]},
        group_attributes={"gname": np.array([7.0, 8.0], dtype=np.float32)},
    )
    return path, polys


def _objects(path):
    """Per output object: its vertices in manifest order, with attribute t."""
    lg = get_resolution_level(open_store(str(path), mode="r"), 0)
    ids, mans = read_object_manifest_rows(lg)
    out = {}
    for oid, man in zip(ids.tolist(), mans):
        pts, ts = [], []
        for cc, f in man:
            pts.append(read_chunk_vertices(lg, tuple(cc), ndim=3)[f])
            ts.append(np.ravel(read_chunk_attributes(
                lg, "t", tuple(cc), dtype="float32", ncols=1)[f]))
        if pts:
            out[oid] = (np.concatenate(pts), np.concatenate(ts))
    return out, lg


def test_group_rechunk_carries_attributes_bridges_and_counts(lines, tmp_path):
    src, polys = lines
    out = tmp_path / "out.zarr"
    summary = rechunk(str(src), RechunkSpec(by="group"), output=str(out))

    objs, lg = _objects(out)
    assert len(objs) == 12
    score = np.asarray(read_object_attributes(lg, "score")).ravel()
    for oid, (pts, t) in objs.items():
        # The vertex attribute names the source polyline and row, so the
        # output object's own vertices and attribute values must agree.
        src_i = int(t[0]) // 100
        np.testing.assert_array_equal(pts, polys[src_i])
        np.testing.assert_array_equal(t, np.arange(len(pts)) + 100 * src_i)
        assert score[oid] == pytest.approx(src_i * 1.5)

    lm = read_level_metadata(open_store(str(out), mode="r"), 0)
    assert lm.vertex_count == summary["total_vertices"] == sum(map(len, polys))

    # One bridge per step between consecutive fragments of an object,
    # joining two consecutive vertices of that polyline.
    chunks, vi = read_link_arrays(lg, delta=0)
    _ids, mans = read_object_manifest_rows(lg)
    assert len(vi) == sum(max(len(m) - 1, 0) for m in mans) > 0
    steps = {(tuple(a), tuple(b)) for p in polys for a, b in zip(p[:-1], p[1:])}
    for c, v in zip(chunks, vi):
        ends = [tuple(np.concatenate(read_chunk_vertices(
            lg, tuple(int(x) for x in c[k]), ndim=3))[v[k]]) for k in range(2)]
        assert (ends[0], ends[1]) in steps

    assert sorted(map(sorted, read_all_groupings(lg))) == [
        list(range(4)), list(range(4, 12))]
    np.testing.assert_array_equal(
        np.ravel(_group_attr(lg, "gname")), [7.0, 8.0])
    assert validate(str(out), level=4).ok


def _group_attr(lg, name):
    from zarr_vectors.core.arrays import read_groupings_attributes
    return read_groupings_attributes(lg, name)


def test_attribute_rechunk_keeps_groupings(lines, tmp_path):
    src, _ = lines
    out = tmp_path / "out.zarr"
    rechunk(str(src), RechunkSpec(by="attribute:score", bins=[0, 5, 100]),
            output=str(out))
    lg = get_resolution_level(open_store(str(out), mode="r"), 0)
    assert sum(len(g) for g in read_all_groupings(lg)) == 12


def test_an_existing_output_is_refused_not_deleted(lines, tmp_path):
    src, _ = lines
    out = tmp_path / "out.zarr"
    out.mkdir()
    (out / "precious.txt").write_text("keep me")
    with pytest.raises(FileExistsError):
        rechunk(str(src), RechunkSpec(by="group"), output=str(out))
    assert (out / "precious.txt").read_text() == "keep me"
    rechunk(str(src), RechunkSpec(by="group"), output=str(out), overwrite=True)
    assert not (out / "precious.txt").exists()


def test_a_pyramid_is_refused_before_anything_is_written(lines, tmp_path):
    from zarr_vectors.multiresolution.coarsen import build_pyramid

    src, _ = lines
    build_pyramid(src, factors=[(2.0, 1.0)])
    out = tmp_path / "out.zarr"
    with pytest.raises(NotImplementedError, match="resolution levels"):
        rechunk(str(src), RechunkSpec(by="group"), output=str(out))
    assert not out.exists()


def _link_positions(path) -> Counter:
    """Every ``links/0`` record as its endpoints' positions, rotated to
    start at the smallest (so a face keeps its winding)."""
    lg = get_resolution_level(open_store(str(path), mode="r"), 0)
    chunks, vi = read_link_arrays(lg, delta=0)
    cache: dict[tuple, np.ndarray] = {}

    def at(c, v):
        key = tuple(int(x) for x in c)
        if key not in cache:
            cache[key] = np.concatenate(read_chunk_vertices(lg, key, ndim=3))
        return tuple(np.round(cache[key][int(v)], 4).tolist())

    out: Counter = Counter()
    for cs, vs in zip(chunks, vi):
        ends = [at(c, v) for c, v in zip(cs, vs)]
        k = ends.index(min(ends))
        out[tuple(ends[k:] + ends[:k])] += 1
    return out


def test_graph_edges_are_carried_across_bins(tmp_path):
    """Explicit links used to be refused. Edges between objects that land
    in different bins become links between bins."""
    from zarr_vectors.types.graphs import write_graph

    rng = np.random.default_rng(0)
    src = tmp_path / "graph.zarr"
    oids = np.arange(40) % 4
    write_graph(src, positions=rng.uniform(0, 100, (40, 3)).astype(np.float32),
                edges=np.array([[i, (i * 7 + 3) % 40] for i in range(40)]),
                object_ids=oids, chunk_shape=(25.0,) * 3,
                object_attributes={"w": np.array([4.0, 3.0, 2.0, 1.0])},
                bounds=([0.0] * 3, [100.0] * 3))
    out = tmp_path / "out.zarr"

    summary = rechunk(str(src), RechunkSpec(by="attribute:w"), output=str(out))

    assert summary["bins_created"] == 4
    assert summary["links"] == 40 and summary["dropped_links"] == 0
    norm = lambda c: Counter({tuple(sorted(k)): n for k, n in c.items()})  # noqa: E731
    assert norm(_link_positions(out)) == norm(_link_positions(src))
    chunks, _ = read_link_arrays(
        get_resolution_level(open_store(str(out), mode="r"), 0), delta=0)
    assert chunks.shape[1:] == (2, 4)
    assert (chunks[:, 0, 0] != chunks[:, 1, 0]).any()   # some cross bins
    assert validate(str(out), level=5).ok, validate(str(out), level=5).summary()


@pytest.mark.slow
def test_mesh_faces_keep_their_winding(tmp_path):
    from zarr_vectors.types.meshes import write_mesh

    rng = np.random.default_rng(3)
    # Two meshes, objects 0 and 1, each with faces of its own vertices.
    verts = rng.uniform(1.0, 99.0, size=(200, 3)).astype(np.float32)
    faces = np.concatenate([
        np.stack([rng.permutation(100)[:3] for _ in range(150)]),
        100 + np.stack([rng.permutation(100)[:3] for _ in range(150)]),
    ])
    src = tmp_path / "mesh.zarr"
    write_mesh(src, verts, faces, chunk_shape=(25.0,) * 3,
               object_ids=np.repeat([0, 1], 100),
               bounds=([0.0] * 3, [100.0] * 3))
    out = tmp_path / "out.zarr"

    summary = rechunk(str(src), RechunkSpec(by="object_id"), output=str(out),)

    assert summary["bins_created"] == 2 and summary["links"] == 300
    assert _link_positions(out) == _link_positions(src)
    assert validate(str(out), level=5).ok, validate(str(out), level=5).summary()


def test_object_ids_and_their_data_are_preserved(lines, tmp_path):
    """Bins interleave the ids, which used to be renumbered in bin order."""
    src, polys = lines
    root = open_store(str(src), mode="r+")
    lg = get_resolution_level(root, 0)
    from zarr_vectors.core.arrays import write_groupings

    groups = [[1, 3, 5, 7, 9, 11], [0, 2, 4, 6, 8, 10]]
    write_groupings(lg, dict(enumerate(groups)))
    out = tmp_path / "out.zarr"

    rechunk(str(src), RechunkSpec(by="group"), output=str(out))

    objs, out_lg = _objects(out)
    assert sorted(objs) == list(range(12))
    score = np.asarray(read_object_attributes(out_lg, "score")).ravel()
    for oid, (pts, t) in objs.items():
        np.testing.assert_array_equal(pts, polys[oid])
        np.testing.assert_array_equal(t, np.arange(len(pts)) + 100 * oid)
        assert score[oid] == pytest.approx(oid * 1.5)
    assert [sorted(g) for g in read_all_groupings(out_lg)] == groups
    # Bin 0 is group 0: the odd ids.
    got = read_polylines(str(out), attribute_filter={"group": 0})
    assert sorted(got["object_ids"]) == groups[0]


def test_sparse_object_ids_are_kept(tmp_path):
    from zarr_vectors.types.points import read_points, write_points

    pos = np.array([[5, 5, 5], [60, 5, 5], [5, 60, 5]], np.float32)
    src = tmp_path / "pts.zarr"
    write_points(str(src), pos, chunk_shape=(50.0,) * 3,
                 object_ids=np.array([2, 5, 9]))
    out = tmp_path / "out.zarr"

    rechunk(str(src), RechunkSpec(by="object_id"), output=str(out))

    lg = get_resolution_level(open_store(str(out), mode="r"), 0)
    ids, mans = read_object_manifest_rows(lg)
    assert sorted(i for i, m in zip(ids.tolist(), mans) if m) == [2, 5, 9]
    for oid, p in zip((2, 5, 9), pos):
        got = read_points(str(out), object_ids=[oid])
        np.testing.assert_array_equal(got["positions"], [p])


def test_a_rechunked_store_can_be_rebinned(lines, tmp_path):
    src, polys = lines
    mid, out = tmp_path / "mid.zarr", tmp_path / "out.zarr"
    rechunk(str(src), RechunkSpec(by="group"), output=str(mid))

    rechunk(str(mid), RechunkSpec(by="attribute:score", bins=[0, 5, 100]),
            output=str(out))

    objs, _ = _objects(out)
    for oid, (pts, _t) in objs.items():
        np.testing.assert_array_equal(pts, polys[oid])
    assert validate(str(out), level=4).ok


def test_a_fragment_shared_across_bins_is_refused(lines, tmp_path):
    """A vertex has one home; two objects in two bins cannot both own it."""
    from zarr_vectors.core.arrays import write_object_index

    src, _ = lines
    lg = get_resolution_level(open_store(str(src), mode="r+"), 0)
    ids, mans = read_object_manifest_rows(lg)
    shared = {int(i): list(m) for i, m in zip(ids.tolist(), mans)}
    shared[1] = shared[0][:1] + shared[1]
    write_object_index(lg, shared, sid_ndim=3)
    out = tmp_path / "out.zarr"
    out.mkdir()
    (out / "keep.txt").write_text("x")

    with pytest.raises(NotImplementedError, match="shares a fragment"):
        rechunk(str(src), RechunkSpec(by="object_id"), output=str(out),
                overwrite=True)
    assert (out / "keep.txt").exists()


def test_by_spatial_is_lossless_on_a_pyramid(tmp_path):
    from zarr_vectors.multiresolution.coarsen import build_pyramid
    from zarr_vectors.types.graphs import write_graph

    rng = np.random.default_rng(0)
    src = tmp_path / "graph.zarr"
    write_graph(src, positions=rng.uniform(0, 100, (40, 3)).astype(np.float32),
                edges=np.array([[i, i + 1] for i in range(39)]),
                object_ids=np.zeros(40, np.int64), chunk_shape=(25.0,) * 3,
                bounds=([0.0] * 3, [100.0] * 3))
    build_pyramid(src, factors=[(2.0, 1.0)])
    out = tmp_path / "out.zarr"
    rechunk(str(src), RechunkSpec(by="spatial", spatial_chunk_shape=(50.0,) * 3),
            output=str(out))
    root = open_store(str(out), mode="r")
    assert list_resolution_levels(root) == [0, 1]
    chunks, vi = read_link_arrays(get_resolution_level(root, 0), delta=0)
    assert len(vi) == 39
    assert read_level_metadata(root, 0).vertex_count == 40
    assert validate(str(out), level=5).ok
