"""``rechunk()`` carries what it can and refuses what it cannot.

It used to copy vertices, manifests and (for ``by="group"`` only)
groupings, and silently drop links, vertex and object attributes, group
attributes and every level above 0; it stamped ``vertex_count: 0``; and
it deleted an existing ``output`` directory without asking.
``RechunkSpec(by="spatial")`` went down the same lossy path.
"""

from __future__ import annotations

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
from zarr_vectors.types.polylines import write_polylines
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


def test_explicit_links_are_refused(tmp_path):
    from zarr_vectors.types.graphs import write_graph

    rng = np.random.default_rng(0)
    src = tmp_path / "graph.zarr"
    write_graph(src, positions=rng.uniform(0, 100, (40, 3)).astype(np.float32),
                edges=np.array([[i, (i * 7 + 3) % 40] for i in range(40)]),
                object_ids=np.zeros(40, np.int64), chunk_shape=(25.0,) * 3,
                bounds=([0.0] * 3, [100.0] * 3))
    with pytest.raises(NotImplementedError, match="links"):
        rechunk(str(src), RechunkSpec(by="object_id", bins=[0, 1]),
                output=str(tmp_path / "out.zarr"))


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
