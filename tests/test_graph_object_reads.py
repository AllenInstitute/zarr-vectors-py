"""``read_graph(object_ids=)``: graphs and skeletons read by object.

The answer is pinned against the input: the named objects' nodes and the
edges among them, compared by position (the skeleton writer reorders
nodes).  Objects are random trees spread over several chunks, with more
than one object per chunk, so the fragment, cross-chunk and implicit-edge
paths are all exercised.
"""

from __future__ import annotations

import numpy as np
import pytest

import zarr_vectors as zv
from zarr_vectors.exceptions import ZVError
from zarr_vectors.types.graphs import read_graph, write_graph

CHUNK = (100.0, 100.0, 100.0)
BOUNDS = ([0.0, 0.0, 0.0], [400.0, 400.0, 400.0])


def _forest(n_objects=12, size=40, seed=0):
    """Random walks grown into trees: ``(positions, edges, object_ids)``,
    edges as ``(child, parent)``, one rooted tree per object."""
    rng = np.random.default_rng(seed)
    pos, edges, oids = [], [], []
    for oid in range(n_objects):
        base = len(pos)
        p = [rng.uniform(40, 360, 3)]
        for i in range(1, size):
            parent = int(rng.integers(max(0, i - 4), i))
            p.append(np.clip(p[parent] + rng.normal(0, 25, 3), 1, 399))
            edges.append((base + i, base + parent))
        pos.extend(p)
        oids.extend([oid] * size)
    return (np.asarray(pos, dtype=np.float32), np.asarray(edges, dtype=np.int64),
            np.asarray(oids, dtype=np.int64))


def _key(p):
    return tuple(np.round(np.asarray(p, dtype=np.float64), 3))


def _edge_set(positions, edges, directed):
    out = set()
    for a, b in edges:
        e = (_key(positions[a]), _key(positions[b]))
        out.add(e if directed else frozenset(e))
    return out


def _expected(positions, edges, oids, wanted, directed, inside=None):
    keep = np.isin(oids, wanted)
    if inside is not None:
        keep &= inside
    nodes = {_key(p) for p in positions[keep]}
    e = edges[keep[edges[:, 0]] & keep[edges[:, 1]]]
    return nodes, _edge_set(positions, e, directed)


@pytest.fixture(params=["graph", "skeleton"])
def store(request, tmp_path):
    positions, edges, oids = _forest()
    path = tmp_path / f"{request.param}.zarrvectors"
    write_graph(path, positions, edges, chunk_shape=CHUNK, bounds=BOUNDS,
                kind=request.param, object_ids=oids)
    return path, request.param, positions, edges, oids


@pytest.mark.parametrize("wanted", [[3], [0, 7, 11], [5, 5, 2]])
def test_objects_read_exactly_their_nodes_and_edges(store, wanted):
    path, kind, positions, edges, oids = store
    got = read_graph(path, object_ids=wanted)
    directed = kind == "skeleton"
    nodes, edge_set = _expected(positions, edges, oids, wanted, directed)
    assert {_key(p) for p in got["positions"]} == nodes
    assert got["node_count"] == len(nodes)
    assert _edge_set(got["positions"], got["edges"], directed) == edge_set
    assert got["edge_count"] == len(edge_set)
    # Grouped by object, in the order asked for.
    order = list(dict.fromkeys(wanted))
    assert list(dict.fromkeys(got["object_ids"].tolist())) == order
    for oid in order:
        mine = got["object_ids"] == oid
        assert {_key(p) for p in got["positions"][mine]} == {_key(p) for p in positions[oids == oid]}


def test_objects_span_several_chunks(store):
    """Guards the fixture: the cross-chunk paths are actually exercised."""
    _, _, positions, _, oids = store
    cells = {tuple(c) for c in (positions[oids == 3] // CHUNK[0]).astype(int)}
    assert len(cells) > 1


def test_an_absent_object_reads_nothing(store):
    path, *_ = store
    got = read_graph(path, object_ids=[999])
    assert got["node_count"] == 0 and got["edge_count"] == 0
    assert len(got["object_ids"]) == 0


def test_skeleton_object_read_is_a_tree(store):
    path, kind, *_ = store
    if kind != "skeleton":
        pytest.skip("skeletons only")
    got = read_graph(path, object_ids=[1, 4])
    # One root per object: every node but one per tree has a parent.
    assert got["edge_count"] == got["node_count"] - 2
    assert len(set(got["edges"][:, 0].tolist())) == got["edge_count"]


def test_bbox_scopes_an_object_read(store):
    path, kind, positions, edges, oids = store
    bbox = (np.array([0.0, 0.0, 0.0]), np.array([200.0, 400.0, 400.0]))
    inside = np.all((positions >= bbox[0]) & (positions <= bbox[1]), axis=1)
    got = read_graph(path, object_ids=[2, 6], bbox=bbox)
    directed = kind == "skeleton"
    nodes, edge_set = _expected(positions, edges, oids, [2, 6], directed, inside)
    assert {_key(p) for p in got["positions"]} == nodes
    assert _edge_set(got["positions"], got["edges"], directed) == edge_set


def test_select_objects_on_a_graph(store):
    path, kind, positions, edges, oids = store
    got = zv.open(path).level(0).select(objects=[3]).read()
    assert got.vertex_count == int((oids == 3).sum())
    assert np.array_equal(np.unique(got.object_ids), [3])
    assert len(got.edges) == int(np.isin(edges[:, 0], np.flatnonzero(oids == 3)).sum())


def test_groups_on_a_graph_still_refuse(store):
    path, *_ = store
    with pytest.raises(ZVError, match=r"select\(groups=\.\.\.\)"):
        zv.open(path).level(0).select(groups=[0]).read()


@pytest.mark.parametrize("octree", [False, True], ids=["indexed", "octree"])
@pytest.mark.parametrize("reader", ["standard", "optimised", "zarr"])
@pytest.mark.parametrize("kind", ["graph", "skeleton"])
def test_sharded_stores_and_every_reader(tmp_path, monkeypatch, octree, reader, kind):
    if octree:
        pytest.importorskip("zarr_octree")
    monkeypatch.setenv("ZARR_VECTORS_SHARD_READER", reader)
    positions, edges, oids = _forest(seed=3)
    ds = zv.create(tmp_path / f"{kind}.zv", schema=zv.Schema(
        bounds=BOUNDS, kind=kind,
        layout=zv.Layout(cells=4, shard_cells=2, compression="zstd", octree=octree)))
    ds.add_graph(positions, edges, object_ids=oids)
    directed = kind == "skeleton"
    got = read_graph(str(tmp_path / f"{kind}.zv"), object_ids=[1, 8])
    nodes, edge_set = _expected(positions, edges, oids, [1, 8], directed)
    assert {_key(p) for p in got["positions"]} == nodes
    assert _edge_set(got["positions"], got["edges"], directed) == edge_set


def test_skeleton_object_read_survives_an_empty_last_fragment(tmp_path):
    """An object manifest may name an empty vertex fragment (a custom
    writer, an edit).  Taken last, its start equalled the row count and
    the skeleton rebuild indexed past the end."""
    from zarr_vectors.core.arrays import (
        read_chunk_vertices, read_object_manifests, write_chunk_vertices,
        write_object_index,
    )
    from zarr_vectors.core.store import get_resolution_level, open_store

    positions = np.array([[50, 50, 50], [60, 50, 50], [70, 50, 50]], dtype=np.float32)
    edges = np.array([[1, 0], [2, 1]], dtype=np.int64)
    path = tmp_path / "empty.zarrvectors"
    write_graph(path, positions, edges, chunk_shape=CHUNK, bounds=BOUNDS,
                kind="skeleton", object_ids=np.zeros(3, dtype=np.int64))
    level = get_resolution_level(open_store(str(path), mode="r+"), 0)
    groups = read_chunk_vertices(level, (0, 0, 0), dtype=np.float32, ndim=3)
    write_chunk_vertices(level, (0, 0, 0), [*groups, np.zeros((0, 3), np.float32)],
                         dtype=np.float32)
    manifests = read_object_manifests(level)
    manifests[0] = [*manifests[0], ((0, 0, 0), len(groups))]
    write_object_index(level, manifests, sid_ndim=3)

    got = read_graph(path, object_ids=[0])
    assert got["node_count"] == 3
    assert _edge_set(got["positions"], got["edges"], True) == _edge_set(positions, edges, True)
