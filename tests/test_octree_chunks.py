"""Octree chunks: an optional chunking of a level, converted to and from uniform.

Pinned here:

* the planner bounds every leaf by the target (unless one cell is heavier),
  covers every cell once and, balanced, keeps adjacent leaves within one
  depth;
* addresses (``d/i.j.k`` and octant paths) and link names (``o`` / ``o^Δ``)
  round-trip;
* ``to_octree`` then ``to_uniform`` gives back every cell's vertex rows,
  fragments, object manifests and links;
* on an octree level, ``read_graph`` / ``read_points`` (whole, box, object)
  answer what the uniform level answers, and per-leaf reads plus seams
  hold every link exactly once.
"""

from __future__ import annotations

import numpy as np
import pytest

import zarr_vectors as zv
from zarr_vectors.constants import CAP_OCTREE_CHUNKS
from zarr_vectors.core.arrays import (
    _parse_chunk_key,
    list_link_offsets,
    read_all_object_manifests,
    read_chunk_vertices,
    read_link_arrays,
)
from zarr_vectors.core.store import get_resolution_level, open_store, read_root_metadata
from zarr_vectors.octree_chunks import (
    OctreeError,
    format_link_segment,
    leaf_table,
    octree_meta,
    parse_link_segment,
    plan_leaves,
    read_octree,
    read_seams,
    to_octree,
    to_uniform,
)
from zarr_vectors.types.graphs import read_graph, write_graph
from zarr_vectors.types.points import read_points, write_points

BOX = 400.0
CELL = 25.0


def _uneven(n=8000, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.concatenate([rng.normal(120, 22, (n, 3)), rng.uniform(0, BOX, (n // 3, 3))])
    return np.clip(pos, 0.5, BOX - 0.5).astype(np.float32)


def _short_edges(pos, reach=5.0, keep=0.5, seed=0):
    """Pairs closer than ``reach`` (well under a cell), found by hashing."""
    rng = np.random.default_rng(seed)
    pairs = []
    # Two hashings, the second offset by half a bucket, so pairs straddling
    # a bucket -- and therefore a chunk -- boundary are found too.
    for origin in (0.0, reach / 2):
        cell = np.floor((pos + origin) / reach).astype(np.int64)
        key = (cell[:, 0] * 4096 + cell[:, 1]) * 4096 + cell[:, 2]
        order = np.argsort(key, kind="stable")
        ks = key[order]
        for shift in range(1, 6):
            a, b = order[:-shift], order[shift:]
            same = ks[:-shift] == ks[shift:]
            pairs.append(np.column_stack([a[same], b[same]]))
    pairs = np.concatenate(pairs)
    d = np.linalg.norm(pos[pairs[:, 0]] - pos[pairs[:, 1]], axis=1)
    pairs = pairs[(d < reach) & (rng.random(len(pairs)) < keep)]
    return np.unique(np.sort(pairs, axis=1), axis=0)


def _key(p):
    return tuple(np.round(np.asarray(p, dtype=np.float64), 4))


def _edge_set(positions, edges):
    return {frozenset((_key(positions[a]), _key(positions[b]))) for a, b in edges}


def _snapshot(path):
    lg = get_resolution_level(open_store(path), 0)
    cells = {}
    for k in lg.list_chunks("vertices"):
        cc = _parse_chunk_key(k)
        cells[cc] = [g.tobytes() for g in read_chunk_vertices(lg, cc, dtype=np.float32, ndim=3)]
    man = [[(tuple(int(c) for c in cc), int(f)) for cc, f in m] for m in read_all_object_manifests(lg)]
    edges = set()
    if list_link_offsets(lg, 0):
        ch, vi = read_link_arrays(lg, delta=0)
        edges = {frozenset([(tuple(ch[r, 0]), int(vi[r, 0])), (tuple(ch[r, 1]), int(vi[r, 1]))])
                 for r in range(len(vi))}
    return cells, man, edges


# -- planner, addresses, names ------------------------------------------------

def test_plan_bounds_covers_and_balances():
    pos = _uneven()
    cells, counts = np.unique(np.floor(pos / CELL).astype(np.int64), axis=0, return_counts=True)
    t = plan_leaves(cells, counts, target=400)
    assert t.weight.max() <= max(400, counts.max())
    lc = t.leaf_of_cells(cells)
    assert (lc >= 0).all()
    assert np.allclose(np.bincount(lc, counts, minlength=len(t)), t.weight)
    for n in range(len(t)):
        assert np.abs(t.depth[t.neighbours(n)] - t.depth[n]).max(initial=0) <= 1
    assert len(set(t.depth.tolist())) > 1, "uneven data should give leaves of several sizes"


def test_a_cell_heavier_than_the_target_stays_one_leaf():
    t = plan_leaves([[0, 0, 0], [3, 3, 3]], [1000, 1], target=10)
    assert int(t.depth.max()) == t.max_depth
    assert t.weight.max() == 1000


def test_addresses_round_trip():
    t = plan_leaves(np.argwhere(np.ones((8, 8, 8))), np.ones(512), target=1)
    n = t.index("3/4.5.6")
    assert t.key(n) == "3/4.5.6"
    assert t.path(n) == "111/001/010"
    assert t.index("111/001/010") == n


@pytest.mark.parametrize("off,delta,text", [((1, 0, 0), 0, "+1.0.0"), ((1, -1, 0), 1, "+1.-1.0^1"),
                                            ((0, 0, 0), 0, "0.0.0")])
def test_link_names(off, delta, text):
    assert format_link_segment(off, delta) == text
    assert parse_link_segment(text) == (off, delta)


def test_link_offsets_follow_the_ancestor_rule():
    # A depth-3 leaf beside a depth-2 one: 3/3.4.4 and 2/2.2.2.
    t = plan_leaves(np.argwhere(np.ones((8, 8, 8))), np.ones(512), target=1)
    from zarr_vectors.octree_chunks.table import LeafTable
    t = LeafTable.from_arrays(max_depth=3, origin_cell=(0, 0, 0), depth=[3, 2], ijk=[[3, 4, 4], [2, 2, 2]],
                              weight=[1, 1])
    o, d = t.link_offsets(np.array([0]), np.array([1]))
    assert o.tolist() == [[1, 0, 0]] and d.tolist() == [1]
    assert t.resolve(0, (1, 0, 0), 1) == 1


# -- conversion ----------------------------------------------------------------

@pytest.fixture(params=["unsharded", "sharded"])
def graph_store(request, tmp_path):
    pos = _uneven()
    edges = _short_edges(pos)
    oids = (np.arange(len(pos)) // 40).astype(np.int64)
    path = tmp_path / "g.zv"
    write_graph(path, pos, edges, chunk_shape=(CELL,) * 3, bounds=([0] * 3, [BOX] * 3), object_ids=oids,
                shard_shape=None if request.param == "unsharded" else 4)
    return path, pos, edges, oids


def test_round_trip_is_exact(graph_store):
    path = graph_store[0]
    before = _snapshot(path)
    summary = to_octree(path, target=300)
    assert summary["leaves"] < len(before[0])
    to_uniform(path)
    after = _snapshot(path)
    assert after[0] == before[0]
    assert after[1] == before[1]
    assert after[2] == before[2]


def test_marked_while_octree_and_unmarked_after(graph_store):
    path = graph_store[0]
    to_octree(path, target=300)
    root = open_store(path)
    assert CAP_OCTREE_CHUNKS in read_root_metadata(root).format_capabilities
    lg = get_resolution_level(root, 0)
    assert octree_meta(lg)["target"] == 300
    segs = list_link_offsets(lg, 0)
    assert len(segs) <= 53 and any("^1" in s for s in segs)
    to_uniform(path)
    root = open_store(path)
    assert CAP_OCTREE_CHUNKS not in read_root_metadata(root).format_capabilities
    assert octree_meta(get_resolution_level(root, 0)) is None


def test_leaves_are_bounded(graph_store):
    path, pos, *_ = graph_store
    s = to_octree(path, target=300)
    assert s["max_leaf_vertices"] <= max(300, s["max_cell_vertices"])
    t = leaf_table(get_resolution_level(open_store(path), 0))
    assert int(t.weight.sum()) == len(pos)


# -- reads on an octree level --------------------------------------------------

def _uniform_and_octree(graph_store, tmp_path):
    path, pos, edges, oids = graph_store
    import shutil
    other = tmp_path / "octree.zv"
    shutil.copytree(path, other)
    to_octree(other, target=300)
    return path, other


@pytest.mark.parametrize("query", [
    {}, {"bbox": ([50, 50, 50], [180, 160, 170])}, {"bbox": ([0, 0, 0], [30, 30, 30])},
    {"object_ids": [3]}, {"object_ids": [0, 17, 150, 199]},
])
def test_graph_reads_match_the_uniform_level(graph_store, tmp_path, query):
    uni, octr = _uniform_and_octree(graph_store, tmp_path)
    a, b = read_graph(uni, **query), read_graph(octr, **query)
    assert {_key(p) for p in a["positions"]} == {_key(p) for p in b["positions"]}
    assert a["node_count"] == b["node_count"]
    assert _edge_set(a["positions"], a["edges"]) == _edge_set(b["positions"], b["edges"])
    if "object_ids" in query:
        ka = sorted(zip(map(_key, a["positions"]), a["object_ids"].tolist()))
        kb = sorted(zip(map(_key, b["positions"]), b["object_ids"].tolist()))
        assert ka == kb


def test_leaves_and_seams_hold_every_link_once(graph_store, tmp_path):
    _, octr = _uniform_and_octree(graph_store, tmp_path)
    path, pos, edges, _ = graph_store
    t = leaf_table(get_resolution_level(open_store(octr), 0))
    inside, across = 0, 0
    for n in range(len(t)):
        inside += len(read_octree(octr, leaves=[n])["edges"])
        across += len(read_seams(octr, n)["row"])
    assert inside + across // 2 == len(edges)
    one = read_octree(octr, leaves=[0], halo=1)
    assert set(one["leaf"].tolist()) >= {0}


def test_select_on_an_octree_level(graph_store, tmp_path):
    uni, octr = _uniform_and_octree(graph_store, tmp_path)
    box = ([40, 40, 40], [160, 160, 160])
    a = zv.open(uni).level(0).select(bbox=box).read()
    b = zv.open(octr).level(0).select(bbox=box).read()
    assert a.vertex_count == b.vertex_count
    assert len(a.edges) == len(b.edges)


def test_uniform_cell_whitelists_are_refused(graph_store, tmp_path):
    _, octr = _uniform_and_octree(graph_store, tmp_path)
    with pytest.raises(OctreeError, match="leaves="):
        read_graph(octr, chunks=[(0, 0, 0)])


def test_point_clouds_round_trip_and_read(tmp_path):
    pos = _uneven(seed=4)
    oids = (np.arange(len(pos)) // 100).astype(np.int64)
    path = tmp_path / "p.zv"
    write_points(path, pos, chunk_shape=(CELL,) * 3, bounds=([0] * 3, [BOX] * 3), object_ids=oids)
    before = _snapshot(path)
    import shutil
    uni = tmp_path / "u.zv"
    shutil.copytree(path, uni)
    to_octree(path, target=250)
    for q in ({}, {"bbox": ([60, 60, 60], [150, 150, 150])}, {"object_ids": [2, 9]}):
        a, b = read_points(uni, **q), read_points(path, **q)
        assert sorted(map(_key, a["positions"])) == sorted(map(_key, b["positions"]))
    to_uniform(path)
    assert _snapshot(path) == before


def test_octree_sharded_store_converts(tmp_path):
    pytest.importorskip("zarr_octree")
    pos = _uneven(n=4000, seed=5)
    edges = _short_edges(pos, seed=5)
    oids = (np.arange(len(pos)) // 40).astype(np.int64)
    ds = zv.create(tmp_path / "o.zv", schema=zv.Schema(
        bounds=([0] * 3, [BOX] * 3), kind="graph",
        layout=zv.Layout(cell_size=[CELL] * 3, shard_cells=4, compression="zstd", octree=True)))
    ds.add_graph(pos, edges, object_ids=oids)
    path = tmp_path / "o.zv"
    before = _snapshot(path)
    to_octree(path, target=300)
    g = read_graph(path)
    assert g["node_count"] == len(pos) and g["edge_count"] == len(edges)
    to_uniform(path)
    assert _snapshot(path) == before


def test_unsupported_kinds_are_refused(tmp_path):
    from zarr_vectors.types.meshes import write_mesh
    rng = np.random.default_rng(0)
    path = tmp_path / "m.zv"
    write_mesh(path, _uneven(n=300), rng.integers(0, 300, (100, 3)), chunk_shape=(CELL,) * 3)
    with pytest.raises(OctreeError, match="point clouds and graphs"):
        to_octree(path, target=100)
