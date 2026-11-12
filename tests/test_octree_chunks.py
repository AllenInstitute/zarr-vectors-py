"""Octree chunks: an optional chunking of a level, converted to and from uniform.

Pinned here:

* the planner bounds every leaf by the target (unless one cell is heavier),
  covers every cell once and, balanced, keeps adjacent leaves within one
  depth;
* addresses (``d/i.j.k`` and octant paths) and link names (``o`` / ``o^Δ``)
  round-trip;
* ``to_octree`` then ``to_uniform`` gives back every cell's vertex rows,
  fragments, object manifests (under their own, possibly sparse, object
  ids) and links;
* a conversion that cannot be done, or fails or is killed part way,
  leaves the level reading exactly as before -- or, once the swap has
  begun, a level the next call finishes; links between levels, into or
  out of the converted one, are refused;
* the root keeps ``octree_chunks`` while any level is octree; an empty
  level converts both ways;
* on an octree level, ``read_graph`` / ``read_points`` (whole, box, object)
  answer what the uniform level answers, and per-leaf reads plus seams
  hold every link exactly once;
* on a sharded level every leaf lies inside one shard -- also for shard
  sides that are not powers of two -- where the uncapped plan would not,
  the cap is recorded, the level keeps its packing both ways, and
  ``validate`` flags leaves a later reshard splits.
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
    read_object_id_table,
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


def _snapshot(path, level=0):
    lg = get_resolution_level(open_store(path), level)
    cells = {}
    for k in lg.list_chunks("vertices"):
        cc = _parse_chunk_key(k)
        cells[cc] = [g.tobytes() for g in read_chunk_vertices(lg, cc, dtype=np.float32, ndim=3, _leaf=True)]
    # Keyed by the real object id: a row-indexed list would not see ids renumbered.
    man = {}
    meta = lg.read_array_meta("object_index") if lg.array_exists("object_index") else {}
    n = int(meta.get("num_objects", 0) or 0)
    if n and "sid_ndim" in meta:
        ids = read_object_id_table(lg)
        rows = read_all_object_manifests(lg)[:n]
        man = {int(ids[r]) if ids is not None else r: [(tuple(int(c) for c in cc), int(f)) for cc, f in m]
               for r, m in enumerate(rows)}
    edges = set()
    if octree_meta(lg) is not None:
        from zarr_vectors.octree_chunks.convert import read_octree_link_cells
        src, r_src, oth, r_oth = read_octree_link_cells(lg, leaf_table(lg))
        edges = {frozenset([(int(a), int(b)), (int(c), int(d))]) for a, b, c, d in zip(src, r_src, oth, r_oth)}
    elif list_link_offsets(lg, 0):
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
    attrs = get_resolution_level(open_store(path), 0).attrs.to_dict()
    summary = to_octree(path, target=300)
    assert summary["leaves"] < len(before[0])
    to_uniform(path)
    after = _snapshot(path)
    assert after[0] == before[0]
    assert after[1] == before[1]
    assert after[2] == before[2]
    assert get_resolution_level(open_store(path), 0).attrs.to_dict() == attrs
    assert _no_staging(path)


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


# -- object ids ------------------------------------------------------------------

def _small_graph(path, oids, *, n=2000, seed=7, shard=None):
    pos = _uneven(n=n, seed=seed)
    edges = _short_edges(pos, seed=seed)
    oids = np.asarray(oids(len(pos)) if callable(oids) else oids, dtype=np.int64)
    write_graph(path, pos, edges, chunk_shape=(CELL,) * 3, bounds=([0] * 3, [BOX] * 3), object_ids=oids,
                shard_shape=shard)
    return pos, edges, oids


def _object_reads(path, ids, reader=read_graph):
    out = {}
    for oid in ids:
        r = reader(path, object_ids=[oid])
        key = sorted(map(_key, r["positions"]))
        if "edges" in r:
            out[oid] = (key, _edge_set(r["positions"], r["edges"]))
        else:
            out[oid] = (key, None)
    return out


@pytest.mark.parametrize("shard", [None, 4])
def test_sparse_object_ids_are_kept_both_ways(tmp_path, shard):
    path = tmp_path / "sparse.zv"
    _, _, oids = _small_graph(path, lambda n: (np.arange(n) // 40) * 1000 + 7, shard=shard)
    ids = np.unique(oids)
    asked = [int(ids[0]), int(ids[1]), int(ids[-1]), 1, 0]   # 1 and 0 are not ids here
    before, reads = _snapshot(path), _object_reads(path, asked)
    assert reads[1007][0] and not reads[1][0]
    table = read_object_id_table(get_resolution_level(open_store(path), 0)).tolist()

    to_octree(path, target=150)
    assert read_object_id_table(get_resolution_level(open_store(path), 0)).tolist() == table
    assert _object_reads(path, asked) == reads
    g = read_graph(path, object_ids=[1007])
    assert set(g["object_ids"].tolist()) == {1007}

    to_uniform(path)
    assert read_object_id_table(get_resolution_level(open_store(path), 0)).tolist() == table
    assert _object_reads(path, asked) == reads
    assert _snapshot(path) == before


def test_sparse_object_ids_on_points(tmp_path):
    pos = _uneven(n=2000, seed=8)
    oids = (np.arange(len(pos)) // 100) * 7919 + 3
    path = tmp_path / "p.zv"
    write_points(path, pos, chunk_shape=(CELL,) * 3, bounds=([0] * 3, [BOX] * 3), object_ids=oids)
    asked = [3, 7922, int(oids[-1]), 1]
    before, reads = _snapshot(path), _object_reads(path, asked, read_points)
    to_octree(path, target=120)
    assert _object_reads(path, asked, read_points) == reads
    to_uniform(path)
    assert _snapshot(path) == before


@pytest.mark.parametrize("oids", [[5, 5, 5], [0, 0, 5], [9, 9, 2]])
def test_repeated_object_ids_in_one_cell(tmp_path, oids):
    pos = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], np.float32)
    path = tmp_path / "one.zv"
    write_graph(path, pos, np.array([[0, 1], [1, 2]]), chunk_shape=(CELL,) * 3, bounds=([0] * 3, [BOX] * 3),
                object_ids=np.array(oids))
    asked = sorted(set(oids) | {0, 5})
    before, reads = _snapshot(path), _object_reads(path, asked)
    to_octree(path, target=1)
    assert _object_reads(path, asked) == reads
    to_uniform(path)
    assert _object_reads(path, asked) == reads
    assert _snapshot(path) == before


@pytest.mark.parametrize("layout", ["vlen", "dense"])
def test_the_object_layer_keeps_its_layout_and_row_shards(tmp_path, layout):
    from zarr_vectors import building as zb
    from zarr_vectors.core import dense_manifests as dense
    from zarr_vectors.core.arrays import write_object_index
    path = tmp_path / "g.zv"
    _small_graph(path, lambda n: (np.arange(n) // 40) * 3 + 11)
    root = open_store(path, mode="r+")
    block = dict(root.attrs.to_dict()["zarr_vectors"])
    root.attrs.update({"zarr_vectors": {**block, "manifest_layout": layout}})
    lg = get_resolution_level(root, 0)
    by_id = _snapshot(path)[1]
    lg.delete_subtree("object_index")
    write_object_index(lg, by_id, sid_ndim=3)
    zb.shard_object_layer(lg, zb.OBJECT_SHARD_ROW_MULTIPLE)
    before = _state(path, ids=(11, 14, 0))
    assert before[2][11][0] and not before[2][0][0]

    def layer():
        lg = get_resolution_level(open_store(path), 0)
        return dense.is_dense(lg), zb.object_shard_rows(lg)

    assert layer() == (layout == "dense", zb.OBJECT_SHARD_ROW_MULTIPLE)
    to_octree(path, target=150)
    assert layer() == (layout == "dense", zb.OBJECT_SHARD_ROW_MULTIPLE)
    assert _object_reads(path, [11, 14, 0]) == before[2]
    to_uniform(path)
    assert layer() == (layout == "dense", zb.OBJECT_SHARD_ROW_MULTIPLE)
    assert _state(path, ids=(11, 14, 0)) == before


# -- refusals and failures leave the level as it was --------------------------

def _state(path, ids=(0, 3, 40)):
    """What a reader sees: the snapshot plus whole and per-object graph reads."""
    g = read_graph(path)
    return _snapshot(path), _edge_set(g["positions"], g["edges"]), _object_reads(path, ids)


def _no_staging(path):
    from zarr_vectors.octree_chunks.convert import STAGING
    return STAGING not in get_resolution_level(open_store(path), 0)


def test_a_link_longer_than_a_cell_is_refused_before_anything_changes(tmp_path):
    pos = np.array([[10, 10, 10], [60, 10, 10], [12, 10, 10], [62, 10, 10]], np.float32)
    path = tmp_path / "long.zv"
    write_graph(path, pos, np.array([[0, 1], [0, 2], [1, 3]]), chunk_shape=(CELL,) * 3,
                bounds=([0] * 3, [100] * 3))
    before = _snapshot(path)
    with pytest.raises(OctreeError, match="not adjacent"):
        to_octree(path, target=1)
    assert _snapshot(path) == before
    assert read_graph(path)["edge_count"] == 3
    assert octree_meta(get_resolution_level(open_store(path), 0)) is None
    assert _no_staging(path)


class _Boom(Exception):
    pass


def _fail_on(monkeypatch, name, nth=1, exc=_Boom):
    """Make ``convert.<name>`` raise on its ``nth`` call; also check that
    every group it is handed is marked as being converted."""
    from zarr_vectors.octree_chunks import convert as cv
    from zarr_vectors.octree_chunks.guard import is_converting
    orig = getattr(cv, name)
    calls = {"n": 0}

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        for a in args:
            if hasattr(a, "zarr_group"):
                assert is_converting(a), f"{name} wrote through an unmarked group"
        if calls["n"] == nth:
            raise exc(f"injected failure in {name}")
        return orig(*args, **kwargs)

    monkeypatch.setattr(cv, name, wrapped)
    return calls


# Failures while the new layout is being staged: the level is untouched.
_STAGE_FAILURES = [
    ("write_chunk_vertices", 1, _Boom), ("write_chunk_vertices", 40, KeyboardInterrupt),
    ("write_object_index", 1, _Boom), ("stamp_fragments_tile", 1, _Boom), ("_seal", 1, KeyboardInterrupt),
]


@pytest.mark.parametrize("name,nth,exc,shard", [
    *[(*f, None) for f in _STAGE_FAILURES], ("_write_octree_links", 1, _Boom, None),
    ("write_chunk_vertices", 40, KeyboardInterrupt, 4), ("_write_octree_links", 1, _Boom, 4),
])
def test_to_octree_failing_while_staging_leaves_the_level_intact(tmp_path, monkeypatch, shard, name, nth, exc):
    path = tmp_path / "g.zv"
    _small_graph(path, lambda n: np.arange(n) // 40, shard=shard)
    before = _state(path)
    _fail_on(monkeypatch, name, nth, exc)
    with pytest.raises(exc):
        to_octree(path, target=150)
    monkeypatch.undo()
    assert _state(path) == before
    root = open_store(path)
    assert octree_meta(get_resolution_level(root, 0)) is None
    assert CAP_OCTREE_CHUNKS not in read_root_metadata(root).format_capabilities
    # Running it again starts afresh, and the result round-trips exactly.
    assert "resumed" not in to_octree(path, target=150)
    assert _no_staging(path)
    assert _edge_set(*(lambda g: (g["positions"], g["edges"]))(read_graph(path))) == before[1]
    to_uniform(path)
    assert _state(path) == before


@pytest.mark.parametrize("name,nth,exc", [*_STAGE_FAILURES, ("write_links", 1, _Boom)])
def test_to_uniform_failing_while_staging_leaves_the_level_intact(tmp_path, monkeypatch, name, nth, exc):
    path = tmp_path / "g.zv"
    _small_graph(path, lambda n: np.arange(n) // 40)
    uniform = _state(path)
    to_octree(path, target=150)
    octree = _state(path)
    _fail_on(monkeypatch, name, nth, exc)
    with pytest.raises(exc):
        to_uniform(path)
    monkeypatch.undo()
    assert _state(path) == octree
    assert octree_meta(get_resolution_level(open_store(path), 0))["leaves"] > 1
    to_uniform(path)
    assert _no_staging(path)
    assert _state(path) == uniform


@pytest.mark.parametrize("nth,shard", [(1, None), (3, 4), (5, None)])
def test_to_octree_killed_during_the_swap_is_finished_by_the_next_call(tmp_path, monkeypatch, shard, nth):
    path = tmp_path / "g.zv"
    _small_graph(path, lambda n: np.arange(n) // 40, shard=shard)
    before = _state(path)
    expect = to_octree(_copy(path, tmp_path / "clean.zv"), target=150)
    _fail_on(monkeypatch, "_move", nth, KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt):
        to_octree(path, target=150)
    monkeypatch.undo()
    # Half swapped: readers refuse the level rather than mixing layouts.
    with pytest.raises(OctreeError, match="part way through to_octree"):
        read_graph(path)
    assert CAP_OCTREE_CHUNKS in read_root_metadata(open_store(path)).format_capabilities
    assert to_octree(path, target=150) == {**expect, "resumed": True}
    assert _no_staging(path)
    assert _snapshot(path) == _snapshot(tmp_path / "clean.zv")
    to_uniform(path)
    assert _state(path) == before


def _not_local(path):
    """``path`` through a store that is no LocalStore, so the swap takes the
    copy-then-delete branch an object store does (a ``memory://`` path does
    not: it opens as a LocalStore)."""
    from zarr.storage import LocalStore, WrapperStore
    return WrapperStore(LocalStore(str(path)))


def test_the_swap_copies_on_a_store_without_renames(tmp_path, monkeypatch):
    path = tmp_path / "g.zv"
    _small_graph(path, lambda n: (np.arange(n) // 40) * 5 + 2, n=1500)
    before = _state(path, ids=(2, 7, 0))
    _fail_on(monkeypatch, "_move", 3, KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt):
        to_octree(_not_local(path), target=150)
    monkeypatch.undo()
    assert to_octree(_not_local(path), target=150)["resumed"] is True
    assert _no_staging(path)
    assert _object_reads(path, [2, 7, 0]) == before[2]
    to_uniform(path)
    assert _state(path, ids=(2, 7, 0)) == before


def test_a_kill_while_deleting_a_moved_staged_array_loses_nothing(tmp_path, monkeypatch):
    # An object store deletes a staged array key by key, its zarr.json
    # (sorting after c/...) last: a kill there leaves the staged array
    # listed but short of cells, after it was copied into place.  The
    # next call must not copy it over the live one again.
    from zarr.core.sync import sync

    from zarr_vectors.core.group import Group
    from zarr_vectors.octree_chunks.convert import STAGING
    path = tmp_path / "g.zv"
    _small_graph(path, lambda n: (np.arange(n) // 40) * 5 + 2, n=1500)
    before = _state(path, ids=(2, 7, 0))
    orig = Group.delete_subtree
    deleted = []

    def partial(self, name):
        if name == "vertices" and self.zarr_group.path.endswith(STAGING):
            store, prefix = self.zarr_group.store, f"{self.zarr_group.path}/vertices/c/"

            async def half():
                keys = sorted([k async for k in store.list_prefix(prefix)])
                for k in keys[: len(keys) // 2]:
                    await store.delete(k)
                    deleted.append(k)

            sync(half())
            raise KeyboardInterrupt
        return orig(self, name)

    monkeypatch.setattr(Group, "delete_subtree", partial)
    with pytest.raises(KeyboardInterrupt):
        to_octree(_not_local(path), target=150)
    monkeypatch.undo()
    assert deleted
    assert to_octree(_not_local(path), target=150)["resumed"] is True
    assert _no_staging(path)
    assert _object_reads(path, [2, 7, 0]) == before[2]
    to_uniform(path)
    assert _state(path, ids=(2, 7, 0)) == before


@pytest.mark.parametrize("repack", ["shard_store", "unshard_store", "set_octree"])
def test_the_shard_repackers_refuse_a_level_part_way_through_a_conversion(tmp_path, monkeypatch, repack):
    # They would repack the live arrays and not the staged ones swapped in
    # later, leaving the level packed two ways.
    from zarr_vectors import sharding
    path = tmp_path / "g.zv"
    _small_graph(path, lambda n: np.arange(n) // 40, shard=4 if repack != "shard_store" else None)
    _fail_on(monkeypatch, "_move", 2, KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt):
        to_octree(path, target=150)
    monkeypatch.undo()
    before = _files(path)
    kwargs = {"shard_store": {"shard_shape": 4}, "unshard_store": {}, "set_octree": {"octree": False}}[repack]
    with pytest.raises(OctreeError, match="part way through to_octree"):
        getattr(sharding, repack)(str(path), **kwargs)
    assert _files(path) == before
    assert to_octree(path, target=150)["resumed"] is True


def test_to_uniform_after_an_interrupted_to_octree_finishes_it_first(tmp_path, monkeypatch):
    path = tmp_path / "g.zv"
    _small_graph(path, lambda n: np.arange(n) // 40)
    before = _state(path)
    _fail_on(monkeypatch, "_move", 2, _Boom)
    with pytest.raises(_Boom):
        to_octree(path, target=150)
    monkeypatch.undo()
    assert to_uniform(path)["cells"] == len(before[0][0])
    assert _state(path) == before
    assert CAP_OCTREE_CHUNKS not in read_root_metadata(open_store(path)).format_capabilities


def test_to_uniform_killed_during_the_swap_is_finished_by_the_next_call(tmp_path, monkeypatch):
    path = tmp_path / "g.zv"
    _small_graph(path, lambda n: (np.arange(n) // 40) * 10 + 1)
    before = _state(path)
    to_octree(path, target=150)
    _fail_on(monkeypatch, "_move", 4, KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt):
        to_uniform(path)
    monkeypatch.undo()
    with pytest.raises(OctreeError, match="part way through to_uniform"):
        read_graph(path)
    assert to_uniform(path)["resumed"] is True
    assert _no_staging(path)
    assert _state(path) == before


def test_a_staged_copy_whose_swap_never_began_is_discarded(tmp_path, monkeypatch):
    # Killed after staging but before the level was marked: the level is as
    # it was (and may be written meanwhile), so the next call redoes the work.
    from zarr_vectors.octree_chunks import convert as cv
    path = tmp_path / "g.zv"
    _small_graph(path, lambda n: np.arange(n) // 40)
    before = _state(path)

    def killed(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(cv, "_swap", killed)
    with pytest.raises(KeyboardInterrupt):
        to_octree(path, target=150)
    monkeypatch.undo()
    assert not _no_staging(path)
    assert _state(path) == before
    assert "resumed" not in to_octree(path, target=150)
    assert _no_staging(path)
    to_uniform(path)
    assert _state(path) == before


def _files(path):
    """Every file under ``path`` and its bytes."""
    from pathlib import Path
    return {str(f.relative_to(path)): f.read_bytes() for f in sorted(Path(path).rglob("*")) if f.is_file()}


def _copy(src, dst):
    import shutil
    shutil.copytree(src, dst)
    return dst


# -- links between levels, the capability, empty levels ---------------------------

def _pyramid(path, cross_level_storage=None):
    _small_graph(path, lambda n: np.arange(n) // 40)
    kw = {} if cross_level_storage is None else {"cross_level_storage": cross_level_storage}
    zv.open(path, mode="r+").build_pyramid(factors=[(2.0, 1.0)], **kw)


def _cross_deltas(path, level):
    lg = get_resolution_level(open_store(path), level)
    return [d for d in (-1, 1) if list_link_offsets(lg, d)]


def test_links_between_levels_are_refused_both_ways(tmp_path):
    path = tmp_path / "pyr.zv"
    _pyramid(path)
    assert _cross_deltas(path, 1), "the pyramid should link level 1 to level 0"
    before = [_snapshot(path, 0), _snapshot(path, 1)]
    for level in (0, 1):
        with pytest.raises(OctreeError, match="links to level"):
            to_octree(path, target=150, level=level)
    assert [_snapshot(path, 0), _snapshot(path, 1)] == before

    # An octree level that another level links into is not converted back.
    other = tmp_path / "plain.zv"
    _pyramid(other, cross_level_storage="none")
    assert not _cross_deltas(other, 0) and not _cross_deltas(other, 1)
    to_octree(other, target=150, level=0)
    import shutil
    shutil.copytree(path / "1" / "links", other / "1" / "links", dirs_exist_ok=True)
    assert _cross_deltas(other, 1)
    octree = _snapshot(other, 0)
    with pytest.raises(OctreeError, match="links to level"):
        to_uniform(other, level=0)
    assert _snapshot(other, 0) == octree


def test_capability_stays_while_any_level_is_octree(tmp_path):
    path = tmp_path / "cap.zv"
    write_points(path, _uneven(n=3000, seed=3), chunk_shape=(CELL,) * 3, bounds=([0] * 3, [BOX] * 3))
    zv.open(path, mode="r+").build_pyramid(factors=[(2.0, 1.0)], cross_level_storage="none")

    def caps():
        return read_root_metadata(open_store(path)).format_capabilities

    to_octree(path, target=150, level=0)
    to_octree(path, target=150, level=1)
    to_uniform(path, level=1)
    assert octree_meta(get_resolution_level(open_store(path), 0)) is not None
    assert CAP_OCTREE_CHUNKS in caps()
    to_uniform(path, level=0)
    assert CAP_OCTREE_CHUNKS not in caps()


@pytest.mark.parametrize("kind", ["graph", "points"])
def test_an_empty_level_converts_both_ways(tmp_path, kind):
    path = tmp_path / "empty.zv"
    box = dict(chunk_shape=(CELL,) * 3, bounds=([0] * 3, [100] * 3))
    if kind == "graph":
        write_graph(path, np.zeros((0, 3), np.float32), np.zeros((0, 2), np.int64), **box)
        read = read_graph
    else:
        write_points(path, np.zeros((0, 3), np.float32), **box)
        read = read_points
    before = _snapshot(path)
    s = to_octree(path, target=10)
    assert s["leaves"] == 0 and s["cells"] == 0
    assert octree_meta(get_resolution_level(open_store(path), 0))["leaves"] == 0
    r = read(path)
    assert len(r["positions"]) == 0
    if kind == "graph":
        assert r["edge_count"] == 0
    assert to_uniform(path)["cells"] == 0
    assert octree_meta(get_resolution_level(open_store(path), 0)) is None
    assert len(read(path)["positions"]) == 0
    assert _snapshot(path) == before


# -- leaves and shards ------------------------------------------------------------

SHARDS = [(2, 2, 2), (4, 4, 4), (4, 2, 8), (6, 6, 6)]
WIDE = 2 * BOX  # 32 cells per side: room for the uncapped plan to make 8-cell leaves


def _skewed(n=8000, seed=0):
    """A dense cluster in one corner and sparse space elsewhere."""
    rng = np.random.default_rng(seed)
    pos = np.concatenate([rng.normal(120, 22, (n, 3)), rng.uniform(0, WIDE, (n // 3, 3))])
    return np.clip(pos, 0.5, WIDE - 0.5).astype(np.float32)


def _cells_and_counts(pos=None):
    pos = _skewed() if pos is None else pos
    return np.unique(np.floor(pos / CELL).astype(np.int64), axis=0, return_counts=True)


@pytest.mark.parametrize("shard", SHARDS)
@pytest.mark.parametrize("shard_origin", [(0, 0, 0), (-3, 1, -2)])
def test_capped_leaves_stay_inside_one_shard(shard, shard_origin):
    from zarr_vectors.octree_chunks.table import shard_leaf_cap

    cells, counts = _cells_and_counts()
    cap = shard_leaf_cap(shard)
    assert cap == {(2, 2, 2): 2, (4, 4, 4): 4, (4, 2, 8): 2, (6, 6, 6): 2}[shard]
    free = plan_leaves(cells, counts, target=1200)
    # The test bites: unconstrained, sparse space gets leaves wider than a shard allows.
    assert int(free.sides().max()) > cap and len(free.straddling(shard, shard_origin))
    t = plan_leaves(cells, counts, target=1200, shard_shape=shard, shard_origin=shard_origin)
    assert t.max_leaf_cells == cap and t.shard_shape == shard and t.shard_origin == shard_origin
    assert int(t.sides().max()) <= cap
    assert not len(t.straddling(shard, shard_origin))
    assert not ((np.asarray(t.origin_cell) - shard_origin) % shard).any()
    lc = t.leaf_of_cells(cells)
    assert (lc >= 0).all() and np.allclose(np.bincount(lc, counts, minlength=len(t)), t.weight)
    assert t.weight.max() <= max(1200, counts.max())
    for n in range(len(t)):  # balancing kept, and never forced a leaf past the cap
        assert np.abs(t.depth[t.neighbours(n)] - t.depth[n]).max(initial=0) <= 1
    assert len(t) > len(free)


def test_leaf_cap_arguments():
    cells, counts = _cells_and_counts()
    t = plan_leaves(cells, counts, target=1200, max_leaf_cells=2)  # unsharded, capped on request
    assert int(t.sides().max()) == 2 and t.max_leaf_cells == 2 and t.shard_shape is None
    assert plan_leaves(cells, counts, target=1200, max_leaf_cells=1, shard_shape=(4, 4, 4)).sides().max() == 1
    assert plan_leaves(cells, counts, target=1200).max_leaf_cells is None
    with pytest.raises(OctreeError, match="power of two"):
        plan_leaves(cells, counts, target=1200, max_leaf_cells=3)
    with pytest.raises(OctreeError, match="inside one shard"):
        plan_leaves(cells, counts, target=1200, max_leaf_cells=4, shard_shape=(6, 6, 6))
    with pytest.raises(OctreeError, match="not on a shard boundary"):
        plan_leaves(cells, counts, target=1200, origin_cell=(1, 0, 0), shard_shape=(4, 4, 4))


@pytest.mark.parametrize("shard", SHARDS)
def test_to_octree_fits_leaves_to_the_shards(tmp_path, shard):
    from zarr_vectors.octree_chunks.convert import array_shard_grid
    from zarr_vectors.octree_chunks.table import shard_leaf_cap
    from zarr_vectors.validate import validate

    pos = _skewed()
    edges = _short_edges(pos)
    oids = (np.arange(len(pos)) // 40).astype(np.int64)
    path = tmp_path / "g.zv"
    # Bounds below zero put the shard grid's origin at cell -1, not 0.
    write_graph(path, pos, edges, chunk_shape=(CELL,) * 3, bounds=([-CELL] * 3, [WIDE] * 3),
                object_ids=oids, shard_shape=shard)
    cells, counts = _cells_and_counts(pos)
    lg = get_resolution_level(open_store(path), 0)
    got_shard, origin = array_shard_grid(lg)
    assert got_shard == shard and origin == (-1, -1, -1)
    free = plan_leaves(cells, counts, target=1200)
    assert int(free.sides().max()) > shard_leaf_cap(shard) and len(free.straddling(shard, origin))
    before = _snapshot(path)
    s = to_octree(path, target=1200)
    cap = shard_leaf_cap(shard)
    assert s["max_leaf_cells"] == cap
    lg = get_resolution_level(open_store(path), 0)
    block = octree_meta(lg)
    assert block["max_leaf_cells"] == cap and block["shard_shape"] == list(shard)
    assert block["shard_origin"] == [-1, -1, -1]
    assert array_shard_grid(lg) == (shard, origin)
    t = leaf_table(lg)
    assert int(t.sides().max()) <= cap and not len(t.straddling(shard, origin))
    result = validate(path, level=3)
    assert result.ok and not result.warnings, (result.errors, result.warnings)
    g = read_graph(path)
    assert g["node_count"] == len(pos) and g["edge_count"] == len(edges)
    to_uniform(path)
    assert _snapshot(path) == before
    assert array_shard_grid(get_resolution_level(open_store(path), 0)) == (shard, origin)


def test_max_leaf_cells_on_an_unsharded_level(tmp_path):
    from zarr_vectors.validate import validate

    path = tmp_path / "g.zv"
    _small_graph(path, lambda n: np.arange(n) // 40)
    before = _snapshot(path)
    with pytest.raises(OctreeError, match="power of two"):
        to_octree(path, target=300, max_leaf_cells=3)
    assert _snapshot(path) == before
    assert to_octree(path, target=300, max_leaf_cells=2)["max_leaf_cells"] == 2
    lg = get_resolution_level(open_store(path), 0)
    assert octree_meta(lg)["max_leaf_cells"] == 2 and octree_meta(lg)["shard_shape"] is None
    assert int(leaf_table(lg).sides().max()) <= 2
    assert validate(path, level=3).ok
    to_uniform(path)
    assert _snapshot(path) == before


def test_a_level_sharded_by_shard_store_keeps_its_shards(tmp_path):
    """Packing follows the level's arrays, not only the root declaration,
    which ``shard_store`` does not write."""
    from zarr_vectors.octree_chunks.convert import array_shard_grid
    from zarr_vectors.sharding.io import shard_store

    path = tmp_path / "g.zv"
    _small_graph(path, lambda n: np.arange(n) // 40)
    shard_store(str(path), shard_shape=4)
    assert read_root_metadata(open_store(path)).shard_shape is None
    before = _snapshot(path)
    assert to_octree(path, target=300)["max_leaf_cells"] == 4
    assert array_shard_grid(get_resolution_level(open_store(path), 0))[0] == (4, 4, 4)
    to_uniform(path)
    assert array_shard_grid(get_resolution_level(open_store(path), 0))[0] == (4, 4, 4)
    assert _snapshot(path) == before


def test_validate_flags_leaves_a_reshard_splits(tmp_path):
    from zarr_vectors.octree_chunks.convert import array_shard_grid
    from zarr_vectors.sharding.io import reshard
    from zarr_vectors.validate import validate

    path = tmp_path / "g.zv"
    _small_graph(path, lambda n: np.arange(n) // 40, n=6000, shard=4)
    before = _snapshot(path)
    to_octree(path, target=1500)
    assert int(leaf_table(get_resolution_level(open_store(path), 0)).sides().max()) == 4
    reshard(str(path), 8)  # coarser shards, still a multiple: every leaf still fits
    assert validate(path, level=3).ok
    reshard(str(path), 2)  # finer shards split the 4-cell leaves
    errors = validate(path, level=3).errors
    assert any("not inside one shard" in e and "to_uniform, then to_octree" in e
               and "resharded since" in e for e in errors), errors
    to_uniform(path)
    assert _snapshot(path) == before
    assert to_octree(path, target=1500)["max_leaf_cells"] == 2
    assert validate(path, level=3).ok
    assert array_shard_grid(get_resolution_level(open_store(path), 0))[0] == (2, 2, 2)


def test_validate_warns_on_uncapped_leaves_a_shard_store_splits(tmp_path):
    from zarr_vectors.sharding.io import shard_store
    from zarr_vectors.validate import validate

    path = tmp_path / "g.zv"
    _small_graph(path, lambda n: np.arange(n) // 40, n=6000)
    to_octree(path, target=1500)
    assert octree_meta(get_resolution_level(open_store(path), 0))["max_leaf_cells"] is None
    shard_store(str(path), shard_shape=2)
    result = validate(path, level=3)
    assert result.ok, result.errors
    assert any("not inside one shard" in w and "without a leaf cap" in w for w in result.warnings)


# -- the guard -------------------------------------------------------------------

def test_guard_marks_handles_and_reads_the_level_block(graph_store):
    from zarr_vectors.octree_chunks import converting, is_converting, is_octree_level
    path = graph_store[0]
    lg = get_resolution_level(open_store(path), 0)
    assert not is_octree_level(lg) and not is_converting(lg)
    with converting(lg):
        assert is_converting(lg)
        with converting(lg):
            assert is_converting(lg)
        assert is_converting(lg)
    assert not is_converting(lg)
    try:
        with converting(lg):
            raise _Boom
    except _Boom:
        pass
    assert not is_converting(lg)
    to_octree(path, target=300)
    assert is_octree_level(get_resolution_level(open_store(path), 0))
