"""Every skeleton the writer accepts reads back exactly, in every read mode.

``kind="skeleton"`` stores only the edges its implicit rule does not give:
``parent[i] = i - 1`` within a fragment, none at a fragment start, and a
stored record wherever one exists.  The rule only holds if the writer
puts each parentless vertex at a fragment start and stores every parent
edge of a vertex that has more than one.  It once did neither: a forest
given in any order but depth-first, or an object whose other vertices
hang off another object's tree, read back with a root joined to the row
before it, and a vertex with two parent edges lost one -- on a full read
as much as a scoped one.

The property test draws random forests -- any vertex order, edges in any
row order, several roots, trees cut into several objects and objects
gathering pieces of several trees, single-vertex objects, objects across
chunks -- and compares every read mode, sharded and not, against the
induced subgraph of the true edge set: the vertices the read selects and
exactly the input edges between them, as a multiset, endpoint order kept.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from zarr_vectors.types.graphs import read_graph, write_graph

CHUNK = (100.0, 100.0, 100.0)
BOUNDS = ([0.0, 0.0, 0.0], [400.0, 400.0, 400.0])


def _key(p):
    return tuple(np.round(np.asarray(p, dtype=np.float64), 3))


def _edges(positions, edges):
    """Directed edge multiset, by endpoint position."""
    return Counter((_key(positions[a]), _key(positions[b])) for a, b in edges)


def _random_forest(rng):
    """``(positions, edges, object_ids)``: a random forest the writer accepts.

    Trees of 1 to ~60 vertices; each tree is cut into pieces at random
    vertices, and pieces are gathered into objects so that an object
    holds at most one tree root (what ``kind="skeleton"`` asks) but may
    hold pieces of several trees.  Vertices come in depth-first, random
    or reversed order, edge rows in random order, as ``(child, parent)``.
    """
    parent: list[int] = []
    for _ in range(int(rng.integers(1, 8))):
        size = int(rng.choice([1, 2, 3, int(rng.integers(4, 60))]))
        base = len(parent)
        reach = int(rng.integers(1, 6))
        parent.append(-1)
        for i in range(1, size):
            parent.append(base + int(rng.integers(max(0, i - reach), i)))
    par = np.asarray(parent, dtype=np.int64)
    n = len(par)

    # Small steps stay in a chunk, large ones hop between chunks.
    step = float(rng.choice([5.0, 40.0]))
    pos = np.zeros((n, 3))
    for i in range(n):
        pos[i] = (rng.uniform(5, 395, 3) if par[i] < 0
                  else np.clip(pos[par[i]] + rng.normal(0, step, 3), 1, 399))

    cut = (par < 0) | (rng.random(n) < float(rng.choice([0.0, 0.05, 0.3])))
    piece = np.zeros(n, dtype=np.int64)
    n_pieces = 0
    for i in range(n):
        if cut[i]:
            piece[i] = n_pieces
            n_pieces += 1
        else:
            piece[i] = piece[par[i]]
    owner = np.arange(n_pieces)
    rooted = np.zeros(n_pieces, dtype=bool)
    rooted[piece[par < 0]] = True
    if rng.random() < 0.5:
        # An unrooted piece may join any object: it adds no root.
        for pc in np.flatnonzero(~rooted):
            if rng.random() < 0.5:
                owner[pc] = owner[int(rng.integers(0, n_pieces))]
        for _ in range(8):
            owner = owner[owner]
    oids = owner[piece] * 3 + 7   # sparse, not 0..k

    edges = np.stack([np.flatnonzero(par >= 0), par[par >= 0]], axis=1)
    order = rng.choice(["dfs", "random", "reversed"])
    if order != "dfs":
        perm = rng.permutation(n) if order == "random" else np.arange(n)[::-1]
        inv = np.argsort(perm)
        pos, oids, edges = pos[perm], oids[perm], inv[edges]
    if len(edges) and rng.random() < 0.5:
        edges = edges[rng.permutation(len(edges))]
    return pos.astype(np.float32), edges.reshape(-1, 2), oids


def _assert_reads(path, positions, edges, oids, rng, n_scopes=3):
    """Every read mode against the induced subgraph of ``edges``."""
    cells = [tuple(c) for c in (positions // CHUNK[0]).astype(int).tolist()]

    def check(selected, **kwargs):
        got = read_graph(path, **kwargs)
        assert got["node_count"] == int(selected.sum()), kwargs
        want = _edges(positions, edges[selected[edges].all(axis=1)])
        assert _edges(got["positions"], got["edges"]) == want, kwargs
        if "object_ids" in kwargs:
            assert set(got["object_ids"].tolist()) <= set(kwargs["object_ids"])

    check(np.ones(len(positions), dtype=bool))
    objects = np.unique(oids)
    for _ in range(n_scopes):
        lo = rng.uniform(0, 250, 3)
        hi = lo + rng.uniform(60, 250, 3)
        bbox = (tuple(lo), tuple(hi))
        in_box = ((positions >= lo) & (positions <= hi)).all(axis=1)
        chunk_set = sorted({c for c in cells if rng.random() < 0.5})
        in_chunks = np.array([c in set(chunk_set) for c in cells])
        wanted = rng.choice(objects, size=int(rng.integers(1, len(objects) + 1)),
                            replace=False).tolist()
        in_objects = np.isin(oids, wanted)
        check(in_box, bbox=bbox)
        check(in_chunks, chunks=chunk_set)
        check(in_objects, object_ids=wanted)
        check(in_objects & in_box, object_ids=wanted, bbox=bbox)
        check(in_objects & in_chunks, object_ids=wanted, chunks=chunk_set)


@pytest.mark.parametrize("sharded", [False, True], ids=["unsharded", "sharded"])
@pytest.mark.parametrize("seed", range(16))
def test_random_forests_read_back_exactly(tmp_path, seed, sharded):
    rng = np.random.default_rng(seed)
    positions, edges, oids = _random_forest(rng)
    path = tmp_path / "forest.zarrvectors"
    write_graph(path, positions, edges, chunk_shape=CHUNK, bounds=BOUNDS,
                kind="skeleton", object_ids=oids,
                shard_shape=2 if sharded else None)
    _assert_reads(path, positions, edges, oids, rng)


def _write(tmp_path, positions, edges, oids=None, **kwargs):
    path = tmp_path / "skel.zarrvectors"
    write_graph(path, np.asarray(positions, dtype=np.float32),
                np.asarray(edges, dtype=np.int64).reshape(-1, 2),
                chunk_shape=CHUNK, bounds=BOUNDS, kind="skeleton",
                object_ids=None if oids is None else np.asarray(oids), **kwargs)
    return path


@pytest.mark.parametrize("order", ["reversed", "random"])
def test_forest_in_any_vertex_order_reads_whole(tmp_path, order):
    """Two chains in one chunk, children before parents.  The reorder
    walked only the first tree; the second kept its order, its root
    landed mid-fragment and read back with the row before it as parent."""
    n = 6
    positions = np.array([[10 + 10 * i, 50, 50] for i in range(n)], dtype=np.float32)
    edges = np.array([[1, 0], [2, 1], [4, 3], [5, 4]])
    oids = np.array([0, 0, 0, 1, 1, 1])
    perm = (np.arange(n)[::-1] if order == "reversed"
            else np.random.default_rng(0).permutation(n))
    inv = np.argsort(perm)
    p, e, o = positions[perm], inv[edges], oids[perm]
    path = _write(tmp_path, p, e, o)
    for kwargs in ({}, {"object_ids": [0, 1]}, {"chunks": [(0, 0, 0)]},
                   {"bbox": ((0, 0, 0), (399, 399, 399))}):
        got = read_graph(path, **kwargs)
        assert _edges(got["positions"], got["edges"]) == _edges(p, e), kwargs


def test_object_root_after_a_piece_of_another_tree(tmp_path):
    """Object 0 is root R with child Y, plus X, a child of object 1's
    root P.  Depth-first from P first, object 0's fragment ran X, R, Y:
    R mid-fragment, read back as X's child.  R now opens the fragment."""
    #             P            X            R            Y
    positions = [[10, 50, 50], [20, 50, 50], [30, 50, 50], [40, 50, 50]]
    edges = [[1, 0], [3, 2]]
    oids = [1, 0, 0, 0]
    path = _write(tmp_path, positions, edges, oids)
    p = np.asarray(positions, dtype=np.float32)
    e = np.asarray(edges)
    for kwargs in ({}, {"object_ids": [0, 1]}, {"chunks": [(0, 0, 0)]}):
        got = read_graph(path, **kwargs)
        assert _edges(got["positions"], got["edges"]) == _edges(p, e), kwargs
    zero = read_graph(path, object_ids=[0])
    assert _edges(zero["positions"], zero["edges"]) == _edges(p, e[1:])


@pytest.mark.parametrize("edges", [
    [[1, 0], [2, 1], [2, 0]],           # two parents (closes a cycle)
    [[1, 0], [1, 0], [2, 1]],           # a repeated edge
    [[1, 0], [2, 1], [0, 2]],           # a cycle, no root at all
    [[1, 0], [2, 1], [1, 1]],           # a self-loop beside a real parent
], ids=["two-parents", "repeated", "cycle", "self-loop"])
def test_vertex_with_several_parent_edges_keeps_them_all(tmp_path, edges):
    """Not trees, but the writer has always accepted them.  An implied
    parent edge left unstored was replaced by its sibling's record."""
    positions = np.array([[10, 50, 50], [20, 50, 50], [30, 50, 50]], dtype=np.float32)
    e = np.asarray(edges)
    path = _write(tmp_path, positions, e,
                  link_attributes={"w": np.arange(len(e), dtype=np.float32)})
    for kwargs in ({}, {"object_ids": [0]}, {"chunks": [(0, 0, 0)]}):
        got = read_graph(path, **kwargs)
        assert _edges(got["positions"], got["edges"]) == _edges(positions, e), kwargs


def test_swc_minus_one_parent_is_a_root(tmp_path):
    """``[root, -1]``, as the skeleton spec's example writes, declares a
    root.  ``-1`` used to index the last vertex: a root-to-leaf edge."""
    positions = np.array([[10, 50, 50], [20, 50, 50], [30, 50, 50]], dtype=np.float32)
    path = _write(tmp_path, positions, [[0, -1], [1, 0], [2, 1]],
                  link_attributes={"w": np.arange(3, dtype=np.float32)})
    got = read_graph(path)
    assert _edges(got["positions"], got["edges"]) == _edges(
        positions, np.array([[1, 0], [2, 1]]),
    )


@pytest.mark.parametrize("edges", [[[1, 0], [-1, 1]], [[1, 0], [2, -2]]],
                         ids=["negative-child", "parent-below-minus-one"])
def test_negative_indices_other_than_a_root_parent_are_refused(tmp_path, edges):
    """Only a parent of ``-1`` means anything; any other negative index
    would wrap round to a vertex from the end."""
    positions = np.array([[10, 50, 50], [20, 50, 50], [30, 50, 50]], dtype=np.float32)
    with pytest.raises(ValueError, match="parent of -1"):
        _write(tmp_path, positions, edges)


@pytest.mark.parametrize("atomic", [True, False])
def test_edit_link_re_points_a_branch(tmp_path, atomic):
    """A branch record names its child's parent, so re-pointing it moves
    the edge; under ``atomic=True`` it is not appended beside the old one,
    which the reader would take as a second parent."""
    from zarr_vectors.core.arrays import read_chunk_links
    from zarr_vectors.core.store import open_store
    from zarr_vectors.ops import LinkRef
    from zarr_vectors.ops.edit import edit_link

    positions = np.array([[10, 50, 50], [20, 50, 50], [30, 50, 50],
                          [40, 50, 50], [50, 50, 50]], dtype=np.float32)
    path = _write(tmp_path, positions, [[1, 0], [2, 1], [3, 2], [4, 1]])
    root = open_store(str(path), mode="r+")
    groups = read_chunk_links(root["0"], (0, 0, 0), delta=0)
    frag = next(i for i, g in enumerate(groups) if len(g))
    assert groups[frag].tolist() == [[4, 1]]
    edit_link(root, LinkRef(level=0, chunk=(0, 0, 0), fragment=frag, row=0, delta=0),
              new_endpoints=(4, 3), atomic=atomic)
    got = read_graph(path)
    assert _edges(got["positions"], got["edges"]) == _edges(
        positions, np.array([[1, 0], [2, 1], [3, 2], [4, 3]]),
    )


def test_edge_between_objects_needs_both(tmp_path):
    """An edge whose ends are different objects comes back when both are
    read and not when one is -- the plain graph's ``object_ids=`` rule,
    the induced subgraph.  Nothing stands in for it."""
    #             A0           A1           B0           B1
    positions = np.array([[10, 50, 50], [20, 50, 50], [30, 50, 50], [150, 50, 50]],
                         dtype=np.float32)
    edges = np.array([[1, 0], [2, 1], [3, 2]])     # B0's parent is A1
    oids = np.array([5, 5, 9, 9])
    skel = _write(tmp_path, positions, edges, oids)
    graph = tmp_path / "graph.zarrvectors"
    write_graph(graph, positions, edges, chunk_shape=CHUNK, bounds=BOUNDS,
                kind="graph", object_ids=oids)
    for wanted in ([5], [9], [5, 9], [9, 5]):
        keep = np.isin(oids, wanted)
        want = _edges(positions, edges[keep[edges].all(axis=1)])
        got = read_graph(skel, object_ids=wanted)
        assert _edges(got["positions"], got["edges"]) == want, wanted
        plain = read_graph(graph, object_ids=wanted)
        undirected = Counter(frozenset(e) for e in want.elements())
        assert Counter(
            frozenset(e) for e in _edges(plain["positions"], plain["edges"]).elements()
        ) == undirected, wanted


def test_single_vertex_objects(tmp_path):
    rng = np.random.default_rng(7)
    positions = rng.uniform(1, 399, (20, 3)).astype(np.float32)
    edges = np.zeros((0, 2), dtype=np.int64)
    oids = np.arange(20) * 2
    path = _write(tmp_path, positions, edges, oids)
    _assert_reads(path, positions, edges, oids, rng)


def test_any_order_stores_no_more_than_depth_first(tmp_path):
    """The writer restores depth-first order for every tree, so a forest
    given shuffled stores the same records as given depth-first -- not
    nearly every edge, as when only the first tree was reordered."""
    rng = np.random.default_rng(3)
    parent, oids = [], []
    for obj in range(20):
        base = len(parent)
        parent.append(-1)
        for i in range(1, 50):
            parent.append(base + i - 1 if rng.random() < 0.8 else base + int(rng.integers(0, i)))
        oids += [obj] * 50
    par = np.asarray(parent)
    pos = np.zeros((len(par), 3))
    for i, p in enumerate(par):
        pos[i] = rng.uniform(20, 380, 3) if p < 0 else np.clip(pos[p] + rng.normal(0, 8, 3), 1, 399)
    edges = np.stack([np.flatnonzero(par >= 0), par[par >= 0]], axis=1)
    oids = np.asarray(oids)
    dfs = write_graph(tmp_path / "dfs.zv", pos.astype(np.float32), edges,
                      chunk_shape=CHUNK, bounds=BOUNDS, kind="skeleton", object_ids=oids)
    perm = rng.permutation(len(par))
    inv = np.argsort(perm)
    shuffled_edges = inv[edges]
    shuffled_edges = shuffled_edges[np.argsort(shuffled_edges[:, 0])]
    shuffled = write_graph(tmp_path / "perm.zv", pos[perm].astype(np.float32),
                           shuffled_edges, chunk_shape=CHUNK, bounds=BOUNDS,
                           kind="skeleton", object_ids=oids[perm])
    assert shuffled["intra_edge_count"] <= dfs["intra_edge_count"]
    assert shuffled["intra_edge_count"] < len(edges) // 4
