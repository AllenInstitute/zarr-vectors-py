"""Scoped skeleton reads return only edges that were written.

A skeleton stores only the edges its implicit rule (``parent[i] = i - 1``
within a fragment) does not give.  A child that is not first in its
fragment but whose parent sits elsewhere has an explicit record, and
when a read takes the child but not the parent, that record is the only
thing that makes the child a root there.  These reads once fetched only
the link cells wholly inside the read, never saw the record, and gave
the child its fragment predecessor as parent -- an edge nobody wrote.

Covered here: ``bbox`` / ``chunks`` / ``attribute_filter`` reads, the
object path with and without a scope, and a tree whose root is another
object's.  A plain graph's scoped reads must not change.
"""

from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors.core.arrays import link_cells_within
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.types.graphs import read_graph, write_graph

CHUNK = (100.0, 100.0, 100.0)
BOUNDS = ([0.0, 0.0, 0.0], [400.0, 400.0, 400.0])


def _key(p):
    return tuple(np.round(np.asarray(p, dtype=np.float64), 3))


def _edge_set(positions, edges):
    return {(_key(positions[a]), _key(positions[b])) for a, b in edges}


def _forest(n_objects=12, size=40, seed=0):
    """Random trees, ``(positions, edges, object_ids)``, edges as
    ``(child, parent)``; children often hop chunks and come back."""
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


# A(chunk 0) <- B(chunk 1) <- C(chunk 0) <- D(chunk 0): the DFS leaves
# chunk 0 and comes back, so C is mid-fragment with its parent elsewhere.
_CHAIN_P = np.array(
    [[50, 50, 50], [150, 50, 50], [60, 50, 50], [70, 50, 50]], dtype=np.float32,
)
_CHAIN_E = np.array([[1, 0], [2, 1], [3, 2]], dtype=np.int64)


@pytest.fixture
def chain(tmp_path):
    path = tmp_path / "chain.zarrvectors"
    write_graph(path, _CHAIN_P, _CHAIN_E, chunk_shape=CHUNK, bounds=BOUNDS,
                kind="skeleton", object_ids=np.zeros(4, dtype=np.int64))
    return path


_ONLY_D_TO_C = {((70.0, 50.0, 50.0), (60.0, 50.0, 50.0))}


@pytest.mark.parametrize("scope", [
    {"bbox": ((0, 0, 0), (99, 99, 99))},
    {"chunks": [(0, 0, 0)]},
])
@pytest.mark.parametrize("objects", [None, [0]])
def test_branch_whose_parent_is_unread_comes_back_a_root(chain, scope, objects):
    got = read_graph(chain, object_ids=objects, **scope)
    assert got["node_count"] == 3
    # Not C->A: C's parent is B, which was not read.
    assert _edge_set(got["positions"], got["edges"]) == _ONLY_D_TO_C


@pytest.mark.parametrize("objects", [None, [0]])
def test_unscoped_chain_reads_whole(chain, objects):
    got = read_graph(chain, object_ids=objects)
    assert _edge_set(got["positions"], got["edges"]) == _edge_set(_CHAIN_P, _CHAIN_E)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("parity", [0, 1])
def test_checkerboard_chunks_read_only_written_edges(tmp_path, seed, parity):
    positions, edges, oids = _forest(seed=seed)
    path = tmp_path / "forest.zarrvectors"
    write_graph(path, positions, edges, chunk_shape=CHUNK, bounds=BOUNDS,
                kind="skeleton", object_ids=oids)
    cc = (positions // 100).astype(int)
    sel = sorted({tuple(c) for c in cc.tolist() if sum(c) % 2 == parity})
    inside = np.array([tuple(c) in set(sel) for c in cc.tolist()])
    want = _edge_set(positions, edges[inside[edges].all(axis=1)])

    full = read_graph(path, chunks=sel)
    assert full["node_count"] == int(inside.sum())
    assert _edge_set(full["positions"], full["edges"]) == want

    obj = read_graph(path, chunks=sel, object_ids=list(range(12)))
    assert obj["node_count"] == int(inside.sum())
    assert _edge_set(obj["positions"], obj["edges"]) == want

    # One object alone: its own in-scope edges, nothing borrowed.
    one = read_graph(path, chunks=sel, object_ids=[3])
    mine = inside & (oids == 3)
    assert _edge_set(one["positions"], one["edges"]) == _edge_set(
        positions, edges[mine[edges].all(axis=1)],
    )


def test_tree_rooted_in_another_object(tmp_path):
    """Root R is object 1; its children X1, X2 are object 0.  Reading
    object 0 alone gives two roots, not X2 -> X1."""
    positions = np.array([[150, 50, 50], [50, 50, 50], [60, 50, 50]], dtype=np.float32)
    edges = np.array([[1, 0], [2, 0]], dtype=np.int64)
    path = tmp_path / "split.zarrvectors"
    write_graph(path, positions, edges, chunk_shape=CHUNK, bounds=BOUNDS,
                kind="skeleton", object_ids=np.array([1, 0, 0]))
    zero = read_graph(path, object_ids=[0])
    assert zero["node_count"] == 2
    assert zero["edge_count"] == 0
    both = read_graph(path, object_ids=[0, 1])
    assert _edge_set(both["positions"], both["edges"]) == _edge_set(positions, edges)


def test_attribute_filter_read_only_written_edges(tmp_path):
    """The chain again, with B another object in another attribute bin
    rather than another chunk (bins are per object): filtering to the
    first bin must not join C to A."""
    positions = np.array(
        [[50, 50, 50], [55, 50, 50], [60, 50, 50], [70, 50, 50]], dtype=np.float32,
    )
    oids = np.array([0, 1, 0, 0], dtype=np.int64)
    path = tmp_path / "attr.zarrvectors"
    write_graph(path, positions, _CHAIN_E, chunk_shape=CHUNK, bounds=BOUNDS,
                kind="skeleton", object_ids=oids,
                vertex_attributes={"label": oids.copy()}, chunk_by_attribute="label")
    for objects in (None, [0]):
        got = read_graph(path, attribute_filter={"label": 0}, object_ids=objects)
        assert got["node_count"] == 3
        assert _edge_set(got["positions"], got["edges"]) == _ONLY_D_TO_C


def test_graph_scoped_reads_unchanged(tmp_path):
    """A plain graph's scoped read keeps the edges wholly inside, as before."""
    positions, edges, oids = _forest(seed=4)
    path = tmp_path / "graph.zarrvectors"
    write_graph(path, positions, edges, chunk_shape=CHUNK, bounds=BOUNDS,
                kind="graph", object_ids=oids)
    cc = (positions // 100).astype(int)
    sel = sorted({tuple(c) for c in cc.tolist() if sum(c) % 2 == 0})
    inside = np.array([tuple(c) in set(sel) for c in cc.tolist()])

    def undirected(p, e):
        return {frozenset(x) for x in _edge_set(p, e)}

    want = undirected(positions, edges[inside[edges].all(axis=1)])
    for objects in (None, list(range(12))):
        got = read_graph(path, chunks=sel, object_ids=objects)
        assert undirected(got["positions"], got["edges"]) == want


def test_link_cells_within_touching(chain):
    """The C -> B record is filed under chunk 0 with offset (1, 0, 0).
    Neither chunk alone holds it wholly, so either alone finds it only
    when touching -- chunk 1 through a source outside ``within``.  Both
    chunks find it either way."""
    level = get_resolution_level(open_store(str(chain)), 0)
    for within in ([(0, 0, 0)], [(1, 0, 0)]):
        wholly = link_cells_within(level, within)
        touching = link_cells_within(level, within, touching=True)
        cross = {seg for seg in touching if seg not in wholly}
        assert cross, (within, wholly, touching)
        for seg in cross:
            assert touching[seg] == ["0.0.0"]
    both = [(0, 0, 0), (1, 0, 0)]
    assert link_cells_within(level, both) == link_cells_within(
        level, both, touching=True,
    )
