"""The ``fragments_tile`` level claim and the three guards around it.

The claim lets a bulk read return each chunk's vertex buffer without
reading its fragment index — worth ~25% of a whole-store read, because
the index costs an object read and a decompression per chunk.

Unlike every other level field it is a claim about what has *not*
happened since, so it is guarded three ways: verified against the store
when stamped, cleared by ``Group.write_bytes`` on any later write to
``vertices`` or ``vertex_fragments``, and re-checked by a reader on the
first chunk it touches.  A stale claim is a wrong answer, not a crash —
extra rows come back and attribute columns desync from positions — so
each guard gets a test.
"""

from __future__ import annotations

import numpy as np

from zarr_vectors.core.arrays import (
    read_chunk_vertex_rows,
    write_chunk_fragments,
)
from zarr_vectors.core.store import (
    get_resolution_level,
    open_store,
    read_level_metadata,
)
from zarr_vectors.types.graphs import write_graph
from zarr_vectors.types.lines import write_lines
from zarr_vectors.types.meshes import write_mesh
from zarr_vectors.types.points import read_points, write_points
from zarr_vectors.types.polylines import write_polylines

CHUNK = (50.0, 50.0, 50.0)


def _points(tmp_path, n=4000, name="p.zv", **kw):
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 200, (n, 3)).astype(np.float32)
    store = str(tmp_path / name)
    write_points(store, pos, chunk_shape=CHUNK, bin_shape=(25.0,) * 3, **kw)
    return store, pos


def _flag(store, level=0):
    return read_level_metadata(open_store(store), level).fragments_tile


def _set_flag(store, value=True, level=0):
    """Force the claim on, without going through a writer."""
    root = open_store(store, mode="a")
    lg = get_resolution_level(root, level)
    meta = dict(lg.attrs.get("zarr_vectors_level"))
    if value:
        meta["fragments_tile"] = True
    else:
        meta.pop("fragments_tile", None)
    lg.attrs.update({"zarr_vectors_level": meta})


def _sorted_rows(a):
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    return a[np.lexsort((a[:, 2], a[:, 1], a[:, 0]))]


# ---------------------------------------------------------------------------
# Stamping
# ---------------------------------------------------------------------------


def test_a_bulk_point_write_sets_the_claim(tmp_path):
    store, _ = _points(tmp_path)
    assert _flag(store) is True


def test_every_bulk_writer_sets_the_claim(tmp_path):
    rng = np.random.default_rng(1)

    pl = str(tmp_path / "pl.zv")
    write_polylines(
        pl, [rng.uniform(0, 200, (8, 3)).astype(np.float32) for _ in range(20)],
        chunk_shape=CHUNK,
    )
    assert _flag(pl) is True

    m = str(tmp_path / "m.zv")
    write_mesh(
        m, rng.uniform(0, 200, (300, 3)).astype(np.float32),
        rng.integers(0, 300, (200, 3)).astype(np.int64), chunk_shape=CHUNK,
    )
    assert _flag(m) is True

    g = str(tmp_path / "g.zv")
    write_graph(
        g, rng.uniform(0, 200, (200, 3)).astype(np.float32),
        rng.integers(0, 200, (300, 2)).astype(np.int64), chunk_shape=CHUNK,
    )
    assert _flag(g) is True

    ln = str(tmp_path / "l.zv")
    write_lines(
        ln, rng.uniform(0, 200, (60, 2, 3)).astype(np.float32),
        chunk_shape=CHUNK,
    )
    assert _flag(ln) is True


def test_an_absent_claim_reads_as_false(tmp_path):
    """A store written before the field existed must not be trusted."""
    store, _ = _points(tmp_path)
    _set_flag(store, False)
    assert _flag(store) is False


# ---------------------------------------------------------------------------
# Reading — with the claim, without it, and against a stale one
# ---------------------------------------------------------------------------


def test_a_bulk_read_agrees_with_and_without_the_claim(tmp_path):
    store, pos = _points(tmp_path, n=6000)
    with_claim = read_points(store)["positions"]
    _set_flag(store, False)
    without = read_points(store)["positions"]

    assert len(with_claim) == len(pos)
    np.testing.assert_array_equal(
        _sorted_rows(with_claim), _sorted_rows(without),
    )


def test_attributes_stay_aligned_with_positions_under_the_claim(tmp_path):
    """The failure a stale claim causes is desync, not just extra rows."""
    rng = np.random.default_rng(2)
    pos = rng.uniform(0, 200, (3000, 3)).astype(np.float32)
    tag = pos[:, 0].astype(np.float32)          # an attribute derived from x
    store = str(tmp_path / "attr.zv")
    write_points(
        store, pos, chunk_shape=CHUNK, bin_shape=(25.0,) * 3,
        vertex_attributes={"tag": tag},
    )
    assert _flag(store) is True

    out = read_points(store, attribute_names=["tag"])
    np.testing.assert_allclose(
        out["vertex_attributes"]["tag"].ravel(), out["positions"][:, 0],
    )


def test_a_narrowed_read_ignores_the_claim(tmp_path):
    """Only a bulk read may take the fast path; a filter needs the index."""
    store, pos = _points(tmp_path, n=5000)
    assert _flag(store) is True
    low, high = [0.0] * 3, [80.0] * 3
    got = read_points(store, bbox=(low, high))["positions"]
    want = pos[np.all((pos >= np.array(low)) & (pos <= np.array(high)), axis=1)]
    assert len(got) == len(want) > 0
    np.testing.assert_array_equal(_sorted_rows(got), _sorted_rows(want))


def test_a_stale_claim_is_caught_by_the_first_chunk_probe(tmp_path):
    """Set the claim by hand on a level whose first chunk does not tile."""
    store, pos = _points(tmp_path, n=3000)
    root = open_store(store, mode="a")
    lg = get_resolution_level(root, 0)
    from zarr_vectors.core.arrays import list_chunk_keys

    first = list_chunk_keys(lg)[0]
    rows = read_chunk_vertex_rows(lg, first, dtype=np.float32, ndim=3)
    # Retire the tail of that chunk: the index no longer covers its buffer.
    write_chunk_fragments(lg, first, [(0, max(1, len(rows) - 2))])
    _set_flag(store, True)          # lie about it

    got = read_points(store)["positions"]
    # The probe must fall back, so the retired rows must NOT come back.
    assert len(got) == len(pos) - 2


# ---------------------------------------------------------------------------
# Clearing — the chokepoint
# ---------------------------------------------------------------------------


def test_appending_fragments_clears_the_claim(tmp_path):
    store, _ = _points(tmp_path)
    assert _flag(store) is True
    root = open_store(store, mode="a")
    lg = get_resolution_level(root, 0)
    from zarr_vectors.core.arrays import list_chunk_keys

    write_chunk_fragments(
        lg, list_chunk_keys(lg)[0], [np.array([0], dtype=np.int64)],
        mode="append",
    )
    assert _flag(store) is False


def test_a_batched_write_clears_the_claim(tmp_path):
    """Batching queues through write_bytes, so it is downstream of the gate."""
    store, _ = _points(tmp_path)
    assert _flag(store) is True
    root = open_store(store, mode="a")
    lg = get_resolution_level(root, 0)
    from zarr_vectors.constants import VERTEX_FRAGMENTS
    from zarr_vectors.core.arrays import list_chunk_keys

    key = ".".join(str(c) for c in list_chunk_keys(lg)[0])
    existing = lg.read_bytes(VERTEX_FRAGMENTS, key)
    with lg.batched_writes():
        lg.write_bytes(VERTEX_FRAGMENTS, key, existing)
    assert _flag(store) is False


def test_writing_vertices_clears_the_claim(tmp_path):
    """A buffer rewritten to a different row count breaks tiling too."""
    store, _ = _points(tmp_path)
    root = open_store(store, mode="a")
    lg = get_resolution_level(root, 0)
    from zarr_vectors.constants import VERTICES
    from zarr_vectors.core.arrays import list_chunk_keys

    key = ".".join(str(c) for c in list_chunk_keys(lg)[0])
    lg.write_bytes(VERTICES, key, lg.read_bytes(VERTICES, key))
    assert _flag(store) is False


# ---------------------------------------------------------------------------
# The validator
# ---------------------------------------------------------------------------


def test_validate_consistency_reports_a_lying_claim(tmp_path):
    from zarr_vectors.validate.consistency import validate_consistency

    store, _ = _points(tmp_path, n=3000)
    root = open_store(store, mode="a")
    lg = get_resolution_level(root, 0)
    from zarr_vectors.core.arrays import list_chunk_keys

    ck = list_chunk_keys(lg)[0]
    rows = read_chunk_vertex_rows(lg, ck, dtype=np.float32, ndim=3)
    write_chunk_fragments(lg, ck, [(0, max(1, len(rows) - 2))])
    _set_flag(store, True)

    report = validate_consistency(store)
    assert any("fragments_tile" in e for e in report.errors), report.errors


def test_validate_consistency_accepts_an_honest_claim(tmp_path):
    from zarr_vectors.validate.consistency import validate_consistency

    store, _ = _points(tmp_path, n=3000)
    assert _flag(store) is True
    report = validate_consistency(store)
    assert not any("fragments_tile" in e for e in report.errors), report.errors
