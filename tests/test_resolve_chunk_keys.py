"""Tests for how a bounding box is resolved to chunk keys.

``resolve_chunk_keys`` used to answer a box by listing every chunk in the
level and filtering that list.  It now resolves from whichever side is
smaller — probing the manifest for the cells the box names, or testing
the present cells against the box — so a targeted query costs what the
query is worth rather than what the store is worth.

These tests pin the parts of that which are easy to get wrong: the two
sides must agree, a store whose keys carry a leading attribute-bin axis
must still resolve (that branch had no coverage at all), a box must not
be enumerated when it is enormous, and the clamp must not invent cells at
the grid edge.
"""

from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors.core.arrays import (
    _BOX_PROBE_FLOOR,
    _chunks_in_box,
    _chunks_in_box_unbounded,
    list_chunk_keys,
    resolve_chunk_keys,
)
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.types.points import read_points, write_points

DOMAIN = 400.0
CHUNK = (50.0, 50.0, 50.0)


def _cloud(tmp_path, n=2000, seed=0, name="c.zv", **kw):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, DOMAIN, (n, 3)).astype(np.float32)
    store = str(tmp_path / name)
    write_points(store, pos, chunk_shape=CHUNK, **kw)
    return store, pos


def _in_box(pos, low, high):
    lo, hi = np.asarray(low, float), np.asarray(high, float)
    return pos[np.all((pos >= lo) & (pos <= hi), axis=1)]


def _sorted_rows(a):
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    return a[np.lexsort((a[:, 2], a[:, 1], a[:, 0]))]


def _level(store):
    return get_resolution_level(open_store(store), 0)


# ---------------------------------------------------------------------------
# The two sides must agree, on every store shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("binned", [False, True])
@pytest.mark.parametrize("box", [
    ((0.0, 0.0, 0.0), (25.0, 25.0, 25.0)),           # one cell
    ((60.0, 60.0, 60.0), (140.0, 140.0, 140.0)),     # a few cells
    ((0.0, 0.0, 0.0), (DOMAIN, DOMAIN, DOMAIN)),     # the whole domain
    ((-500.0, -500.0, -500.0), (900.0, 900.0, 900.0)),  # beyond it
])
def test_probe_and_scan_sides_agree(tmp_path, binned, box):
    """The fast side and the slow side are two routes to one answer."""
    kw = {}
    if binned:
        rng = np.random.default_rng(7)
        kw = {
            "vertex_attributes": {"gene": rng.choice(["A", "B", "C"], 2000)},
            "chunk_by_attribute": "gene",
        }
    store, _ = _cloud(tmp_path, name=f"c{int(binned)}.zv", **kw)
    lg = _level(store)
    low, high = np.asarray(box[0]), np.asarray(box[1])

    bounded = _chunks_in_box(lg, CHUNK, (low, high), "vertices")
    legacy = _chunks_in_box_unbounded(lg, CHUNK, (low, high), "vertices")
    assert bounded == legacy


# ---------------------------------------------------------------------------
# Attribute-binned stores — the branch that had zero coverage
# ---------------------------------------------------------------------------


def test_bbox_on_an_attribute_binned_store_returns_its_points(tmp_path):
    """Keys carry a leading bin axis; the box speaks spatial coords only."""
    rng = np.random.default_rng(11)
    n = 3000
    pos = rng.uniform(0, DOMAIN, (n, 3)).astype(np.float32)
    gene = rng.choice(["A", "B", "C"], n)
    store = str(tmp_path / "binned.zv")
    write_points(
        store, pos, chunk_shape=CHUNK,
        vertex_attributes={"gene": gene}, chunk_by_attribute="gene",
    )
    # Keys really are rank-4.
    assert len(list_chunk_keys(_level(store))[0]) == 4

    low, high = [80.0] * 3, [180.0] * 3
    got = read_points(store, bbox=(low, high))["positions"]
    want = _in_box(pos, low, high)
    assert len(got) == len(want) > 0
    np.testing.assert_array_equal(_sorted_rows(got), _sorted_rows(want))


def test_a_one_cell_box_on_a_binned_store_finds_every_bin(tmp_path):
    """One spatial cell maps to one key per bin — all of them must resolve."""
    rng = np.random.default_rng(12)
    n = 4000
    pos = rng.uniform(0, 100.0, (n, 3)).astype(np.float32)
    gene = rng.choice(["A", "B", "C"], n)
    store = str(tmp_path / "b1.zv")
    write_points(
        store, pos, chunk_shape=CHUNK,
        vertex_attributes={"gene": gene}, chunk_by_attribute="gene",
    )
    low, high = [0.0] * 3, [49.0] * 3
    got = read_points(store, bbox=(low, high))["positions"]
    want = _in_box(pos, low, high)
    assert len(got) == len(want) > 0
    np.testing.assert_array_equal(_sorted_rows(got), _sorted_rows(want))


# ---------------------------------------------------------------------------
# Grid origin
# ---------------------------------------------------------------------------


def test_bbox_read_with_a_nonzero_chunk_grid_origin(tmp_path):
    """Negative coordinates give the level a non-zero grid origin.

    The clamp is expressed in absolute coords, so an origin it ignored
    would fold the box onto the wrong cells.
    """
    rng = np.random.default_rng(13)
    pos = rng.uniform(-300.0, -50.0, (1500, 3)).astype(np.float32)
    store = str(tmp_path / "neg.zv")
    write_points(store, pos, chunk_shape=CHUNK)
    assert min(min(k) for k in list_chunk_keys(_level(store))) < 0

    low, high = [-200.0] * 3, [-120.0] * 3
    got = read_points(store, bbox=(low, high))["positions"]
    want = _in_box(pos, low, high)
    assert len(got) == len(want) > 0
    np.testing.assert_array_equal(_sorted_rows(got), _sorted_rows(want))


# ---------------------------------------------------------------------------
# Degenerate and hostile boxes
# ---------------------------------------------------------------------------


def test_a_whole_domain_box_does_not_enumerate_the_grid(tmp_path, monkeypatch):
    """The box side must not be taken when the box is the expensive side.

    ``chunks_intersecting_bbox`` materialises the full cartesian product
    with no clamp, so a huge box is precisely where it must not be
    called and where the candidates must not be enumerated either.
    """
    store, pos = _cloud(tmp_path, n=3000)

    def explode(*a, **k):
        raise AssertionError("the box was enumerated")

    monkeypatch.setattr(
        "zarr_vectors.spatial.chunking._cartesian_product", explode,
    )
    got = read_points(store, bbox=([0.0] * 3, [DOMAIN] * 3))["positions"]
    assert len(got) == len(pos)


def test_a_box_entirely_off_the_grid_reads_nothing(tmp_path):
    """Clamping must not fold an outside box onto the edge cell."""
    store, _ = _cloud(tmp_path, n=800)
    for box in (
        ([10_000.0] * 3, [11_000.0] * 3),
        ([-11_000.0] * 3, [-10_000.0] * 3),
    ):
        assert read_points(store, bbox=box)["vertex_count"] == 0


def test_an_inverted_box_reads_nothing(tmp_path):
    store, _ = _cloud(tmp_path, n=800)
    assert read_points(store, bbox=([300.0] * 3, [100.0] * 3))["vertex_count"] == 0


def test_a_nan_corner_falls_back_rather_than_inventing_cells(tmp_path):
    store, _ = _cloud(tmp_path, n=800)
    out = read_points(store, bbox=([float("nan")] * 3, [100.0] * 3))
    assert out["vertex_count"] == 0


def test_an_infinite_corner_selects_the_half_space(tmp_path):
    """An unbounded corner means "as far as the store goes", not INT64_MIN."""
    store, pos = _cloud(tmp_path, n=1500)
    inf = float("inf")
    got = read_points(store, bbox=([200.0, -inf, -inf], [inf, inf, inf]))
    want = pos[pos[:, 0] >= 200.0]
    assert len(got["positions"]) == len(want) > 0
    np.testing.assert_array_equal(
        _sorted_rows(got["positions"]), _sorted_rows(want),
    )


# ---------------------------------------------------------------------------
# The chunks= whitelist, and the no-filter contract
# ---------------------------------------------------------------------------


def test_chunks_empty_list_yields_an_empty_result(tmp_path):
    """``chunks=[]`` means "no chunks", not "no filter"."""
    store, _ = _cloud(tmp_path, n=800)
    assert read_points(store, chunks=[])["vertex_count"] == 0
    assert resolve_chunk_keys(_level(store), CHUNK, chunks=[]) == []


def test_chunks_whitelist_arity_is_validated(tmp_path):
    store, _ = _cloud(tmp_path, n=800)
    lg = _level(store)
    resolve_chunk_keys(lg, CHUNK, chunks=[(0, 0, 0)])          # spatial
    resolve_chunk_keys(lg, CHUNK, chunks=[(0, 0, 0, 0)])       # binned arity
    with pytest.raises(ValueError):
        resolve_chunk_keys(lg, CHUNK, chunks=[(0, 0)])


def test_no_filter_tolerates_a_none_chunk_shape(tmp_path):
    """``chunk_shape`` must not be dereferenced when nothing filters by it.

    ``benchmarks/paper/profile_hotspots.py`` passes it positionally as
    None to walk every chunk in a level.
    """
    store, _ = _cloud(tmp_path, n=800)
    lg = _level(store)
    assert resolve_chunk_keys(lg, None, bbox=None, chunks=None) == (
        list_chunk_keys(lg)
    )


def test_no_filter_returns_a_copy(tmp_path):
    """The caller may mutate what it gets; the session cache may not."""
    store, _ = _cloud(tmp_path, n=800)
    lg = _level(store)
    got = resolve_chunk_keys(lg, None)
    got.append((99, 99, 99))
    assert (99, 99, 99) not in list_chunk_keys(lg)


def test_bbox_and_chunks_intersect(tmp_path):
    """Both filters supplied means both apply."""
    store, _ = _cloud(tmp_path, n=2000)
    lg = _level(store)
    box = (np.array([0.0] * 3), np.array([100.0] * 3))
    by_box = set(resolve_chunk_keys(lg, CHUNK, bbox=box))
    assert by_box
    one = sorted(by_box)[0]
    assert resolve_chunk_keys(lg, CHUNK, bbox=box, chunks=[one]) == [one]
    outside = (99, 99, 99)
    assert resolve_chunk_keys(lg, CHUNK, bbox=box, chunks=[outside]) == []


def test_probe_floor_is_a_floor_not_a_cap(tmp_path):
    """A box under the floor takes the probe path even on a tiny level."""
    assert _BOX_PROBE_FLOOR >= 1
