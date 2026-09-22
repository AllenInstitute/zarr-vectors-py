"""Allocating into a level whose keys carry a leading attribute-bin axis.

A level chunked by an attribute puts the bin first in every chunk key --
``gene.z.y.x`` -- and its per-chunk arrays are one axis wider to match.
``open_write_session(bin_count=)`` does that; `_derive_native_config`,
the layout source used when NO session is open, derived a purely spatial
grid and allocated one axis too narrow.

The paths that meet this are the decentralised ones: an attribute array
created directly by a worker (the pattern the HPC how-to teaches), the
lazy writer's attribute fan-out, an edit adding a links segment. None of
them opens a session, and none of them was covered -- every existing
attribute-chunking test creates a store and only reads from it.

The consequence was bimodal, which is why it went unnoticed. With an
all-zero grid origin the rank mismatch is caught by a length check and
raises. With a non-zero origin -- negative bounds, or a min corner at
least one chunk in -- ``_coord_to_index`` zipped the coords against a
shorter origin, ``zip`` truncated, and the result was a plausible-looking
index that dropped the last spatial axis: every key differing only in
that axis aliased onto one cell, last write winning, and all four
validators passed the store clean.

Covers:

* An array allocated with no session matches the level's own rank.
* The three reachable allocation paths, not just the direct one.
* A non-zero-origin store does not alias cells.
* A rank disagreement raises rather than truncating, batched or not.
* A spatial-only level gains no axis it should not have.
"""

from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors.building import (
    create_attribute_array,
    create_store,
    get_resolution_level,
    open_store,
)
from zarr_vectors.exceptions import StoreError
from zarr_vectors.types.points import write_points

_CHUNK = (50.0, 50.0, 50.0)
# Min corner one chunk in on two axes -> grid origin (0, 2, 2), which is
# stored, which is what makes the truncating zip reachable.
_OFFSET_BOUNDS = ([0.0, 100.0, 100.0], [150.0, 200.0, 200.0])
_ZERO_BOUNDS = ([0.0, 0.0, 0.0], [100.0, 100.0, 100.0])


def _attr_store(path, bounds=_ZERO_BOUNDS, n=40, bins=("A", "B", "C")):
    """A point store chunked by a categorical vertex attribute."""
    rng = np.random.default_rng(0)
    lo, hi = bounds
    pos = np.column_stack(
        [rng.uniform(lo[i], hi[i], n) for i in range(3)]
    ).astype("float32")
    genes = np.array(list(bins))[rng.integers(0, len(bins), n)]
    write_points(
        str(path), pos, bounds=[list(lo), list(hi)], chunk_shape=_CHUNK,
        vertex_attributes={"gene": genes}, chunk_by_attribute="gene",
    )
    return str(path)


def _level(path, mode="r+"):
    return get_resolution_level(open_store(path, mode=mode), 0)


# --- the rank the level actually has ----------------------------------


def test_a_lazily_created_attribute_array_matches_the_level_rank(tmp_path):
    """No session open: this is how a decentralised worker allocates."""
    path = _attr_store(tmp_path / "g.zarrvectors")
    lg = _level(path)
    vertices = lg._sharded_chunk_array("vertices")
    assert vertices.ndim == 4, "fixture is not attribute-chunked"

    create_attribute_array(lg, "score", dtype="float32")

    score = lg._sharded_chunk_array("vertex_attributes/score")
    assert score.shape == vertices.shape
    assert dict(score.attrs).get("chunk_grid_origin") == dict(
        vertices.attrs
    ).get("chunk_grid_origin")


def test_a_real_key_round_trips_through_the_lazily_created_array(tmp_path):
    """The rank being right is only useful if writes land."""
    path = _attr_store(tmp_path / "g.zarrvectors")
    lg = _level(path)
    create_attribute_array(lg, "score", dtype="float32")

    key = lg.list_chunks("vertices")[0]
    assert len(key.split(".")) == 4
    lg.write_bytes("vertex_attributes/score", key, b"payload")
    assert lg.read_bytes("vertex_attributes/score", key) == b"payload"


def test_a_lazily_created_links_segment_matches_the_level_rank(tmp_path):
    """The other reachable family: an edit adding a links segment.

    ``EditSession.flush`` opens only ``batched_writes``, never a write
    session, so ``create_links_array`` and the ``link_fragments`` sidecar
    it allocates both come off the derived config.
    """
    from zarr_vectors.building import create_links_array

    path = _attr_store(tmp_path / "g.zarrvectors")
    lg = _level(path)
    rank = lg._sharded_chunk_array("vertices").ndim

    create_links_array(lg, link_width=2, sid_ndim=rank, offsets=[(0,) * rank])

    segment = next(
        n for n in _per_chunk_paths(lg) if n.startswith("links/")
    )
    assert lg._sharded_chunk_array(segment).ndim == rank
    assert lg._sharded_chunk_array("link_fragments").ndim == rank


def _per_chunk_paths(level_group):
    from zarr_vectors.building import per_chunk_array_paths

    return per_chunk_array_paths(level_group)


def test_a_spatial_only_level_gains_no_leading_axis(tmp_path):
    """Guards against inventing a bin axis where there is none."""
    path = str(tmp_path / "plain.zarrvectors")
    rng = np.random.default_rng(1)
    write_points(
        path, rng.uniform(0, 100, (30, 3)).astype("float32"),
        bounds=[[0, 0, 0], [100, 100, 100]], chunk_shape=_CHUNK,
    )
    lg = _level(path)
    create_attribute_array(lg, "score", dtype="float32")

    assert lg._sharded_chunk_array("vertex_attributes/score").ndim == 3


# --- the silent branch ------------------------------------------------


def test_a_nonzero_origin_store_does_not_alias_cells(tmp_path):
    """The corruption this bug actually caused.

    With a stored grid origin, a rank-4 key zipped against a rank-3
    origin truncated to a rank-3 index that passed every check and
    dropped the last spatial axis. Two keys differing only in that axis
    landed in one cell. Before the fix both reads returned ``SECOND``.
    """
    path = _attr_store(tmp_path / "g.zarrvectors", bounds=_OFFSET_BOUNDS)
    lg = _level(path)
    assert dict(lg._sharded_chunk_array("vertices").attrs).get(
        "chunk_grid_origin"
    ), "fixture has a zero origin; the silent branch is unreachable"

    create_attribute_array(lg, "score", dtype="float32")
    first, second = "1.2.2.2", "1.2.2.3"        # differ only in the last axis
    lg.write_bytes("vertex_attributes/score", first, b"FIRST")
    lg.write_bytes("vertex_attributes/score", second, b"SECOND")

    assert lg.read_bytes("vertex_attributes/score", first) == b"FIRST"
    assert lg.read_bytes("vertex_attributes/score", second) == b"SECOND"


def test_the_manifest_does_not_outrank_its_array(tmp_path):
    """A rank-4 key recorded against a rank-3 array is the tell."""
    path = _attr_store(tmp_path / "g.zarrvectors", bounds=_OFFSET_BOUNDS)
    lg = _level(path)
    create_attribute_array(lg, "score", dtype="float32")
    lg.write_bytes("vertex_attributes/score", "1.2.2.2", b"x")

    arr = lg._sharded_chunk_array("vertex_attributes/score")
    for key in lg.list_chunks("vertex_attributes/score"):
        assert len(key.split(".")) == arr.ndim


# --- a rank skew from any cause is loud -------------------------------


def _mis_allocated(tmp_store_path):
    """A rank-3 array in a store whose keys we will address at rank 4."""
    root = create_store(str(tmp_store_path))
    root.create_sharded_chunk_array("narrow", (4, 4, 4), origin=(0, 2, 2))
    return root


def test_a_rank_mismatch_raises_rather_than_truncating(tmp_store_path):
    root = _mis_allocated(tmp_store_path)
    with pytest.raises(StoreError, match="rank"):
        root.write_bytes("narrow", "1.2.2.2", b"x")


def test_a_rank_mismatch_raises_inside_batched_writes(tmp_store_path):
    """The batch flush re-derives the index with its own copy of the zip."""
    root = _mis_allocated(tmp_store_path)
    with pytest.raises(StoreError, match="rank"):
        with root.batched_writes():
            root.write_bytes("narrow", "1.2.2.2", b"x")


def test_a_rank_mismatch_reads_as_absent(tmp_store_path):
    """Reads treat it as missing rather than aliasing onto a cell."""
    root = _mis_allocated(tmp_store_path)
    assert not root.chunk_exists("narrow", "1.2.2.2")


def test_the_rank_error_names_the_cause(tmp_store_path):
    """"Out of grid" reads as a bad coordinate; the fault is allocation."""
    root = _mis_allocated(tmp_store_path)
    with pytest.raises(StoreError) as excinfo:
        root.write_bytes("narrow", "1.2.2.2", b"x")
    message = str(excinfo.value)
    assert "rank" in message
    assert "leading bin axis" in message


def test_a_matching_rank_still_writes(tmp_store_path):
    """The guard must not reject the ordinary case."""
    root = _mis_allocated(tmp_store_path)
    root.write_bytes("narrow", "2.2.2", b"payload")
    assert root.read_bytes("narrow", "2.2.2") == b"payload"
