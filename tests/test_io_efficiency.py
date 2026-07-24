"""Equivalence tests for the I/O hot-path optimizations.

These guard behavior-preserving performance fixes: the optimized paths must
return results identical to the unoptimized/sibling paths.

* B — ``read_polylines`` selective (object/group subset) path vs the full-read
  path: same fragments, composing correctly with ``bbox`` / ``chunks``.
* A — ``write_links(mode="append")`` in batches vs a single ``replace`` of the
  union: same records, ``num_links``, ``first_new``.
* C — ``read_mesh(bbox=...)`` vectorized vertex/face remap: faces reference only
  kept vertices, remapped consistently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zarr_vectors.core.arrays import (
    read_links,
    write_links,
)
from zarr_vectors.core.paths import links_group_path
from zarr_vectors.core.store import create_store, get_resolution_level
from zarr_vectors.types.meshes import read_mesh, write_mesh
from zarr_vectors.types.polylines import read_polylines, write_polylines

CHUNK = (100.0, 100.0, 100.0)


def _polys_across_chunks(n: int, rng: np.random.Generator) -> list[np.ndarray]:
    """n polylines, each a short path placed in a distinct region so they
    span several chunks (and some cross chunk boundaries)."""
    out = []
    for i in range(n):
        base = np.array([(i % 5) * 90.0, ((i // 5) % 5) * 90.0, 0.0])
        steps = np.array([[0, 0, 0], [30, 10, 0], [60, 20, 0], [95, 30, 0]],
                         dtype=np.float64)
        out.append((base + steps).astype(np.float32))
    return out


def _concat(frags: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(frags, axis=0) if frags else np.zeros((0, 3), np.float32)


# ---------------------------------------------------------------------------
# B. read_polylines selective path == full path
# ---------------------------------------------------------------------------

def test_selective_object_ids_match_full_read(tmp_path: Path):
    store = str(tmp_path / "p.zarrvectors")
    rng = np.random.default_rng(0)
    polys = _polys_across_chunks(20, rng)
    write_polylines(store, polys, chunk_shape=CHUNK)

    full = read_polylines(store)  # full-read path, all oids in order
    assert full["polyline_count"] == 20

    subset = [2, 5, 7, 11, 19]
    sel = read_polylines(store, object_ids=subset)  # selective path
    assert sel["polyline_count"] == len(subset)

    for pos, oid in enumerate(subset):
        np.testing.assert_array_equal(
            _concat(sel["polylines"][pos]), _concat(full["polylines"][oid]),
        )


def test_selective_group_ids_match_full_read(tmp_path: Path):
    store = str(tmp_path / "pg.zarrvectors")
    rng = np.random.default_rng(1)
    polys = _polys_across_chunks(12, rng)
    groups = {0: [0, 2, 4, 6], 1: [1, 3, 5]}
    write_polylines(store, polys, chunk_shape=CHUNK, groups=groups)

    full = read_polylines(store)
    sel = read_polylines(store, group_ids=[0])  # selective via group resolution

    assert sel["polyline_count"] == len(groups[0])
    for pos, oid in enumerate(sorted(groups[0])):
        np.testing.assert_array_equal(
            _concat(sel["polylines"][pos]), _concat(full["polylines"][oid]),
        )


def test_selective_with_bbox_matches_full_with_bbox(tmp_path: Path):
    store = str(tmp_path / "pb.zarrvectors")
    rng = np.random.default_rng(2)
    polys = _polys_across_chunks(20, rng)
    write_polylines(store, polys, chunk_shape=CHUNK)

    bbox = (np.array([0.0, 0.0, 0.0]), np.array([120.0, 120.0, 50.0]))
    full = read_polylines(store, bbox=bbox)
    full_count = full["polyline_count"]

    # Selective over all ids + same bbox must yield the same set of polylines
    # as the full bbox read.
    sel = read_polylines(store, object_ids=list(range(20)), bbox=bbox)
    assert sel["polyline_count"] == full_count
    assert sel["vertex_count"] == full["vertex_count"]


def test_selective_empty_subset(tmp_path: Path):
    store = str(tmp_path / "pe.zarrvectors")
    polys = _polys_across_chunks(5, np.random.default_rng(3))
    write_polylines(store, polys, chunk_shape=CHUNK)
    # Out-of-range oids → skipped, empty result (no crash, no full scan).
    sel = read_polylines(store, object_ids=[999, 1000])
    assert sel["polyline_count"] == 0
    assert sel["vertex_count"] == 0


# ---------------------------------------------------------------------------
# A. write_links append batches == replace of the union
# ---------------------------------------------------------------------------

def _make_links(n: int, rng: np.random.Generator):
    """n edge records ((chunkA, viA), (chunkB, viB)) spanning chunk pairs."""
    recs = []
    for i in range(n):
        ca = (int(i % 3), 0, 0)
        cb = (int(i % 3) + 1, 0, 0)
        recs.append(((ca, int(i)), (cb, int(i * 2))))
    return recs


def _links_level(tmp_path: Path, name: str):
    """A level group whose chunk grid actually holds chunks (0..3, 0, 0).

    The links family is a rank-D array over the level's chunk grid, so a
    source chunk outside it is rejected — bounds must cover the coords
    ``_make_links`` uses.
    """
    root = create_store(
        str(tmp_path / name),
        bounds=([0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]),
        chunk_shape=CHUNK,
        geometry_types=["graph"],
        ndim=3,
    )
    return get_resolution_level(root, 0)


def test_links_append_batches_equal_replace_union(tmp_path: Path):
    rng = np.random.default_rng(4)
    all_recs = _make_links(30, rng)

    # Reference: single replace of the union.
    ref = _links_level(tmp_path, "links_ref.zarrvectors")
    write_links(ref, all_recs, sid_ndim=3, delta=0, mode="replace")
    ref_out = read_links(ref, delta=0)
    ref_meta = ref.read_array_meta(links_group_path(0))

    # Streaming: replace first batch, then append the rest in chunks.
    strm = _links_level(tmp_path, "links_strm.zarrvectors")
    write_links(strm, all_recs[:10], sid_ndim=3, delta=0, mode="replace")
    total_first_new = []
    for start in range(10, 30, 7):
        part = write_links(
            strm, all_recs[start:start + 7], sid_ndim=3, delta=0, mode="append",
        )
        total_first_new.append(part.first_new)

    strm_out = read_links(strm, delta=0)
    strm_meta = strm.read_array_meta(links_group_path(0))

    # Same total record count and the meta num_links matches.
    assert strm_meta["num_links"] == ref_meta["num_links"] == 30
    # first_new of each append reflects the running total (10, 17, 24).
    assert total_first_new == [10, 17, 24]
    # Same multiset of records (cell ordering is deterministic but compare
    # as sets to be robust).
    assert sorted(map(repr, strm_out)) == sorted(map(repr, ref_out))
    # Every record here straddles a chunk boundary, so nothing landed in
    # the intra (all-zero offsets) array and the family IS the 30 records.
    assert len(ref_out) == 30


# ---------------------------------------------------------------------------
# C. read_mesh bbox vectorized remap correctness
# ---------------------------------------------------------------------------

def test_mesh_bbox_remap_keeps_only_in_box_faces(tmp_path: Path):
    store = str(tmp_path / "m.zarrvectors")
    # Two triangles in different chunks; vertices straddle a bbox edge.
    verts = np.array([
        [10, 10, 10], [20, 20, 10], [30, 10, 10],     # tri 0 (in box)
        [150, 150, 10], [160, 160, 10], [170, 150, 10],  # tri 1 (out of box)
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    write_mesh(store, verts, faces, chunk_shape=CHUNK)

    full = read_mesh(store)
    assert full["vertex_count"] == 6
    assert full["face_count"] == 2

    bbox = (np.array([0.0, 0.0, 0.0]), np.array([100.0, 100.0, 100.0]))
    out = read_mesh(store, bbox=bbox)

    # Only the in-box triangle survives; its 3 vertices are kept and faces
    # are remapped into [0, kept).
    assert out["vertex_count"] == 3
    assert out["face_count"] == 1
    assert out["faces"].shape == (1, 3)
    assert set(out["faces"].ravel().tolist()) == {0, 1, 2}
    # Every kept vertex lies in the box.
    assert np.all(out["vertices"] >= bbox[0]) and np.all(out["vertices"] <= bbox[1])
