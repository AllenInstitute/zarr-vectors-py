"""Tests for skeleton ingest, segment-id object index, and skeleton-aware
coarsening (path simplification)."""

from __future__ import annotations

import shutil
import tempfile
from collections import defaultdict

import numpy as np
import pytest

from zarr_vectors.core.arrays import (
    list_chunk_keys,
    read_all_object_manifests,
    read_chunk_fragment_attributes,
    read_chunk_vertices,
)
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.multiresolution.strategies.skeletons import (
    build_skeleton_pyramid,
    coarsen_skeleton_level,
    decimate_skeleton,
    simplify_skeleton,
)
from zarr_vectors.types import skeletons as sk


# --------------------------------------------------------------------------
# Pure simplification
# --------------------------------------------------------------------------

def test_simplify_collapses_straight_chain():
    pos = np.zeros((11, 3), np.float32)
    pos[:, 0] = np.arange(11) * 10.0
    edges = np.array([[i, i - 1] for i in range(1, 11)])
    r = simplify_skeleton(pos, edges, tolerance=1.0)
    # A perfectly straight chain collapses to its two endpoints.
    assert r["kept_source_indices"].tolist() == [0, 10]
    assert r["edges"].tolist() == [[1, 0]]


def test_simplify_preserves_branch_and_endpoints():
    posY = np.array(
        [[0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 5, 0],
         [40, 10, 0], [30, -5, 0], [40, -10, 0]], np.float32)
    edgesY = np.array([[1, 0], [2, 1], [3, 2], [4, 3], [5, 2], [6, 5]])
    rad = np.array([1, 2, 9, 3, 4, 5, 6], np.float32)
    r = simplify_skeleton(posY, edgesY, tolerance=1.0, attributes={"radius": rad})
    kept = set(r["kept_source_indices"].tolist())
    # root(0), branch(2), and both leaves(4,6) must survive.
    assert {0, 2, 4, 6} <= kept
    # radius is max-aggregated onto survivors.
    assert r["attributes"]["radius"].max() == pytest.approx(9.0)


def test_decimate_keeps_every_kth_plus_anchors():
    # straight chain of 33 vertices → keep endpoints + every 8th interior
    pos = np.zeros((33, 3), np.float32)
    pos[:, 0] = np.arange(33) * 10.0
    edges = np.array([[i, i - 1] for i in range(1, 33)])
    r = decimate_skeleton(pos, edges, stride=8)
    kept = r["kept_source_indices"].tolist()
    assert kept[0] == 0 and kept[-1] == 32          # endpoints kept
    assert kept == [0, 8, 16, 24, 32]               # every 8th + last
    # all kept vertices are original positions (no new points)
    assert {tuple(v) for v in r["positions"]} <= {tuple(v) for v in pos}


def test_decimate_preserves_branch_and_forced():
    # Y: branch at 2, leaves 4 & 6; force-keep vertex 1 (a chunk boundary)
    posY = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 5, 0],
                     [40, 10, 0], [30, -5, 0], [40, -10, 0]], np.float32)
    edgesY = np.array([[1, 0], [2, 1], [3, 2], [4, 3], [5, 2], [6, 5]])
    r = decimate_skeleton(posY, edgesY, stride=8, forced_keep=[1])
    kept = set(r["kept_source_indices"].tolist())
    assert {0, 2, 4, 6} <= kept     # root + branch + both leaves
    assert 1 in kept                # forced (chunk-boundary) vertex


def test_split_components_two_disjoint_paths():
    pos = np.array([[0, 0, 0], [1, 0, 0], [5, 5, 5], [6, 5, 5]], np.float32)
    edges = np.array([[0, 1], [2, 3]])  # two separate edges
    pieces = sk.split_components(pos, edges)
    assert len(pieces) == 2
    assert all(len(p["positions"]) == 2 for p in pieces)


# --------------------------------------------------------------------------
# Level-0 writer + segment-id object index + pull-by-id
# --------------------------------------------------------------------------

@pytest.fixture
def tmp_store():
    d = tempfile.mkdtemp(suffix=".zv")
    shutil.rmtree(d)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _write_two_skeletons(d):
    chunk_shape = (100.0, 100.0, 100.0)
    bounds = ([0.0, 0.0, 0.0], [200.0, 100.0, 100.0])
    root, lg = sk.init_skeleton_store(
        d, chunk_shape=chunk_shape, bounds=bounds, ndim=3,
        attribute_dtypes={"radius": "float32"})
    # seg 12345: chain crossing x=100 → two chunks.
    p0 = np.array([[x, 50, 50] for x in range(10, 100, 10)], np.float32)
    e0 = np.array([[i, i - 1] for i in range(1, len(p0))])
    r0 = np.linspace(1, 5, len(p0)).astype(np.float32)
    p1 = np.array([[x, 50, 50] for x in range(110, 200, 10)], np.float32)
    e1 = np.array([[i, i - 1] for i in range(1, len(p1))])
    r1 = np.linspace(5, 9, len(p1)).astype(np.float32)
    # seg 999: a Y wholly in chunk (0,0,0).
    pY = np.array([[20, 20, 20], [30, 20, 20], [40, 20, 20],
                   [50, 30, 20], [50, 10, 20]], np.float32)
    eY = np.array([[1, 0], [2, 1], [3, 2], [4, 2]])
    rY = np.array([2, 2, 3, 1, 1], np.float32)
    records = []
    recs, _ = sk.write_skeleton_chunk(lg, (0, 0, 0), [
        {"segment_id": 12345, "positions": p0, "edges": e0, "attributes": {"radius": r0}},
        {"segment_id": 999, "positions": pY, "edges": eY, "attributes": {"radius": rY}},
    ], attr_dtypes={"radius": np.float32})
    records += recs
    recs, _ = sk.write_skeleton_chunk(lg, (1, 0, 0), [
        {"segment_id": 12345, "positions": p1, "edges": e1, "attributes": {"radius": r1}},
    ], attr_dtypes={"radius": np.float32})
    records += recs
    oid_of = sk.build_object_index(lg, records, ndim=3)
    sk.finalize_skeleton_store(root)
    return oid_of, (r0, r1)


def test_pull_by_segment_id_roundtrip(tmp_store):
    oid_of, (r0, r1) = _write_two_skeletons(tmp_store)
    assert set(oid_of) == {999, 12345}

    res = sk.read_skeleton_by_segment_id(tmp_store, 12345)
    assert res["fragment_count"] == 2          # one piece per chunk
    assert len(res["positions"]) == 18
    assert len(res["edges"]) == 16             # 8 per fragment, root dropped
    assert np.allclose(np.sort(res["attributes"]["radius"]),
                       np.sort(np.concatenate([r0, r1])))

    resY = sk.read_skeleton_by_segment_id(tmp_store, 999)
    assert len(resY["positions"]) == 5
    assert len(resY["edges"]) == 4

    assert sk.read_skeleton_by_segment_id(tmp_store, 7777) is None


def test_segment_id_fragment_attribute_written(tmp_store):
    """Each fragment carries its owning segment id as a per-fragment
    attribute, in fragment order — drives the renderer's per-fragment
    colouring.  A branching (Y) skeleton decomposes into >1 path fragment,
    all of which must report the same owning segment id."""
    chunk_shape = (100.0, 100.0, 100.0)
    bounds = ([0.0, 0.0, 0.0], [100.0, 100.0, 100.0])
    root, lg = sk.init_skeleton_store(
        tmp_store, chunk_shape=chunk_shape, bounds=bounds, ndim=3,
        attribute_dtypes={})
    big = 720575940612786691  # flywire-scale uint64
    # A straight chain (one fragment) + a Y (two path fragments).
    pA = np.array([[10, 10, 10], [20, 10, 10], [30, 10, 10]], np.float32)
    eA = np.array([[1, 0], [2, 1]])
    pY = np.array([[20, 20, 20], [30, 20, 20], [40, 20, 20],
                   [50, 30, 20], [50, 10, 20]], np.float32)
    eY = np.array([[1, 0], [2, 1], [3, 2], [4, 2]])
    recs, _ = sk.write_skeleton_chunk(lg, (0, 0, 0), [
        {"segment_id": 999, "positions": pA, "edges": eA, "attributes": {}},
        {"segment_id": big, "positions": pY, "edges": eY, "attributes": {}},
    ])
    sk.build_object_index(lg, recs, ndim=3)
    sk.finalize_skeleton_store(root)

    lg2 = get_resolution_level(open_store(tmp_store), 0)
    seg_ids = read_chunk_fragment_attributes(
        lg2, "segment_id", (0, 0, 0), dtype=np.uint64)
    # One fragment per record, in fragment order.
    assert seg_ids.tolist() == [r[0] for r in recs]
    # The Y produced two fragments, both owned by `big`.
    assert (seg_ids == big).sum() == 2
    assert (seg_ids == 999).sum() == 1


def test_segment_id_uint64_preserved(tmp_store):
    """Large flywire-style uint64 IDs survive the object index."""
    big = 720575940600000000  # > 2**32
    chunk_shape = (100.0, 100.0, 100.0)
    bounds = ([0.0, 0.0, 0.0], [100.0, 100.0, 100.0])
    root, lg = sk.init_skeleton_store(
        tmp_store, chunk_shape=chunk_shape, bounds=bounds, ndim=3,
        attribute_dtypes={})
    p = np.array([[10, 10, 10], [20, 10, 10], [30, 10, 10]], np.float32)
    e = np.array([[1, 0], [2, 1]])
    recs, _ = sk.write_skeleton_chunk(lg, (0, 0, 0), [
        {"segment_id": big, "positions": p, "edges": e, "attributes": {}}])
    sk.build_object_index(lg, recs, ndim=3)
    sk.finalize_skeleton_store(root)
    res = sk.read_skeleton_by_segment_id(tmp_store, big)
    assert res is not None
    assert len(res["positions"]) == 3


# --------------------------------------------------------------------------
# Pyramid
# --------------------------------------------------------------------------

def _build_random_skeleton_store(d, n_seg=30, seed=0):
    rng = np.random.default_rng(seed)
    chunk_shape = (256.0, 256.0, 256.0)
    bounds = ([0.0, 0.0, 0.0], [1024.0, 1024.0, 256.0])
    root, lg = sk.init_skeleton_store(
        d, chunk_shape=chunk_shape, bounds=bounds, ndim=3,
        attribute_dtypes={"radius": "float32"})
    records = []
    chunk_pieces = defaultdict(list)
    for s in range(n_seg):
        seg = 1000 + s * 7
        npts = int(rng.integers(40, 120))
        pos = np.cumsum(rng.normal(0, 8, size=(npts, 3)), axis=0)
        pos = np.clip(pos + rng.uniform([10, 10, 10], [1000, 1000, 240]),
                      [1, 1, 1], [1023, 1023, 255]).astype(np.float32)
        cc_of = np.floor(pos / np.array(chunk_shape)).astype(int)
        i = 0
        while i < len(pos):
            j = i
            while j + 1 < len(pos) and tuple(cc_of[j + 1]) == tuple(cc_of[i]):
                j += 1
            sp = pos[i:j + 1]
            se = np.array([[k, k - 1] for k in range(1, len(sp))])
            rad = rng.uniform(50, 300, len(sp)).astype(np.float32)
            chunk_pieces[tuple(int(x) for x in cc_of[i])].append(
                {"segment_id": seg, "positions": sp, "edges": se,
                 "attributes": {"radius": rad}})
            i = j + 1
    for cc, pieces in chunk_pieces.items():
        recs, _ = sk.write_skeleton_chunk(lg, cc, pieces,
                                          attr_dtypes={"radius": np.float32})
        records += recs
    sk.build_object_index(lg, records, ndim=3)
    sk.finalize_skeleton_store(root)
    return n_seg


def _level_stats(d, lvl):
    g = get_resolution_level(open_store(d), lvl)
    mans = read_all_object_manifests(g)
    nv = sum(len(f) for cc in list_chunk_keys(g, "vertices")
             for f in read_chunk_vertices(g, cc, ndim=3))
    return {"objs": sum(1 for m in mans if m), "verts": nv, "slots": len(mans)}


def test_pyramid_preserves_oids_and_shrinks(tmp_store):
    n_seg = _build_random_skeleton_store(tmp_store, n_seg=30)
    l0 = _level_stats(tmp_store, 0)
    assert l0["objs"] == n_seg

    build_skeleton_pyramid(
        tmp_store, strides=[8, 8], chunk_scale_factors=[2, 2],
        sparsity_factors=[1.0, 1.0], sparsity_strategy="length")

    l1 = _level_stats(tmp_store, 1)
    l2 = _level_stats(tmp_store, 2)
    # Coarsen-only levels keep every object …
    assert l1["objs"] == n_seg
    assert l2["objs"] == n_seg
    # … the OID slot space is stable …
    assert l0["slots"] == l1["slots"] == l2["slots"] == n_seg
    # … and total vertices do not grow.
    assert l1["verts"] <= l0["verts"]
    assert l2["verts"] <= l1["verts"]


def test_pyramid_sparsity_drops_objects(tmp_store):
    n_seg = _build_random_skeleton_store(tmp_store, n_seg=30, seed=3)
    build_skeleton_pyramid(
        tmp_store, strides=[8], chunk_scale_factors=[2],
        sparsity_factors=[3.0], sparsity_strategy="length", sparsity_seed=1)
    l1 = _level_stats(tmp_store, 1)
    assert l1["slots"] == n_seg          # OID space preserved
    assert l1["objs"] < n_seg            # but fewer objects survive
    assert l1["objs"] == pytest.approx(n_seg // 3, abs=2)


def test_pull_by_id_works_at_coarse_levels(tmp_store):
    _build_random_skeleton_store(tmp_store, n_seg=20, seed=5)
    build_skeleton_pyramid(
        tmp_store, strides=[8], chunk_scale_factors=[2],
        sparsity_factors=[1.0])
    seg = 1000 + 5 * 7
    r0 = sk.read_skeleton_by_segment_id(tmp_store, seg, level=0)
    r1 = sk.read_skeleton_by_segment_id(tmp_store, seg, level=1)
    assert r0 is not None and r1 is not None
    assert len(r1["positions"]) <= len(r0["positions"])


def test_ingest_driver_offline(tmp_store):
    """End-to-end ETL (extract→write→reduce→pyramid) via an in-memory
    .frags reader, with a flywire-shaped spatial index and a phase-offset
    chunk grid (so one zarr chunk collects pieces from several .frags)."""
    from zarr_vectors.ingest.precomputed_skeletons import (
        InMemoryFragsReader, SkeletonInfo, enumerate_frag_keys, run_ingest,
    )

    info = SkeletonInfo(
        base_url="mem://x",
        resolution_nm=(32.0, 32.0, 40.0),
        chunk_size_nm=(16384.0, 16384.0, 20480.0),
        vertex_attributes=[
            {"id": "radius", "data_type": "float32", "num_components": 1},
            {"id": "cross_sectional_area", "data_type": "float32", "num_components": 1},
        ])
    cs_nm = np.array(info.chunk_size_nm)
    rng = np.random.default_rng(0)
    anchor = (17910, 8912, 3088)             # phase-offset from origin 0
    keys = enumerate_frag_keys(info, anchor, (2, 2, 1))
    corners = [(x, y, z)
               for x in (anchor[0], anchor[0] + 512)
               for y in (anchor[1], anchor[1] + 512)
               for z in (anchor[2],)]
    chunks = {}
    seg = 720575940000000000
    all_segs = set()
    for k, (vx, vy, vz) in zip(keys, corners):
        origin = np.array([vx, vy, vz]) * np.array(info.resolution_nm)
        d = {}
        for _ in range(6):
            sid = seg; seg += 13; all_segs.add(sid)
            npts = int(rng.integers(30, 80))
            start = origin + rng.uniform(0.1, 0.9, 3) * cs_nm
            pos = np.clip(np.cumsum(rng.normal(0, 200, (npts, 3)), axis=0) + start,
                          origin + 10, origin + cs_nm - 10).astype(np.float32)
            d[sid] = {
                "vertices": pos,
                "edges": np.array([[i, i - 1] for i in range(1, npts)]),
                "radius": rng.uniform(50, 400, npts).astype(np.float32),
                "cross_sectional_area": rng.uniform(1e3, 1e5, npts).astype(np.float32),
            }
        chunks[k] = d

    reader = InMemoryFragsReader(info, chunks)
    bounds = ([anchor[a] * info.resolution_nm[a] for a in range(3)],
              [(anchor[a] + (2 if a < 2 else 1) * 512) * info.resolution_nm[a]
               for a in range(3)])
    summary = run_ingest(
        reader, tmp_store, keys, bounds_nm=bounds,
        strides=[8, 8], chunk_scale_factors=[2, 2],
        sparsity_factors=[1.0, 2.0], progress=False)

    assert summary["objects"] == len(all_segs)
    # every segment pulls back out by its uint64 id, with both attributes
    for sid in all_segs:
        r0 = sk.read_skeleton_by_segment_id(tmp_store, sid, level=0)
        assert r0 is not None and len(r0["positions"]) > 0
        assert "radius" in r0["attributes"]
        assert "cross_sectional_area" in r0["attributes"]
    # coarsen-only level keeps all objects; sparsity level drops some
    levels = summary["pyramid"]["levels"]
    assert levels[0]["object_count"] == len(all_segs)
    assert levels[1]["object_count"] < len(all_segs)


def test_cross_chunk_edges_merge_fragments_one_level_up(tmp_store):
    """A segment split across a zarr-chunk boundary at level 0 is stored as
    two fragments + one cross-chunk link; one level up (chunks ×2) both
    pieces fall in the same chunk and the link re-merges them into a single
    connected fragment — no proximity heuristic involved."""
    from zarr_vectors.ingest.precomputed_skeletons import (
        InMemoryFragsReader, SkeletonInfo, run_ingest,
    )
    from zarr_vectors.core.store import open_store, get_resolution_level
    from zarr_vectors.core.arrays import read_cross_chunk_links

    info = SkeletonInfo(
        base_url="mem://x", resolution_nm=(32.0, 32.0, 40.0),
        chunk_size_nm=(16384.0, 16384.0, 20480.0))
    # Straight line in x crossing the origin-0 boundary at 35*16384=573440.
    xs = np.arange(573300.0, 573600.0, 20.0)
    pos = np.stack([xs, np.full_like(xs, 335000.0), np.full_like(xs, 130000.0)], 1
                   ).astype(np.float32)
    edges = np.array([[i, i - 1] for i in range(1, len(xs))])
    seg = 720575940000000123
    key = "17910-18422_10448-10960_3088-3600.frags"
    reader = InMemoryFragsReader(info, {key: {seg: {"vertices": pos, "edges": edges}}})
    bounds = ([573120.0, 334336.0, 123520.0], [589504.0, 350720.0, 144000.0])

    # align=False keeps the origin-0 grid so the line is phase-split into
    # two chunks, producing the cross-chunk edge that drives merging.
    summary = run_ingest(reader, tmp_store, [key], bounds_nm=bounds,
                         strides=[2], chunk_scale_factors=[2],
                         sparsity_factors=[1.0], align=False, progress=False)
    assert summary["level0_cross_chunk_edges"] == 1

    # Level 0: two fragments, link present.
    r0 = sk.read_skeleton_by_segment_id(tmp_store, seg, level=0)
    assert r0["fragment_count"] == 2
    g0 = get_resolution_level(open_store(tmp_store), 0)
    assert len(read_cross_chunk_links(g0, delta=0)) == 1

    # Level 1: merged to one fragment, fully connected (n-1 edges, 1 comp).
    r1 = sk.read_skeleton_by_segment_id(tmp_store, seg, level=1)
    assert r1["fragment_count"] == 1
    n1 = len(r1["positions"])
    assert len(r1["edges"]) == n1 - 1   # a single tree over all vertices
    # endpoints preserved → x-extent still spans the original line
    assert r1["positions"][:, 0].min() < 573440 < r1["positions"][:, 0].max()


def test_alignment_no_split_and_world_roundtrip(tmp_store):
    """With align=True the chunk grid is shifted to the .frag grid, so a
    piece within one .frag stays a single fragment (no phase-split, no
    cross-chunk edges), and reads return absolute world coordinates."""
    from zarr_vectors.ingest.precomputed_skeletons import (
        InMemoryFragsReader, SkeletonInfo, run_ingest,
    )
    info = SkeletonInfo(
        base_url="mem://x", resolution_nm=(32.0, 32.0, 40.0),
        chunk_size_nm=(16384.0, 16384.0, 20480.0))
    # A line that WOULD cross the origin-0 boundary at 573440 (abs coords).
    xs = np.arange(573300.0, 573600.0, 20.0)
    pos = np.stack([xs, np.full_like(xs, 335000.0), np.full_like(xs, 130000.0)], 1
                   ).astype(np.float32)
    edges = np.array([[i, i - 1] for i in range(1, len(xs))])
    seg = 720575940000000123
    key = "17910-18422_10448-10960_3088-3600.frags"
    reader = InMemoryFragsReader(info, {key: {seg: {"vertices": pos, "edges": edges}}})
    bounds = ([573120.0, 334336.0, 123520.0], [589504.0, 350720.0, 144000.0])

    summary = run_ingest(reader, tmp_store, [key], bounds_nm=bounds,
                         strides=[], align=True, progress=False)
    # aligned → the .frag piece is wholly inside one zarr chunk
    assert summary["level0_cross_chunk_edges"] == 0
    r0 = sk.read_skeleton_by_segment_id(tmp_store, seg, level=0)
    assert r0["fragment_count"] == 1
    # reads come back in absolute world coordinates (offset added back)
    assert r0["positions"][:, 0].min() == pytest.approx(573300.0, abs=1.0)
    assert r0["positions"][:, 0].max() == pytest.approx(573580.0, abs=1.0)


def test_coincident_boundary_vertices_become_cross_chunk_edges(tmp_store):
    """Aligned ingest recovers cross-chunk edges from exact coincident
    boundary vertices (as igneous writes them): a segment present in two
    adjacent .frags with a shared boundary vertex gets a level-0
    cross-chunk link, and the two fragments merge into one connected
    fragment one level up."""
    from zarr_vectors.ingest.precomputed_skeletons import (
        InMemoryFragsReader, SkeletonInfo, enumerate_frag_keys, run_ingest,
    )
    info = SkeletonInfo(
        base_url="mem://x", resolution_nm=(32.0, 32.0, 40.0),
        chunk_size_nm=(16384.0, 16384.0, 20480.0))
    keys = enumerate_frag_keys(info, (17398, 10448, 3088), (2, 1, 1))  # kA, kB in x
    # Duplicated boundary vertices sit at the chunk face + 0.5 voxel (the
    # voxel-center convention), exactly as flywire stores them.
    bnd = 17910 * 32 + 16  # face 573120 + 0.5*32 = 573136
    y, z = 340000.0, 130000.0
    seg = 720575940000000999
    shared = [bnd, y, z]
    # A: ... up to the shared boundary vertex
    aA = np.array([[560000, y, z], [565000, y, z], [570000, y, z], shared], np.float32)
    eA = np.array([[i, i - 1] for i in range(1, len(aA))])
    # B: starts at the same shared vertex, continues
    aB = np.array([shared, [576000, y, z], [580000, y, z]], np.float32)
    eB = np.array([[i, i - 1] for i in range(1, len(aB))])
    reader = InMemoryFragsReader(info, {
        keys[0]: {seg: {"vertices": aA, "edges": eA}},
        keys[1]: {seg: {"vertices": aB, "edges": eB}},
    })
    bounds = ([17398 * 32.0, 10448 * 32.0, 3088 * 40.0],
              [18422 * 32.0, 10960 * 32.0, 3600 * 40.0])

    summary = run_ingest(reader, tmp_store, keys, bounds_nm=bounds,
                         strides=[2], chunk_scale_factors=[2],
                         sparsity_factors=[1.0], align=True, progress=False)
    assert summary["level0_cross_chunk_edges"] >= 1

    r0 = sk.read_skeleton_by_segment_id(tmp_store, seg, level=0)
    assert r0["fragment_count"] == 2          # one per .frag, aligned 1:1
    r1 = sk.read_skeleton_by_segment_id(tmp_store, seg, level=1)
    assert r1["fragment_count"] == 1          # merged via the cross-chunk edge
    n1 = len(r1["positions"])
    assert len(r1["edges"]) == n1 - 1         # single connected tree


def test_links_have_no_phantom_indices(tmp_store):
    """Regression: a branching skeleton must be stored as linear-path
    fragments + branch links with NO ``-1`` / out-of-range link indices
    (those rendered as edges to a phantom origin vertex in neuroglancer).
    Reconstruction must yield the connected tree (components == 1)."""
    from zarr_vectors.core.store import open_store, get_resolution_level
    from zarr_vectors.core.arrays import read_chunk_links, read_chunk_vertices

    chunk_shape = (1000.0, 1000.0, 1000.0)
    bounds = ([0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0])
    root, lg = sk.init_skeleton_store(
        tmp_store, chunk_shape=chunk_shape, bounds=bounds, ndim=3,
        attribute_dtypes={})
    # A bushy tree: trunk with two side branches → multiple paths.
    pos = np.array([[10, 10, 10], [20, 10, 10], [30, 10, 10], [40, 20, 10],
                    [50, 30, 10], [40, 0, 10], [50, -10, 10], [40, 10, 30]],
                   np.float32)
    edges = np.array([[1, 0], [2, 1], [3, 2], [4, 3], [5, 2], [6, 5], [7, 2]])
    recs, _ = sk.write_skeleton_chunk(lg, (0, 0, 0), [
        {"segment_id": 42, "positions": pos, "edges": edges, "attributes": {}}])
    sk.build_object_index(lg, recs, ndim=3)
    sk.finalize_skeleton_store(root)

    g = get_resolution_level(open_store(tmp_store), 0)
    nverts = sum(len(f) for f in read_chunk_vertices(g, (0, 0, 0), ndim=3))
    for grp in read_chunk_links(g, (0, 0, 0), link_width=2, delta=0):
        if len(grp) == 0:
            continue
        flat = np.asarray(grp).reshape(-1)
        assert flat.min() >= 0, "link references a phantom (negative) vertex"
        assert flat.max() < nverts, "link references an out-of-range vertex"

    r = sk.read_skeleton_by_segment_id(tmp_store, 42)
    assert len(r["positions"]) == 8
    # connected tree: components = n - edges == 1
    assert len(r["positions"]) - len(r["edges"]) == 1


def test_coarse_manifest_maps_correct_fragments(tmp_store):
    """Regression: a branching object decomposes into multiple path
    fragments per chunk, so the coarse-level object index must point each
    object to ITS OWN fragments (built from the writer's records, not the
    piece order).  With two branching skeletons in one chunk, reading each
    back at a coarse level must return only that object's vertices."""
    chunk_shape = (1000.0, 1000.0, 1000.0)
    bounds = ([0.0, 0.0, 0.0], [2000.0, 1000.0, 1000.0])
    root, lg = sk.init_skeleton_store(
        tmp_store, chunk_shape=chunk_shape, bounds=bounds, ndim=3,
        attribute_dtypes={})

    def make_y(ox):
        # trunk (0..3) with two diverging arms branching at vertex 3
        pos = np.array([
            [ox + 10, 500, 500], [ox + 30, 500, 500], [ox + 50, 500, 500],
            [ox + 70, 500, 500],
            [ox + 90, 560, 500], [ox + 110, 620, 500],   # arm A
            [ox + 90, 440, 500], [ox + 110, 380, 500],   # arm B
        ], np.float32)
        edges = np.array([[1, 0], [2, 1], [3, 2],
                          [4, 3], [5, 4], [6, 3], [7, 6]])
        return pos, edges

    pA, eA = make_y(0)
    pB, eB = make_y(300)
    recs, _ = sk.write_skeleton_chunk(lg, (0, 0, 0), [
        {"segment_id": 111, "positions": pA, "edges": eA, "attributes": {}},
        {"segment_id": 222, "positions": pB, "edges": eB, "attributes": {}},
    ])
    sk.build_object_index(lg, recs, ndim=3)
    sk.finalize_skeleton_store(root)

    # small tolerance keeps branch + endpoints; the object still has >1 path
    build_skeleton_pyramid(tmp_store, strides=[2], chunk_scale_factors=[2],
                           sparsity_factors=[1.0])

    for seg in (111, 222):
        r0 = sk.read_skeleton_by_segment_id(tmp_store, seg, level=0)
        r1 = sk.read_skeleton_by_segment_id(tmp_store, seg, level=1)
        s0 = {tuple(np.rint(v).astype(int)) for v in r0["positions"]}
        # every coarse vertex must be one of THIS object's level-0 vertices
        assert all(tuple(np.rint(v).astype(int)) in s0 for v in r1["positions"]), (
            f"seg {seg}: coarse manifest points to wrong fragments"
        )
        assert len(r1["positions"]) >= 2


def test_coarsen_level_routes_to_skeleton(tmp_store):
    """The generic coarsen_level dispatches skeleton stores to the
    skeleton coarsener (coarsen_factor interpreted as tolerance)."""
    from zarr_vectors.multiresolution.coarsen import coarsen_level
    from zarr_vectors.constants import COARSEN_SKELETON
    _build_random_skeleton_store(tmp_store, n_seg=15, seed=7)
    summary = coarsen_level(tmp_store, 0, 1, coarsen_factor=200.0,
                            chunk_scale_factor=2)
    assert summary["method"] == COARSEN_SKELETON
    assert _level_stats(tmp_store, 1)["slots"] == 15
