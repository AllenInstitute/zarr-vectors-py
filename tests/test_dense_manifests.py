"""The dense object-index layout reads back what the vlen layout does.

The vlen layout is the reference. The same manifests written each way
must read back identically through every manifest reader -- by id, by
row, all at once, as CSR, as a presence mask -- after full writes,
appends (with gaps and torn flushes), patches and appender runs. On top
of that: the arrays are what the spec says, an index keeps its layout,
and writing a million single-fragment objects builds no Python object
per object (BRIDGE R4).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from zarr_vectors.building import (
    create_store,
    get_resolution_level,
    open_store,
    read_all_object_manifests,
    read_all_object_manifests_csr,
    write_object_manifests,
)
from zarr_vectors.constants import CAP_DENSE_MANIFESTS, OBJECT_INDEX
from zarr_vectors.core import dense_manifests as dense
from zarr_vectors.core.arrays import (
    object_ids_for_rows,
    object_present_mask,
    patch_object_manifests,
    read_object_manifest,
    read_object_manifest_rows,
    read_object_manifests,
    write_object_index,
)
from zarr_vectors.core.store import read_root_metadata
from zarr_vectors.exceptions import ArrayError

pytestmark = pytest.mark.vlen_only  # these pick their layouts themselves


def _level(tmp_path, name, layout):
    (tmp_path / name).mkdir()
    path = tmp_path / name / "s.zarrvectors"
    root = create_store(
        path, bounds=([0.0] * 3, [100.0] * 3), chunk_shape=(50.0,) * 3,
        geometry_types=["polyline"], manifest_layout=layout,
    )
    return path, get_resolution_level(root, 0)


def _manifests(rng, n, *, empty_share=0.2, max_blocks=4):
    out = {}
    for oid in range(n):
        k = 0 if rng.random() < empty_share else int(rng.integers(1, max_blocks + 1))
        out[oid] = [
            (tuple(int(c) for c in rng.integers(0, 2, 3)), int(rng.integers(0, 2**40)))
            for _ in range(k)
        ]
    return out


def _commit(lg, n):
    meta = lg.read_array_meta(OBJECT_INDEX) or {}
    lg.write_array_meta(OBJECT_INDEX, {
        "layout": "vlen_manifests_v1",  # what a vlen writer's commit stamps
        **meta, "zv_array": "object_index", "num_objects": n, "num_present": n,
        "sid_ndim": 3,
    })


def _same_reads(a, b, ids=None):
    """Every manifest reader agrees between two levels."""
    assert read_all_object_manifests(a) == read_all_object_manifests(b)
    ids_a, rows_a = read_object_manifest_rows(a)
    ids_b, rows_b = read_object_manifest_rows(b)
    assert ids_a.tolist() == ids_b.tolist() and rows_a == rows_b
    assert read_object_manifests(a) == read_object_manifests(b)
    probe = ids if ids is not None else ids_a.tolist()[::3]
    assert read_object_manifests(a, ids=probe) == read_object_manifests(b, ids=probe)
    for oid in [o for o in probe if o in set(ids_a.tolist())][:10]:
        assert read_object_manifest(a, oid) == read_object_manifest(b, oid)
    csr_a, csr_b = read_all_object_manifests_csr(a), read_all_object_manifests_csr(b)
    for x, y in zip(csr_a, csr_b):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(object_present_mask(a), object_present_mask(b))


def test_the_store_declares_the_layout(tmp_path):
    path, _ = _level(tmp_path, "d", "dense")
    meta = read_root_metadata(open_store(str(path)))
    assert meta.manifest_layout == "dense"
    assert CAP_DENSE_MANIFESTS in meta.format_capabilities
    vlen_path, _ = _level(tmp_path, "v", None)
    assert read_root_metadata(open_store(str(vlen_path))).manifest_layout is None


def test_a_full_write_reads_back_as_the_vlen_one(tmp_path):
    manifests = _manifests(np.random.default_rng(0), 300)
    _, v = _level(tmp_path, "v", None)
    _, d = _level(tmp_path, "d", "dense")
    for lg in (v, d):
        write_object_index(lg, manifests, 3)
    assert d.array_exists(dense.SPANS_PATH) and not d.array_exists(f"{OBJECT_INDEX}/manifests")
    assert d.read_array_meta(OBJECT_INDEX)["layout"] == dense.OBJECT_INDEX_LAYOUT_DENSE
    assert (
        d.read_array_meta(OBJECT_INDEX)["num_present"]
        == v.read_array_meta(OBJECT_INDEX)["num_present"]
    )
    _same_reads(v, d)


def test_the_arrays_are_what_the_spec_says(tmp_path):
    _, d = _level(tmp_path, "d", "dense")
    write_object_index(d, {
        0: [((1, 0, 0), 4), ((0, 1, 1), 7)], 1: [], 2: [((1, 1, 1), 2)],
    }, 3)
    spans = d.read_array(dense.SPANS_PATH)
    blocks = d.read_array(dense.BLOCKS_PATH)
    assert spans.dtype == np.int64 and spans.tolist() == [[0, 2], [2, 0], [2, 1]]
    assert blocks.dtype == np.int64
    assert blocks.tolist() == [[1, 0, 0, 4], [0, 1, 1, 7], [1, 1, 1, 2]]
    assert object_ids_for_rows(d).tolist() == [0, 1, 2]


def test_sparse_ids_and_total_objects(tmp_path):
    manifests = {5: [((0, 0, 0), 1)], 2**40: [((1, 1, 1), 3)]}
    _, v = _level(tmp_path, "v", None)
    _, d = _level(tmp_path, "d", "dense")
    for lg in (v, d):
        write_object_index(lg, manifests, 3, total_objects=8)
    _same_reads(v, d, ids=[0, 5, 7, 2**40, 99])


def test_array_appends_with_gaps_and_a_torn_flush(tmp_path):
    rng = np.random.default_rng(1)
    _, v = _level(tmp_path, "v", None)
    _, d = _level(tmp_path, "d", "dense")
    n_total = 0
    for step, gap in enumerate([0, 0, 4, 0, -3, 0]):
        k = int(rng.integers(0, 30))
        counts = rng.integers(0, 4, k)
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        coords = rng.integers(0, 2, (int(offsets[-1]), 3)).astype(np.int64)
        frags = rng.integers(0, 1000, int(offsets[-1])).astype(np.int64)
        at = max(0, n_total + gap)
        for lg in (v, d):
            got = write_object_manifests(
                lg, chunk_coords=coords, fragment_idx=frags,
                manifest_offsets=offsets, mode="append", at=at,
            )
            assert got == (at, k)
            _commit(lg, at + k)
        n_total = at + k
    _same_reads(v, d)


def test_appends_to_an_index_that_stores_sparse_ids(tmp_path):
    _, v = _level(tmp_path, "v", None)
    _, d = _level(tmp_path, "d", "dense")
    for lg in (v, d):
        write_object_index(lg, {10: [((0, 0, 0), 1)], 20: [((1, 0, 0), 2)]}, 3)
        write_object_manifests(
            lg, chunk_coords=[[1, 1, 1]], fragment_idx=[5], ids=[40],
            mode="append", at=2,
        )
        meta = lg.read_array_meta(OBJECT_INDEX)
        lg.write_array_meta(OBJECT_INDEX, {**meta, "num_objects": 3, "num_present": 3})
        with pytest.raises(ArrayError, match="pass ids="):
            write_object_manifests(
                lg, chunk_coords=[[0, 0, 0]], fragment_idx=[1], mode="append", at=3,
            )
    _same_reads(v, d, ids=[10, 20, 40])


def test_patches_touch_only_what_they_change(tmp_path):
    manifests = _manifests(np.random.default_rng(2), 200)
    _, v = _level(tmp_path, "v", None)
    _, d = _level(tmp_path, "d", "dense")
    updates = {
        3: [((1, 1, 1), 9)], 50: [],
        199: [((0, 0, 1), 2), ((1, 0, 1), 3)], 500: [((0, 0, 0), 1)],
    }
    blocks_before = None
    for lg in (v, d):
        write_object_index(lg, manifests, 3)
        if lg is d:
            blocks_before = dense.num_blocks(d)
        patch_object_manifests(lg, updates, 3)
    # Appended, never rewritten: the new blocks follow the old ones.
    assert dense.num_blocks(d) == blocks_before + 4
    assert (
        d.read_array_meta(OBJECT_INDEX)["num_present"]
        == v.read_array_meta(OBJECT_INDEX)["num_present"]
    )
    _same_reads(v, d, ids=[3, 50, 199, 500, 0])


def test_a_rewrite_compacts_patched_away_blocks(tmp_path):
    _, d = _level(tmp_path, "d", "dense")
    write_object_index(d, {0: [((0, 0, 0), 1)], 1: [((1, 0, 0), 2)]}, 3)
    patch_object_manifests(d, {0: [((1, 1, 1), 5)]}, 3)
    assert dense.num_blocks(d) == 3
    ids, rows = read_object_manifest_rows(d)
    write_object_index(d, dict(zip(ids.tolist(), rows)), 3)
    assert dense.num_blocks(d) == 2
    assert read_object_manifest(d, 0) == [((1, 1, 1), 5)]


def test_an_index_keeps_its_layout(tmp_path):
    """A vlen index on a dense store stays vlen, and the other way round."""
    _, d = _level(tmp_path, "d", "dense")
    write_object_index(d, {0: [((0, 0, 0), 1)]}, 3, layout="vlen")
    write_object_manifests(d, chunk_coords=[[1, 0, 0]], fragment_idx=[2], mode="append", at=1)
    assert d.array_exists(f"{OBJECT_INDEX}/manifests")
    assert not d.array_exists(dense.SPANS_PATH)

    _, v = _level(tmp_path, "v", None)
    write_object_index(v, {0: [((0, 0, 0), 1)]}, 3, layout="dense")
    write_object_manifests(v, chunk_coords=[[1, 0, 0]], fragment_idx=[2], mode="append", at=1)
    assert v.array_exists(dense.SPANS_PATH)
    assert dense.num_rows(v) == 2


def test_a_stale_vlen_stamp_does_not_hide_a_dense_index(tmp_path):
    """A caller committing ``layout: vlen_manifests_v1`` over a dense index."""
    _, d = _level(tmp_path, "d", "dense")
    write_object_manifests(d, chunk_coords=[[0, 0, 0]], fragment_idx=[3], mode="append")
    d.write_array_meta(OBJECT_INDEX, {
        "zv_array": "object_index", "num_objects": 1, "num_present": 1,
        "sid_ndim": 3, "layout": "vlen_manifests_v1",
    })
    assert read_all_object_manifests(d) == [[((0, 0, 0), 3)]]


def test_the_appender_writes_dense_too(tmp_path):
    from zarr_vectors.core.streaming import ObjectIndexAppender

    manifests = _manifests(np.random.default_rng(3), 40)
    _, v = _level(tmp_path, "v", None)
    _, d = _level(tmp_path, "d", "dense")
    for lg in (v, d):
        write_object_index(lg, {i: manifests[i] for i in range(10)}, 3)
        with ObjectIndexAppender(lg, 6, 3, list(range(6))) as app:
            app.append([manifests[i] for i in range(10, 40)], [1] * 30)
    _same_reads(v, d)


def test_a_million_single_fragment_objects_build_no_blob_each(tmp_path, monkeypatch):
    """BRIDGE R4: the write builds no per-object Python object, and the
    CSR read is a gather, not a decode."""
    from zarr_vectors.encoding import fragments

    def _forbidden(*a, **k):
        raise AssertionError("a per-object manifest blob was built")

    monkeypatch.setattr(fragments, "encode_object_manifest_blocks", _forbidden)
    monkeypatch.setattr(fragments, "encode_object_manifests_csr", _forbidden)
    _, d = _level(tmp_path, "d", "dense")
    n = 1_000_000
    rng = np.random.default_rng(4)
    coords = rng.integers(0, 2, (n, 3)).astype(np.int64)
    frags = rng.integers(0, 5000, n).astype(np.int64)
    t = time.perf_counter()
    assert write_object_manifests(
        d, chunk_coords=coords, fragment_idx=frags, mode="append",
    ) == (0, n)
    _commit(d, n)
    csr = read_all_object_manifests_csr(d)
    elapsed = time.perf_counter() - t
    np.testing.assert_array_equal(csr.offsets, np.arange(n + 1))
    np.testing.assert_array_equal(csr.chunk_coords, coords)
    np.testing.assert_array_equal(csr.fragment_idx, frags)
    assert elapsed < 10, elapsed


def test_validation_passes_a_dense_store_and_catches_a_bad_span(tmp_path):
    from zarr_vectors.types.polylines import write_polylines
    from zarr_vectors.validate import validate

    path = tmp_path / "p" / "s.zarrvectors"
    path.parent.mkdir()
    root = create_store(
        path, bounds=([0.0] * 3, [100.0] * 3), chunk_shape=(50.0,) * 3,
        geometry_types=["polyline"], manifest_layout="dense",
    )
    rng = np.random.default_rng(5)
    write_polylines(
        root, [rng.uniform(5, 95, (6, 3)).astype("float32") for _ in range(8)],
        chunk_shape=(50.0,) * 3,
    )
    lg = get_resolution_level(open_store(str(path), mode="r+"), 0)
    assert dense.is_dense(lg)
    assert validate(str(path), level=5).ok
    node = lg.zarr_group[dense.SPANS_PATH]
    node[0] = [0, 10_000]
    result = validate(str(path), level=3)
    assert not result.ok
    assert "manifest_blocks" in result.summary()
