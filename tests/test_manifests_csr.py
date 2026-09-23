"""Object manifests as CSR arrays: the per-object codec's bytes, no per-object Python."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from zarr_vectors.encoding.fragments import (
    decode_object_manifest_blocks,
    decode_object_manifests_csr,
    encode_object_manifest_blocks,
    encode_object_manifests_csr,
)
from zarr_vectors.exceptions import ArrayError


def _objects(rng, n, sid, *, max_blocks=4):
    """CSR arrays for n objects, some with none, and the per-object form."""
    counts = rng.integers(0, max_blocks + 1, n)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    total = int(offsets[-1])
    coords = rng.integers(-3, 9, (total, sid)).astype(np.int64)
    frags = rng.integers(0, 2**40, total).astype(np.int64)  # wide values
    per_object = [
        [(tuple(coords[b].tolist()), int(frags[b])) for b in range(offsets[o], offsets[o + 1])]
        for o in range(n)
    ]
    return coords, frags, offsets, per_object


@pytest.mark.parametrize("n", [0, 1, 5, 300])
@pytest.mark.parametrize("sid", [3, 4])
def test_csr_blobs_match_the_per_object_encoder(n, sid):
    coords, frags, offsets, per_object = _objects(np.random.default_rng(n + sid), n, sid)
    got = encode_object_manifests_csr(coords, frags, offsets, sid_ndim=sid)
    want = [encode_object_manifest_blocks(m, sid_ndim=sid) for m in per_object]
    assert got.dtype == object and len(got) == n
    assert list(got) == want


def test_one_block_per_object_by_default():
    coords = np.array([[0, 0, 0], [1, 2, 3]])
    got = encode_object_manifests_csr(coords, [5, 0])
    assert list(got) == [
        encode_object_manifest_blocks([((0, 0, 0), 5)], sid_ndim=3),
        encode_object_manifest_blocks([((1, 2, 3), 0)], sid_ndim=3),
    ]


def test_an_empty_manifest_keeps_its_zero_bytes():
    """A fixed-width byte string would strip these to b''."""
    got = encode_object_manifests_csr(np.empty((0, 3), np.int64), [], [0, 0, 0], sid_ndim=3)
    assert list(got) == [b"\0\0\0\0", b"\0\0\0\0"]


@pytest.mark.parametrize("sid", [3, 4])
def test_decode_matches_the_block_decoder(sid):
    coords, frags, offsets, per_object = _objects(np.random.default_rng(sid), 200, sid)
    blobs = list(encode_object_manifests_csr(coords, frags, offsets, sid_ndim=sid))
    got_off, got_cc, got_fi = decode_object_manifests_csr(blobs, sid)
    np.testing.assert_array_equal(got_off, offsets)
    np.testing.assert_array_equal(got_cc, coords)
    np.testing.assert_array_equal(got_fi, frags)


def _expanded(blob, sid):
    out = []
    for cc, ref in decode_object_manifest_blocks(blob, sid):
        if isinstance(ref, int):
            out.append((tuple(cc), ref))
        elif isinstance(ref, tuple):
            out.extend((tuple(cc), ref[0] + k) for k in range(ref[1]))
        else:
            out.extend((tuple(cc), int(v)) for v in ref)
    return out


def test_range_and_explicit_blocks_are_expanded_in_place():
    sid = 3
    blobs = [
        encode_object_manifest_blocks([((0, 0, 0), 4)], sid_ndim=sid),
        encode_object_manifest_blocks([((1, 0, 0), (2, 3)), ((1, 1, 0), 0)], sid_ndim=sid),
        b"\0\0\0\0",
        encode_object_manifest_blocks([((2, 0, 0), np.array([9, 1, 5]))], sid_ndim=sid),
        encode_object_manifest_blocks([((3, 3, 3), 7), ((3, 3, 4), 8)], sid_ndim=sid),
        b"",
    ]
    off, cc, fi = decode_object_manifests_csr(blobs, sid)
    for o, blob in enumerate(blobs):
        want = _expanded(blob, sid) if blob else []
        got = [
            (tuple(cc[r].tolist()), int(fi[r])) for r in range(off[o], off[o + 1])
        ]
        assert got == want, o


def test_a_blob_whose_length_looks_single_is_not_trusted_on_length():
    """[range, explicit(0), explicit(0)] is exactly 4 + 3 * width bytes long."""
    sid = 3
    width = sid * 8 + 9
    coords = struct.pack("<3q", 1, 2, 3)
    blob = struct.pack("<I", 3) + b"".join([
        coords + struct.pack("<B", 1) + struct.pack("<qq", 5, 2),       # 41 bytes
        coords + struct.pack("<B", 2) + struct.pack("<I", 0),           # 29 bytes
        coords + struct.pack("<B", 2) + struct.pack("<I", 0),           # 29 bytes
    ])
    assert len(blob) == 4 + 3 * width
    off, cc, fi = decode_object_manifests_csr([blob], sid)
    assert off.tolist() == [0, 2] and fi.tolist() == [5, 6]


def test_bad_input_is_refused():
    with pytest.raises(ArrayError, match=">= 0"):
        encode_object_manifests_csr([[0, 0, 0]], [-1])
    with pytest.raises(ArrayError, match="rank"):
        encode_object_manifests_csr([[0, 0]], [1], sid_ndim=3)
    with pytest.raises(ArrayError, match="header"):
        decode_object_manifests_csr([b"\x01"], 3)


# --- the store ------------------------------------------------------------


def _store(tmp_path, name):
    from zarr_vectors.building import create_store, get_resolution_level

    (tmp_path / name).mkdir()
    path = tmp_path / name / "s.zarrvectors"
    root = create_store(
        path, bounds=([0.0] * 3, [100.0] * 3), chunk_shape=(50.0,) * 3,
        geometry_types=["polyline"],
    )
    return path, get_resolution_level(root, 0)


def _commit(level, n, sid=3, layout=None):
    from zarr_vectors.constants import OBJECT_INDEX
    from zarr_vectors.core.arrays import OBJECT_INDEX_LAYOUT_V1

    level.write_array_meta(OBJECT_INDEX, {
        "zv_array": "object_index", "num_objects": n, "num_present": n,
        "sid_ndim": sid, "layout": layout or OBJECT_INDEX_LAYOUT_V1,
    })


def test_array_appends_leave_the_store_blob_appends_do(tmp_path):
    from tests._store_compare import assert_stores_identical
    from zarr_vectors.building import write_object_manifests

    path_a, lg_a = _store(tmp_path, "blobs")
    path_b, lg_b = _store(tmp_path, "arrays")
    rng = np.random.default_rng(11)
    n_total = 0
    for flush in range(5):
        coords, frags, offsets, per_object = _objects(rng, int(rng.integers(0, 40)), 3)
        at = n_total + (3 if flush == 2 else 0)  # one flush leaves a gap
        blobs = [encode_object_manifest_blocks(m, sid_ndim=3) for m in per_object]
        write_object_manifests(lg_a, blobs, mode="append", at=at)
        got = write_object_manifests(
            lg_b, chunk_coords=coords, fragment_idx=frags, manifest_offsets=offsets,
            mode="append", at=at,
        )
        assert got == (at, len(per_object))
        n_total = at + len(per_object)
        _commit(lg_a, n_total)
        _commit(lg_b, n_total)
    assert_stores_identical(path_a, path_b)


def test_the_csr_read_is_the_flattened_manifest_read(tmp_path):
    from zarr_vectors.building import read_all_object_manifests, write_object_manifests
    from zarr_vectors.core.arrays import read_all_object_manifests_csr

    _, lg = _store(tmp_path, "r")
    blobs = [
        encode_object_manifest_blocks([((0, 1, 0), 4), ((1, 1, 0), 2)], sid_ndim=3),
        encode_object_manifest_blocks([((1, 0, 0), (2, 3))], sid_ndim=3),
        b"\0\0\0\0",
        encode_object_manifest_blocks([((0, 0, 1), np.array([7, 1]))], sid_ndim=3),
        encode_object_manifest_blocks([((1, 1, 1), 9)], sid_ndim=3),  # residue
    ]
    write_object_manifests(lg, blobs)
    _commit(lg, 4)  # the last row is past the commit point
    csr = read_all_object_manifests_csr(lg)
    offsets, coords, frags = csr
    assert csr.object_ids is None
    want = read_all_object_manifests(lg)[:4]
    for o, manifest in enumerate(want):
        got = [
            (tuple(coords[r].tolist()), int(frags[r]))
            for r in range(offsets[o], offsets[o + 1])
        ]
        assert got == [(tuple(c), int(f)) for c, f in manifest]
    assert len(offsets) == 5


def _v2_level(tmp_path, ids):
    from zarr_vectors.core.arrays import write_object_index

    _, lg = _store(tmp_path, "v2")
    write_object_index(lg, {int(i): [((0, 0, 0), k)] for k, i in enumerate(ids)}, 3)
    return lg


def test_a_v2_append_extends_the_id_table(tmp_path):
    from zarr_vectors.building import write_object_manifests
    from zarr_vectors.core.arrays import (
        OBJECT_INDEX_LAYOUT_V2,
        object_ids_for_rows,
        object_rows_for_ids,
        read_object_manifests,
    )

    lg = _v2_level(tmp_path, [10, 20, 30])
    write_object_manifests(
        lg, chunk_coords=[[1, 1, 1], [1, 0, 1]], fragment_idx=[5, 6],
        ids=[40, 50], mode="append", at=3,
    )
    _commit(lg, 5, layout=OBJECT_INDEX_LAYOUT_V2)
    assert object_ids_for_rows(lg).tolist() == [10, 20, 30, 40, 50]
    assert object_rows_for_ids(lg, [50])[1].tolist() == [4]
    assert set(read_object_manifests(lg)) == {10, 20, 30, 40, 50}


def test_a_v2_append_without_ids_raises_unless_the_table_is_the_rows(tmp_path):
    from zarr_vectors.building import write_object_manifests
    from zarr_vectors.core.arrays import object_ids_for_rows

    sparse = _v2_level(tmp_path, [10, 20, 30])
    with pytest.raises(ArrayError, match="pass ids="):
        write_object_manifests(
            sparse, chunk_coords=[[0, 0, 0]], fragment_idx=[1], mode="append", at=3,
        )
    with pytest.raises(ArrayError, match="cannot pad"):
        write_object_manifests(
            sparse, chunk_coords=[[0, 0, 0]], fragment_idx=[1], ids=[99],
            mode="append", at=5,
        )

    dense_dir = tmp_path / "dense"
    dense_dir.mkdir()
    dense = _v2_level(dense_dir, [0, 1, 2])
    write_object_manifests(
        dense, chunk_coords=[[0, 0, 0]], fragment_idx=[1], mode="append", at=3,
    )
    assert object_ids_for_rows(dense).tolist()[:4] == [0, 1, 2, 3]


def test_positional_ids_must_be_the_rows(tmp_path):
    from zarr_vectors.building import write_object_manifests

    _, lg = _store(tmp_path, "v1")
    with pytest.raises(ArrayError, match="positionally"):
        write_object_manifests(
            lg, chunk_coords=[[0, 0, 0]], fragment_idx=[1], ids=[7], mode="append", at=0,
        )


def test_device_arrays_are_copied_off_once(tmp_path):
    from tests._fake_device import FakeDeviceArray
    from zarr_vectors import _xp
    from zarr_vectors.building import write_object_manifests

    _, lg = _store(tmp_path, "d")
    coords, frags, offsets, _ = _objects(np.random.default_rng(1), 20, 3)
    with _xp.count_transfers() as stats:
        write_object_manifests(
            lg, chunk_coords=FakeDeviceArray(coords), fragment_idx=FakeDeviceArray(frags),
            manifest_offsets=FakeDeviceArray(offsets), mode="append",
        )
    assert stats.d2h_calls == 3
