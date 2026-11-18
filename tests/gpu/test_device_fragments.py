"""``read_cells(fragments=True, device="cuda")`` decodes ZVFG blobs on the device.

The host decoder (:func:`zarr_vectors.encoding.fragments.decode_fragments`
and ``ChunkFragmentIndex.flat``) is the reference: every field of the
:class:`FragmentColumn` and every error must match it, for empty, range,
explicit and mixed indices, across byte boundaries of the bitmap, for
repeated, absent and off-grid cells, and for each way a blob can be
malformed.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from tests.gpu._cuda import CUDA, cupy
from zarr_vectors.core.arrays import list_chunk_keys
from zarr_vectors.core.cells import read_cells
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.encoding.fragments import encode_fragments
from zarr_vectors.exceptions import ArrayError
from zarr_vectors.types.points import write_points

pytestmark = CUDA

_FIELDS = ("cell_offsets", "starts", "counts", "index_offsets", "indices")


def _store(path, compressor, shard):
    if compressor == "zstd":
        pytest.importorskip("nvidia.nvcomp")
    rng = np.random.default_rng(0)
    write_points(
        str(path), rng.uniform(0, 100, (700, 3)).astype("float32"),
        chunk_shape=(25.0, 25.0, 25.0), bounds=[[0, 0, 0], [100, 100, 100]],
        compressor=compressor, shard_shape=shard,
    )
    lg = get_resolution_level(open_store(str(path), mode="r+"), 0)
    return lg, sorted(tuple(k) for k in list_chunk_keys(lg))


def _blobs(rng):
    """Fragment indices of every shape the layout has."""
    mixed = []
    for f in range(1000):
        kind = rng.integers(3)
        if kind == 0:
            mixed.append((int(rng.integers(0, 50)), int(rng.integers(0, 20))))
        elif kind == 1:
            mixed.append(rng.integers(0, 60, int(rng.integers(0, 6))))
        else:
            mixed.append(np.zeros(0, dtype=np.int64))
    return [
        encode_fragments([]),                                        # F = 0
        encode_fragments([(3, 4)]),                                  # one range
        encode_fragments([np.array([5, 1, 5])], force_explicit=True),  # one explicit
        encode_fragments([(i, 1) for i in range(8)]),                # a full byte
        encode_fragments([(i, 2) for i in range(67)]),               # bitmap past 8 bytes
        encode_fragments([np.array([2, 0])] * 9, force_explicit=True),  # repeated rows
        encode_fragments([(0, 4), np.array([12, 7, 19]), (20, 8)]),  # the spec's example
        encode_fragments([np.zeros(0, dtype=np.int64), (1, 0), np.array([4])]),
        encode_fragments(mixed),
    ]


def _write(lg, cells, blobs):
    for cell, blob in zip(cells, blobs):
        lg.write_bytes("vertex_fragments", ".".join(map(str, cell)), blob)


def _request(present):
    extra = [present[0], present[3], (0, 3, 3), (7, 7, 7), (-1, 0, 0)]
    return np.asarray(list(present) + extra, dtype=np.int64)


def _decode(compressor):
    return "device" if compressor == "zstd" else "auto"


def _assert_same(host, dev):
    def errors(batch):
        return sorted((e.array, e.chunk_key, e.message) for e in batch.errors)

    assert errors(host) == errors(dev)
    for name in _FIELDS:
        d, h = getattr(dev.fragments, name), getattr(host.fragments, name)
        assert isinstance(d, cupy.ndarray), name
        assert d.dtype == h.dtype == np.int64, name
        np.testing.assert_array_equal(d.get(), h, err_msg=name)


@pytest.mark.parametrize("compressor", [None, "zstd"])
@pytest.mark.parametrize("shard", [None, (2, 2, 2)])
def test_every_kind_of_index_decodes_like_the_host(tmp_path, compressor, shard):
    lg, present = _store(tmp_path / "p.zarrvectors", compressor, shard)
    blobs = _blobs(np.random.default_rng(1))
    assert len(present) > len(blobs)
    _write(lg, present, blobs)
    cells = _request(present)
    host = read_cells(lg, cells, ["vertices"], fragments=True)
    dev = read_cells(
        lg, cells, ["vertices"], device="cuda", decode=_decode(compressor), fragments=True,
    )
    assert not host.errors
    _assert_same(host, dev)
    assert dev.io[-1].array == "vertex_fragments" and dev.io[-1].path != "host"
    # Range and explicit fragments both present, explicit rows repeated.
    starts = host.fragments.starts
    assert (starts == -1).any() and (starts >= 0).any()
    assert len(host.fragments.indices) > len(np.unique(host.fragments.indices))


def test_a_batch_with_no_fragments(tmp_path):
    lg, present = _store(tmp_path / "p.zarrvectors", None, None)
    _write(lg, present, [encode_fragments([])] * len(present))
    for cells in (_request(present), np.zeros((0, 3), dtype=np.int64)):
        host = read_cells(lg, cells, ["vertices"], fragments=True)
        dev = read_cells(lg, cells, ["vertices"], device="cuda", fragments=True)
        _assert_same(host, dev)
        assert int(dev.fragments.cell_offsets[-1]) == 0


def _header(magic=0x5A564647, version=1, flags=0, f=3, r=2):
    return struct.pack("<IHHII", magic, version, flags, f, r)


def _malformed():
    good = encode_fragments([(0, 4), np.array([12, 7, 19]), (20, 8)])
    body = good[16:]
    flipped = bytearray(good)
    flipped[16] = 0x07                      # three range bits, R = 2
    decreasing = bytearray(encode_fragments(
        [np.array([1, 2]), np.array([3])], force_explicit=True,
    ))
    decreasing[24 + 4: 24 + 8] = struct.pack("<I", 9)   # offsets [0, 9, 3]
    nonzero = bytearray(encode_fragments([np.array([1])], force_explicit=True))
    nonzero[24:28] = struct.pack("<I", 1)                # offsets start at 1
    return {
        20: good[:10],
        21: _header(magic=0x12345678) + body,
        22: _header(version=2) + body,
        23: _header(flags=1) + body,
        24: _header(f=1, r=2) + body,
        25: _header(f=200, r=0) + b"\x00" * 8,
        26: good[:16 + 8 + 16],
        27: good[:16 + 8 + 32 + 4],
        28: good[:-1],
        29: bytes(flipped),
        30: bytes(decreasing),
        31: bytes(nonzero),
    }


@pytest.mark.parametrize("compressor", [None, "zstd"])
def test_malformed_blobs_are_the_hosts_errors(tmp_path, compressor):
    lg, present = _store(tmp_path / "p.zarrvectors", compressor, None)
    bad = _malformed()
    assert len(present) > len(bad) + 1
    _write(lg, present, list(bad.values()) + [encode_fragments([(0, 2), (2, 3)])])
    cells = _request(present)
    host = read_cells(lg, cells, ["vertices"], fragments=True)
    dev = read_cells(
        lg, cells, ["vertices"], device="cuda", decode=_decode(compressor), fragments=True,
    )
    assert len(host.errors) == len(bad)
    assert len({e.message for e in host.errors}) == len(bad) - 1   # 30 and 31 share one
    _assert_same(host, dev)
    # The well-formed cell after them still reads.
    assert int(dev.fragments.cell_offsets[-1]) >= 2
    for decode, device in ((_decode(compressor), "cuda"), ("host", "cpu")):
        with pytest.raises(ArrayError):
            read_cells(lg, cells, ["vertices"], device=device, decode=decode,
                       fragments=True, on_error="raise")
