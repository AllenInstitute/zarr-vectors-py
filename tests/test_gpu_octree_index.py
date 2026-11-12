"""The device reader's store path locates cells in both shard index formats.

``gpu/_fetch.py`` reads arrays that are not on a local filesystem through
the store's own ranged ``get``. It once parsed every shard's index as
``sharding_indexed``'s dense one, so a zarr-octree shard's sparse index
gave garbage byte ranges. These tests run without a GPU: cupy is replaced
by a numpy stand-in for the one upload ``_fetch_store`` does, and the
store is a :class:`~zarr.storage.WrapperStore` around a local one, which
the reader treats as remote.
"""

from __future__ import annotations

import itertools
import sys
import types

import numpy as np
import pytest
import zarr
from numcodecs import VLenBytes, Zstd
from zarr.codecs import BytesCodec, ShardingCodec, VLenBytesCodec, ZstdCodec
from zarr.storage import LocalStore, WrapperStore

from zarr_vectors.core._cells_on_disk import cell_source, locate_local

zarr_octree_codec = pytest.importorskip("zarr_octree.codec")

SHAPE = (8, 8, 8)
SHARD = (4, 4, 4)
KINDS = ["dense_end", "dense_start_nocrc", "octree", "octree_runs"]


@pytest.fixture
def fetch_module():
    """``zarr_vectors.gpu._fetch`` imported against a numpy stand-in for cupy,
    and forgotten afterwards so no other test sees the stand-in."""
    fake = types.ModuleType("cupy")
    fake.asarray = np.asarray
    fake.empty = np.empty
    fake.uint8 = np.uint8
    names = [n for n in sys.modules if n == "cupy" or n.startswith(("cupy.", "zarr_vectors.gpu"))]
    saved = {n: sys.modules.pop(n) for n in names}
    package = sys.modules["zarr_vectors"]
    had_gpu = hasattr(package, "gpu")
    old_gpu = getattr(package, "gpu", None)
    sys.modules["cupy"] = fake
    try:
        from zarr_vectors.gpu import _fetch

        yield _fetch
    finally:
        for n in [n for n in sys.modules if n == "cupy" or n.startswith(("cupy.", "zarr_vectors.gpu"))]:
            del sys.modules[n]
        sys.modules.update(saved)
        if had_gpu:
            package.gpu = old_gpu
        elif hasattr(package, "gpu"):
            del package.gpu


def _serializer(kind: str):
    inner = [VLenBytesCodec(), ZstdCodec()]
    if kind == "dense_end":
        return ShardingCodec(chunk_shape=(1, 1, 1), codecs=inner)
    if kind == "dense_start_nocrc":
        return ShardingCodec(chunk_shape=(1, 1, 1), codecs=inner, index_codecs=[BytesCodec()], index_location="start")
    if kind == "octree":
        return zarr_octree_codec.OctreeShardingCodec(chunk_shape=(1, 1, 1), codecs=inner, target=2_000)
    return zarr_octree_codec.OctreeShardingCodec(
        chunk_shape=(1, 1, 1), codecs=inner, layout="runs", min_weight=100, max_weight=2_000,
    )


def _values() -> dict[tuple[int, ...], bytes]:
    """Cells of varied sizes; shards with x >= 4 are never written, and
    about a third of the other cells are not either."""
    rng = np.random.default_rng(0)
    out = {}
    for c in itertools.product(range(4), range(8), range(8)):
        if rng.random() < 0.35:
            continue
        n = int(rng.choice([1, 50, 600, 3_000]))
        out[c] = rng.integers(0, 256, n, dtype=np.uint8).tobytes()  # incompressible
    return out


@pytest.fixture(params=KINDS)
def stored(request, tmp_path):
    """``(kind, local source, wrapped source, values)`` for one shard format."""
    kind = request.param
    vals = _values()
    arr = zarr.create_array(
        LocalStore(str(tmp_path / kind)), name="a", shape=SHAPE, chunks=SHARD,
        dtype="bytes", serializer=_serializer(kind), compressors=None,
    )
    idx = np.array(list(vals)).T
    data = np.empty(len(vals), dtype=object)
    data[:] = list(vals.values())
    arr.vindex[tuple(idx)] = data
    local = cell_source("a", zarr.open_array(LocalStore(str(tmp_path / kind), read_only=True), path="a"))
    wrapped = cell_source(
        "a", zarr.open_array(WrapperStore(LocalStore(str(tmp_path / kind), read_only=True)), path="a"),
    )
    assert local is not None and wrapped is not None and wrapped.local_root is None
    assert wrapped.index_format == ("zarr_octree" if kind.startswith("octree") else "dense")
    return kind, local, wrapped, vals


def _all_cells() -> np.ndarray:
    return np.array(list(itertools.product(*(range(s) for s in SHAPE))), dtype=np.int64)


def _decode(raw: bytes) -> bytes:
    return VLenBytes().decode(Zstd().decode(raw))[0]


def test_store_ranges_match_the_local_index_reader(fetch_module, stored):
    kind, local, wrapped, vals = stored
    cells = _all_cells()
    ranges, errors = fetch_module.locate_store(wrapped, cells)
    assert errors == {}
    flat, local_errors = locate_local([(local, cells)])
    assert local_errors == [{}]
    want = {i: (off, size) for _, i, _, off, size in flat}
    got = {i: (r.offset, r.size) for i, r in enumerate(ranges) if r is not None}
    assert got == want
    # Every stored cell is located, and nothing else: unwritten cells and missing shards are None.
    assert sorted(tuple(cells[i]) for i in got) == sorted(vals)
    # The ranges are the cells' bytes.
    root = local.local_root
    for i, r in enumerate(ranges):
        if r is None:
            continue
        with open(f"{root}/{r.key}", "rb") as fh:
            fh.seek(r.offset)
            raw = fh.read(r.size)
        assert len(raw) == r.size and _decode(raw) == vals[tuple(cells[i])]


def test_store_fetch_returns_each_cells_stored_bytes(fetch_module, stored):
    kind, _, wrapped, vals = stored
    rng = np.random.default_rng(1)
    cells = _all_cells()[rng.permutation(np.prod(SHAPE))[:300]]  # a scattered subset, in no order
    raw = fetch_module._fetch_store(wrapped, cells)
    assert raw.errors == {}
    data = np.asarray(raw.data)
    for c, start, size in zip(cells.tolist(), raw.starts, raw.sizes):
        if tuple(c) in vals:
            assert size > 0 and _decode(data[start:start + size].tobytes()) == vals[tuple(c)]
        else:
            assert size == 0


def test_a_corrupt_octree_index_is_an_error_per_cell_not_a_garbage_range(fetch_module, tmp_path):
    vals = _values()
    arr = zarr.create_array(
        LocalStore(str(tmp_path)), name="a", shape=SHAPE, chunks=SHARD,
        dtype="bytes", serializer=_serializer("octree"), compressors=None,
    )
    idx = np.array(list(vals)).T
    data = np.empty(len(vals), dtype=object)
    data[:] = list(vals.values())
    arr.vindex[tuple(idx)] = data
    path = tmp_path / "a" / "c" / "0" / "0" / "0"
    blob = bytearray(path.read_bytes())
    blob[-20] ^= 0xFF  # inside the index body: its checksum fails
    path.write_bytes(bytes(blob))
    src = cell_source("a", zarr.open_array(WrapperStore(LocalStore(str(tmp_path), read_only=True)), path="a"))
    cells = _all_cells()
    ranges, errors = fetch_module.locate_store(src, cells)
    in_bad = (cells < 4).all(axis=1)
    assert set(errors) == set(np.flatnonzero(in_bad).tolist())
    assert all("checksum" in e for e in errors.values())
    assert all(ranges[i] is None for i in np.flatnonzero(in_bad))
    # The other shards are unaffected.
    good = [i for i in np.flatnonzero(~in_bad) if tuple(cells[i]) in vals]
    assert good and all(ranges[i] is not None for i in good)


def test_a_range_past_the_end_of_an_object_is_a_short_read(fetch_module, stored, monkeypatch):
    kind, _, wrapped, vals = stored
    cells = np.array([next(iter(vals))], dtype=np.int64)
    real = fetch_module.locate_store

    def past_the_end(src, cells):
        ranges, errors = real(src, cells)
        r = ranges[0]
        r.offset += 10**9  # what a misread index gave: a store returns nothing, and no error
        return ranges, errors

    monkeypatch.setattr(fetch_module, "locate_store", past_the_end)
    raw = fetch_module._fetch_store(wrapped, cells)
    assert raw.sizes.tolist() == [0] and "short read" in raw.errors[0]
